import json
import logging
import multiprocessing
import time
import uuid
from typing import Any, Dict, List, Tuple, Union

import backend.datapointstreamhandler as shandler
import backend.datastructures as ds
import backend.disorientationdetectors as detectors
import backend.utils as utils
import backend.xchanges as xchanges
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut


def read_all(json_file: str) -> Dict[int, pd.DataFrame]:
    data_in = None
    with open(json_file, "r") as f:
        data_in = json.loads(f.read())
    batches = {}
    # Read in each batch of trajectories from the json file
    for i, batch in enumerate(data_in):
        df_batch = pd.read_json(batch, orient="split")
        _data = []
        # Deserialize each trajectory contained in the dataframe into its own
        # dataframe and nest it inside the 'df_batch' dataframe
        for _, row in df_batch.iterrows():
            df_traj = pd.read_json(row["df"], orient="split")
            _data.append(df_traj)
        df_batch["df"] = _data
        batches[i] = df_batch
    return batches
    

class LeaveOneOutDetectorEvaluator:
    def __init__(
        self,
        data_in: pd.DataFrame,
        user: str,
        method: str,
        params: Dict[str, Any],
        bits_per_char: int = 2,
    ) -> None:
        self.data_in = data_in
        self.user = user
        # A common geohashing parameter value used by both detection methods
        self._bits_per_char = bits_per_char
        # A dictionary of parameters specific to a particular detection method
        self.params = params
        self.method = method
        self.user = user
        self.data_out: Any = None
        # Define a variable to keep track of whether the method "set_labels()"
        # has been called
        self._set_labels_called = False

    def set_labels(
        self, anom_uids: List, df_batch: pd.DataFrame
    ) -> pd.DataFrame:
        for i, row in df_batch.iterrows():
            if isinstance(row["uid"], str):
                uid = uuid.UUID(row["uid"])
            else:
                uid = row["uid"]
            if uid in anom_uids:
                df_batch.loc[i, "outcome"] = True
        self._set_labels_called = True
        return df_batch

    def compute_auc_score(self, df_batch: pd.DataFrame) -> Union[None, float]:
        if self._set_labels_called:
            y_vals = [int(_) for _ in df_batch["outcome"].values]
            y_pred = [int(_) for _ in df_batch["anomalous"].values]
            try:
                score = roc_auc_score(y_vals, y_pred)
            except ValueError as e:
                logging.info(
                    f"A problem was encountered in the calculation of the ROC \
                    AUC score: {e}"
                )
                score = 0.0
            return score
        else:
            logging.info("Method .fit needs to be called first!")
            return None

    def extract_trajectories(
        self, uids: List, trajectories: List[ds.Trajectory]
    ) -> Tuple[List[ds.Trajectory], List[ds.Trajectory]]:
        train_uids, test_uids = uids
        train_traj, test_traj = [], []
        for tajectory in trajectories:
            # TODO: Check that uid is of type uuid.UUID
            if tajectory.uid in test_uids["uid"].values:
                test_traj.append(tajectory)
            elif tajectory.uid in train_uids["uid"].values:
                train_traj.append(tajectory)
            else:
                raise ValueError(
                    "Trajectory UUID not in any train or test set!"
                )
        return train_traj, test_traj

    def call_dd0(
        self, train_traj: List[ds.Trajectory], test_traj: List[ds.Trajectory],
    ) -> Tuple[pd.DataFrame, List, float, float]:
        dd0 = detectors.iBDD(
            # NOTE: Set to 0. Value is estimated/determined internally
            sampling_rate=0,
            # NOTE: Set to 0. Value is estimated/determined internally
            max_seperation_distance=0,
            theta=self.params["theta"],
            precision=self.params["precision"],
            bits_per_char=self._bits_per_char,
        )
        fit_time_start = time.perf_counter()
        dd0.fit(trajectories=train_traj)
        fit_time_end = time.perf_counter()
        fit_time = fit_time_end - fit_time_start

        ###
        ###
        dxs = []
        sampling_rates = []
        # Estimate parameters based on all trajectories in training set
        for trajectory in train_traj:
            dxs.extend([datapoint.dx for datapoint in trajectory.datapoints])
            coordinates = [
                ds.latlon_to_utm(datapoint)
                for datapoint in trajectory.datapoints
            ]
            rate = dd0.estimate_sampling_rate(data=coordinates,)
            if rate is not None:
                sampling_rates.append(rate)
        # Determine the max seperation distance between two points
        max_seperation_distance = dd0.estimate_max_seperation_distance(dxs)
        # Determine average sampling rate, i.e. the average number of datapoints
        # recorded per meter
        sampling_rate = np.mean(sampling_rates)
        ###
        ###
        dd0.sampling_rate = sampling_rate
        dd0.max_seperation_distance = max_seperation_distance
        ###
        ###

        detect_time_start = time.perf_counter()
        df = dd0.detect(trajectories=test_traj, user=self.user)
        detect_time_end = time.perf_counter()
        detect_time = detect_time_end - detect_time_start

        return df, dd0.support, fit_time, detect_time

    def call_dd1(
        self, train_traj: List[ds.Trajectory], test_traj: List[ds.Trajectory],
    ) -> Tuple[pd.DataFrame, List, float, float]:
        dd1 = detectors.DisorientationDetector(
            threshold_high=self.params["threshold_high"],
            window_size=self.params["window_size"],
            spd=None,
            precision=self.params["precision"],
            bits_per_char=self._bits_per_char,
            sequence_interpolator=self.params["sequence_interpolator"],
        )
        fit_time_start = time.perf_counter()
        dd1.fit(
            trajectories=train_traj, min_frequency=self.params["min_frequency"],
        )
        fit_time_end = time.perf_counter()
        fit_time = fit_time_end - fit_time_start

        detect_time_start = time.perf_counter()
        df = dd1.detect(trajectories=test_traj, user=self.user)
        detect_time_end = time.perf_counter()
        detect_time = detect_time_end - detect_time_start

        return df, dd1.support, fit_time, detect_time

    def cross_validation(self, trajectories: List[ds.Trajectory],) -> List[List[pd.DataFrame]]:
        df = detectors.BaseDetector._to_dataframe(trajectories=trajectories,)
        for column_name in ["uid"]:
            if column_name not in df.columns:
                raise KeyError(
                    f"Column name {column_name} is not present in the \
                    dataframe!"
                )
        trajectory_uids = [{"uid": uid_} for uid_ in df.uid.unique()]
        # Dataframe containing all trajectory ids
        uid_df = pd.DataFrame(data=trajectory_uids)
        # Split the dataframe into training and testing dataframes:
        loo = LeaveOneOut()
        # List of tuples containing indices specifying a training set
        # and a testing set
        loo_splits = list(loo.split(uid_df))
        train_test = self.retrieve_train_test_sets(uid_df, loo_splits)
        return train_test

    def retrieve_train_test_sets(self, df: pd.DataFrame, kf_splits: Any) -> List[List[pd.DataFrame]]:
        list_ = []
        for item in kf_splits:
            train = item[0]
            test = item[1]
            train_ids = df.iloc[train]
            train_df = df.loc[df["uid"].isin(train_ids["uid"].values)]
            test_ids = df.iloc[test]
            test_df = df.loc[df["uid"].isin(test_ids["uid"].values)]
            list_.append([train_df, test_df])
        return list_

    def run_dd0(self, processes: int = 15) -> Dict[str, Any]:
        results = {}
        # Process each batch
        for i_index, batch in enumerate(self.data_in):
            df_batch = self.data_in[batch]
            trajectories = []
            dict_anom_start0 = {}
            dict_anom_start1 = {}
            # Process each trajectory in a batch
            for j_index in range(df_batch.shape[0]):
                temp_df = df_batch["df"].iloc[j_index]
                temp_df["external_timestamp"] = temp_df.index
                # Get the uid of the trajectory
                uid = temp_df["uid"].values[0]
                anom_start = temp_df[temp_df["anom_start"] == True]  # type: ignore # noqa
                if anom_start.shape[0] > 0:
                    dict_anom_start0[uuid.UUID(uid)] = anom_start.iloc[0][
                        "external_timestamp"
                    ]
                trajectory = self._dd0_preprocess(temp_df, uid)
                trajectories.append(trajectory)
            dfs = self.cross_validation(trajectories)
            anom_uids = []
            timings = {}
            detect_res = {}
            # Generate required function arguments up front, then execute 
            # function calls in parallel using the python multiprocessing
            # library
            arg_list = []
            for pair in dfs:
                train_traj, test_traj = self.extract_trajectories(
                    pair, trajectories
                )
                arg_list.append([train_traj, test_traj])
            with multiprocessing.Pool(processes=processes) as p:
                det_res = p.starmap(self.call_dd0, arg_list,)
            # Organize results
            for result, support, fit_time, detect_time in det_res:
                test_uid = result["uid"].values[0]
                timings[test_uid] = {
                    "fit_time": fit_time,
                    "detect_time": detect_time,
                }
                detect_res[test_uid] = {"support": support, "result": result}
                anomalous = result[result["anomaly"] == True]  # type: ignore # noqa
                if anomalous.shape[0] > 0:
                    dict_anom_start1[
                        anomalous["uid"].values[0]
                    ] = anomalous.index[0]
                anom_uid = np.unique(result[result["anomaly"] == True]["uid"])  # type: ignore # noqa
                anom_uids.extend(anom_uid)
            df_batch["fit_time"] = 0.0
            df_batch["detect_time"] = 0.0
            df_batch["outcome"] = False
            df_batch["delay"] = np.inf
            df_batch["detect_df"] = [
                pd.DataFrame() for _ in range(df_batch.shape[0])
            ]
            df_batch["support"] = [[] for _ in range(df_batch.shape[0])]
            df_out = self.set_labels(anom_uids, df_batch)
            for _, row in df_out.iterrows():
                if row["outcome"] is True and row["anomalous"] is True:
                    _uid = uuid.UUID(row["uid"])
                    if _uid in dict_anom_start0 and _uid in dict_anom_start1:
                        df_out.at[_, "delay"] = float(
                            (
                                dict_anom_start1[_uid]
                                - dict_anom_start0[_uid].to_pydatetime()
                            ).total_seconds()
                        )
                        # If the detection delay is negative then correct the 
                        # detection result by registering it as a false positive
                        if df_out.at[_, "delay"] < 0:
                            df_out.at[_, "outcome"] = False
            auc_score = self.compute_auc_score(df_out)
            # Set timings
            for k_index in range(df_out.shape[0]):
                _uid = uuid.UUID(df_out.at[k_index, "uid"])
                fit_time = timings[_uid]["fit_time"]
                detect_time = timings[_uid]["detect_time"]
                detect_df = detect_res[_uid]["result"]
                support = detect_res[_uid]["support"]
                df_out.at[k_index, "fit_time"] = fit_time
                df_out.at[k_index, "detect_time"] = detect_time
                df_out.at[k_index, "detect_df"] = detect_df
                df_out.at[k_index, "support"] = support
            delays = [
                _
                for _ in df_out[df_out["delay"] != np.inf]["delay"].values
                if _ >= 0.0
            ]
            if len(delays) >= 0:
                median_delay = np.median(delays)
            else:
                median_delay = np.inf
            results[i_index] = {
                "df": df_out,
                "score": auc_score,
                "median_fit_time": np.median(df_out["fit_time"].values),
                "median_detect_time": np.median(df_out["detect_time"].values),
                "median_delay": median_delay,
            }
        self.data_out = results
        return results

    def run_dd1(self, processes: int = 15) -> Dict[str, Any]:
        results = {}
        # Process each batch of synthetically generated trajectories
        for i_index, batch in enumerate(self.data_in):
            exchange = xchanges.InMemoryMessageExchangeWrapper()
            dpsh = shandler.InMemoryDataPointStreamHandler(
                exchange=exchange,
                **{"max_merging_distance": self.params["max_merging_distance"]},
            )
            df_batch = self.data_in[batch]
            trajectories = []
            dict_anom_start0 = {}
            dict_anom_start1 = {}
            # Process each trajectory in a batch
            for j_index in range(df_batch.shape[0]):
                temp_df = df_batch["df"].iloc[j_index]
                temp_df["external_timestamp"] = temp_df.index
                # Get the uid of the trajectory
                uid = temp_df["uid"].values[0]
                anom_start = temp_df[temp_df["anom_start"] == True]  # type: ignore # noqa
                if anom_start.shape[0] > 0:
                    dict_anom_start0[uuid.UUID(uid)] = anom_start.iloc[0][
                        "external_timestamp"
                    ]
                trajectory = self._dd1_preprocess(
                    temp_df=temp_df,
                    uid=uid,
                    user=self.user,
                    exchange=exchange,
                    dpsh=dpsh,
                )
                trajectories.append(trajectory)
            dfs = self.cross_validation(trajectories)
            anom_uids = []
            timings = {}
            detect_res = {}
            # Generate required function arguments up front, then execute 
            # function calls in parallel using the python multiprocessing
            # library
            arg_list = []
            for pair in dfs:
                train_traj, test_traj = self.extract_trajectories(
                    pair, trajectories
                )
                arg_list.append([train_traj, test_traj])
            with multiprocessing.Pool(processes=processes) as p:
                det_res = p.starmap(self.call_dd1, arg_list,)
            # Organize results
            for result, support, fit_time, detect_time in det_res:
                test_uid = result["uid"].values[0]
                timings[test_uid] = {
                    "fit_time": fit_time,
                    "detect_time": detect_time,
                }
                detect_res[test_uid] = {"support": support, "result": result}
                anomalous = result[result["anomaly"] == True]  # type: ignore # noqa
                if anomalous.shape[0] > 0:
                    dict_anom_start1[
                        anomalous["uid"].values[0]
                    ] = anomalous.index[0]
                anom_uid = np.unique(result[result["anomaly"] == True]["uid"])  # type: ignore # noqa
                anom_uids.extend(anom_uid)
            df_batch["fit_time"] = 0.0
            df_batch["detect_time"] = 0.0
            df_batch["outcome"] = False
            df_batch["delay"] = np.inf
            df_batch["detect_df"] = [
                pd.DataFrame() for _ in range(df_batch.shape[0])
            ]
            df_batch["support"] = [[] for _ in range(df_batch.shape[0])]
            df_out = self.set_labels(anom_uids, df_batch)
            for _, row in df_out.iterrows():
                if row["outcome"] is True and row["anomalous"] is True:
                    _uid = uuid.UUID(row["uid"])
                    if _uid in dict_anom_start0 and _uid in dict_anom_start1:
                        df_out.at[_, "delay"] = float(
                            (
                                dict_anom_start1[_uid]
                                - dict_anom_start0[_uid].to_pydatetime()
                            ).total_seconds()
                        )
                        # If the detection delay is negative then correct the 
                        # detection result by registering it as a false positive
                        if df_out.at[_, "delay"] < 0:
                            df_out.at[_, "outcome"] = False
            auc_score = self.compute_auc_score(df_out)
            # Set timings
            for k_index in range(df_out.shape[0]):
                _uid = uuid.UUID(df_out.at[k_index, "uid"])
                fit_time = timings[_uid]["fit_time"]
                detect_time = timings[_uid]["detect_time"]
                detect_df = detect_res[_uid]["result"]
                support = detect_res[_uid]["support"]
                df_out.at[k_index, "fit_time"] = fit_time
                df_out.at[k_index, "detect_time"] = detect_time
                df_out.at[k_index, "detect_df"] = detect_df
                df_out.at[k_index, "support"] = support
            delays = [
                _
                for _ in df_out[df_out["delay"] != np.inf]["delay"].values
                if _ >= 0.0
            ]
            if len(delays) >= 0:
                median_delay = np.median(delays)
            else:
                median_delay = np.inf
            results[i_index] = {
                "df": df_out,
                "score": auc_score,
                "median_fit_time": np.median(df_out["fit_time"].values),
                "median_detect_time": np.median(df_out["detect_time"].values),
                "median_delay": median_delay,
            }
        self.data_out = results
        return results

    def evaluate_detector(self) -> pd.DataFrame:
        if self.method == "dd0":
            df_out = self.run_dd0()
        elif self.method == "dd1":
            df_out = self.run_dd1()
        else:
            raise ValueError(
                "No valid detection method specified. The specified 'method' \
                needs to be either 'dd0' or 'dd1'."
            )
        return df_out

    def _dd0_preprocess(self, temp_df: pd.DataFrame, uid: str) -> ds.Trajectory:
        datapoints = []
        for _, row in temp_df.iterrows():
            datapoints.append(
                ds.DataPoint(
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    external_timestamp=row["external_timestamp"],
                )
            )
        datapoints[0].dx = 0
        for i in range(1, len(datapoints)):
            dx = ds.haversine_distance(
                lat_1=datapoints[i - 1].latitude,
                lon_1=datapoints[i - 1].longitude,
                lat_2=datapoints[i].latitude,
                lon_2=datapoints[i].longitude,
            )
            datapoints[i].dx = dx
        traj_obj = ds.Trajectory(datapoints=datapoints)
        traj_obj.uid = uuid.UUID(uid)
        return traj_obj

    # TODO: Update w.r.t. method 'bulk_process_datapoint' in file
    #       disorientationdetector.py
    def _dd1_preprocess(
        self,
        temp_df: pd.DataFrame,
        uid: str,
        user: str,
        exchange: xchanges.InMemoryMessageExchangeWrapper,
        dpsh: shandler.InMemoryDataPointStreamHandler,
    ) -> ds.Trajectory:
        if exchange.client is not None:
            detectors.bulk_insert_datapoint(
                df=temp_df, user=user, exchange=exchange
            )
            queue_size = exchange.client.llen(
                name=utils.get_queue(
                    namespace=settings.STREAMHANDLER_NAMESPACE,
                    user=user,
                    name=settings.INGRESS_QUEUE,
                )
            )
            for _ in range(queue_size):
                dpsh.update(user=user)
            # Flush last data contained in the ingress queue
            dpsh.finish_trajectory(user=user)
            key = next(reversed(dpsh._dicts[user]))
            traj_obj = dpsh._dicts[user][key]
            datapoints = []
            for datapoint in traj_obj["datapoints"]:
                datapoints.append(ds.DataPoint.from_dict(datapoint))
            traj_obj = ds.Trajectory(datapoints=datapoints)
            traj_obj.uid = uuid.UUID(uid)
            return traj_obj
        else:
            error = f"exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)
