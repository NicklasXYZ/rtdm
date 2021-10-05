import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import backend.datastructures as ds
import backend.geohashsequenceinterpolator as ip
import backend.models as django_models
import backend.sequencescorer as seqscorer
import backend.utils as utils
import backend.xchanges as xchanges
import msgpack
import numpy as np
import pandas as pd
from django.conf import settings
from shapely.geometry import Point, Polygon

# TODO: Geohash interpolator object should be passed as
#       a function argument and not be hardcoded inside
#       the respective functions below...


class DataFrameInterface:
    @staticmethod
    def set_dfdata(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        data: Dict[str, Any],
        user: str,
        obj_type: str,
    ) -> None:
        """
        Set a dataframe in Redis.

        Args:
            data (Dict[str, Any]): A dictionary containing a dataframe and \
                possibly extra accompanying data that should be set in Redis.
            user (str): A unique identifier of a user.
            obj_type (str): A descriptor of the data that should be set in \
                Redis.

        Raises:
            ValueError: If the returned data is None.
        """
        if data is not None:
            # Make sure a dataframe 'df' key is present in the given dictionary
            if "dataframe" not in data:
                raise ValueError(
                    "No dataframe 'dataframe' key was contained in the given \
                    dictionary!"
                )
            else:
                # Create a copy of the data, so we do not change the original
                # that was passed in as a function argument
                # TODO: Figure out a better way to do this. Creating a copy of
                #  a large DataFrame can be costly.
                data_ = data.copy()
                # Serialize dataframe: pd.DataFrame --> String data
                data_["dataframe"] = data_["dataframe"].to_json()
                # Convert all dictionary data to bytes
                serialized_data = msgpack.dumps(data_)
                exchange.kv_set(
                    data=serialized_data, name=f"{user}:{obj_type}",
                )
        else:
            raise ValueError(
                "The given data is None. The given data can thus not be set."
            )

    @staticmethod
    def get_dfdata(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        user: str,
        obj_type: str,
        timeout: Union[None, float] = None,
    ) -> Union[None, Dict[str, Any]]:
        """
        Get a dataframe in Redis.

        Args:
            user (str): A unique identifier of a user.
            obj_type (str): A descriptor of the data that should be set in \
                Redis.
            timeout (Union[None, float], optional): The amount of time we are \
                ready to wait before timing out. Defaults to None.

        Returns:
            Union[None, pd.DataFrame]: A dataframe or None.
        """
        data = exchange.kv_get(
            name=f"{user}:{obj_type}", timeout=timeout  # type: ignore # noqa
        )
        if data is not None:
            # Convert all byte data to a dictionary
            deserialized_data = msgpack.loads(data)
            # Make sure a dataframe 'df' key is present in the dictionary
            if "dataframe" not in deserialized_data:
                # NOTE: This should not happen. For debugging purposes
                raise ValueError(
                    "No dataframe 'dataframe' key was contained in the \
                    retrieved dictionary!"
                )
            else:
                # Deserialize dataframe: String data --> pd.DataFrame
                deserialized_data["dataframe"] = pd.read_json(
                    deserialized_data["dataframe"], dtype=object,
                )
            return deserialized_data
        else:
            return None

    @staticmethod
    def clear_dfdata(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        user: str,
        obj_type: str,
    ) -> None:
        """
        Clear a dataframe in Redis.

        Args:
            user (str): A unique identifier of a user.
            obj_type (str): A descriptor of the data that should be set in \
                Redis.
        """
        if exchange.client is not None:
            exchange.client.delete(f"{user}:{obj_type}")
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)


# Class for namespacing operations on a certain type of dataframe.
# None of the methods in this class should be used by themselves.
# The a proper control flow, i.e., the proper sequence of operations
# on this type of dataframe should be handled in an outer scope.
class BreakPointDataFrame:
    @staticmethod
    def _scan_df(
        data: Dict[str, Any],
        start_index: Union[None, datetime],
        scan_index: Union[None, datetime],
        time_period: str,
    ) -> Tuple[int, Union[None, Dict[str, Any]]]:
        """
        Scan through a dataframe.

        Note:
            Scan through a dataframe of datapoints such that we can determine \
            if a user is entering or leaving a certain area containing stay \
            points belonging to the user. If the user is entering or leaving \
            a certain area then the current trajectory is segmented into two \
            or more sub-trajectories.

        Args:
            dataframe (pd.DataFrame): The current working dataframe containing \
                incoming datapoints.
            start_index (datetime): The start index of a datapoint that falls \
                within an area containing significant stay points belonging \
                to the user.
            scan_index (datetime): The current scan index indicating the last \
                datapoint that fell within an area containing significant stay \
                points belonging to the user.
            time_period (str): The size a moving window in minutes.

        Returns:
            Tuple[int, Dict[str, Any]]: A return code signaling whether a user \
                is outside, entering or leaving a certain area, along with \
                relevant data.
        """
        dataframe = data["dataframe"]
        return_code = 0
        return_data = None
        last_index = None
        if start_index is not None:
            label = None
            update_data = False
            dataframe_out = dataframe.iloc[dataframe.index >= start_index]
            windows = (
                dataframe_out[dataframe_out.index >= scan_index]
                .rolling(time_period)
                .label
            )
            # for window in windows:
            #     print("Window: ", window)
            for window in windows:
                last_index = window.index[-1]
                # TODO: Make sure null value in dataframe is always 'None'.
                #       Somehow 'None' is converted to a 'NaN' value after
                #       serialization and deserialization. For now just check
                #       for 'NaN' as well...
                values = window.isin([label, np.nan, None])
                if bool(values.all()) is False:
                    scan_index = values.iloc[~values.values].index[-1]
                    # TODO: Use the mode to determine if a person actually
                    # stayed near a cluster of stay points for a long time
                    # change_period = df_out.iloc[df_out.index <= scan_index]
                    # mode = change_period["label"].mode().\
                    # value_counts().to_dict()
                    update_data = True
                    return_code = 1
                elif bool(values.all()) is True:
                    return_code = 2
                    break
            if update_data is True:
                return_data = {
                    # The possible START timestamp index of when a person
                    # enters a certain region
                    "start_index": start_index,
                    # The current timestamp index we have gotten to. It is a
                    # possible END index of when a person might leave a
                    # certain region
                    "scan_index": scan_index,
                    # The timestamp index of the very last value in the last
                    # window
                    "last_index": last_index,
                    # The current working dataframe
                    "dataframe": dataframe_out,
                }
        return return_code, return_data

    @staticmethod
    def _create_dfdata(
        datapoints: List[ds.DataPoint], polygons: List[Polygon],
    ) -> Dict[str, Any]:
        if len(datapoints) == 0:
            raise ValueError(
                "The list of given datapoints has length 0. A dataframe \
                can thus not be created.",
            )
        data = []
        external_timestamps = []
        for datapoint in datapoints:
            dict_ = None
            point = Point([datapoint.latitude, datapoint.longitude])
            for polygon, label in polygons:
                if polygon.contains(point):
                    dict_ = {
                        "latitude": datapoint.latitude,
                        "longitude": datapoint.longitude,
                        # Set the label of the cluster of significant stay
                        # points, that that the datapoint falls within
                        "label": label,
                    }
                    # print("--> Inside polygon. label: ", label)
                    break
            if dict_ is None:
                dict_ = {
                    "latitude": datapoint.latitude,
                    "longitude": datapoint.longitude,
                    # The datapoint does not fall within any cluster of stay
                    # points so set the label to None
                    "label": None,
                }
            data.append(dict_)
            external_timestamps.append(datapoint.external_timestamp)
        # return {
        #     "dataframe": pd.DataFrame(
        #         data = data,
        #         index = external_timestamps
        #     ),
        # }
        return_df = {
            "dataframe": pd.DataFrame(
                data=data, index=external_timestamps, dtype=object,
            ),
        }
        return return_df

    @staticmethod
    def save_breakpoints(
        user: str,
        trajectory: str,
        return_data: Dict[str, Any],
        return_code: int,
    ) -> None:
        if return_code == 2:
            print("Saving 'BreakPoint' for trajectory: ", trajectory)
            traj_obj = django_models.Trajectory.objects.get(uid=trajectory)
            breakpoint_obj = django_models.BreakPoint(
                # Convert timestamp indices to strings
                start_index=return_data["start_index"],
                scan_index=return_data["scan_index"],
                last_index=return_data["last_index"],
                # TODO: Is it necessary to save this as well?
                label=return_code,
                # Associate the breakpoint with a user
                user=user,
                # Associate the breakpoint with a trajectory
                trajectory=traj_obj,
            )
            breakpoint_obj.save()
            # TODO: Should we  clear current sequence of geohash values?

    @staticmethod
    def _find_breakpoints(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        user: str,
        trajectory: str,
        data: Dict[str, Any],
        start_index: Union[None, datetime],
        scan_index: Union[None, datetime],
        time_period: str,
        save_breakpoints: bool = False,
    ) -> Union[None, datetime]:
        """
        Segment the current trajectory we are building in memory.

        Note:
            Segment the current trajectory we are building into two or more \
            sub-trajectories. We do this by scanning through the current \
            dataframe and determining the breakpoints in the trajectory.

        Args:
            exchange (RedisQueueMessageExchange):
            user (str):
            trajectory (str):
            dataframe (pd.DataFrame): The current working dataframe of \
                incoming datapoints.
            start_index (datetime): The start index of a datapoint that falls \
                within an area containing significant stay points belonging to \
                the user.
            scan_index (datetime): The current scan index indicating the last \
                datapoint that fell within an area containing significant stay \
                points belonging to the user.
            time_period (str): The size a moving window in minutes.
            save_breakpoints (bool):

        Returns:
            Union[None, datetime]: The last index (most recent timestamp) in \
                the dataframe that we have seen so far.
        """
        return_code, return_data = BreakPointDataFrame._scan_df(
            data=data,
            start_index=start_index,
            scan_index=scan_index,
            time_period=time_period,
        )
        if return_code != 0 and return_data is not None:
            # Remove possibly duplicate rows in the dataframe
            return_data["dataframe"] = return_data["dataframe"][
                ~return_data["dataframe"].index.duplicated(keep="last")
            ]
            last_index = return_data["last_index"]
            # Convert datetime objects to strings
            return_data["start_index"] = str(return_data["start_index"])
            return_data["scan_index"] = str(return_data["scan_index"])
            return_data["last_index"] = str(return_data["last_index"])
            # Set the return data in Redis
            DataFrameInterface.set_dfdata(
                exchange=exchange,
                user=user,
                obj_type=BreakPointDataFrame.__name__,
                data=return_data,
            )
            # TODO: Should be handled in an outer scope!
            # Save to the database all the breakpoints that we found
            if save_breakpoints:
                BreakPointDataFrame.save_breakpoints(
                    user=user,
                    trajectory=trajectory,
                    return_data=return_data,
                    return_code=return_code,
                )
            return last_index
        else:
            return None

    @staticmethod
    def _update_dfdata(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        user: str,
        datapoints: Union[None, List[ds.DataPoint]],
        polygons: List[Polygon],
        timeout: Union[None, float],
    ) -> Union[None, Dict[str, Any]]:
        data = DataFrameInterface.get_dfdata(
            exchange=exchange,
            user=user,
            obj_type=BreakPointDataFrame.__name__,
            timeout=timeout,
        )
        if data is None and datapoints is not None:
            data = BreakPointDataFrame._create_dfdata(
                datapoints=datapoints, polygons=polygons
            )
        elif data is not None and datapoints is not None:
            data_ = BreakPointDataFrame._create_dfdata(
                datapoints=datapoints, polygons=polygons
            )
            data["dataframe"].update(data_["dataframe"])
        elif data is not None and datapoints is None:
            # If df is not None and datapoints is None
            # TODO: Raise error
            pass
        else:
            # If df is None and datapoints is None
            # TODO: Raise error
            pass
        return data

    @staticmethod
    def _retrieve_indices(
        data: Dict[str, Any], last_index: Union[None, datetime] = None,
    ) -> Union[Tuple[None, None], Tuple[datetime, datetime]]:
        dataframe = data["dataframe"]
        if last_index is not None:
            dataframe = dataframe.iloc[dataframe.index > last_index]
        indices = dataframe.iloc[~dataframe.label.isnull().values]
        if indices.shape[0] > 0:
            start_index = indices.index[0]
            scan_index = start_index
        else:
            start_index = None
            scan_index = None
        return start_index, scan_index

    @staticmethod
    def _clear_dfdata(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        user: str,
    ) -> None:
        """
        Clear a dataframe in Redis.

        Args:
            user (str): A unique identifier of a user.
        """
        DataFrameInterface.clear_dfdata(
            exchange=exchange, user=user, obj_type=BreakPointDataFrame.__name__,
        )


class GeohashDataFrame:
    @staticmethod
    def _create_dfdata(
        datapoints: List[ds.DataPoint], precision: int, bits_per_char: int,
    ) -> Dict[str, Any]:
        data = []
        external_timestamps = []
        for datapoint in datapoints:
            geohashed_datapoint = datapoint.get_geohashed_datapoint(
                precision=precision, bits_per_char=bits_per_char,
            )
            dict_: Dict[str, Any] = {
                "latitude": datapoint.latitude,
                "longitude": datapoint.longitude,
                "geohash": geohashed_datapoint.geohash,
                "between": [],
            }
            data.append(dict_)
            external_timestamps.append(datapoint.external_timestamp)
        return {"dataframe": pd.DataFrame(data=data, index=external_timestamps)}

    @staticmethod
    def _extend_df(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        user: str,
        datapoints: Union[None, List[ds.DataPoint]],
        timeout: Union[None, float],
        precision: int,
        bits_per_char: int,
    ) -> pd.DataFrame:
        data = DataFrameInterface.get_dfdata(
            exchange=exchange,
            user=user,
            obj_type=GeohashDataFrame.__name__,
            timeout=timeout,
        )
        if data is None and datapoints is not None:
            data = GeohashDataFrame._create_dfdata(
                datapoints=datapoints,
                precision=precision,
                bits_per_char=bits_per_char,
            )
            # Reduce dataframe by removing consecutive and identical geohash
            # values
            # NOTE:
            # * shift(-1) to keep only the LAST occurrence (geohash, timestamp,
            # lat/lon location, etc).
            # data["dataframe"] = data["dataframe"].loc[
            #     data["dataframe"]["geohash"].shift(-1) != \
            #     data["dataframe"]["geohash"]
            # ]
            # NOTE:
            # * shift()   to keep only the FIRST occurrence (geohash, timestamp,
            # lat/lon location, etc).
            data["dataframe"] = data["dataframe"].loc[
                data["dataframe"]["geohash"].shift()
                != data["dataframe"]["geohash"]
            ]
            # Run through the whole dataframe the first time around
            for i in range(1, data["dataframe"].shape[0]):
                geohash0 = data["dataframe"]["geohash"].iat[i - 1]
                geohash1 = data["dataframe"]["geohash"].iat[i]
                (
                    sequence,
                    _,
                ) = ip.NaiveGeohashSequenceInterpolator._interpolate(
                    start_geohash=geohash0,
                    end_geohash=geohash1,
                    precision=precision,
                    bits_per_char=bits_per_char,
                )
                # Exclude start and end geohash values they are already
                # included in the 'geohash' column. We can get the start and
                # end geohash values from there
                data["dataframe"]["between"].iat[i] = sequence[1:-1]
            # Set the last index that we reached the first time around
            data["geohash_scan_index"] = str(data["dataframe"].index[-1])
        elif data is not None and datapoints is not None:
            data_ = GeohashDataFrame._create_dfdata(
                datapoints=datapoints,
                precision=precision,
                bits_per_char=bits_per_char,
            )
            try:
                # Get 'scan_index'
                # scan_index = data["scan_index"]
                # TODO: Fetching last row should be equivalent to this:
                # temp_df = data["dataframe"][data["dataframe"].index >= \
                # scan_index]
                # 'last_row' is the oldest so put it first then append
                # more recent rows...
                last_row = data["dataframe"].iloc[-1]
                temp_df = pd.DataFrame(
                    data=[last_row.to_dict()], index=[last_row.name]
                ).append(
                    data_["dataframe"],
                    # Raise ValueError if there are overlap betweeen the two
                    # dataframes, i.e. duplicate rows are present in the
                    # resulting dataframe
                    verify_integrity=True,
                )
                # Reduce dataframe by removing consecutive and identical
                # geohash values.
                # NOTE:
                # * shift(-1) to keep only the LAST occurrence (geohash,
                # timestamp, lat/lon location, etc).
                # temp_df = temp_df.loc[temp_df["geohash"].shift(-1) != \
                # temp_df["geohash"]]
                # NOTE:
                # * shift()   to keep only the FIRST occurrence (geohash,
                # timestamp, lat/lon location, etc).
                temp_df = temp_df.loc[
                    temp_df["geohash"].shift() != temp_df["geohash"]
                ]
                for i in range(1, temp_df.shape[0]):
                    geohash0 = temp_df["geohash"].iat[i - 1]
                    geohash1 = temp_df["geohash"].iat[i]
                    (
                        sequence,
                        _,
                    ) = ip.NaiveGeohashSequenceInterpolator._interpolate(
                        start_geohash=geohash0,
                        end_geohash=geohash1,
                        precision=precision,
                        bits_per_char=bits_per_char,
                    )
                    # Exclude start and end geohash values they are already
                    # included in the 'geohash' column. We can get the start
                    # and end geohash values from there
                    temp_df["between"].iat[i] = sequence[1:-1]
                # Set the last index that we reached the first time around
                data["geohash_scan_index"] = str(temp_df.index[-1])
                # Extend dataframe with new data
                data["dataframe"] = data["dataframe"].append(
                    # Skip first row. It is already contained in
                    # 'data['datafrane']'
                    temp_df.iloc[1:],
                    # Raise ValueError if there are overlap betweeen the two
                    # dataframes, i.e.
                    # duplicate rows are present in the resulting dataframe
                    verify_integrity=True,
                )
                print("Removing consecutive and identical geohash values")
            except ValueError as e:
                # logging.error(
                #     "Duplicate time indices was found in the dataframe. " + \
                #     "The Dataframe was thus not updated!",
                # )
                print(
                    "Duplicate time indices was found in dataframe. Dataframe \
                    was not updated!"
                )
                print(e)
        elif data is not None and datapoints is None:
            # If df is not None and datapoints is None
            # TODO: Raise error
            pass
        else:
            # If df is None and datapoints is None
            # TODO: Raise error
            pass
        return data

    @staticmethod
    def handle(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        user: str,
        datapoints: List[ds.DataPoint],
        precision: int,
        bits_per_char: int,
    ) -> pd.DataFrame:
        data = GeohashDataFrame._extend_df(
            exchange=exchange,
            user=user,
            datapoints=datapoints,
            timeout=0,
            precision=precision,
            bits_per_char=bits_per_char,
        )
        DataFrameInterface.set_dfdata(
            exchange=exchange,
            data=data,
            user=user,
            obj_type=GeohashDataFrame.__name__,
        )
        return data


def extract_sequence(row: pd.Series) -> List[str]:
    # Middle sequence + start geohash
    return row["between"] + [row["geohash"]]


# Class for namespacing operations on a certain type of dataframe.
# None of the methods in this class should be used by themselves.
# The a proper control flow, i.e., the proper sequence of operations
# on this type of dataframe should be handled in an outer scope
class ScoringDataFrame:
    @staticmethod
    def _create_dfdata(
        user: str,
        data: Dict[str, Any],
        ss: seqscorer.SequenceScorer,
        threshold_high: float,
    ) -> Dict[str, Any]:
        geohash_dataframe = data["dataframe"]
        scan_index = data["score_scan_index"]
        min_score = data["min_score"]
        windows = geohash_dataframe.rolling(settings.TIMESERIES_WINDOW_SIZE)
        timestamps = []
        rows = []
        last_window = None
        for window in windows:
            # Continue here if we have already processed data and should
            # continue processing new incoming data
            if scan_index is not None:
                if isinstance(scan_index, str):
                    scan_index = datetime.fromisoformat(scan_index)
                if window.index[-1] > scan_index:
                    sequence_list = window.apply(
                        extract_sequence, axis=1, args=()
                    )
                    sequence_flat = [
                        item for sublist in sequence_list for item in sublist
                    ]
                    timestamps.append(window.index[-1])
                    similarity_score = ss.score_sequence(
                        uid=user, sequence=sequence_flat
                    )
                    if similarity_score < min_score:
                        min_score = similarity_score
                    anomaly_score = 1.0 - min_score
                    rows.append(
                        {
                            "sequence": sequence_flat,
                            "similarity_score": similarity_score,
                            "anomaly_score": anomaly_score,
                            "anomaly": True
                            if anomaly_score > threshold_high
                            else False,
                            "latitude": window["latitude"].values[-1],
                            "longitude": window["longitude"].values[-1],
                        }
                    )
                # The data has already been processed...
                # ...
            # Else evaluate whatever data is passed and has not yet been
            # processed
            else:
                sequence_list = window.apply(extract_sequence, axis=1, args=())
                sequence_flat = [
                    item for sublist in sequence_list for item in sublist
                ]
                timestamps.append(window.index[-1])
                similarity_score = ss.score_sequence(
                    uid=user, sequence=sequence_flat
                )
                if similarity_score < min_score:
                    min_score = similarity_score
                anomaly_score = 1.0 - min_score
                rows.append(
                    {
                        "sequence": sequence_flat,
                        "similarity_score": similarity_score,
                        "anomaly_score": anomaly_score,
                        "anomaly": True
                        if anomaly_score > threshold_high
                        else False,
                        "latitude": window["latitude"].values[-1],
                        "longitude": window["longitude"].values[-1],
                    }
                )
            last_window = window
        # Set the very last seen index as the current 'scan_index'
        if last_window is not None:
            scan_index = last_window.index[-1]
        return {
            "min_score": min_score,
            "score_scan_index": scan_index,
            "dataframe": pd.DataFrame(data=rows, index=timestamps),
        }

    @staticmethod
    def _extend_dfdata(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        user: str,
        data_in: Dict[str, Any],
        ss: seqscorer.SequenceScorer,
        timeout: Union[None, float],
        threshold_high: float,
    ) -> Dict[str, Any]:
        data = DataFrameInterface.get_dfdata(
            exchange=exchange,
            user=user,
            obj_type=ScoringDataFrame.__name__,
            timeout=timeout,
        )
        if data is None and data_in is not None:
            # Set the scan_index to None to start with. The first dictionary
            # that is ever passed to this function is missing this key
            data_in["score_scan_index"] = None
            data_in["min_score"] = np.inf
            data = ScoringDataFrame._create_dfdata(
                user=user, data=data_in, ss=ss, threshold_high=threshold_high,
            )
        elif data is not None and data_in is not None:
            data_in["score_scan_index"] = data["score_scan_index"]
            data_in["min_score"] = data["min_score"]
            data_ = ScoringDataFrame._create_dfdata(
                user=user, data=data_in, ss=ss, threshold_high=threshold_high,
            )
            try:
                data["score_scan_index"] = data_["score_scan_index"]
                data["min_score"] = data_["min_score"]
                data["dataframe"] = data["dataframe"].append(
                    data_["dataframe"], verify_integrity=True,
                )
            except ValueError:
                logging.error(
                    "Duplicate time indices was found in the dataframe. "
                    + "The Dataframe was thus not updated!",
                )
        elif data is not None and data_in is None:
            # If df is not None and datapoints is None
            # TODO: Raise error
            pass
        else:
            # If df is None and datapoints is None
            # TODO: Raise error
            pass
        return data

    @staticmethod
    @utils.timing
    def handle(
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        user: str,
        datapoints: List[ds.DataPoint],
        ss: seqscorer.SequenceScorer,
        precision: int,
        bits_per_char: int,
        threshold_high: float = 0.6,
    ) -> pd.DataFrame:
        data_in = GeohashDataFrame.handle(
            exchange=exchange,
            user=user,
            datapoints=datapoints,
            precision=precision,
            bits_per_char=bits_per_char,
        )
        data = ScoringDataFrame._extend_dfdata(
            exchange=exchange,
            user=user,
            data_in=data_in,
            ss=ss,
            timeout=0,
            threshold_high=threshold_high,
        )
        # Convert datetime objects to strings
        data["score_scan_index"] = str(data["score_scan_index"])
        DataFrameInterface.set_dfdata(
            exchange=exchange,
            data=data,
            user=user,
            obj_type=ScoringDataFrame.__name__,
        )
        # TODO: Integrate the detection method with the SOD backend...
        # if 1.0 - data["min_score"] > threshold_high:
        # ... Then the trajectory is anomalous.
        # ... Send a websocket message with an alert to relatives
        #     and possibly volunteers.
        return data
