import logging
from typing import Any, Dict, List, Tuple, Union

import backend.dataframeinterfaces as dfinterfaces
import backend.datapointstreamhandler as shandler
import backend.datastructures as ds
import backend.geohashsequenceinterpolator as ip
import backend.models as django_models
import backend.sequencescorer as seqscorer
import backend.staypointdetector as spdetector
import backend.utils as utils
import backend.xchanges as xchanges
import msgpack
import numpy as np

import pandas as pd
import utm

# from adtk.visualization import plot
from backend.typealias import CoordinatePair, Pattern, Sequence

from Bio import pairwise2
from django.conf import settings
from shapely.geometry import LineString


class BaseDetector:
    def __init__(self, precision: int, bits_per_char: int) -> None:
        self.precision = precision
        self.bits_per_char = bits_per_char
        # Set constant distance delta. Over large latitudinal distances this
        # value should be determined dynamically as the distance between
        # geohash grid cells change w.r.t to Earth's latitude
        self.distance_delta = ds.interpolation_distance_delta(
            precision=precision, bits_per_char=bits_per_char,
        )
        self.support: List = []
        self._fit_called = False

    @staticmethod
    def _to_dataframe(trajectories: List[ds.Trajectory],) -> pd.DataFrame:
        data = []
        for trajectory in trajectories:
            for datapoint in trajectory.datapoints:
                dict_ = {
                    "latitude": datapoint.latitude,
                    "longitude": datapoint.longitude,
                    "external_timestamp": datapoint.external_timestamp,
                    "uid": trajectory.uid,
                }
                data.append(dict_)
        return pd.DataFrame(data=data)

    def _check_vars(self) -> None:
        """Check and validate all class arguments on class instantiation."""
        if not isinstance(self.bits_per_char, int):
            error = f"ARG 'precision' is of type {type(self.bits_per_char)} \
                but should be of type 'int'!"
            try:
                self.bits_per_char = int(self.bits_per_char)
            except Exception:
                raise TypeError(error)
        else:
            valid_bits_per_char = [2, 4, 6]
            if self.bits_per_char not in valid_bits_per_char:
                raise ValueError(
                    f"The given 'bits_per_char' value is: \
                    {self.bits_per_char}. It needs to be either of \
                    {valid_bits_per_char}."
                )
        if not isinstance(self.precision, int):
            error = f"ARG 'precision' is of type {type(self.precision)} \
                but should be of type 'int'!"
            try:
                self.precision = int(self.precision)
            except Exception:
                raise TypeError(error)
        else:
            valid_precision = {
                2: list(range(32)),
                4: list(range(16)),
                6: list(range(10)),
            }
            if self.precision not in valid_precision[self.bits_per_char]:
                raise ValueError(
                    f"The given 'precision' value is: {self.precision}. \
                    It needs to be either of \
                    {valid_precision[self.bits_per_char]}."
                )

    def _to_geohash(self, coordinate: Union[CoordinatePair, pd.Series]) -> str:
        """
        Convert a latitude/longitude coordinate pair to a geohash.

        Args:
            coordinate (CoordinatePair): A latitude/longitude coordinate pair.

        Raises:
            ValueError: If the required input size is wrong.

        Returns:
            str: A geohash value.
        """
        if isinstance(coordinate, list):
            if not len(coordinate) == 2:
                raise ValueError(
                    "The given list/tuple must contain a (latitude, \
                    longitude)-coordinate pair!",
                )
            latitude = coordinate[0]
            longitude = coordinate[1]
        elif isinstance(coordinate, pd.Series):
            latitude = coordinate["latitude"]
            longitude = coordinate["longitude"]
        else:
            raise ValueError(
                f"The method did not expect to get input of type: {coordinate} \
                The method expects a list or a pandas dataframe row!"
            )
        return ds.geohash_encode(
            latitude=latitude,
            longitude=longitude,
            precision=self.precision,
            bits_per_char=self.bits_per_char,
        )

    # # TODO: _xPLOTTING_
    # def plot_support(
    #     self,
    #     center: CoordinatePair,
    #     opacity: float = 0.10,
    #     map_: Union[None, folium.Map] = None,
    #     ) -> folium.Map:
    #     if map_ is None:
    #         map_ = utils.setup_map(center = center)
    #     if len(self.support) > 0:
    #         # TODO: Make compatible with 'DisorientationDetector0'
    #         for _, seq in self.support:
    #             map_ = self.plot_sequence(
    #                 seq,
    #                 center = center,
    #                 opacity = opacity,
    #                 map_ = map_,
    #             )
    #         return map_
    #     else:
    #         return None

    # # TODO: _xPLOTTING_
    # def plot_trajectory_anomalies(
    #     self,
    #     df: pd.DataFrame,
    #     anomaly_indices,
    #     center: CoordinatePair,
    #     weight: float = 2.0,
    #     opacity: float = 1,
    #     map_: Union[None, folium.Map] = None,
    #     ) -> folium.Map:
    #     if map_ is None:
    #         map_ = utils.setup_map(center)
    #     anomalous_datapoints = []
    #     datapoints = []
    #     for _, row in df.iterrows():
    #         dp = datastructures.DataPoint(
    #             latitude = row["latitude"],
    #             longitude = row["longitude"],
    #             external_timestamp = str(row.index),
    #         )
    #         for anomaly_index in anomaly_indices:
    #             if row.name == anomaly_index:
    #                 anomalous_datapoints.append(dp)
    #         datapoints.append(dp)
    #     trajectory_ = datastructures.Trajectory(anomalous_datapoints)
    #     trajectory = datastructures.Trajectory(datapoints)
    #     trajectory.plot(
    #         center = center,
    #         map_ = map_,
    #     )
    #     for datapoint in trajectory_.datapoints:
    #         color = "#" + str(utils.color_hex())
    #         map_ = datapoint.plot(
    #             center = center,
    #             map_ = map_,
    #             color = color,
    #         )
    #     return map_


class iBDD(BaseDetector):  # noqa
    """A class that implements the iBDD method."""

    def __init__(
        self,
        sampling_rate: float,
        max_seperation_distance: float,
        theta: float,
        rd_max: int = 3,
        rl_max: int = 3,
        with_heuristic: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Algorithm parameters
        self.sampling_rate = sampling_rate
        self.max_seperation_distance = max_seperation_distance
        self.theta = theta
        self.rl_max = rl_max
        self.rd_max = rd_max
        # Internal state
        self.current_sequence: List = []
        self.working_sequence: List = []
        self.working_support: List = []
        self.rl: int = 0
        self.rd: int = 0
        # Subsequence finder heuristic
        self.with_heuristic = with_heuristic
        # Check the validity of the given input
        self._check_vars()

    @staticmethod
    def estimate_max_seperation_distance(
        data: List[float], mdeviations: float = 3.0
    ) -> float:
        """
        Estimate max seperation distance parameter.

        Note:
            Estimate the maximum allowed seperation distance between two \
            datapoints. The maximum allowed seperation distance between two \
            datapoints is used to classify the latest datapoint as being noisy.

        Args:
            data (List[float]): A list of distance deltas (distances, in \
                meters, between consecutive datapoints).
            mdeviations (float, optional): Number of standard deviations from
                the mean. Defaults to 3.

        Returns:
            float: An estimate of the maximum allowed seperation distance \
                between two datapoints.
        """
        max_seperation_distance = np.mean(data) + mdeviations * np.std(data)
        return float(max_seperation_distance)

    @staticmethod
    def estimate_sampling_rate(
        data: List[CoordinatePair],
    ) -> Union[None, float]:
        """
        Estimate the sampling rate.

        Note:
            Estimate the sampling rate such that we can use it to classify \
            whether a list of datapoint constitute a critical point.

        Args:
            data (List[CoordinatePair]): A list of UTM coordinates \
                (transformed latitude, longitude coordinates).


        Returns:
            float: The sampling rate (datapoint per meter).
        """
        if len(data) <= 1:
            return None
        else:
            sampling_rate = LineString(data).length / len(data)
            return float(sampling_rate)

    def update(self, datapoint: ds.DataPoint) -> Tuple[int, int]:
        new, token = self._tokenize(datapoint=datapoint)
        if new == 1 and token is not None:
            label, self.working_support = self._detection(
                token=token, working_support=self.working_support,
            )
        else:
            label = 0
        return label, new

    def fit(self, trajectories: List[ds.Trajectory]) -> None:
        self.support = self._extract_support(
            trajectories=trajectories,
            precision=self.precision,
            bits_per_char=self.bits_per_char,
        )
        self._fit_called = True

    def detect(
        self, trajectories: List[ds.Trajectory], user: str,
    ) -> Union[None, pd.DataFrame]:
        if self._fit_called:
            df = self._to_dataframe(trajectories=trajectories)
            df = df.merge(
                self._bulk_detect(df=df, user=user),
                on=["external_timestamp", "uid"]
                # Merge on the timestamp column
            ).set_index("external_timestamp")
            return df
        else:
            logging.info("Method .fit needs to be called first!")
            return None

    def _check_vars(self) -> None:
        """Check and validate all class arguments on class instantiation."""
        super()._check_vars()
        if not isinstance(self.sampling_rate, float):
            error = f"ARG 'sampling_rate' is of type \
                {type(self.sampling_rate)} but should be of type 'float'!"
            try:
                self.sampling_rate = float(self.sampling_rate)
            except Exception:
                raise TypeError(error)
        else:
            if self.sampling_rate <= 0:
                raise ValueError(
                    f"The given 'sampling_rate' has value: \
                    {self.sampling_rate}. It needs to be larger than 0."
                )

        if not isinstance(self.max_seperation_distance, float):
            error = f"ARG 'max_seperation_distance' is of type \
                {type(self.max_seperation_distance)} but should be of type \
                'float'!"
            try:
                self.max_seperation_distance = float(
                    self.max_seperation_distance
                )
            except Exception:
                raise TypeError(error)
        else:
            if self.max_seperation_distance <= 0:
                raise ValueError(
                    f"The given 'max_seperation_distance' has value: \
                    {self.max_seperation_distance}. It needs to be larger \
                    than 0."
                )
        if not isinstance(self.theta, float):
            error = f"ARG 'theta' is of type {type(self.theta)} \
                but should be of type 'float'!"
            try:
                self.theta = float(self.theta)
            except Exception:
                raise TypeError(error)
        else:
            if not (self.theta >= 0 and self.theta <= 1):
                raise ValueError(
                    f"The given 'theta' value is: {self.theta}. \
                    It needs to be a percentage between and 0 and 1."
                )
        if not isinstance(self.rl_max, int):
            error = f"ARG 'rl_max' is of type {type(self.rl_max)} \
                but should be of type 'int'!"
            try:
                self.rl_max = int(self.rl_max)
            except Exception:
                raise TypeError(error)
        else:
            if self.rl_max <= 0:
                raise ValueError(
                    f"The given 'rl_max' has value: {self.rl_max}. \
                    It needs to be larger than 0."
                )
        if not isinstance(self.rd_max, int):
            error = f"ARG 'rd_max' is of type {type(self.rd_max)} \
                but should be of type 'int'!"
            try:
                self.rd_max = int(self.rd_max)
            except Exception:
                raise TypeError(error)
        else:
            if self.rd_max <= 0:
                raise ValueError(
                    f"The given 'rd_max' has value: {self.rd_max}. \
                    It needs to be larger than 0."
                )
        if not isinstance(self.with_heuristic, bool):
            error = f"ARG 'with_heuristic' is of type \
                {type(self.with_heuristic)} but should be of type 'bool'!"
            try:
                self.rd_max = int(self.rd_max)
            except Exception:
                raise TypeError(error)
        else:
            if self.rd_max <= 0:
                raise ValueError(
                    f"The given 'rd_max' has value: {self.rd_max}. \
                    It needs to be larger than 0."
                )

    def _subfinder(self, sequence0: Sequence, sequence1: Sequence) -> Sequence:
        """
        Given a sequence determine if it is a subsequence of a known sequence.

        Args:
            sequence0 (Sequence): A known historically recorded sequence.
            sequence1 (Sequence): A new on-going sequence that we need to \
                determine if it is a subsequence of a known sequence.

        Returns:
            Sequence: A sequence that consists of the common parts of given \
                input sequence0 and sequence1.
        """
        matches = set()
        for i in range(len(sequence0)):
            if (
                sequence0[i] == sequence1[0]
                and sequence0[i : i + len(sequence1)] == sequence1  # noqa
            ):
                matches.add(tuple(sequence1))
        return list(matches)  # type: ignore

    def _find_support(
        self, current_sequence: Sequence, support: List[Sequence]
    ) -> List[Sequence]:
        support_set = []
        for sequence in support:
            values = self._subfinder(sequence, current_sequence)
            # If we find that 'current_sequence' is a part of 'sequence'
            # then copy the whole 'sequence' to working support set
            if len(values) > 0:
                support_set.extend([sequence])
        return support_set

    def _calculate_support_score(
        self, current_sequence: Sequence, support: List[Sequence]
    ) -> Tuple[float, List[Sequence]]:
        """
        Calculate the support score for the on-going trajectory/sequence.

        Args:
            current_sequence (Sequence): An on-going trajectory that has been \
                converted into a sequence of traversed geohash grid cells.
            support (List[Sequence]): The current set of historically recorded \
                sequences of geohash grid cells that support the current \
                on-going sequence.

        Returns:
            Tuple[float, List[Sequence]]: The support score that determines if \
                the on-going trajectory/sequence contains disorientation \
                behavior.
        """
        support_set = self._find_support(current_sequence, support)
        if len(support) == 0:
            support_score = 0.0
        else:
            support_score = len(support_set) * 1.0 / len(support)
        return support_score, support_set

    def _test_support_score(self) -> None:
        t1: Sequence = ["1", "2", "3", "4"]
        t2: Sequence = ["1", "2", "3", "4", "5"]
        t3: Sequence = ["2", "3", "4", "5"]
        t0: List[Sequence] = [t1, t2, t3]
        test1: Sequence = ["1", "2", "3"]
        test2: Sequence = ["2", "3", "4"]
        support_score0, _ = self._calculate_support_score(test1, t0)
        assert np.isclose(support_score0, 0.666666)
        support_score1, _ = self._calculate_support_score(test2, t0)
        assert np.isclose(support_score1, 1.000000)

    def _traversed_distance(self, datapoints: List[ds.DataPoint]) -> float:
        """
        Return the distance traversed by a given ordered list of datapoints.

        Args:
            datapoints (List[ds.DataPoint]): A chronologically ordered list of \
                datapoints.

        Returns:
            float: The distance of the path in meters.
        """
        transformed = []
        if len(datapoints) < 2:
            return 0.0
        else:
            for datapoint in datapoints:
                value = utm.from_latlon(datapoint.latitude, datapoint.longitude)
                transformed.append([value[0], value[1]])
            ls = LineString(transformed)
            return ls.length

    def _is_noisy(
        self, datapoints: List[ds.DataPoint], datapoint: ds.DataPoint,
    ) -> bool:
        """
        Determine if a datapoint is noisy.

        Args:
            datapoints (List[ds.DataPoint]): A list of the latest \
                chronologically ordered datapoints.
            datapoint (ds.DataPoint): The datapoint that we have to determine \
                is noisy or not.

        Returns:
            bool: Whether the given datapoint is noisy or not.
        """
        dpm1 = datapoints[-1]
        dpm0 = datapoint
        value = (
            self._traversed_distance([dpm0, dpm1])
            >= self.max_seperation_distance
        )
        return value

    def _is_critical(self, datapoints: List[ds.DataPoint]) -> bool:
        """
        Determine if list of datapoints constitute a critical point.

        Args:
            datapoints (List[ds.DataPoint]): A list of the latest \
                chronologically ordered datapoints.

        Returns:
            bool: Whether the given list of datapoints constitute a critical \
                datapoint.
        """
        value = (
            self._traversed_distance(datapoints) <= self.distance_delta / 3.0
        ) and (
            len(datapoints) <= self.distance_delta / (5.0 * self.sampling_rate)
        )
        return value

    def _tokenize(
        self, datapoint: ds.DataPoint
    ) -> Tuple[int, Union[None, Dict[str, Any]]]:
        """
        Convert a new datapoint in an on-going trajectory to a geohash.

        Args:
            datapoint (ds.DataPoint): A new datapoint in the on-going \
                trajectory.

        Returns:
            Tuple[int, Union[None, Dict[str, Any]]]: A tuple consisting of an \
                integer value that indicates whether a new geohash grid cell \
                is traveresed. If a new grid cell is traversed the geohash \
                value is returned as well, otherwise it is None.
        """
        new = 0
        token = self._to_geohash([datapoint.latitude, datapoint.longitude])
        if len(self.current_sequence) == 0:
            self.current_sequence.append(
                {"token": token, "datapoints": [datapoint]}
            )
        else:
            if token == self.current_sequence[-1]["token"]:
                self.current_sequence[-1]["datapoints"].append(datapoint)
            else:
                self.current_sequence.append(
                    {"token": token, "datapoints": [datapoint]}
                )
                current_datapoints = self.current_sequence[-2]["datapoints"]
                is_invalid = self._is_noisy(
                    current_datapoints, datapoint
                ) or self._is_critical(current_datapoints)
                if is_invalid:
                    if len(self.current_sequence) >= 3:
                        if (
                            self.current_sequence[-1]["token"]
                            == self.current_sequence[-3]["token"]
                        ):
                            self.current_sequence = self.current_sequence[:-1]
                else:
                    new = 1
        if len(self.current_sequence) >= 2:
            return new, self.current_sequence[-2]
        else:
            return new, None

    def _detection(
        self, token: Dict[str, Any], working_support: List[Sequence]
    ) -> Tuple[int, List[Sequence]]:
        self.working_sequence.append(token)
        temp_seq = [_["token"] for _ in self.working_sequence]
        if self.with_heuristic is True:
            if len(self.working_sequence) >= 3:
                temp_seq = temp_seq[-3:]
        support_score, ttsupp = self._calculate_support_score(
            temp_seq, working_support,
        )
        if support_score < self.theta:
            is_looping = False
            if len(self.working_sequence) >= 3:
                if self.working_sequence[-1] == self.working_sequence[-3]:
                    self.working_sequence = [
                        self.working_sequence[-2],
                        self.working_sequence[-1],
                    ]
                    self.rl += 1
                    is_looping = True
            if is_looping is False:
                self.working_sequence = []
                self.rd += 1
            working_support = self.support
        else:
            working_support = ttsupp
        if self.rd >= self.rd_max or self.rl >= self.rl_max:
            # Label as outlying or looping datapoint
            label = 1
            # Reset counters
            self.rd = 0
            self.rl = 0
        else:
            # Label as normal datapoint
            label = 0
        return label, working_support

    def _extract_support(
        self,
        trajectories: List[ds.Trajectory],
        precision: int,
        bits_per_char: int,
    ) -> List[Sequence]:
        support = []
        dxs = []
        # sampling_rates = list()
        # Estimate parameters based on all trajectories
        # for trajectory in trajectories:
        #     dxs.extend([datapoint.dx for datapoint in trajectory.datapoints])
        #     coordinates = [
        #         datastructures.latlon_to_utm(datapoint)
        #         for datapoint in trajectory.datapoints
        #     ]
        #     rate = self.estimate_sampling_rate(
        #         data = coordinates,
        #     )
        #     if rate is not None:
        #         sampling_rates.append(rate)
        # Determine the max seperation distance between two points
        # self.max_seperation_distance = self.estimate_max_seperation_distance(
        #   dxs
        # )
        # Determine average sampling rate, i.e. the average number of datapoints
        # recorded per meter
        # self.sampling_rate = np.mean(sampling_rates)
        for trajectory in trajectories:
            # Determine the max seperation distance between two points
            dxs = [datapoint.dx for datapoint in trajectory.datapoints]
            max_seperation_distance = self.estimate_max_seperation_distance(dxs)
            # Determine the sampling rate, i.e. number of datapoints per meter
            coordinates = [
                ds.latlon_to_utm(datapoint)
                for datapoint in trajectory.datapoints
            ]
            sampling_rate = self.estimate_sampling_rate(data=coordinates)
            if sampling_rate is not None:
                # Reset class parameters and internal state
                self = self.__class__(
                    sampling_rate=sampling_rate,
                    # NOTE: This value is not used at this step, so just set
                    # to 1...
                    theta=1.0,
                    max_seperation_distance=max_seperation_distance,
                    precision=precision,
                    bits_per_char=bits_per_char,
                )
                # Reset class parameters and internal state
                # _self = self.__class__(
                #     sampling_rate = self.sampling_rate,
                #     # NOTE: This value is not used at this step, so just set
                #     # to 1...
                #     theta = 1.0,
                #     max_seperation_distance = self.max_seperation_distance,
                #     precision = precision,
                #     bits_per_char = bits_per_char,
                # )
                is_new = []
                for datapoint in trajectory.datapoints:
                    new, token = self._tokenize(datapoint)
                    if new == 1 and token is not None:
                        is_new.append(token)
                sequence = []
                for cell in self.current_sequence:
                    sequence.append(cell["token"])
                support.append(sequence)
        return support

    def _bulk_detect(self, df: pd.DataFrame, user: str) -> pd.DataFrame:
        temp_df = pd.DataFrame()
        for _, df_ in df.groupby("uid"):
            # Reset vars
            self.rd = 0
            self.rl = 0
            self.working_sequence = []
            self.current_sequence = []
            self.working_support = []
            # Run detection alg.
            ndf = self._detect(df=df_)
            temp_df = temp_df.append(ndf)
        return temp_df

    def _detect(self, df: pd.DataFrame) -> pd.DataFrame:
        # Initialize
        data: List[Dict[str, Any]] = []
        self.working_support = self.support
        # Loop through the location data for each trajectory
        for _, row in df.iterrows():
            datapoint = ds.DataPoint(
                latitude=row["latitude"],
                longitude=row["longitude"],
                external_timestamp=row["external_timestamp"],
            )
            label, _ = self.update(datapoint=datapoint)
            # If trajectory is anomalous
            if label == 1:
                data.append(
                    {
                        "anomaly": True,
                        "external_timestamp": row["external_timestamp"],
                    }
                )
            else:
                data.append(
                    {
                        "anomaly": False,
                        "external_timestamp": row["external_timestamp"],
                    }
                )
        temp_df = pd.DataFrame(data=data)
        if temp_df.shape[0] == 0:
            # Return an empty dataframe with columns defined, but no data
            return pd.DataFrame(columns=["anomaly", "external_timestamp"])
        else:
            ndf = df.merge(temp_df, on=["external_timestamp"],)
            return ndf[["external_timestamp", "anomaly", "uid"]]


class DisorientationDetector(BaseDetector):
    """A new approach to detecting disorientation behavior."""

    def __init__(
        self,
        threshold_high: float,
        window_size: str,
        sequence_interpolator: Union[
            ip.GeohashSequenceInterpolator, ip.NaiveGeohashSequenceInterpolator,
        ],
        spd: Union[None, spdetector.StayPointDetector] = None,
        fns: Union[
            None, Dict[str, Tuple[seqscorer.ScoringFunction, float]],
        ] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sequence_interpolator = sequence_interpolator
        self.threshold_high = threshold_high
        self.window_size = window_size
        self.spd = spd
        self.scorer = seqscorer.ExtendedInMemorySequenceScorer(
            **{
                # TODO: Pass exchange from same scope as 'spd' so 'exchange'
                #       can be shared among the two objects
                "exchange": xchanges.InMemoryMessageExchangeWrapper(),
                "fns": fns,
            }
        )
        self.df = None
        # Check the validity of the given input
        self._check_vars()
        # TODO: Temporary variable for debugging...
        self.database: List = []

    def fit(
        self, trajectories: List[ds.Trajectory], min_frequency: int = 1,
    ) -> None:
        self.df, self.database = self._process(trajectories)
        self.support = self._extract_support(
            database=self.database,
            min_frequency=min_frequency,
            closed=True,
            generator=False,
        )
        self._fit_called = True

    def combine_columns(self, row):
        # middle + end
        return row["between"] + [row["geohash"]]

    def detect(
        self, trajectories: List[ds.Trajectory], user: str,
    ) -> Union[None, pd.DataFrame]:
        if self._fit_called:
            # df = self._to_dataframe(trajectories = trajectories)
            # df = self._process_detect(df).set_index("external_timestamp")
            # df = self._to_dataframe(trajectories = trajectories)
            df = self._process_detect(
                trajectories
            )  # .set_index("external_timestamp")
            return df
        else:
            logging.warning("Method .fit needs to be called first!")
            return None

    def score_sequence(
        self, in_seq: Sequence, support: Pattern
    ) -> Union[None, float]:
        return self.scorer.score_sequence(
            # NOTE: Make sure a python list is passed else Biopython will crash!
            sequence=list(in_seq),
            uid="placeholder",
        )

    # def plot_score(self, df: pd.DataFrame, anomalies: pd.DataFrame) -> Any:
    #     return plot(
    #         df,
    #         anomaly=anomalies,
    #         ts_linewidth=1,
    #         ts_markersize=5.0,
    #         anomaly_color="red",
    #     )

    def _check_vars(self) -> None:
        """Check and validate all class arguments on class instantiation."""
        super()._check_vars()
        if not isinstance(self.threshold_high, float):
            error = f"ARG 'threshold_high' is of type {type(self.threshold_high)} \
                but should be of type 'float'!"
            try:
                self.theta = float(self.threshold_high)
            except Exception:
                raise TypeError(error)
        else:
            if not (self.threshold_high > 0 and self.threshold_high <= 1):
                raise ValueError(
                    f"The given 'threshold_high' value is: \
                    {self.threshold_high}. It needs to be a percentage between \
                    and 0 and 1."
                )
        if not isinstance(self.window_size, str):
            error = (
                f"ARG 'window_size' is of type {type(self.window_size)} "
                + "but should be of type 'str'!"
            )
            raise ValueError(error)

    def _process(
        self, trajectories: List[ds.Trajectory]
    ) -> Tuple[pd.DataFrame, List[Sequence]]:
        """
        Process a list of trajectories in bulk by turning them into sequences.

        Note:
            Assume that a complete dataset is given such that trajectory data \
            can be processed in bulk: Segmented trajectories are given as \
            input and can be processed one at a time. We only need to do a \
            single pass over each trajectory and do not need to store any \
            temporary data.

        Args:
            trajectories (List[ds.Trajectory]): A list of trajectories to \
                process.

        Returns:
            Tuple[pd.DataFrame, List[Sequence]]: A tuple consisting of a \
                dataframe that contains all the trajectory data. A database \
                of the trajectories that have been turned into geohash \
                sequences are returned as well.
        """

        database = []
        traj_df = None
        for trajectory in trajectories:
            exchange = xchanges.InMemoryMessageExchangeWrapper()
            data = dfinterfaces.GeohashDataFrame.handle(
                exchange=exchange,
                user="temporary",
                datapoints=trajectory.datapoints,
                precision=self.precision,
                bits_per_char=self.bits_per_char,
            )
            df = data["dataframe"]
            df["uid"] = trajectory.uid
            _seq = df.apply(self.combine_columns, axis=1, args=())
            support_seq = [item for sublist in _seq for item in sublist]
            database.append(support_seq)
            if traj_df is None:
                traj_df = df
            else:
                traj_df = traj_df.append(df, verify_integrity=True)
        return traj_df, database

    def _process_detect(
        self, trajectories: List[ds.Trajectory]
    ) -> pd.DataFrame:
        """
        Determine the trajectories contained in a list are anomalous.

        Args:
            trajectories (List[ds.Trajectory]): A list of trajectories to \
                process and detect anomalies in.

        Returns:
            pd.DataFrame: A pandas dataframe containing the anomaly detection \
                results.
        """
        for trajectory in trajectories:
            exchange = xchanges.InMemoryMessageExchangeWrapper()
            f1 = seqscorer.ScoringFunction(pairwise2.align.localxx)
            fns = {"f1": (f1, 1.00)}
            # TODO: Object also instantiated in __init__ method and not used!
            scorer = seqscorer.ExtendedInMemorySequenceScorer(
                **{"exchange": exchange, "fns": fns}
            )
            scorer._add_support(
                name=scorer.get_key(
                    obj_type="support", uid="testing"
                ),  # TODO: Find a better name for arg 'uid'
                support=msgpack.dumps(self.support),
            )
            data = dfinterfaces.ScoringDataFrame.handle(
                exchange=exchange,
                # TODO: Find a better name for arg 'user'
                user="testing",
                ss=scorer,
                datapoints=trajectory.datapoints,
                precision=self.precision,
                bits_per_char=self.bits_per_char,
                threshold_high=self.threshold_high,
            )
            df = data["dataframe"]
            df["uid"] = trajectory.uid
        return df  # TODO: What if None?

    def _extract_support(
        self,
        database: List[Sequence],
        min_frequency: int,
        closed: bool = True,
        generator: bool = False,
    ) -> List[Pattern]:
        return self.scorer._create_support(
            # TODO: Find a better name for arg 'uid'
            uid="placeholder",
            sequences=database,
            min_frequency=min_frequency,
            closed=closed,
            generator=generator,
        )

    # def _detect(self, df: pd.DataFrame, user: str) -> pd.DataFrame:
    #     threshold_ad = ThresholdAD(
    #         high = self.threshold_high,
    #         low = self.threshold_low,
    #     )
    #     anomalies = threshold_ad.detect(df["anomaly_score"])
    #     # Check if anomalies are inside a 'stay point cluster'. We should
    #     # discard these as a user is stationary inside these areas.
    #     anomalies = anomalies.to_frame(name = "anomaly")
    #     df_ = df.merge(anomalies, on = ["external_timestamp"])
    #     # TODO: Do not detect anomalies when a person is inside a geofenced
    #     # region
    #     # if self.spd is not None:
    #     #     for i, row in df_[df_["anomaly"] == True].iterrows():
    #     #         datapoints = [
    #     #             datastructures.DataPoint(
    #     #                 latitude = row["latitude"],
    #     #                 longitude = row["longitude"],
    #     #                 external_timestamp = i,
    #     #             )
    #     #         ]
    #     #         polygons = self.spd.extract_geofences(
    #     #             datapoints = datapoints,
    #     #             **{"user": user},
    #     #         )
    #     #         for datapoint in datapoints:
    #     #             point = Point([datapoint.latitude, datapoint.longitude])
    #     #             for polygon, _ in polygons:
    #     #                 if polygon.contains(point):
    #     #                     df_.loc[i, "anomaly"] = False
    #     return df_


def bulk_insert_datapoint(
    df: pd.DataFrame,
    user: str,
    exchange: Union[
        xchanges.RedisMessageExchangeWrapper,
        xchanges.InMemoryMessageExchangeWrapper,
    ],
) -> None:
    # Method for inserting location data into the ingress queue
    def insert_datapoint(
        datapoint: ds.DataPoint, user: str, pipe: Any, name: str,
    ) -> None:
        pipe.rpush(
            utils.get_queue(
                namespace=settings.STREAMHANDLER_NAMESPACE,
                user=user,
                name=name,
            ),
            datapoint.to_msgpack(),
        )

    for column_name in ["longitude", "latitude", "external_timestamp"]:
        if column_name not in df.columns:
            raise KeyError(
                f"Column name {column_name} is not present in the dataframe!"
            )
    # Create a pipeline to execute operations in bulk
    if exchange.client is not None:
        pipe = exchange.client.pipeline()
        # Convert dataframe rows to datapoints and insert into Redis named queue
        for _, row in df.iterrows():
            datapoint = ds.DataPoint(
                latitude=row["latitude"],
                longitude=row["longitude"],
                # Give the datapoint the most recently known timestamp
                external_timestamp=row["external_timestamp"],
                user=user,
            )
            insert_datapoint(
                datapoint=datapoint,
                user=user,
                pipe=pipe,
                name=settings.INGRESS_QUEUE,
            )
        _ = pipe.execute()
    else:
        logging.warning("Object exchange.client is None!")
        raise ValueError("Object exchange.client is None!")


def bulk_process_datapoint(
    user: str,
    dpsh: Union[
        shandler.DataPointStreamHandler,
        shandler.InMemoryDataPointStreamHandler,
    ],
    exchange: Union[
        xchanges.RedisMessageExchangeWrapper,
        xchanges.InMemoryMessageExchangeWrapper,
    ],
) -> List[ds.Trajectory]:
    if exchange.client is not None:
        queue_name = utils.get_queue(
            namespace=settings.STREAMHANDLER_NAMESPACE,
            user=user,
            name=settings.INGRESS_QUEUE,
        )
        queue_size = exchange.client.llen(name=queue_name)
        if queue_size == 0:
            # If no datapoints are present in the INGRESS queue then
            # raise an error...
            raise ValueError(
                f"The INGRESS queue with name {queue_name} has size 0 no \
                datapoints. No datapoints can thus be processed."
            )
        else:
            # If there are datapoints present in the INGRESS queue then
            # process all the datapoints
            for _ in range(queue_size):
                dpsh.update(user=user)
            # Flush the last data contained in the INGRESS queue
            dpsh.finish_trajectory(user=user)
            # Retrieve trajectories from storage and return them
            if isinstance(
                exchange, xchanges.InMemoryMessageExchangeWrapper
            ) and isinstance(dpsh, shandler.InMemoryDataPointStreamHandler):
                return _processed_datapoints_from_memory(
                    user=user, exchange=exchange, dpsh=dpsh,
                )
            elif isinstance(
                exchange, xchanges.RedisMessageExchangeWrapper
            ) and isinstance(dpsh, shandler.DataPointStreamHandler):
                return _processed_datapoints_from_database(
                    user=user, exchange=exchange, dpsh=dpsh,
                )
            else:
                exchange_error = ""
                if not isinstance(
                    exchange, xchanges.InMemoryMessageExchangeWrapper
                ) or not isinstance(
                    exchange, xchanges.RedisMessageExchangeWrapper
                ):
                    # If an 'exchange' of an unknown type is passed, then raise
                    # anerror...
                    exchange_error = f"The input ARG 'exchange' is of type \
                        {type(exchange)}, but should be of type \
                        'RedisMessageExchangeWrapper' or \
                        'InMemoryMessageExchangeWrapper'."
                streamhandler_error = ""
                if not isinstance(
                    exchange, shandler.InMemoryDataPointStreamHandler
                ) or not isinstance(exchange, shandler.DataPointStreamHandler):
                    streamhandler_error = f"The input ARG 'dpsh' is of type \
                        {type(dpsh)}, but should be of type \
                        'DataPointStreamHandler' or \
                        'InMemoryDataPointStreamHandler'."
                raise ValueError(exchange_error + "\n" + streamhandler_error)
    else:
        logging.warning("")
        raise ValueError("")


def _processed_datapoints_from_memory(
    user: str,
    exchange: xchanges.InMemoryMessageExchangeWrapper,
    dpsh: shandler.InMemoryDataPointStreamHandler,
) -> List[ds.Trajectory]:
    # If the 'InMemoryMessageExchange' and 'InMemoryDataPointStreamHandler'
    # has been used then retrieve trajectories from memory
    trajectories = []
    for key in reversed(dpsh._dicts[user]):
        trajectory = dpsh._dicts[user][key]
        datapoints = []
        for datapoint in trajectory["datapoints"]:
            datapoints.append(ds.DataPoint.from_dict(datapoint))
        trajectory = ds.Trajectory(datapoints=datapoints)
        trajectories.append(trajectory)
    return trajectories


# TODO: Test function. How is UUID assigned. How do we distinguish between
#       trajectories?
def _processed_datapoints_from_database(
    user: str,
    exchange: xchanges.RedisMessageExchangeWrapper,
    dpsh: shandler.DataPointStreamHandler,
) -> List[ds.Trajectory]:
    # If the 'RedisMessageExchange' and 'DataPointStreamHandler'
    # has been used then retrieve trajectories from database
    trajectories = django_models.Trajectory.objects.filter(user=user,)
    trajectories = [ds.Trajectory.from_model(_) for _ in trajectories]
    return trajectories
