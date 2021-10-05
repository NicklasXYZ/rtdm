import logging
from typing import Any, List, Union

import backend.dataframeinterfaces as dfinterfaces
import backend.datastructures as ds
import backend.managers as django_managers
import backend.models as django_models
import backend.staypointdetector as spdetector
import backend.utils as utils
import backend.xchanges as xchanges
from backend.typealias import Sequence
from shapely.geometry import Polygon


class BaseTrajectorySegmentor:
    def __init__(
        self,
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        min_chunk_size: int = 1,
        max_chunk_size: int = 1,
        on_key: str = django_models.BreakPoint._meta.label,
    ) -> None:
        self._exchange = exchange
        self._on_key = on_key
        self._bulk_manager = django_managers.BulkCreateManager(
            min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size,
        )

    def _check_vars(self) -> None:
        """Check and validate all class arguments on class instantiation."""


class TrajectorySegmentor(BaseTrajectorySegmentor):
    def __init__(self, time_period: str = "2.5T", **kwargs: Any) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            time_period (str): The minimum duration a user has to be \
                stationary before he/she is considered as staying in a \
                certain area. Defaults to "2.5T", which is 2.5 minutes.
        """
        self.time_period = time_period
        super().__init__(**kwargs)
        self._check_vars()

    def handle(self, user: str) -> None:
        pass

    def _check_vars(self) -> None:
        """Check and validate all class arguments on class instantiation."""
        super()._check_vars()
        if not isinstance(self.time_period, str):
            error = (
                f"ARG 'time_period' is of type {type(self.time_period)} "
                + "but should be of type 'str'!"
            )
            try:
                self.time_period = str(self.time_period)
            except Exception:
                raise TypeError(error)
        else:
            str_list = self.time_period.split("T")
            correct_format = True
            if len(str_list) == 2:
                if utils.str_is_number(str_list[0]) is False:
                    correct_format = False
            else:
                correct_format = False
            if correct_format is False:
                raise ValueError(
                    f"The given 'time_period' value is: {self.time_period}. \
                    It needs to follow a specific format. For example: \
                    '12.5T' is 12 minues and 30 seconds."
                )

    # def handle(self, user: str) -> None:
    #     """ Handling incoming data for a certain user...
    #     """
    #     # Hardcode percentile lower bound and radius for now...
    #     spd = StayPointDetector(
    #         percentile = 85,
    #         radius = 100., **{"exchange": self._exchange},
    #     )
    #     datapoints = self._exchange.client.lrange(
    #         _get_queue(
    #             # Get datapoints belonging to a certain user that have been
    #             # forwarded to the PROCESS_0 queue from the
    #             # 'DataPointStreamHandler' class.
    #             namespace = "DataPointStreamHandler",
    #             user = user,
    #             name = "process_0",
    #         ),
    #         # Start, End
    #         0, -1,
    #     )
    #     # pipe = self._exchange.client.pipeline()
    #     # key = _get_queue(
    #     #     # Get datapoints belonging to a certain user that have been
    #     #     # forwarded to the PROCESS_0 queue from the
    #     #     # 'DataPointStreamHandler' class.
    #     #     namespace = "DataPointStreamHandler",
    #     #     user = user,
    #     #     name = "process_0",
    #     # )
    #     # pipe.lrange(key, 0, -1)
    #     # pipe.delete(key)
    #     # datapoints, _ = pipe.execute()
    #     if datapoints is not None:
    #         trajectory = []
    #         # Process all trajectories
    #         for datapoint in datapoints:
    #             try:
    #                 trajectory.append(
    #                     DataPoint.from_msgpack(datapoint)
    #                 )
    #             except Exception:
    #                 polygons = spd.extract_geofences(
    #                     datapoints = trajectory, **{"user": user},
    #                 )
    #                 self._segment_trajectory_online(
    #                     user,
    #                     datapoints = trajectory,
    #                     polygons = polygons
    #                 )
    #                 self.clear_df(user = user)
    #                 trajectory = []
    #     else:
    #         logging.debug(
    #             "The data taken from the queue had value None!"
    #         )

    def _segment_trajectory_online(
        self,
        user: str,
        datapoints: List[ds.DataPoint],
        polygons: List[Polygon],
        save_breakpoints: bool = False,
    ) -> None:
        trajectory = datapoints[0].trajectory
        if trajectory is None:
            raise ValueError(
                "The unique identifier of the trajectory associated with a \
                datapoint is None. Breakpoints can thus not be associated with \
                a particular trajectory."
            )
        data = dfinterfaces.BreakPointDataFrame._update_dfdata(
            exchange=self._exchange,
            user=user,
            datapoints=datapoints,
            polygons=polygons,
            timeout=0,
        )
        if data is not None:
            (
                start_index,
                scan_index,
            ) = dfinterfaces.BreakPointDataFrame._retrieve_indices(
                data=data, last_index=None,
            )
            # print(start_index, scan_index)

            last_index = dfinterfaces.BreakPointDataFrame._find_breakpoints(
                exchange=self._exchange,
                user=user,
                trajectory=trajectory,
                data=data,
                start_index=start_index,
                scan_index=scan_index,
                time_period=self.time_period,
                save_breakpoints=save_breakpoints,
            )
            # Process the remaining part of the dataframe if necessary
            if last_index is not None:
                while True:
                    # If the person is outside of a cluster of stay points,
                    # then we do not need to keep a window of locations to
                    # determine if he is actually near a cluster of stay points
                    if last_index is None:
                        dfinterfaces.BreakPointDataFrame._clear_dfdata(
                            exchange=self._exchange, user=user,
                        )
                        break
                    # If no more data is currently available stop processing
                    # data for now
                    elif last_index >= data["dataframe"].index[-1]:  # type: ignore # noqa
                        # Save the index of the first datapoint inside a
                        # geofenced region
                        if save_breakpoints:
                            dfinterfaces.BreakPointDataFrame.save_breakpoints(
                                user=user,
                                trajectory=trajectory,
                                return_data={
                                    "start_index": start_index,
                                    "scan_index": start_index,
                                    "last_index": last_index,
                                },
                                return_code=2,
                            )
                        break
                    # Else keep processing datapoints until we have no more to
                    # process...
                    data = dfinterfaces.BreakPointDataFrame._update_dfdata(
                        exchange=self._exchange,
                        user=user,
                        datapoints=None,
                        polygons=polygons,
                        timeout=None,
                    )
                    if data is not None:
                        (
                            start_index,
                            scan_index,
                        ) = dfinterfaces.BreakPointDataFrame._retrieve_indices(
                            data=data, last_index=last_index,
                        )
                        last_index = dfinterfaces.BreakPointDataFrame._find_breakpoints(  # noqa
                            exchange=self._exchange,
                            user=user,
                            trajectory=trajectory,
                            data=data,
                            start_index=start_index,
                            scan_index=scan_index,
                            time_period=self.time_period,
                            save_breakpoints=save_breakpoints,
                        )
                    else:
                        logging.warning("DataFrame data is None!")
        else:
            logging.warning("DataFrame data is None!")

    # KEEP
    # def _segment_trajectory(self, user: str) -> None:
    #     pass
    #     # traj_obj = django_models.Trajectory.objects.last()
    #     # traj_obj.start_timestamp = dps[0].external_timestamp
    #     # traj_obj.end_timestamp = dps[-1].external_timestamp
    #     # traj_obj.save(update_fields = ["start_timestamp", "end_timestamp"])
    #     # for dp in dps:
    #     #     # Associate each datapoint with the trajectory
    #     #     setattr(dp, "trajectory", traj_obj)
    #     #     obj_list.append(dp)
    #     # self._bulk_manager.add_bulk(obj_list = obj_list)
    #     # # Save whenever we have 1 complete trajectory
    #     # self._bulk_manager.done()
    #     # # Remove the uuid of current 'raw' trajectory that we were building
    #     # self._exchange.client.delete(
    #     #     self._get_key(
    #     #         namespace = self.namespace,
    #     #         user = user,
    #     #         name = CURRENT_TRAJECTORY,
    #     #     )
    #     # )
    # KEEP

    # def _clear_db(self, user: str):
    #     # Clear all old subtrajectories that were previously found. Due to
    #     # new incoming data we might find new semantic breakpoints in
    #     # trajectories and thus find new and different subtrajectories.
    #     django_models.Trajectory.objects.filter(
    #         user = user,
    #         tag = "subtrajectory",
    #     ).delete()
    #     # Clear all existing trajectory breakpoints. These were previously
    #     # used to segment trajectories into subtrajectories.
    #     django_models.BreakPoint.objects.filter(
    #         user = user,
    #     ).delete()
    #     # Clear frequent sequential patterns of geohash sequences
    #     django_models.SupportSet.objects.filter(
    #         user = user,
    #     ).delete()

    # def _segment_trajectory_offline(self,  user: str, spd: StayPointDetector,
    # min_trajectory_size: int = 10) -> None:
    def _segment_trajectory_offline(
        self,
        trajectories: List[ds.Trajectory],
        user: str,
        spd: spdetector.StayPointDetector,
    ) -> None:
        # Clean up old database data
        # self._clear_db(user = user)
        # ss = SequenceScorer()
        # ss.clear(uid = user)

        # Retrieve "raw" trajectories
        # trajectories = list(
        #     django_models.Trajectory.objects.select_related(
        #     ).annotate(
        #         total = Count("datapoints__pk"),
        #     ).filter(
        #         total__gt = min_trajectory_size,
        #         user = user,
        #         tag = "raw",
        #     )
        # )

        # TODO: The 'StayPointDetector' should be passed as a function argument!
        # spd = StayPointDetector(
        #     percentile = 85,
        #     radius = 100.,
        #     **{"exchange": self._exchange},
        # )
        # Run through all retrieved trajectories and segment each trajectory
        # into possibly several subtrajectories
        for trajectory in trajectories:
            datapoints = trajectory.datapoints
            # datapoints = datastructures.Trajectory.from_model(
            #   trajectory
            # ).datapoints
            polygons = spd.extract_geofences(
                datapoints=datapoints, **{"user": user}
            )
            print("Polygons: ", polygons)
            # TODO: Save breakpoints in bulk!
            self._segment_trajectory_online(
                user=user,
                datapoints=datapoints,
                polygons=polygons,
                # Save the newly generated breakpoints
                save_breakpoints=True,
            )
            # Make sure we clear old data that is being stored in Redis for
            # determining trajectory breakpoints. We should not be using data
            # pertaining to a previous trajectory
            dfinterfaces.BreakPointDataFrame._clear_dfdata(
                exchange=self._exchange, user=user,
            )
        # TODO: Make sure breakpoints were created before we proceed!
        # Extract geohash sequences from corresponding subtrajectories

        # TODO: Do these steps at subseqent point. That is, do not call these
        # methods here!
        # sequences = self._bulk_interpolate_geohash_sequences(
        #     user = user,
        #     trajectories = trajectories,
        # )
        # ss.create_support(uid = user, sequences = sequences)
        # return ss

    # def _bulk_interpolate_geohash_sequences(
    #     self,
    #     user: str,
    #     trajectories: List[django_models.Trajectory],
    #     precision: int,
    #     bits_per_char: int,
    # ) -> List[Sequence]:
    #     distance_delta = ds.interpolation_distance_delta(
    #         precision=precision, bits_per_char=bits_per_char,
    #     )
    #     sequences_list = []
    #     for trajectory in trajectories:
    #         subtrajectories = self._segment_trajectory(
    #             user=user, trajectory=trajectory,
    #         )
    #         self._subtrajectories_to_db(user, subtrajectories)
    #         sequences = self._to_geohash_sequences(
    #             user=user,
    #             subtrajectories=subtrajectories,
    #             precision=precision,
    #             bits_per_char=bits_per_char,
    #         )
    #         for sequence in sequences:
    #             (
    #                 new_sequence,
    #                 _,
    #             ) = ip.GeohashSequenceInterpolator._interpolate_geohash_sequence(  # noqa
    #                 sequence=sequence,
    #                 distance_delta=distance_delta,
    #                 precision=precision,
    #                 bits_per_char=bits_per_char,
    #             )
    #             sequences_list.append(new_sequence)
    #     return sequences_list

    def _segment_trajectory(
        self, user: str, trajectory: django_models.Trajectory
    ) -> List[ds.Trajectory]:
        datapoints = trajectory.datapoints.all()
        # If the trajectory starts at a stay point cluster, then we start
        # with a breakpoint that should just be ignored. Filter out this
        # breakpoint and use all the others to segment a "raw" trajectory:
        breakpoints = trajectory.breakpoints.filter(
            scan_index__gt=datapoints.first().external_timestamp,
        )
        breakpoints_count = breakpoints.count()
        subtrajectories = []
        if breakpoints_count == 0:
            dps0 = datapoints.all()
            subtrajectory = ds.Trajectory(
                [ds.DataPoint.from_model(_) for _ in dps0],
            )
            subtrajectory.trajectory = str(trajectory.uid)
            subtrajectories.append(subtrajectory)
        else:
            for i in range(0, breakpoints_count):
                # Case: Start segment
                if i == 0:
                    # Only a single breakpoint
                    if i == (breakpoints_count - 1):
                        dps0 = datapoints.filter(
                            # Remember to include segment start and endpoints
                            external_timestamp__lte=breakpoints[i].start_index,
                        )
                        subtrajectory = ds.Trajectory(
                            [ds.DataPoint.from_model(_) for _ in dps0],
                        )
                        subtrajectory.trajectory = str(trajectory.uid)
                        subtrajectories.append(subtrajectory)

                        dps1 = datapoints.filter(
                            # Remember to include segment start and endpoints
                            external_timestamp__gte=breakpoints[i].scan_index,
                        )
                        subtrajectory = ds.Trajectory(
                            [ds.DataPoint.from_model(_) for _ in dps1],
                        )
                        subtrajectory.trajectory = str(trajectory.uid)
                        subtrajectories.append(subtrajectory)
                    # Many following breakpoints
                    else:
                        dps0 = datapoints.filter(
                            # Remember to include segment start and endpoints
                            external_timestamp__lte=breakpoints[i].start_index,
                        )
                        subtrajectory = ds.Trajectory(
                            [ds.DataPoint.from_model(_) for _ in dps0],
                        )
                        subtrajectory.trajectory = str(trajectory.uid)
                        subtrajectories.append(subtrajectory)
                # Case: End segment
                elif i == (breakpoints_count - 1):
                    dps0 = datapoints.filter(
                        # Remember to include segment start and endpoints
                        external_timestamp__gte=breakpoints[i - 1].scan_index,
                        external_timestamp__lte=breakpoints[i].start_index,
                    )
                    subtrajectory = ds.Trajectory(
                        [ds.DataPoint.from_model(_) for _ in dps0],
                    )
                    subtrajectory.trajectory = str(trajectory.uid)
                    subtrajectories.append(subtrajectory)
                # Case: Middle segment
                else:
                    dps0 = datapoints.filter(
                        # Remember to include segment start and endpoints
                        external_timestamp__gte=breakpoints[i - 1].scan_index,
                        external_timestamp__lte=breakpoints[i].start_index,
                    )
                    subtrajectory = ds.Trajectory(
                        [ds.DataPoint.from_model(_) for _ in dps0],
                    )
                    subtrajectory.trajectory = str(trajectory.uid)
                    subtrajectories.append(subtrajectory)
        return subtrajectories

    def _subtrajectories_to_db(
        self, user: str, subtrajectories: List[django_models.Trajectory]
    ) -> None:
        obj_list = []
        for subtrajectory in subtrajectories:
            traj_obj = django_models.Trajectory.objects.get(
                uid=subtrajectory.trajectory,
            )
            dps = [_.to_model() for _ in subtrajectory.datapoints]
            subtraj_obj = django_models.Trajectory(
                start_timestamp=dps[0].external_timestamp,
                end_timestamp=dps[-1].external_timestamp,
                user=user,
                tag="subtrajectory",
                # Set a reference to the parent trajectory the subtrajctory was
                # made from
                trajectory=traj_obj,
            )
            obj_list.append(subtraj_obj)
            for dp in dps:
                # Associate each datapoint with the trajectory
                setattr(dp, "trajectory", subtraj_obj)  # noqa
                obj_list.append(dp)
        # Save all the subtrajectories and corresponding datapoints to the
        # database
        self._bulk_manager.add_bulk(obj_list=obj_list)
        self._bulk_manager.done()

    def _to_geohash_sequences(
        self,
        user: str,
        subtrajectories: django_models.Trajectory,
        precision: int,
        bits_per_char: int,
    ) -> List[Sequence]:
        # TODO: Make min_subtrajectory_size a GLOBAL variable
        min_subtrajectory_size = 5
        geohash_sequences = []
        for subtrajectory in subtrajectories:
            geohash_sequence: Sequence = []
            for datapoint in list(subtrajectory.datapoints):
                geohash = datapoint.get_geohashed_datapoint(
                    precision=precision, bits_per_char=bits_per_char,
                ).geohash
                if len(geohash_sequence) >= 1:
                    if geohash_sequence[-1] != geohash:
                        geohash_sequence.append(geohash)
                # Else skip, so we do not collect duplicate geohash
                # sequences
                else:
                    geohash_sequence.append(geohash)
            if len(geohash_sequence) >= min_subtrajectory_size:
                geohash_sequences.append(geohash_sequence)
        return geohash_sequences
