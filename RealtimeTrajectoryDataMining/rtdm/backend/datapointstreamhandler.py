import logging
import multiprocessing
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import backend.datastructures as ds
import backend.managers as django_managers
import backend.models as django_models
import backend.utils as utils
import backend.xchanges as xchanges
import msgpack
import numpy as np
import ruuid
from django.conf import settings

# Import django 'stream_handler' application such that we can import
# and use appropriate database models, managers, etc.
# from stream_handler import setup
# setup()


class BaseDataPointStreamHandler:
    def __init__(
        self,
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        namespace: Union[None, str] = None,
        max_merging_distance: float = 28.0,  # In meters
        max_separation_distance: float = 500.0,  # In meters
        max_separation_time: float = 300.0,  # In seconds
        max_acceleration: float = 7.0,  # In meters per. second
        max_radius: float = 100.0,  # In meters
    ) -> None:
        self._exchange = exchange
        if namespace is None:
            self.namespace = settings.STREAMHANDLER_NAMESPACE
        else:
            self.namespace = namespace
        self.max_merging_distance = max_merging_distance
        self.max_separation_distance = max_separation_distance
        self.max_separation_time = max_separation_time
        self.max_acceleration = max_acceleration
        self.max_radius = max_radius
        # Validate input arguments
        self._check_vars()

    def update(self, user: str) -> Union[None, int]:
        """
        Aggregate and clean data based on new incoming data.

        Note:
            Given a unique identifier of a user, aggregate and clean data \
            based on new incoming data belonging to the user.

        Args:
            user (str): A unique identifier of the user for which we \
                should process new incoming data.
        """
        (
            ingress_queue_size,
            aggregate_queue_size,
            filter_queue_size,
        ) = self._queue_sizes(user=user)
        if ingress_queue_size >= 2:
            dp0_, dp1_ = self._next_pair(
                user=user, ingress_queue_size=ingress_queue_size,
            )
            # If two datapoints are far away from each other temporally or
            # geospatially, then we cut off the trajectory we are currently
            # building and start a new
            if (
                dp1_.dt > self.max_separation_time
                or dp1_.dx > self.max_separation_distance
            ):
                self._separate_trajectory(
                    dp0_=dp0_,
                    dp1_=dp1_,
                    user=user,
                    aggregate_queue_size=aggregate_queue_size,
                    filter_queue_size=filter_queue_size,
                )
                # Signal end of trajectory
                return 0
            # TODO: Implement 'max_radius' check for 'stay points'
            elif None:

                return 1
            # If two datapoints are close geospatially, then we merge them by
            # inserting them into the AGGREGATE queue
            elif (
                dp1_.dx <= self.max_merging_distance
                and self._distance_to_median(dp0_, dp1_) < self.max_radius
            ):
                self._aggregate_trajectory(
                    dp0_=dp0_,
                    dp1_=dp1_,
                    user=user,
                    aggregate_queue_size=aggregate_queue_size,
                    filter_queue_size=filter_queue_size,
                )
                # Signal continuation of trajtectory
                return 1
            else:
                # Only add the first datapoint 'dp0_' to the FILTER queue if:
                # (i) If the AGGREGATE queue has size > 0 then datapoints have
                # only been added to the AGGREGATE queue up until this point.
                # In this case we should remember to add the aggregate point
                # 'dp0_' to the trajectory we are currently building before
                # adding a new datapoint 'dp1_'. (ii) If the FILTER queue is
                # currently empty, then we recently flushed the queue to start
                # a new trajectory. In this case we should remember to add the
                # first datapoint 'dp0_' to the new trajectory we are building.
                self._extend_trajectory(
                    dp0_=dp0_,
                    dp1_=dp1_,
                    user=user,
                    aggregate_queue_size=aggregate_queue_size,
                    filter_queue_size=filter_queue_size,
                )
                # Signal continuation of trajectory
                return 1
        # Signal nothing to be done...
        return None

    def bulk_process(self, datapoints: List[ds.DataPoint], user: str) -> None:
        if self._exchange.client is not None:
            # Method for processing a stream of datapoints in bulk:
            # - Insert datapoints into ingress queue
            pipe = self._exchange.client.pipeline()
            for datapoint in datapoints:
                pipe.rpush(
                    self._get_key(
                        namespace=self.namespace,
                        user=user,
                        name=settings.INGRESS_QUEUE,
                    ),
                    datapoint.to_msgpack(),
                )
            _ = pipe.execute()
            # - Aggregate and filter through the datapoints
            for _ in range(len(datapoints)):
                self.update(user=user)
            # - Save and flush the last remaining datapoints contained in the
            # ingress queue
            self.finish_trajectory(user=user)
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def finish_trajectory(self, user: str) -> None:
        """
        Cut a trajectory short and save it to the db.

        Args:
            user (str): A unique identifier of the user for which we should \
                "finish" a trajectory for.
        """
        if self._exchange.client is not None:
            _, aggregate_queue_size, filter_queue_size = self._queue_sizes(
                user=user
            )
            if aggregate_queue_size > 0:
                adp = self._aggregate_datapoints(user=user)
                if filter_queue_size > 0:
                    dp = self._exchange.client.lindex(
                        index=-1,
                        name=self._get_key(
                            namespace=self.namespace,
                            user=user,
                            name=settings.INGRESS_QUEUE,
                        ),
                    )
                    if dp is not None:
                        dp = ds.DataPoint.from_msgpack(dp)
                        adp.compute_metrics(dp)
                        self._exchange.client.rpush(
                            self._get_key(
                                namespace=self.namespace,
                                user=user,
                                name=settings.FILTER_QUEUE,
                            ),
                            adp.to_msgpack(),
                        )
                    else:
                        logging.warning("")
            values = self._exchange.client.lrange(
                name=self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.FILTER_QUEUE,
                ),
                start=0,
                end=-1,
            )
            # Take all the data in the FILTER queue and create a new trajectory
            # and insert it into the database for long-term storage
            if values is not None:
                self._to_db(user=user, values=values)
            else:
                error = f"values are None!"
                logging.warning(error)
                raise ValueError(error)
            # After we have saved everything to the database, we clean up the
            # queues
            self._clear_queues(user=user)
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def _check_vars(self) -> None:
        """Check and validate all class arguments on class instantiation."""
        # max_mergining_distance
        if not isinstance(self.max_merging_distance, float):
            error = f"ARG 'max_merging_distance' is of type \
                {type(self.max_merging_distance)} but should be of type \
                'float'!"
            try:
                self.max_merging_distance = float(self.max_merging_distance)
            except Exception:
                raise TypeError(error)
        else:
            if self.max_merging_distance < 0.0:
                raise ValueError(
                    f"The given 'max_merging_distance' value is: \
                    {self.max_merging_distance}. It needs to be larger than \
                    0.0"
                )
        # max_seperation_distance
        if not isinstance(self.max_separation_distance, float):
            error = f"ARG 'max_merging_distance' is of type \
                {type(self.max_merging_distance)} but should be of type \
                'float'!"
            try:
                self.max_separation_distance = float(
                    self.max_separation_distance
                )
            except Exception:
                raise TypeError(error)
        else:
            if self.max_separation_distance < 0.0:
                raise ValueError(
                    f"The given 'max_separation_distance' value is: \
                    {self.max_separation_distance}. It needs to be larger \
                    than 0.0"
                )
        # max_seperation_time
        if not isinstance(self.max_separation_time, float):
            error = f"ARG 'max_separation_time' is of type \
                {type(self.max_separation_time)} but should be of type \
                'float'!"
            try:
                self.max_separation_time = float(self.max_separation_time)
            except Exception:
                raise TypeError(error)
        else:
            if self.max_separation_time < 0.0:
                raise ValueError(
                    f"The given 'max_separation_time' value is: \
                    {self.max_separation_time}. It needs to be larger than \
                    0.0"
                )
        # max_acceleration
        if not isinstance(self.max_acceleration, float):
            error = f"ARG 'max_acceleration' is of type \
                {type(self.max_acceleration)} but should be of type 'float'!"
            try:
                self.max_acceleration = float(self.max_acceleration)
            except Exception:
                raise TypeError(error)
        else:
            if self.max_acceleration < 0.0:
                raise ValueError(
                    f"The given 'max_acceleration' value is: \
                    {self.max_acceleration}. It needs to be larger than 0.0"
                )
        # max_radius
        if not isinstance(self.max_radius, float):
            error = f"ARG 'max_radius' is of type {type(self.max_radius)} but \
                should be of type 'float'!"
            try:
                self.max_acceleration = float(self.max_radius)
            except Exception:
                raise TypeError(error)
        else:
            if self.max_radius < 0.0:
                raise ValueError(
                    f"The given 'max_radius' value is: {self.max_radius}. It \
                    needs to be larger than 0.0"
                )

    def _get_key(self, namespace: str, user: str, name: str) -> str:
        """
        Construct a proper key for accessing a certain queue in Redis.

        Args:
            namespace (str): A namespace used to identify the class that
                manages the queue.
            user (str): A unique identifier of a user.
            name (str): The name of the queue that represent a certain step
                in the data-processing pipleline.

        Returns:
            str: The key to a queue in Redis.
        """
        return utils.get_queue(namespace=namespace, user=user, name=name)

    def _queue_sizes(self, user: str) -> Tuple[int, int, int]:
        """
        Retrieve the current size of the INGRESS, AGGREGATE and FILTER queues.

        Args:
            user (str): A unique identifier of a user.

        Returns:
            Tuple[int, int]: The size of the INGRESS, AGGREGATE and FILTER
                Redis queues belonging to a certain user, respectively.
        """
        if self._exchange.client is not None:
            pipe = self._exchange.client.pipeline()
            # Retrieve working queues sizes:
            # - Queue containing an all incoming datapoints
            pipe.llen(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.INGRESS_QUEUE,
                ),
            )
            # - Queue containing an aggregation of possibly several datapoints
            pipe.llen(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.AGGREGATE_QUEUE,
                ),
            )
            # - Queue containing filtered datapoints representing a trajectory
            pipe.llen(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.FILTER_QUEUE,
                ),
            )
            # Execute the full pipeline of Redis commands
            (
                ingress_queue_size,
                aggregate_queue_size,
                filter_queue_size,
            ) = pipe.execute()
            return ingress_queue_size, aggregate_queue_size, filter_queue_size
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def _forward_data(self) -> bool:
        return settings.FORWARD_DATA

    def _to_db(self, user: str, values: List[bytes]) -> None:
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    def _initialize_trajectory(self, user: str) -> Dict[str, Any]:
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    def _parallel_deserialize(
        self, dps: List[bytes], threshold: int = 10_00_00,
    ) -> List[ds.DataPoint]:
        available_cores = multiprocessing.cpu_count()
        # In the EXTREME case where we need to deserialize more than
        # 'threshold' == 10_00_00 datapoints and we have more than 4 cores
        # available, then use half of the cores available on deserializing
        # the data. This should result in a small performance boost. If
        # 'threshold' < 10_00_00 and 'cores' < 3 then the overhead of doing
        # the deserialization in parallel will NOT be worth it.
        # TODO: If we need to deserialize >> 25_00_00 datapoints, then some
        # other strategy should be used to decrease the time of deserializing
        # this many datapoints. For example: Split the current set of
        # aggregated datapoints into two separate aggregate datapoints.
        cores = int(np.ceil(available_cores / 2))
        if available_cores > 4 and len(dps) > threshold:
            pool = multiprocessing.Pool(cores)
            results = pool.map(ds.DataPoint.from_msgpack, dps)  # type: ignore
            pool.close()
            pool.join()
        else:
            results = list(
                map(lambda v: ds.DataPoint.from_msgpack(v), dps)  # type: ignore
            )
        return results

    # TODO: Figure out a better strategy for aggregating datapoints.
    # Continously deserializing and serializing datapoints will be a
    # performance bottleneck if the number of datapoints is very large.
    def _aggregate_datapoints(self, user: str) -> ds.DataPoint:
        """
        Aggregate two or more datapoints.

        Note:
            Aggregate two or more datapoints by (i) taking the median of the \
            latitude and longitude coordinates, and (ii) using the most recent \
            timestamp among the datapoints to ultimately create a new \
            'aggregate' datapoint.

        Args:
            user (str): A unique identifier of a user.

        Returns:
            DataPoint: A single aggregate datapoint.
        """
        if self._exchange.client is not None:
            # Merge all datapoints currently in the AGGREGATE queue, as they
            # are all in the vicinicty of each other
            dps = self._exchange.client.lrange(
                name=self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.AGGREGATE_QUEUE,
                ),
                start=0,
                end=-1,
            )
            if dps is not None:
                values = self._parallel_deserialize(dps=dps)
                coordinate_pair = list(
                    map(lambda v: [v.latitude, v.longitude], values)
                )
                latitude, longitude = np.median(coordinate_pair, axis=0)
                dp = ds.DataPoint(
                    latitude=latitude,
                    longitude=longitude,
                    # Give the datapoint the most recently known timestamp
                    external_timestamp=values[-1].external_timestamp,
                    user=user,
                )
                # Compute the weight of the datapoint as the time spent in the
                # vicinity of a certain location
                if (
                    values[-1].external_timestamp is not None
                    and values[0].external_timestamp is not None
                ):
                    dp.weight = (
                        values[-1].external_timestamp
                        - values[0].external_timestamp
                    ).total_seconds()
                # Save all datapoints that were used to create the aggregate
                # datapoint
                dp.datapoints = values
                # Set the trajectory the datapoint should be associated with
                dp.trajectory = values[0].trajectory
                return dp
            else:
                error = f"dps is None!"
                logging.warning(error)
                raise ValueError(error)
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def _clear_queues(self, user: str) -> None:
        """
        Clear the AGGREGATE and FILTER Redis queues belonging to a certain user.

        Args:
            user (str): A unique identifier of a user.
        """
        if self._exchange.client is not None:
            pipe = self._exchange.client.pipeline()
            pipe.delete(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.AGGREGATE_QUEUE,
                ),
            )
            pipe.delete(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.FILTER_QUEUE,
                ),
            )
            # Execute the full pipeline of Redis commands
            pipe.execute()
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def _set_uuid(
        self, dp: ds.DataPoint, data: Dict[str, Any],
    ) -> ds.DataPoint:
        dp.trajectory = data["traj_uuid"]
        dp.user = data["user_uuid"]
        return dp

    def _get_data(self, user: str) -> Dict[str, Any]:
        data = self._exchange.kv_get(
            name=self._get_key(
                namespace=self.namespace,
                user=user,
                name=settings.CURRENT_TRAJECTORY_KVSTORE,
            ),
            # Do not wait until data becomes available
            timeout=0,  # type: ignore
        )
        if data is None:
            # INGRESS queue must have length 1
            data = self._initialize_trajectory(user)
        else:
            data = msgpack.loads(data)
        return data

    def _distance_to_median(
        self, dp0_: ds.DataPoint, dp1_: ds.DataPoint,
    ) -> float:
        # If dp0_ is a single point and not an aggregated point
        # access the lat/lon coords directly
        if len(dp0_.datapoints) == 0:
            coordinate_pair = [[dp0_.latitude, dp0_.longitude]]
        # Else if dp0_ is not a single point but an aggregate point
        # collect all lat/lon coordinates from the associated list
        else:
            coordinate_pair = list(
                map(lambda v: [v.latitude, v.longitude], dp0_.datapoints)
            )
        latitude, longitude = np.median(
            coordinate_pair + [[dp1_.latitude, dp1_.longitude]], axis=0,
        )
        distance = ds.haversine_distance(
            lat_1=latitude,
            lon_1=longitude,
            lat_2=dp1_.latitude,
            lon_2=dp1_.longitude,
        )
        return distance

    def _next_pair(
        self, user: str, ingress_queue_size: int,
    ) -> Tuple[ds.DataPoint, ds.DataPoint]:
        if self._exchange.client is not None:
            # Retrieve identifiers for the current trajectory we are building
            data = self._get_data(user)
            # POP the currently oldest datapoint in the INGRESS queue
            pipe = self._exchange.client.pipeline()
            pipe.lpop(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.INGRESS_QUEUE,
                ),
            )
            # ACCESS the currently oldest datapoint in the INGRESS queue
            pipe.lindex(
                name=self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.INGRESS_QUEUE,
                ),
                index=0,
            )
            # Execute the full pipeline of Redis commands
            dp0, dp1 = pipe.execute()
            # Adjust current INGRESS queue size (due to pop operation on the
            # queue)
            ingress_queue_size -= 1
            # Deserialize datapoint byte data into Python DataPoint objects,
            # then compute metrics for the two consecutive datapoints, such as:
            # (i) change in time and (ii) change in distance, etc...
            dp0_ = ds.DataPoint.from_msgpack(dp0)
            dp1_ = ds.DataPoint.from_msgpack(dp1)
            dp1_.compute_metrics(dp0_)
            dp0_ = self._set_uuid(dp0_, data)
            return dp0_, dp1_
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def _aggregate_trajectory(
        self,
        dp0_: ds.DataPoint,
        dp1_: ds.DataPoint,
        user: str,
        aggregate_queue_size: int,
        filter_queue_size: int,
    ) -> None:
        if self._exchange.client is not None:
            # If the AGGREGATE and FILTER queues are currently empty, then we
            # previously created a complete trajectory and flushed both queues.
            # In this case we should place the first datapoint into the
            # AGGREGATE queue as it is close to the next datapoint
            if aggregate_queue_size == 0:
                if filter_queue_size == 0:
                    self._exchange.client.rpush(
                        self._get_key(
                            namespace=self.namespace,
                            user=user,
                            name=settings.AGGREGATE_QUEUE,
                        ),
                        dp0_.to_msgpack(),
                    )
                    # Adjust current AGGREGATE queue size (due to push operation
                    # on the queue)
                    aggregate_queue_size += 1
                # If the AGGREGATE queue is currently empty, but the FILTER
                # queue is not, then we previously added a datapoint to the
                # trajectory we are currently building. In this case we should
                # remove the most recent datapoint in the trajectory and add
                # it to the AGGREGATE queue instead, it is close to the new
                # datapoint 'dp1_'
                else:
                    # Right pop
                    dp = self._exchange.client.rpop(
                        self._get_key(
                            namespace=self.namespace,
                            user=user,
                            name=settings.FILTER_QUEUE,
                        ),
                    )
                    if dp is not None:
                        # Adjust current FILTER queue size (due to pop
                        # operation on the queue)
                        filter_queue_size -= 1
                        # Insert the datapoint into the aggregate queue
                        self._exchange.client.rpush(
                            self._get_key(
                                namespace=self.namespace,
                                user=user,
                                name=settings.AGGREGATE_QUEUE,
                            ),
                            dp,
                        )
                        # Adjust current AGGREGATE queue size (due to push
                        # operation on the queue)
                        aggregate_queue_size += 1
                    else:
                        error = f"dp is None!"
                        logging.warning(error)
                        raise ValueError(error)
            # Place the newest datapoint 'dp1_' into the AGGREGATE queue. It is
            # close to the previous datapoint 'dp0_' that was inserted into the
            # AGGREGATE queue just now or previously
            self._exchange.client.rpush(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.AGGREGATE_QUEUE,
                ),
                dp1_.to_msgpack(),
            )
            # Adjust current AGGREGATE queue size (due to push operation on the
            # queue)
            aggregate_queue_size += 1
            # As there are now several datapoints in the AGGREGATE queue,
            # compute the aggregate datapoint
            adp = self._aggregate_datapoints(user=user)
            # If there are > 0 datapoints in the trajectory we are currently
            # building, then update the metrics for the new aggregate datapoint.
            # To do this we use the most recent datapoint that was added to the
            # trajectory that we are currently building. The aggregate datapoint
            # will be the next datapoint that is inserted into the trajectory.
            if filter_queue_size > 0:
                dp = self._exchange.client.lindex(
                    index=-1,
                    name=self._get_key(
                        namespace=self.namespace,
                        user=user,
                        name=settings.FILTER_QUEUE,
                    ),
                )
                if dp is not None:
                    dp = ds.DataPoint.from_msgpack(dp)
                    adp.compute_metrics(dp)
                else:
                    error = f"dp is None!"
                    logging.warning(error)
                    raise ValueError(error)
            # Set the aggregate datapoint, as the next datapoint we take out of
            # the INGRESS queue
            self._exchange.client.lset(
                index=0,
                value=adp.to_msgpack(),
                name=self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.INGRESS_QUEUE,
                ),
            )
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def _separate_trajectory(
        self,
        dp0_: ds.DataPoint,
        dp1_: ds.DataPoint,
        user: str,
        aggregate_queue_size: int,
        filter_queue_size: int,
    ) -> None:
        if self._exchange.client is not None:
            pipe = self._exchange.client.pipeline()
            if aggregate_queue_size > 0 or filter_queue_size == 0:
                dp0_serialized = dp0_.to_msgpack()
                pipe.rpush(
                    self._get_key(
                        namespace=self.namespace,
                        user=user,
                        name=settings.FILTER_QUEUE,
                    ),
                    dp0_serialized,
                )
                # Adjust current FILTER queue size (due to push operation on
                # the queue)
                filter_queue_size += 1
                if self._forward_data():
                    # NOTE: Forward to next analysis queue
                    pipe.rpush(
                        self._get_key(
                            namespace=self.namespace,
                            user=user,
                            name=settings.PROCESS0_QUEUE,
                        ),
                        dp0_serialized,
                    )
            if self._forward_data():
                # NOTE: Forward location data to next analysis queue
                pipe.rpush(
                    self._get_key(
                        namespace=self.namespace,
                        user=user,
                        name=settings.PROCESS0_QUEUE,
                    ),
                    # Special case: Signal the end of the trajectory
                    msgpack.dumps("END"),
                )
            pipe.lrange(
                name=self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.FILTER_QUEUE,
                ),
                start=0,
                end=-1,
            )
            # Execute the full pipeline of Redis commands
            return_data = pipe.execute()
            # Access the the output from the last command sent to Redis:
            # --> lrange: We get all the datapoints from the FILTER queue
            values = return_data[-1]
            # Take all the data in the FILTER queue and create a new trajectory
            # and insert it into the database
            self._to_db(user=user, values=values)
            # After we have saved everything to the database, we clean up the
            # queues
            self._clear_queues(user=user)
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def _extend_trajectory(
        self,
        dp0_: ds.DataPoint,
        dp1_: ds.DataPoint,
        user: str,
        aggregate_queue_size: int,
        filter_queue_size: int,
    ) -> None:
        if self._exchange.client is not None:
            pipe = self._exchange.client.pipeline()
            if aggregate_queue_size > 0 or filter_queue_size == 0:
                dp0_serialized = dp0_.to_msgpack()
                pipe.rpush(
                    self._get_key(
                        namespace=self.namespace,
                        user=user,
                        name=settings.FILTER_QUEUE,
                    ),
                    dp0_serialized,
                )
                # Adjust current FILTER queue size (due to push operation on
                # the queue)
                filter_queue_size += 1
                if self._forward_data():
                    pipe.rpush(
                        self._get_key(
                            namespace=self.namespace,
                            user=user,
                            name=settings.PROCESS0_QUEUE,
                        ),
                        dp0_serialized,
                    )
            if np.abs(dp1_.acceleration) <= self.max_acceleration:
                # Add the new datapoint to the trajectory we are currently
                # building
                dp1_serialized = dp1_.to_msgpack()
                pipe.rpush(
                    self._get_key(
                        namespace=self.namespace,
                        user=user,
                        name=settings.FILTER_QUEUE,
                    ),
                    dp1_serialized,
                )
                # Adjust current FILTER queue size (due to push operation on the
                # queue)
                filter_queue_size += 1
                if self._forward_data():
                    # NOTE: Forward to next analysis queue
                    pipe.rpush(
                        self._get_key(
                            namespace=self.namespace,
                            user=user,
                            name=settings.PROCESS0_QUEUE,
                        ),
                        dp1_serialized,
                    )
            # After we have saved everything to the db and Redis we clean up the
            # queues
            pipe.delete(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.AGGREGATE_QUEUE,
                ),
            )
            # Execute the full pipeline of Redis commands
            pipe.execute()
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)


class InMemoryDataPointStreamHandler(BaseDataPointStreamHandler):
    def __init__(self, **kwargs: Any) -> None:
        # Use a dictionary of ordered dictionaries for data storage
        # --> Each entry in the dictionary '_dicts' is associated with a user
        # --> Each entry is an ordered dictionary
        # --> An ordered dictionary will contain a set of chronologically
        #     ordered trajectories
        self._dicts: Dict[str, Any] = {}
        super().__init__(**kwargs)

    def _to_db(self, user: str, values: List[bytes]) -> None:
        if self._exchange.client is not None:
            # TODO: Figure out if data (trajectory and associated datapoints)
            # should be saved incrementally to the database. Instead of in bulk
            # (whenever a trajectory has been 'finalized'), as it is currently
            # done.
            # --> It will add more overhead, but it will reduce the memory used.
            dps = [ds.DataPoint.from_msgpack(_).to_dict() for _ in values]
            # Retrive the trajectory we initialize at the very start
            # and update its fields before we save all associated data
            key = next(reversed(self._dicts[user]))
            traj_obj = self._dicts[user][key]
            traj_obj["start_timestamp"] = dps[0]["external_timestamp"]
            traj_obj["end_timestamp"] = dps[-1]["external_timestamp"]
            traj_obj["datapoints"] = dps
            # TODO: Save to local storage/file
            # ... Data is currently only kept in memory and not persisted.
            # Remove the uuid of current 'raw' trajectory that we were building
            self._exchange.client.delete(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.CURRENT_TRAJECTORY_KVSTORE,
                )
            )
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def _initialize_trajectory(self, user: str) -> Dict[str, Any]:
        if self._exchange.client is not None:
            # Retrieve the very first datapoint that should be assigned to a new
            # 'raw' trajectory
            dp = self._exchange.client.lindex(
                name=self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.INGRESS_QUEUE,
                ),
                index=0,
            )
            if dp is not None:
                dp = ds.DataPoint.from_msgpack(dp)
                # Use the timestamp of the datapoint as a temporary start time
                # of the trajectory
                traj_uuid = str(ruuid.uuid4())
                # TODO: Make a 'ds.Trajectory' object instrad of dict
                traj_data = {
                    "start_timestamp": dp.external_timestamp,
                    "end_timestamp": None,
                    "user": user,
                    "uid": traj_uuid,
                    "tag": "raw",
                    "datapoints": None,
                }
                # Initialize entry with an ordered dictionary
                if user not in self._dicts:
                    self._dicts[user] = OrderedDict()
                    self._dicts[user][traj_uuid] = traj_data
                # Otherwise just add to the existing ordered dictionary
                else:
                    self._dicts[user][traj_uuid] = traj_data
                data = {
                    "traj_uuid": traj_uuid,
                    "user_uuid": user,
                }
                serialized_data = msgpack.dumps(data)
                # Set uuid of current 'raw' trajectory that we are building
                self._exchange.kv_set(
                    data=serialized_data,
                    name=self._get_key(
                        namespace=self.namespace,
                        user=user,
                        name=settings.CURRENT_TRAJECTORY_KVSTORE,
                    ),
                )
                return data
            else:
                error = f"dp is None!"
                logging.warning(error)
                raise ValueError(error)
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)


# TODO: Rename 'RedisDataPointStreamHandler'
class DataPointStreamHandler(BaseDataPointStreamHandler):
    def __init__(
        self, min_chunk_size: int = 1, max_chunk_size: int = 1, **kwargs: Any,
    ) -> None:
        self._bulk_manager = django_managers.BulkCreateManager(
            min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size,
        )
        super().__init__(**kwargs)

    def _to_db(self, user: str, values: List[bytes]) -> None:
        if self._exchange.client is not None:
            # TODO: Figure out if data (trajectory and associated datapoints)
            # should be saved incrementally to the database. Instead as in bulk
            # (whenever a trajectory has been 'finalized'), as it is currently
            # done.
            # --> It will add more overhead, but will reduce memory used.
            obj_list = []
            dps = [ds.DataPoint.from_msgpack(_).to_model() for _ in values]
            # Retrive the trajectory associated to a particular user and that
            # was initialized at the very start. Do this before we update its
            # fields and before we save all associated data.
            # NOTE: Make sure that the trajectories are ordered in ascending
            # order such that the last trajectory we retrieve from the database
            # is the most recent!
            traj_obj = django_models.Trajectory.objects.filter(uid=user).last()
            traj_obj.start_timestamp = dps[0].external_timestamp
            traj_obj.end_timestamp = dps[-1].external_timestamp
            traj_obj.save(update_fields=["start_timestamp", "end_timestamp"])
            for dp in dps:
                if dp.magnetometer is not None:
                    obj_list.append(dp.magnetometer)
                if dp.devicemotion is not None:
                    obj_list.append(dp.devicemotion)
                # Associate each datapoint with the trajectory
                # TODO: Make sure 'traj_obj' is not None
                setattr(dp, "trajectory", traj_obj)  # noqa
                obj_list.append(dp)
            self._bulk_manager.add_bulk(obj_list=obj_list)
            # Save whenever we have 1 complete trajectory
            self._bulk_manager.done()
            # Remove the uuid of current 'raw' trajectory that we were building
            self._exchange.client.delete(
                self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.CURRENT_TRAJECTORY_KVSTORE,
                )
            )
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)

    def _initialize_trajectory(self, user: str) -> Dict[str, Any]:
        if self._exchange.client is not None:
            # Retrieve the very first datapoint that should be assigned to a new
            # 'raw' trajectory
            dp = self._exchange.client.lindex(
                name=self._get_key(
                    namespace=self.namespace,
                    user=user,
                    name=settings.INGRESS_QUEUE,
                ),
                index=0,
            )
            if dp is not None:
                dp = ds.DataPoint.from_msgpack(dp)
                # Use the timestamp of the datapoint as a temporary start time
                # of the trajectory
                traj_obj = django_models.Trajectory(
                    start_timestamp=dp.external_timestamp,
                    end_timestamp=None,
                    user=user,
                    tag="raw",  # Tag trajectory data as "raw"
                )
                traj_obj.save()
                data = {
                    "traj_uuid": str(traj_obj.uid),
                    "user_uuid": user,
                }
                serialized_data = msgpack.dumps(data)
                # Set uuid of current 'raw' trajectory that we are building
                self._exchange.kv_set(
                    data=serialized_data,
                    name=self._get_key(
                        namespace=self.namespace,
                        user=user,
                        name=settings.CURRENT_TRAJECTORY_KVSTORE,
                    ),
                )
                return data
            else:
                error = f"dp is None!"
                logging.warning(error)
                raise ValueError(error)
        else:
            error = f"self._exchange.client is None!"
            logging.warning(error)
            raise ValueError(error)
