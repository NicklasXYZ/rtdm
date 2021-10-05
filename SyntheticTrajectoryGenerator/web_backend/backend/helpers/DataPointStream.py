# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
#                     Website    : Nicklas.xyz                                 #
#                     Github     : github.com/NicklasXYZ                       #
# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import logging
import uuid

import numpy as np
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
from shapely.geometry import Point

# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from .DataPoint import DataPoint
from .valhalla import BUFFER_DISTANCE, ValhallaInterface

# import matplotlib.pyplot as plt
# from descartes import PolygonPatch
# from redisxchange.xchanges import (
#     RedisQueueMessageExchange,
#     dump,
#     load,
# )


# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
logging.basicConfig(level=logging.DEBUG)
# exchange = RedisQueueMessageExchange()
MAX_ACCELERATION = 2.5
MAX_DISTANCE_STAY_POINT = 10.0


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class Track:
    def __init__(self, datapoints) -> None:
        # Set a unique identifier of the segement
        self.uuid = uuid.uuid4()
        # Store all the datapoints in the segment
        self.datapoints = datapoints
        # The length of the segment in meters
        self.euclidean_length = 0
        self.manhatten_length = 0
        # The length in terms of datapoints
        self.length = len(datapoints)
        # Compute the different metrics
        self.compute_metrics()

    def compute_metrics(self):
        for datapoint in self.datapoints:
            self.euclidean_length += datapoint.dx_euclidean
            self.manhatten_length += datapoint.dx_manhatten


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class DataPointStream:
    def __init__(
        self,
        min_time: float,
        min_distance: float,
        window_size: int = 3,
        buffer_size: int = 3,
    ) -> None:
        self.valhalla = ValhallaInterface()
        self.name = "tracking"
        # Set parameters that are used for spatio-temporal segmentation
        self.min_time = min_time
        self.min_distance = min_distance
        # Set buffer and window size
        self.window_size = window_size
        self.buffer_size = buffer_size
        # Creates a queues
        self.queue0 = []
        self.queue1 = []
        self.path = []
        # Stay datapoint queue
        self.sdp_queue = []
        # Create list to store tracks
        self.tracks = []
        logging.debug(
            "\n\nInitializing 'DataPointStream' with min_time = "
            + str(self.min_time)
            + ", min_distance = "
            + str(self.min_distance)
            + "\n",
        )

    # def merge_datapoints(self):
    #     pass

    # def generate_linestring(self):
    #         rel = float(content1["trip"]["legs"][0]["summary"]["length"]) * 1000. #convert to meters
    #         if end1_dp.dx_manhatten != 0:
    #             rel_m = np.abs(rel - end1_dp.dx_manhatten) / end1_dp.dx_manhatten
    #             print("Relative measure: ", rel_m, rel, end1_dp.dx_manhatten)
    #         else:
    #             rel_m = 0
    #         path1 = LineString(decode_polyline(content1["trip"]["legs"][0]["shape"]))
    #         path2 = LineString(decode_polyline(content2["trip"]["legs"][0]["shape"]))
    #         path1_buffered = path1.buffer(distance = distance)
    #         match = path1_buffered.intersection(path2).buffer(distance = distance)
    #         new_point = False
    #         if rel_m <= 5 and not self.skip:
    #             for point in path1.coords:
    #                 p = Point(point)
    #                 if match.contains(p):
    #                     build_path.append([point[0], point[1]])
    #                     new_point = True
    #             if new_point:
    #                 # Set current 'path1' endpoint as the new starting point
    #                 # next time this method is called.
    #                 self.queue1[-2].latitude = build_path[-1][0]
    #                 self.queue1[-2].longitude = build_path[-1][1]
    #                 for pair in build_path:
    #                     self.full_path.append(
    #                         DataPoint(
    #                             latitude = pair[0],
    #                             longitude = pair[1],
    #                             time = None,
    #                         )
    #                     )
    #         else:
    #             self.skip = True

    # def process(self):
    #     datapoints0 = []
    #     # Take the latest datapoints
    #     # --> First time:
    #     if len(self.queue0) == self.buffer_size:
    #         datapoints0 = self.queue0
    #     # --> Proceeding:
    #     elif len(self.queue0) > self.buffer_size:
    #         datapoints0 = [self.queue0[-1]]
    #     # Check consistency of latest datapoints
    #     for dp0 in datapoints0:
    #         # Accerleration is in meters per. second.
    #         # Upper bound on the acceleration is hardcoded for now...
    #         # TODO: Should be decided dynamically, based on datapoints in buffer...
    #         if np.abs(dp0.acceleration) <= 2.5:
    #             self.queue1.append(dp0)
    #             if len(self.queue1) == self.window_size:
    #                 for i in range(0, self.window_size):
    #                     if self.queue1[i].dt > self.min_time:
    #                         ### Save the raw track
    #                         self.tracks.append(
    #                             Track(self.queue1[:i].copy()),
    #                         )
    #                         self.skip = False
    #                         ### TODO: Do something with raw track...
    #                         # Reset queue1. Keep remaining datapoints.
    #                         self.queue2 = self.queue1[i:].copy()
    #                         self.queue1 = self.queue2.copy()

    #             if len(self.queue1) >= self.window_size:
    #                 if self.queue1[-1].dt > self.min_time:
    #                     ### Save the raw track
    #                     track0 = Track(self.queue1[:-1].copy())
    #                     self.tracks.append(
    #                         track0,
    #                     )
    #                     # self.generate_linestring_last()
    #                     track1 = Track(self.full_path)
    #                     track1.uuid = track0.uuid
    #                     ### Save the filtered track
    #                     self.tracks.append(
    #                         track1,
    #                     )
    #                     # Reset list
    #                     self.full_path = []
    #                     self.skip = False

    #                     ### TODO: Do something with the raw track...
    #                     # Reset queue1. Keep remaining datapoints.
    #                     self.queue2 = self.queue1[-1:].copy()
    #                     self.queue1 = self.queue2.copy()
    #                 else:
    #                     self.generate_linestring()

    # def update(self, dp_p01):
    #     # Process the data if we have more than one datapoint in the queue
    #     if len(self.queue0) >= 1:
    #         # dp_m01 = self.queue0[-1]
    #         # dp_p01.compute_metrics(dp_m01)
    #         self.queue0.append(dp_p01)
    #         if len(self.queue0) >= self.buffer_size:
    #             self.process()
    #     else:
    #         self.queue0.append(dp_p01)

    def update(self, dp_p01):
        # Process the data if we have more than one datapoint in the queue
        if len(self.queue0) >= 1:
            # dp_m01 = self.queue0[-1]
            # dp_p01.compute_metrics(dp_m01)
            self.queue0.append(dp_p01)
            self.queue1.append(dp_p01)
            # if len(self.queue0) >= self.buffer_size:
            self.process()
        else:
            self.queue0.append(dp_p01)
            self.queue1.append(dp_p01)

    def process(self):
        dp_am01 = self.queue0[-2]  # Raw dp stream. 2nd last.
        dp_ap01 = self.queue0[-1]  # Raw dp stream.     last.
        dp_ap01.compute_metrics(dp_am01)
        if (
            np.abs(dp_ap01.acceleration) <= MAX_ACCELERATION
        ):  # Decision based on current dp uses queue0
            dp_bm01 = self.queue1[-2]
            dp_bp01 = self.queue1[-1]
            dp_bp01.compute_metrics(dp_bm01)
            if (
                dp_bp01.dx_manhatten <= MAX_DISTANCE_STAY_POINT
            ):  # Decision based on current sdp uses queue1
                self.sdp_queue.append(dp_bp01)
                self.queue1[-2] = DataPoint(
                    latitude=np.mean([p.latitude for p in self.sdp_queue]),
                    longitude=np.mean([p.longitude for p in self.sdp_queue]),
                    time=self.sdp_queue[-1].time,
                )
                # Discard the latest point. It was merged with the previous point.
                self.queue1.pop()
            else:
                # Not a sdp so reset sdp queue.
                self.sdp_queue = []
                # If we have collected enough dp, then process the new point which is not a sdp...
                if len(self.queue1) >= self.window_size:
                    if self.queue1[-1].dt > self.min_time:
                        ### Save the raw track
                        track0 = Track(self.queue1[:-1].copy())
                        self.tracks.append(track0,)
                        if len(self.path) > 0:
                            track1 = Track(self.path)
                            track1.uuid = track0.uuid
                            ### Save the filtered track
                            self.tracks.append(track1,)
                        # Reset list
                        self.path = []
                        # self.skip = False
                        ### TODO: Do something with the raw track...
                        # Reset queue1. Keep remaining datapoints.
                        self.queue1 = self.queue1[-1:].copy()
                    else:
                        build_path = []
                        start_dp = self.queue1[-3]
                        middle_dp = self.queue1[-2]
                        end_dp = self.queue1[-1]
                        (
                            path_1,
                            path_2,
                            match,
                        ) = self.valhalla.generate_linestrings(
                            start_dp=start_dp,
                            middle_dp=middle_dp,
                            end_dp=end_dp,
                        )
                        # Convert trip length to meters
                        length_1 = (
                            self.valhalla.content_1["trip"]["legs"][0][
                                "summary"
                            ]["length"]
                            * 1000.0
                        )
                        length_1_rel = (
                            np.abs(length_1 - middle_dp.dx_manhatten)
                            / middle_dp.dx_manhatten
                        )
                        print(
                            "Relative measure: ",
                            length_1_rel,
                            length_1,
                            middle_dp.dx_manhatten,
                        )
                        path1_buffered = path_1.buffer(distance=BUFFER_DISTANCE)
                        match = path1_buffered.intersection(path_2).buffer(
                            distance=BUFFER_DISTANCE
                        )
                        new_point = False
                        if length_1_rel <= 5.0:
                            for point in path_1.coords:
                                p = Point(point)
                                if match.contains(p):
                                    build_path.append([point[0], point[1]])
                                    new_point = True
                            if new_point:
                                # Set current 'path1' endpoint as the new starting point
                                # next time this method is called.
                                # self.queue1[-2].latitude = build_path[-1][0]
                                # self.queue1[-2].longitude = build_path[-1][1]
                                for pair in build_path:
                                    self.path.append(
                                        DataPoint(
                                            latitude=pair[0],
                                            longitude=pair[1],
                                            time=None,
                                        )
                                    )
                            else:
                                self.path.append(middle_dp)
                                self.path.append(end_dp)
                        else:
                            self.path.append(middle_dp)
                            self.path.append(end_dp)
        else:
            # Discard the latest point. The acceleration is too large...
            self.queue0.pop()
            self.queue1.pop()

    # def process(self):
    #     datapoints0 = []
    #     # Take the latest datapoints
    #     # --> First time:
    #     if len(self.queue0) == self.buffer_size:
    #         datapoints0 = self.queue0
    #     # --> Proceeding:
    #     elif len(self.queue0) > self.buffer_size:
    #         datapoints0 = [self.queue0[-1]]
    #     # Check consistency of latest datapoints
    #     for i in range(0, len(datapoints0)):
    #         # Accerleration is in meters per. second.
    #         # Upper bound on the acceleration is hardcoded for now...
    #         # TODO: Should be decided dynamically, based on datapoints in buffer...
    #         if np.abs(datapoints0[i].acceleration) <= MAX_ACCELERATION:
    #             # Merge datapoints if they are close in terms of the manhatten distance
    #             if datapoints0[i].dx_manhatten <= MAX_DISTANCE_STAY_POINT:
    #                 self.stay_datapoint.append(datapoints0[i])
    #             else:
    #                 # New datapoint is not a stay datapoint
    #                 if len(self.stay_datapoint) == 0:
    #                     dp = DataPoint(
    #                         latitude = datapoints0[i].latitude,
    #                         longitude = datapoints0[i].longitude,
    #                         time = datapoints0[i].time,
    #                     )
    #                     dp.compute_metrics(self.queue1[-1])
    #                     self.queue1.append(dp)
    #                 else:
    #                     sdp = DataPoint(
    #                         latitude = np.mean([p.latitude for p in self.stay_datapoint]),
    #                         longitude = np.mean([p.longitude for p in self.stay_datapoint]),
    #                         time = self.stay_datapoint[-1].time,
    #                     )
    #                     self.queue1.append(sdp)
    #                     self.stay_point = []

    #     if len(self.queue1) == self.window_size:
    #         for i in range(0, self.window_size):
    #             if self.queue1[i].dt > self.min_time:
    #                 ### Save the raw track for visualization purposes
    #                 self.tracks.append(
    #                     Track(self.queue1[:i].copy()),
    #                 )
    #                 ### TODO: Do something with raw track...
    #                 # Reset queue1. Keep remaining datapoints.
    #                 self.queue1 = self.queue1[i:].copy()
    #                 print("1 Queue 1: ", self.queue1)
    #     if len(self.queue1) >= self.window_size:
    #         if self.queue1[-1].dt > self.min_time:
    #             ### Save the raw track
    #             track0 = Track(self.queue1[:-1].copy())
    #             self.tracks.append(
    #                 track0,
    #             )
    #             track1 = Track(self.path)
    #             track1.uuid = track0.uuid
    #             ### Save the filtered track
    #             self.tracks.append(
    #                 track1,
    #             )
    #             # Reset list
    #             self.path = []
    #             self.skip = False
    #             ### TODO: Do something with the raw track...
    #             # Reset queue1. Keep remaining datapoints.
    #             self.queue1 = self.queue1[-1:].copy()
    #             print("2 Queue 1: ", self.queue1)
    #         else:
    #             build_path = []

    #             start_dp = self.queue1[-3]
    #             middle_dp = self.queue1[-2]
    #             end_dp = self.queue1[-1]
    #             path_1, path_2, match = self.valhalla.generate_linestrings(
    #                 start_dp = start_dp,
    #                 middle_dp = middle_dp,
    #                 end_dp = end_dp,
    #             )
    #             if not path_1 is None:
    #                 rel = self.valhalla.content_1["trip"]["legs"][0]["summary"]["length"] * 1000. # Convert to meters
    #                 if middle_dp.dx_manhatten != 0:
    #                     rel_m = np.abs(rel - middle_dp.dx_manhatten) / middle_dp.dx_manhatten
    #                     print("Relative measure: ", rel_m, rel, middle_dp.dx_manhatten)
    #                 else:
    #                     rel_m = 0
    #                 path1_buffered = path_1.buffer(distance = BUFFER_DISTANCE)
    #                 match = path1_buffered.intersection(path_2).buffer(distance = BUFFER_DISTANCE)
    #                 new_point = False
    #                 if rel_m <= 5. and not self.skip:
    #                     for point in path_1.coords:
    #                         p = Point(point)
    #                         if match.contains(p):
    #                             build_path.append([point[0], point[1]])
    #                             new_point = True
    #                     if new_point:
    #                         # Set current 'path1' endpoint as the new starting point
    #                         # next time this method is called.
    #                         # self.queue1[-2].latitude = build_path[-1][0]
    #                         # self.queue1[-2].longitude = build_path[-1][1]
    #                         for pair in build_path:
    #                             self.path.append(
    #                                 DataPoint(
    #                                     latitude = pair[0],
    #                                     longitude = pair[1],
    #                                     time = None,
    #                                 )
    #                             )
    #                 else:
    #                     self.skip = True

    # def process(self):
    #     datapoints0 = []
    #     # Take the latest datapoints
    #     # --> First time:
    #     if len(self.queue0) == self.buffer_size:
    #         datapoints0 = self.queue0
    #     # --> Proceeding:
    #     elif len(self.queue0) > self.buffer_size:
    #         datapoints0 = [self.queue0[-1]]
    #     # Check consistency of latest datapoints
    #     for dp0 in datapoints0:
    #         print(dp0)
    #         # Accerleration is in meters per. second.
    #         # Upper bound on the acceleration is hardcoded for now...
    #         # TODO: Should be decided dynamically, based on datapoints in buffer...
    #         print(np.abs(dp0.acceleration))
    #         if np.abs(dp0.acceleration) <= MAX_ACCELERATION:
    #             self.queue1.append(dp0)
    #             if len(self.queue1) == self.window_size:
    #                 for i in range(0, self.window_size):
    #                     if self.queue1[i].dt > self.min_time:
    #                         ### Save the raw track for visualization purposes
    #                         self.tracks.append(
    #                             Track(self.queue1[:i].copy()),
    #                         )
    #                         ### TODO: Do something with raw track...
    #                         # Reset queue1. Keep remaining datapoints.
    #                         self.queue1 = self.queue1[i:].copy()
    #                         print("1 Queue 1: ", self.queue1)
    #             if len(self.queue1) >= self.window_size:
    #                 if self.queue1[-1].dt > self.min_time:
    #                     ### Save the raw track
    #                     track0 = Track(self.queue1[:-1].copy())
    #                     self.tracks.append(
    #                         track0,
    #                     )
    #                     track1 = Track(self.path)
    #                     track1.uuid = track0.uuid
    #                     ### Save the filtered track
    #                     self.tracks.append(
    #                         track1,
    #                     )
    #                     # Reset list
    #                     self.path = []
    #                     self.skip = False
    #                     ### TODO: Do something with the raw track...
    #                     # Reset queue1. Keep remaining datapoints.
    #                     self.queue1 = self.queue1[-1:].copy()
    #                     print("2 Queue 1: ", self.queue1)
    #                 else:
    #                     build_path = []

    #                     start_dp = self.queue1[-3]
    #                     middle_dp = self.queue1[-2]
    #                     end_dp = self.queue1[-1]
    #                     path_1, path_2, match = self.valhalla.generate_linestrings(
    #                         start_dp = start_dp,
    #                         middle_dp = middle_dp,
    #                         end_dp = end_dp,
    #                     )
    #                     if not path_1 is None:
    #                         rel = self.valhalla.content_1["trip"]["legs"][0]["summary"]["length"] * 1000. # Convert to meters
    #                         if middle_dp.dx_manhatten != 0:
    #                             rel_m = np.abs(rel - middle_dp.dx_manhatten) / middle_dp.dx_manhatten
    #                             print("Relative measure: ", rel_m, rel, middle_dp.dx_manhatten)
    #                         else:
    #                             rel_m = 0
    #                         path1_buffered = path_1.buffer(distance = BUFFER_DISTANCE)
    #                         match = path1_buffered.intersection(path_2).buffer(distance = BUFFER_DISTANCE)
    #                         new_point = False
    #                         if rel_m <= 5. and not self.skip:
    #                             for point in path_1.coords:
    #                                 p = Point(point)
    #                                 if match.contains(p):
    #                                     build_path.append([point[0], point[1]])
    #                                     new_point = True
    #                             if new_point:
    #                                 # Set current 'path1' endpoint as the new starting point
    #                                 # next time this method is called.
    #                                 # self.queue1[-2].latitude = build_path[-1][0]
    #                                 # self.queue1[-2].longitude = build_path[-1][1]
    #                                 for pair in build_path:
    #                                     self.path.append(
    #                                         DataPoint(
    #                                             latitude = pair[0],
    #                                             longitude = pair[1],
    #                                             time = None,
    #                                         )
    #                                     )
    #                         else:
    #                             self.skip = True
