# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
#                     Website    : Nicklas.xyz                                 #
#                     Github     : github.com/NicklasXYZ                       #
# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
#
import json
import logging
import uuid

# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
import numpy as np
import requests
from infostop import Infostop
from shapely.geometry import LineString, Point

from .DataPoint import DataPoint
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from .kalman import ConstantVelocityKlamanFilter

# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
logging.basicConfig(level=logging.DEBUG)


# Six degrees of precision in valhalla
inv = 1.0 / 1e6

# Decode an encoded string
# def decode(encoded):
#     decoded = []; previous = [0, 0]; i = 0
#     # For each byte...
#     while i < len(encoded):
#         # For each coord (lat, lon)
#         ll = [0, 0]
#         for j in [0, 1]:
#             shift = 0
#             byte = 0x20
#             # Keep decoding bytes until you have this coord
#             while byte >= 0x20:
#                 print("ll: ", ll)
#                 byte = ord(encoded[i]) - 63
#                 i += 1
#                 ll[j] |= (byte & 0x1f) << shift
#                 shift += 5
#             # Get the final value adding the previous offset and remember it for the next
#             ll[j] = previous[j] + (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
#             previous[j] = ll[j]
#             # Scale by the precision and chop off long coords also flip the positions so
#             # its the far more standard lon,lat instead of lat,lon
#             decoded.append([float("%.6f" % (ll[1] * inv)), float("%.6f" % (ll[0] * inv))])
#         # Hand back the list of coordinates
#         return decoded


def decode_polyline(polyline_str):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {"latitude": 0, "longitude": 0}
    # Coordinates have variable length when encoded, so just keep
    # track of whether we've hit the end of the string. In each
    # while loop iteration, a single coordinate is decoded.
    while index < len(polyline_str):
        # Gather lat/lon changes, store them in a dictionary to apply them later
        for unit in ["latitude", "longitude"]:
            shift, result = 0, 0

            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if not byte >= 0x20:
                    break

            if result & 1:
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = result >> 1

        lat += changes["latitude"]
        lng += changes["longitude"]

        coordinates.append((lat / 1000000.0, lng / 1000000.0))

    return coordinates


# TODO: Rename 'Segment' --> 'Track'
class Segment:
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
    def __init__(self, min_time: float, min_distance: float) -> None:
        # Set parameters that are used for spatio-temporal segmentation
        self.min_time = min_time
        self.min_distance = min_distance
        # Initialize queue which hold several datapoints for processing
        self.queue = []
        self.datapoints = []
        # Initialize constant velocity Kalman filter
        self.filter = ConstantVelocityKlamanFilter()
        # Initialize list which is going to hold separate segments of datapoints
        self.segments = []
        # Initialize statistics trackers for mean and variance
        logging.debug(
            "Initializing 'DataPointStream' with min_time = "
            + str(self.min_time)
            + ", min_distance = "
            + str(self.min_distance)
        )
        self.model = Infostop()
        self.labels = None

    # def update(self, dp_p01):
    #     # Process the data if we have more than one datapoint in the queue
    #     if len(self.queue) >= 1:
    #         dp_m01 = self.queue[-1]
    #         # if not dp_m01.time > dp_p01.time:
    #         dp_p01.compute_metrics(dp_m01)
    #         print(
    #             "Manhatten distance: ", dp_p01.dx_euclidean,
    #             "Euclidean distance: ", dp_p01.dx_manhatten,
    #             "Change in time    : ", dp_p01.dt,
    #             "Speed             : ", dp_p01.velocity,
    #             "Acceleration      : ", dp_p01.acceleration,
    #         )
    #         print()
    #         if np.abs(dp_p01.acceleration) <= 2.5: # Accerleration is in meters per. second
    #             # Update the Kalman filter and retrive the new state
    #             # based on the given (latitude, longitude) measurements
    #             predicted_value = self.filter.predict_and_update(
    #                 [dp_p01.latitude, dp_p01.longitude],
    #                 dp_p01.dt,
    #             )
    #             # Create a corresponding 'KalmanFilteredDataPoint' object
    #             # that holds the new (latitude, longitude) state
    #             dp_p01.create_filtered_datapoint(
    #                 latitude = predicted_value[0][0],
    #                 longitude = predicted_value[1][0],
    #             )
    #             # Compute additional metrics based on this new (latitude,
    #             # longitude) state
    #             dp_p01.filtered_datapoint.compute_metrics(
    #                 dp_m01.filtered_datapoint
    #             )
    #             # Collect the datapoint for stop point detection
    #             self.datapoints.append(
    #                 [
    #                     dp_p01.filtered_datapoint.latitude,
    #                     dp_p01.filtered_datapoint.longitude,
    #                     dp_p01.get_unix_timestamp(),
    #                 ],
    #             )
    #             # Determine if we should segement the stream of datapoints into a new segement
    #             if dp_p01.dt > self.min_time:
    #                 # Perform map matching using the newly formed segement
    #                 d = {
    #                     "longitude": [p.longitude for  p in self.queue],
    #                     "latitude": [p.latitude for  p in self.queue],
    #                     "time": [p.get_unix_timestamp() for  p in self.queue],
    #                 }
    #                 # d = {
    #                 #     "longitude": [p.get_geohashed_datapoint(precision = 9).longitude for  p in self.queue],
    #                 #     "latitude": [p.get_geohashed_datapoint(precision = 9).latitude for  p in self.queue],
    #                 #     "time": [p.get_unix_timestamp() for  p in self.queue],
    #                 # }
    #                 # d = {
    #                 #     "longitude": [p.filtered_datapoint.longitude for  p in self.queue],
    #                 #     "latitude": [p.filtered_datapoint.latitude for  p in self.queue],
    #                 #     "time": [p.filtered_datapoint.get_unix_timestamp() for  p in self.queue],
    #                 # }
    #                 df = pd.DataFrame(data = d)

    #                 df_trip_for_meili = df[['longitude', 'latitude', 'time']].copy()
    #                 df_trip_for_meili.columns = ['lon', 'lat', 'time']

    #                 meili_coordinates = df_trip_for_meili.to_json(orient = "records")
    #                 meili_head = '{"shape":'
    #                 # meili_tail = ""","search_radius": 10, "shape_match":"map_snap", "costing":"auto", "format":"osrm"}"""
    #                 # meili_tail = ""","search_radius": 10, "shape_match":"map_snap", "costing":"pedestrian", "format":"osrm"}"""
    #                 # meili_tail = ',"search_radius": 5, "shape_match":"map_snap", "costing":"pedestrian", "format":"osrm"}'
    #                 # meili_tail = ',"search_radius": 50, "shape_match":"walk_or_snap", "costing":"pedestrian", "format":"osrm"}'
    #                 meili_durations = ""","durations":""" + json.dumps([self.queue[i].dt for i in range(1, len(self.queue))]) + ""","begin_time":""" + str(self.queue[0].get_unix_timestamp())
    #                 meili_tail = ""","search_radius": 10, "shape_match":"map_snap", "costing":"pedestrian", "use_timestamps": true, "format":"osrm"}"""
    #                 meili_request_body = meili_head + meili_coordinates + meili_durations + meili_tail
    #                 # Sending a request to Valhalla's Meili
    #                 url = "http://localhost:8002/trace_route"
    #                 # url = "http://localhost:8002/trace_attributes"
    #                 headers = {"Content-type": "application/json"}
    #                 data = str(meili_request_body)
    #                 r = requests.post(url, data = data, headers = headers)
    #                 # print("Request: ", r.content)

    #                 print("Writing std output to file...")
    #                 f = open("./testout.txt", "w")
    #                 f.write(json.dumps(json.loads(r.content), indent=4))
    #                 f.close()
    #                 # quit()

    #                 # Parsing the response from Valhalla's Meili
    #                 if r.status_code == 200:
    #                     self.queue = []
    #                     response_text = json.loads(r.text)
    #                     resp = str(response_text['tracepoints'])
    #                     resp = resp.replace("'waypoint_index': None", "'waypoint_index': '#'")
    #                     resp = resp.replace("None", "{'matchings_index': '#', 'name': '', 'waypoint_index': '#', 'alternatives_count': 0, 'distance': 0, 'location': [0.0, 0.0]}")
    #                     resp = resp.replace("'", '"')
    #                     resp = json.dumps(resp)
    #                     resp = json.loads(resp)
    #                     df_response = pd.read_json(resp)
    #                     df_response = df_response[['name', 'distance', 'location']]
    #                     df_trip_optimized = pd.merge(
    #                         df_trip_for_meili, df_response,
    #                         left_index = True,
    #                         right_index = True,
    #                     )
    #                     df_trip_optimized["longitude"] = 0.0
    #                     df_trip_optimized["latitude"] = 0.0
    #                     print("df shape:", df_trip_optimized.shape, df.shape)
    #                     for i, _ in df_trip_optimized.iterrows():
    #                         df_trip_optimized.at[i, 'longitude'] = df_trip_optimized.at[i,'location'][0]
    #                         df_trip_optimized.at[i, 'latitude'] = df_trip_optimized.at[i,'location'][1]
    #                         # Overwrite all the geopoints that couldn't be snapped to road
    #                         if df_trip_optimized.at[i, 'longitude'] == 0.0:
    #                         #     df_trip_optimized.at[i, 'longitude'] = df_trip_optimized.at[i, 'lon']
    #                         #     df_trip_optimized.at[i, 'latitude'] = df_trip_optimized.at[i, 'lat']
    #                         # # self.queue[i].latitude = df_trip_optimized.at[i, 'latitude']
    #                         # # self.queue[i].longitude = df_trip_optimized.at[i, 'longitude']
    #                         # if df_trip_optimized.at[i, 'latitude'] is None or df_trip_optimized.at[i, 'longitude'] is None:
    #                             print("A point is None!")
    #                         else:

    #                             self.queue.append(
    #                                 DataPoint(
    #                                     latitude = df_trip_optimized.at[i, 'latitude'],
    #                                     longitude = df_trip_optimized.at[i, 'longitude'],
    #                                     time = None,
    #                                 )
    #                             )
    #                     # df_trip_optimized = df_trip_optimized.drop(['location', 'lon', 'lat'], 1)

    #                     # for i, _ in df_trip_optimized.iterrows():
    #                     #     self.queue[i].latitude = df_trip_optimized.at[i, 'latitude']
    #                     #     self.queue[i].longitude = df_trip_optimized.at[i, 'longitude']

    #                 # Save the segement
    #                 self.segments.append(
    #                     Segment(self.queue),
    #                 )

    #                 # Reset
    #                 self.queue = []
    #                 self.filter = ConstantVelocityKlamanFilter()
    #             else:
    #                 self.queue.append(dp_p01)
    #                 # Collect the datapoint for stop point detection
    #                 self.datapoints.append(
    #                     [
    #                         dp_p01.filtered_datapoint.latitude,
    #                         dp_p01.filtered_datapoint.longitude,
    #                         dp_p01.get_unix_timestamp(),
    #                     ],
    #                 )
    #     else:
    #         predicted_value = self.filter.predict_and_update(
    #             [dp_p01.latitude, dp_p01.longitude],
    #             dp_p01.dt,
    #         )
    #         # Create a corresponding 'KalmanFilteredDataPoint' object
    #         # that holds the new (latitude, longitude) state
    #         dp_p01.create_filtered_datapoint(
    #             latitude = predicted_value[0][0],
    #             longitude = predicted_value[1][0],
    #         )
    #         self.queue.append(dp_p01)
    #         self.datapoints.append(
    #             [
    #                 dp_p01.filtered_datapoint.latitude,
    #                 dp_p01.filtered_datapoint.longitude,
    #                 dp_p01.get_unix_timestamp(),
    #             ],
    #         )

    def get_linestring(self, w):
        new_queue = self.queue.copy()
        build_path = []
        url = "http://localhost:8002/optimized_route"
        for k in range(0, len(self.queue) - w):
            start = k
            # middle = (k + 1)
            end1 = k + w - 2
            end2 = k + w - 1
            d1 = {
                # "locations":[{"lat":p.latitude, "lon": p.longitude} for p in self.queue],
                "locations": [
                    {
                        "lat": new_queue[start].latitude,
                        "lon": new_queue[start].longitude,
                    },
                    {
                        "lat": new_queue[end1].latitude,
                        "lon": new_queue[end1].longitude,
                    },
                ],
                "costing": "pedestrian",
            }
            d2 = {
                # "locations":[{"lat":p.latitude, "lon": p.longitude} for p in self.queue],
                "locations": [
                    {
                        "lat": new_queue[start].latitude,
                        "lon": new_queue[start].longitude,
                    },
                    {
                        "lat": new_queue[end2].latitude,
                        "lon": new_queue[end2].longitude,
                    },
                ],
                "costing": "pedestrian",
            }
            data1 = json.dumps(d1)
            data2 = json.dumps(d2)
            headers = {"Content-type": "application/json"}
            r1 = requests.post(url, data=data1, headers=headers)
            r2 = requests.post(url, data=data2, headers=headers)
            if r1.status_code == 200 and r2.status_code == 200:
                # for leg in content["trip"]["legs"]:
                #     path = decode_polyline(leg["shape"])
                #     full_path.append(new_queue[start])
                distance = 10 / 1000000.0
                content1 = json.loads(r1.content)
                content2 = json.loads(r2.content)

                # print("Writing std output to file...")
                # f = open("./testout.txt", "w")
                # f.write(json.dumps(content1, indent = 4))
                # f.close()

                if float(content1["trip"]["legs"][0]["summary"]["time"]) != 0:
                    route_vel = (
                        float(content1["trip"]["legs"][0]["summary"]["length"])
                        * 1000.0
                        / float(content1["trip"]["legs"][0]["summary"]["time"])
                    )
                else:
                    route_vel = 0
                print(
                    "Leg lst len      :",
                    len(content1["trip"]["legs"]),
                    "\n",
                    "Leg path len     :",
                    float(content1["trip"]["legs"][0]["summary"]["length"])
                    * 1000.0,
                    "\n",
                    "Routing time     :",
                    content1["trip"]["legs"][0]["summary"]["time"],
                    "\n",
                    "Routing velocity :",
                    route_vel,
                    "\n",
                    "Actual length euc:",
                    new_queue[end1].dx_euclidean,
                    "\n",
                    "Actual length man:",
                    new_queue[end1].dx_manhatten,
                    "\n",
                    "Actual time      :",
                    new_queue[end1].dt,
                    "\n",
                    "Actual velocity  :",
                    new_queue[end1].velocity,
                    "\n",
                )
                # rel = float(content1["trip"]["legs"][0]["summary"]["time"])
                rel = (
                    float(content1["trip"]["legs"][0]["summary"]["length"])
                    * 1000.0
                )
                # if new_queue[end1].dt != 0:
                #     rel_m = rel / new_queue[end1].dt
                #     print("Relative measure: ", rel_m)
                # else:
                #     rel_m = 0
                if new_queue[end1].dx_manhatten != 0:
                    rel_m = (
                        np.abs(rel - new_queue[end1].dx_manhatten)
                        / new_queue[end1].dx_manhatten
                    )
                    print(
                        "Relative measure: ",
                        rel_m,
                        rel,
                        new_queue[end1].dx_manhatten,
                    )
                else:
                    rel_m = 0

                path1 = LineString(
                    decode_polyline(content1["trip"]["legs"][0]["shape"])
                )
                path2 = LineString(
                    decode_polyline(content2["trip"]["legs"][0]["shape"])
                )
                path1_buffered = path1.buffer(distance=distance)
                match = path1_buffered.intersection(path2).buffer(
                    distance=distance
                )
                # patch1 = PolygonPatch(path1_buffered, fc='blue', ec='blue', alpha=0.5, zorder=2)
                # ax.add_patch(patch1)
                # x,y=track1.xy
                # ax.plot(x,y,'b.')
                # x,y=track2.xy
                # ax.plot(x,y,'g.')
                # plt.show()
                match = path1_buffered.intersection(path2).buffer(distance)
                # print("Match: ", dir(match))
                # print("Match: ", match.simplify(0.5))
                # print("Match: ", match.geoms.shape_factory())
                # match = match.simplify(0.75, preserve_topology = False)
                # print("Match: ", match)
                # print("Envelope: ", match.envelope)

                # fig=plt.figure()
                # ax = fig.add_subplot(111)
                # patch1 = PolygonPatch(match, fc = "green", ec = "green", alpha = 0.5, zorder = 2)
                # ax.add_patch(patch1)
                # x,y=path1.xy
                # ax.plot(x,y,'b.')
                # x,y=path2.xy
                # ax.plot(x,y,'g.')
                # plt.show()
                new_point = False
                if rel_m <= 10:
                    # print(match)
                    # for polygon in match:
                    # build_path = []
                    for point in path1.coords:
                        p = Point(point)
                        if match.contains(p):
                            build_path.append([point[0], point[1]])
                            new_point = True
                    if new_point:
                        new_queue[end1].latitude = build_path[-1][0]
                        new_queue[end1].longitude = build_path[-1][1]
                else:
                    break
                    # new_queue[end1].latitude = new_queue[start].latitude
                    # new_queue[end1].longitude = new_queue[start].longitude
                    # new_queue[start].compute_metrics(new_queue[end2])
                    # new_queue[end1] = new_queue[start]

        if len(build_path) > 0:
            full_path = []
            for pair in build_path:
                full_path.append(
                    DataPoint(latitude=pair[0], longitude=pair[1], time=None,)
                )
            self.segments.append(Segment(full_path),)

        # if len(build_path) > 0:
        #     full_path = []
        #     for pair in build_path:
        #         full_path.append(
        #             DataPoint(
        #                 latitude = pair[0],
        #                 longitude = pair[1],
        #                 time = None,
        #             )
        #         )
        #     self.segments.append(
        #         Segment(full_path),
        #     )

    def update(self, dp_p01):
        # Process the data if we have more than one datapoint in the queue
        if len(self.queue) >= 1:
            dp_m01 = self.queue[-1]
            # if not dp_m01.time > dp_p01.time:
            dp_p01.compute_metrics(dp_m01)
            # print(
            #     "Manhatten distance: ", dp_p01.dx_euclidean,
            #     "Euclidean distance: ", dp_p01.dx_manhatten,
            #     "Change in time    : ", dp_p01.dt,
            #     "Speed             : ", dp_p01.velocity,
            #     "Acceleration      : ", dp_p01.acceleration,
            # )
            # print()
            if (
                np.abs(dp_p01.acceleration) <= 2.5
            ):  # Accerleration is in meters per. second
                # Update the Kalman filter and retrive the new state
                # based on the given (latitude, longitude) measurements
                predicted_value = self.filter.predict_and_update(
                    [dp_p01.latitude, dp_p01.longitude], dp_p01.dt,
                )
                # Create a corresponding 'KalmanFilteredDataPoint' object
                # that holds the new (latitude, longitude) state
                dp_p01.create_filtered_datapoint(
                    latitude=predicted_value[0][0],
                    longitude=predicted_value[1][0],
                )
                # Compute additional metrics based on this new (latitude,
                # longitude) state
                dp_p01.filtered_datapoint.compute_metrics(
                    dp_m01.filtered_datapoint
                )
                # Collect the datapoint for stop point detection
                self.datapoints.append(
                    [
                        dp_p01.filtered_datapoint.latitude,
                        dp_p01.filtered_datapoint.longitude,
                        dp_p01.get_unix_timestamp(),
                    ],
                )
                # Determine if we should segement the stream of datapoints into a new segement
                if dp_p01.dt > self.min_time:
                    # Perform map matching using the newly formed segement
                    # d = {
                    #     "longitude": [p.longitude for  p in self.queue],
                    #     "latitude": [p.latitude for  p in self.queue],
                    #     "time": [p.get_unix_timestamp() for  p in self.queue],
                    # }
                    # # d = {
                    # #     "longitude": [p.get_geohashed_datapoint(precision = 9).longitude for  p in self.queue],
                    # #     "latitude": [p.get_geohashed_datapoint(precision = 9).latitude for  p in self.queue],
                    # #     "time": [p.get_unix_timestamp() for  p in self.queue],
                    # # }
                    # # d = {
                    # #     "longitude": [p.filtered_datapoint.longitude for  p in self.queue],
                    # #     "latitude": [p.filtered_datapoint.latitude for  p in self.queue],
                    # #     "time": [p.filtered_datapoint.get_unix_timestamp() for  p in self.queue],
                    # # }
                    # df = pd.DataFrame(data = d)

                    # df_trip_for_meili = df[['longitude', 'latitude', 'time']].copy()
                    # df_trip_for_meili.columns = ['lon', 'lat', 'time']

                    # meili_coordinates = df_trip_for_meili.to_json(orient = "records")
                    # meili_head = '{"shape":'
                    # # meili_tail = ""","search_radius": 10, "shape_match":"map_snap", "costing":"auto", "format":"osrm"}"""
                    # # meili_tail = ""","search_radius": 10, "shape_match":"map_snap", "costing":"pedestrian", "format":"osrm"}"""
                    # # meili_tail = ',"search_radius": 5, "shape_match":"map_snap", "costing":"pedestrian", "format":"osrm"}'
                    # # meili_tail = ',"search_radius": 50, "shape_match":"walk_or_snap", "costing":"pedestrian", "format":"osrm"}'
                    # meili_durations = ""","durations":""" + json.dumps([self.queue[i].dt for i in range(1, len(self.queue))]) + ""","begin_time":""" + str(self.queue[0].get_unix_timestamp())
                    # meili_tail = ""","search_radius": 10, "shape_match":"map_snap", "costing":"pedestrian", "use_timestamps": true, "format":"osrm"}"""
                    # meili_request_body = meili_head + meili_coordinates + meili_durations + meili_tail
                    # # Sending a request to Valhalla's Meili
                    # url = "http://localhost:8002/trace_route"
                    # # url = "http://localhost:8002/trace_attributes"
                    # headers = {"Content-type": "application/json"}
                    # data = str(meili_request_body)
                    # r = requests.post(url, data = data, headers = headers)
                    # # print("Request: ", r.content)

                    # print("Writing std output to file...")
                    # f = open("./testout.txt", "w")
                    # f.write(json.dumps(json.loads(r.content), indent=4))
                    # f.close()
                    # # quit()

                    # # Parsing the response from Valhalla's Meili
                    # if r.status_code == 200:
                    #     self.queue = []
                    #     response_text = json.loads(r.text)
                    #     resp = str(response_text['tracepoints'])
                    #     resp = resp.replace("'waypoint_index': None", "'waypoint_index': '#'")
                    #     resp = resp.replace("None", "{'matchings_index': '#', 'name': '', 'waypoint_index': '#', 'alternatives_count': 0, 'distance': 0, 'location': [0.0, 0.0]}")
                    #     resp = resp.replace("'", '"')
                    #     resp = json.dumps(resp)
                    #     resp = json.loads(resp)
                    #     df_response = pd.read_json(resp)
                    #     df_response = df_response[['name', 'distance', 'location']]
                    #     df_trip_optimized = pd.merge(
                    #         df_trip_for_meili, df_response,
                    #         left_index = True,
                    #         right_index = True,
                    #     )
                    #     df_trip_optimized["longitude"] = 0.0
                    #     df_trip_optimized["latitude"] = 0.0
                    #     print("df shape:", df_trip_optimized.shape, df.shape)
                    #     for i, _ in df_trip_optimized.iterrows():
                    #         df_trip_optimized.at[i, "longitude"] = df_trip_optimized.at[i,"location"][0]
                    #         df_trip_optimized.at[i, "latitude"] = df_trip_optimized.at[i,"location"][1]
                    #         # Overwrite all the geopoints that couldn't be snapped to road
                    #         if df_trip_optimized.at[i, "longitude"] == 0.0:
                    #         #     df_trip_optimized.at[i, 'longitude'] = df_trip_optimized.at[i, 'lon']
                    #         #     df_trip_optimized.at[i, 'latitude'] = df_trip_optimized.at[i, 'lat']
                    #         # # self.queue[i].latitude = df_trip_optimized.at[i, 'latitude']
                    #         # # self.queue[i].longitude = df_trip_optimized.at[i, 'longitude']
                    #         # if df_trip_optimized.at[i, 'latitude'] is None or df_trip_optimized.at[i, 'longitude'] is None:
                    #             print("A point is None!")
                    #         else:

                    #             self.queue.append(
                    #                 DataPoint(
                    #                     latitude = df_trip_optimized.at[i, "latitude"],
                    #                     longitude = df_trip_optimized.at[i, "longitude"],
                    #                     time = None,
                    #                 )
                    #             )
                    #     df_trip_optimized = df_trip_optimized.drop(["location", "lon", "lat"], 1)

                    # {
                    # "coordinates": [
                    #     [ 13.288925, 52.438512 ],
                    #     [ 13.288938, 52.438938 ],
                    #     [ 13.288904, 52.439169 ],
                    #     [ 13.288821, 52.439398 ],
                    #     [ 13.288824, 52.439491 ],
                    #     [ 13.288824, 52.439563 ]
                    # ]
                    # }
                    # d = {
                    #     "coordinates": [[p.longitude, p.latitude ] for  p in self.queue],
                    # }
                    # data = json.dumps(d)
                    # url = "http://localhost:8002?search_radius=50&mode=pedestrian"
                    # headers = {"Content-type": "application/json"}
                    # r = requests.post(url, data = data, headers = headers)
                    # print("Writing std output to file...")
                    # f = open("./testout.txt", "w")
                    # f.write(json.dumps(json.loads(r.content), indent = 4))
                    # f.close()
                    #
                    # url = "http://localhost:8002/optimized_route?mode=pedestrian&json="

                    ### Save the raw track
                    self.segments.append(Segment(self.queue),)

                    ###
                    w = 3
                    if len(self.queue) > w:
                        self.get_linestring(w)

                        # new_queue[middle] = DataPoint(
                        #     latitude = np.mean([p[0] for p in path]),
                        #     longitude = np.mean([p[1] for p in path]),
                        #     time = None,
                        # )
                        # print("Writing std output to file...")
                        # f = open("./testout.txt", "w")
                        # content = json.loads(r.content)
                        # f.write(json.dumps(content, indent = 4))
                        # f.close()

                        # self.queue = []

                        # for leg in content["trip"]["legs"]:
                        #     path = decode_polyline(leg["shape"])
                        #     for pair in path:
                        #         self.queue.append(
                        #             DataPoint(
                        #                 latitude = pair[0],
                        #                 longitude = pair[1],
                        #                 time = None,
                        #             )
                        #         )

                    # Save the segement
                    # self.segments.append(
                    #     Segment(new_queue),
                    # )
                    # self.segments.append(
                    #     Segment(full_path),
                    # )

                    # Reset
                    self.queue = []
                    self.filter = ConstantVelocityKlamanFilter()
                else:
                    self.queue.append(dp_p01)
                    # Collect the datapoint for stop point detection
                    self.datapoints.append(
                        [
                            dp_p01.filtered_datapoint.latitude,
                            dp_p01.filtered_datapoint.longitude,
                            dp_p01.get_unix_timestamp(),
                        ],
                    )
        else:
            predicted_value = self.filter.predict_and_update(
                [dp_p01.latitude, dp_p01.longitude], dp_p01.dt,
            )
            # Create a corresponding 'KalmanFilteredDataPoint' object
            # that holds the new (latitude, longitude) state
            dp_p01.create_filtered_datapoint(
                latitude=predicted_value[0][0], longitude=predicted_value[1][0],
            )
            self.queue.append(dp_p01)
            self.datapoints.append(
                [
                    dp_p01.filtered_datapoint.latitude,
                    dp_p01.filtered_datapoint.longitude,
                    dp_p01.get_unix_timestamp(),
                ],
            )

    def retrieve_clusters(self):
        # Fit model to all the collected datapoints and predict
        # possible stop points
        # self.labels = self.model.fit_predict(
        #     np.array(self.datapoints),
        # )
        # return self.labels
        return []
