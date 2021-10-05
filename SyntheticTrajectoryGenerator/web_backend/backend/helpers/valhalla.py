# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
#                     Website    : Nicklas.xyz                                 #
#                     Github     : github.com/NicklasXYZ                       #
# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import json
import logging

import requests
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
from shapely.geometry import LineString

# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#


# import matplotlib.pyplot as plt
# from shapely.geometry import LineString
# from descartes import PolygonPatch
# from shapely.ops import shared_paths

# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
logging.basicConfig(level=logging.DEBUG)
OPTIMIZED_ROUTE_URL = "http://localhost:8002/optimized_route"
DEFAULT_HEADERS = {"Content-type": "application/json"}
# Use six degrees of precision when using Valhalla for routing
VALHALLA_PRECISION = 1.0 / 1e6
# Max distance two geometrics objects can be away from each other and still be
# characterized as being intersecting
BUFFER_DISTANCE = 10.0 / 1e6


class ValhallaInterface:
    def __init__(self) -> None:
        self.content_1 = None
        self.content_2 = None

    def decode_polyline(self, polyline_string):
        index = 0
        latitude = 0
        longitude = 0
        coordinates = []
        changes = {"latitude": 0, "longitude": 0}
        # Coordinates have variable length when encoded, so just keep
        # track of whether we have hit the end of the string. In each
        # while loop iteration a single coordinate is decoded.
        while index < len(polyline_string):
            # Gather latitude/longitufe changes, store them in a dictionary to apply them later
            for unit in ["latitude", "longitude"]:
                shift, result = 0, 0
                while True:
                    byte = ord(polyline_string[index]) - 63
                    index += 1
                    result |= (byte & 0x1F) << shift
                    shift += 5
                    if not byte >= 0x20:
                        break
                if result & 1:
                    changes[unit] = ~(result >> 1)
                else:
                    changes[unit] = result >> 1
            latitude += changes["latitude"]
            longitude += changes["longitude"]
            coordinates.append(
                [VALHALLA_PRECISION * latitude, VALHALLA_PRECISION * longitude],
            )
        return coordinates

    def send_optimized_route_request(self, dp1, dp2):
        def build_optimized_route_request(dp1, dp2):
            return json.dumps(
                {
                    "locations": [
                        {"lat": dp1.latitude, "lon": dp1.longitude},
                        {"lat": dp2.latitude, "lon": dp2.longitude},
                    ],
                    "costing": "pedestrian",
                    "directions_options": {"units": "kilometers"},
                }
            )

        d = build_optimized_route_request(dp1, dp2)
        response = requests.post(
            OPTIMIZED_ROUTE_URL, data=d, headers=DEFAULT_HEADERS,
        )
        if response.status_code == 200:
            content = json.loads(response.content)
        else:
            content = None
        return content

    def extract_route_data(self, content):
        pass

    def generate_linestrings(self, start_dp, middle_dp, end_dp):
        self.content_1 = self.send_optimized_route_request(start_dp, middle_dp)
        self.content_2 = self.send_optimized_route_request(start_dp, end_dp)
        if not self.content_1 is None and not self.content_2 is None:
            path_1 = LineString(
                self.decode_polyline(
                    self.content_1["trip"]["legs"][0]["shape"]
                ),
            )
            path_2 = LineString(
                self.decode_polyline(
                    self.content_2["trip"]["legs"][0]["shape"]
                ),
            )
            path1_buffered = path_1.buffer(distance=BUFFER_DISTANCE)
            match = path1_buffered.intersection(path_2).buffer(
                distance=BUFFER_DISTANCE
            )
            return path_1, path_2, match
        else:
            return None, None, None


# rel = content_1["trip"]["legs"][0]["summary"]["length"] * 1000. # Convert to meters
# if middle_dp.dx_manhatten != 0:
#     rel_m = np.abs(rel - middle_dp.dx_manhatten) / middle_dp.dx_manhatten
#     print("Relative measure: ", rel_m, rel, middle_dp.dx_manhatten)
# else:
#     rel_m = 0

# path1_buffered = path_1.buffer(distance = distance)
# match = path1_buffered.intersection(path_2).buffer(distance = distance)
# new_point = False
# if rel_m <= 5. and not self.skip:
#     for point in path_1.coords:
#         p = Point(point)
#         if match.contains(p):
#             path.append([point[0], point[1]])
#             new_point = True
#     if new_point:
#         # Set current 'path1' endpoint as the new starting point
#         # next time this method is called.
#         # self.queue1[-2].latitude = build_path[-1][0]
#         # self.queue1[-2].longitude = build_path[-1][1]
#         for pair in build_path:
#             self.full_path.append(
#                 DataPoint(
#                     latitude = pair[0],
#                     longitude = pair[1],
#                     time = None,
#                 )
#             )
# else:
#     self.skip = True
