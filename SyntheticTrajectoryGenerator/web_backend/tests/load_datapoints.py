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
import os

# from ..backend.models import DataPoint
# from ..backend.serializers import DataPointSerializer
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pylab as plt
# import numpy as np
import gpxpy
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from settings import EPOCH

# import requests         # pip install requests
# from requests.auth import HTTPBasicAuth


# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
logging.basicConfig(level=logging.DEBUG)


# ------------------------------------------------------------------------------#
def parse_gpx(rel_path):
    """"""
    sorted_data = list()
    for gpx_file in os.listdir(rel_path):
        filepath = os.path.join(rel_path, gpx_file)
        with open(filepath, "r") as f:
            gpx = gpxpy.parse(f)
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        d = {
                            "timestamp": (point.time - EPOCH).total_seconds(),
                            "point": point,
                        }
                        sorted_data.append(d)
    sorted_data.sort(key=lambda x: x.get("timestamp"))
    points = []
    for item in sorted_data:
        latitude = item["point"].latitude
        longitude = item["point"].longitude
        time = item["point"].time
        points.append([latitude, longitude, time])
    return points


# def load_datapoints(datapoints):
#     lst = []
#     for datapoint in datapoints:
#         latitude = datapoint[0]
#         longitude = datapoint[1]
#         time = datapoint[2]

#         data = {
#             "longitude": float(longitude),
#             "latitude": float(latitude),
#             "external_timestamp": str(time)
#         }
#         dp = DataPointSerializer(data = data)
#         dp.is_valid(raise_exception=True)
#         dp.save()
#     #     lst.append(dp)
#     # DataPoint.objects.bulk_create(lst)

# def main():
#     rel_path = "./gpsdata"
#     datapoints = parse_gpx(rel_path)
#     load_datapoints(datapoints)
