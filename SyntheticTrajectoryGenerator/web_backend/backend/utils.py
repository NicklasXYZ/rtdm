# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import random
import uuid
from datetime import datetime

# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
from infostop import utils

from .helpers.DataPoint import DataPoint
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from .helpers.DataPointStream import DataPointStream

# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django std. lib.               #
# ------------------------------------------------------------------------------#
#
# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django extensions              #
# ------------------------------------------------------------------------------#


def rand_24_bit():
    """Returns a random 24-bit integer."""
    return random.randrange(0, 16 ** 6)


def color_dec():
    """Alias of rand_24 bit()"""
    return rand_24_bit()


def color_hex(num=rand_24_bit()):
    """Returns a 24-bit int in hex."""
    return "%06x" % num


def color_rgb(num=rand_24_bit()):
    """Returns three 8-bit numbers, one for each channel in RGB."""
    hx = color_hex(num)
    barr = bytearray.fromhex(hx)
    return (barr[0], barr[1], barr[2])


def coords_to_geojson(data):
    # feature_array_one = []
    # for datapoint in data:
    #     feature = {
    #         "type": "Feature",
    #         "properties": {
    #             "color": "#33C9EB",
    #         },
    #         "geometry": {
    #             "type": "LineString",
    #             "coordinates": (datapoint["latitude"], datapoint["longitude"]),
    #         },
    #     }
    #     feature_array_one.extend([feature])
    # line_data = {
    #     "type":"FeatureCollection",
    #     "features": feature_array_one
    # }
    feature_array_two = []
    for datapoint in data:
        feature = {
            "type": "Feature",
            "properties": {"color": "#33C9EB",},
            "geometry": {
                "type": "Point",
                "coordinates": [datapoint["longitude"], datapoint["latitude"]],
            },
        }
        feature_array_two.append(feature)
    point_data = {"type": "FeatureCollection", "features": feature_array_two}
    return point_data


# def segments_to_geojson(data):
#     # feature_array_one = []
#     # for datapoint in data:
#     #     feature = {
#     #         "type": "Feature",
#     #         "properties": {
#     #             "color": "#33C9EB",
#     #         },
#     #         "geometry": {
#     #             "type": "LineString",
#     #             "coordinates": (datapoint["latitude"], datapoint["longitude"]),
#     #         },
#     #     }
#     #     feature_array_one.extend([feature])
#     # line_data = {
#     #     "type":"FeatureCollection",
#     #     "features": feature_array_one
#     # }
#     feature_array_two = []
#     for datapoint in data:
#         feature = {
#             "type": "Feature",
#             "properties": {
#                 "color": "#33C9EB",
#             },
#             "geometry": {
#                 "type": "Point",
#                 "coordinates": [datapoint["longitude"], datapoint["latitude"]],
#             },
#         }
#         feature_array_two.append(feature)
#     point_data = {
#         "type":"FeatureCollection",
#         "features": feature_array_two
#     }
#     return point_data


def parse_date(text):
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError("No valid date format found")


# def handle_datapoints(points):
#     dps = DataPointStream(
#         min_time = 300,
#         min_distance = 250,
#     )
#     for point in points:
#         time = try_parsing_date(point["external_timestamp"])
#         dp = DataPoint(
#             latitude = point["latitude"],
#             longitude = point["longitude"],
#             time = time,
#         )
#         dps.update(dp)
#     feature_array_one = []
#     for segment in dps.segments:
#         v = []; c = color_dec(); x = color_hex(c)
#         for dp in segment:
#             v.append([dp.longitude, dp.latitude])
#         feature = {
#             "type": "Feature",
#             "properties": {
#                 "color": "#" + str(x), # "#33C9EB",
#             },
#             "geometry": {
#                 "type": "LineString",
#                 "coordinates": v,
#             },
#         }
#         feature_array_one.append(feature)
#     line_data = {
#         "type":"FeatureCollection",
#         "features": feature_array_one
#     }
#     return line_data


def handle_datapoints(points):
    dps = DataPointStream(min_time=600, min_distance=300,)
    new_points = []
    for i in range(1, len(points)):
        time = parse_date(points[i]["external_timestamp"])
        # dp0 = DataPoint(
        #     latitude = points[i - 1]["latitude"],
        #     longitude = points[i]["longitude"],
        #     time = time,
        # )
        # new_points.append(dp0)
        dp1 = DataPoint(
            latitude=points[i]["latitude"],
            longitude=points[i]["longitude"],
            time=time,
        )
        new_points.append(dp1)

        # dp2 = DataPoint(
        #     latitude = points[i]["latitude"],
        #     longitude = points[i - 1]["longitude"],
        #     time = time,
        # )
        # new_points.append(dp2)

    for point in new_points:
        dps.update(point)
        if len(dps.tracks) > 20:
            break

    # for point in points:
    # for point in new_points:
    #     time = parse_date(point["external_timestamp"])
    #     dp = DataPoint(
    #         latitude = point["latitude"],
    #         longitude = point["longitude"],
    #         time = time,
    #     )
    #     dps.update(dp)
    dps.retrieve_clusters()
    # ppoints = dps.model._stat_coords[dps.model._stat_labels >= 0]
    # llabels = dps.model._stat_labels[dps.model._stat_labels >= 0]
    # feature_array_one = []
    # feature_array_two = render_polygons(ppoints, llabels)
    # print("True: All segements: ", len(dps.segments))

    feature_array_one = []
    feature_array_two = []
    print("True: All segements: ", len(dps.tracks))

    for segment in dps.tracks:
        v = []
        c = color_dec()
        x = color_hex(c)
        lon_min = 0  # np.amin([dp.longitude for dp in segment])
        lat_min = 0  # np.amin([dp.latitude for dp in segment])
        lon_max = 0  # np.amax([dp.longitude for dp in segment])
        lat_max = 0  # np.amax([dp.latitude for dp in segment])
        decoded_geohash_lon_min = 0  # np.amin(np.array([dp.decoded_geohash_lon for dp in segment]).astype(np.float))
        decoded_geohash_lat_min = 0  # np.amin(np.array([dp.decoded_geohash_lat for dp in segment]).astype(np.float))
        dx_euclidean_min = 0  # np.amin([dp.dx_euclidean for dp in segment])
        dx_manhatten_min = 0  # np.amin([dp.dx_manhatten for dp in segment])
        dt_min = 0  # np.amin([dp.dt for dp in segment])
        acceleration_min = 0  # np.amin([dp.acceleration for dp in segment])
        velocity_min = 0  # np.amin([dp.velocity for dp in segment])
        compass_bearing_min = (
            0  #  np.amin([dp.compass_bearing for dp in segment])
        )
        # print("Segment length: ", len(segment))
        if len(segment.datapoints) > 0:
            segment.datapoints[0].get_unix_timestamp()
            datapoint_template_dict = {
                "meta": {"id": str(uuid.uuid4()),},
                "raw_data": {
                    "longitude": segment.datapoints[0].longitude,
                    "latitude": segment.datapoints[0].latitude,
                    # "time": str(segment.datapoints[0].time),
                    # "timestamp": segment.datapoints[0].get_unix_timestamp(),
                    "decoded_geohash_lon": 0,  # segment.datapoints[0].decoded_geohash_lon,
                    "decoded_geohash_lat": 0,  # segment.datapoints[0].decoded_geohash_lat,
                    "dx_euclidean": segment.datapoints[0].dx_euclidean,
                    "dx_manhatten": segment.datapoints[0].dx_manhatten,
                    "dt": segment.datapoints[0].dt,
                    "acceleration": segment.datapoints[0].acceleration,
                    "velocity": segment.datapoints[0].velocity,
                    "compass_bearing": segment.datapoints[0].compass_bearing,
                },
                "regularized_data": {
                    "longitude": segment.datapoints[0].longitude - lon_min,
                    "latitude": segment.datapoints[0].latitude - lat_min,
                    # "time": str(segment.datapoints[0].time),
                    # "timestamp": 0,
                    "decoded_geohash_lon": 0,  # float(segment.datapoints[0].decoded_geohash_lon) - decoded_geohash_lon_min,
                    "decoded_geohash_lat": 0,  # float(segment.datapoints[0].decoded_geohash_lat) - decoded_geohash_lat_min,
                    "dx_euclidean": segment.datapoints[0].dx_euclidean
                    - dx_euclidean_min,
                    "dx_manhatten": segment.datapoints[0].dx_manhatten
                    - dx_manhatten_min,
                    "dt": segment.datapoints[0].dt - dt_min,
                    "acceleration": segment.datapoints[0].acceleration
                    - acceleration_min,
                    "velocity": segment.datapoints[0].velocity - velocity_min,
                    "compass_bearing": segment.datapoints[0].compass_bearing
                    - compass_bearing_min,
                },
            }
            for key in datapoint_template_dict["raw_data"]:
                if key != "time":
                    datapoint_template_dict["raw_data"][key] = float(
                        datapoint_template_dict["raw_data"][key]
                    )
            for key in datapoint_template_dict["regularized_data"]:
                if key != "time":
                    datapoint_template_dict["regularized_data"][key] = float(
                        datapoint_template_dict["regularized_data"][key]
                    )
            v.append(datapoint_template_dict)
            for i in range(1, len(segment.datapoints)):
                datapoint_template_dict = {
                    "meta": {"id": str(uuid.uuid4()),},
                    "raw_data": {
                        "longitude": segment.datapoints[i].longitude,
                        "latitude": segment.datapoints[i].latitude,
                        # "time": str(segment.datapoints[i].time),
                        # "timestamp": segment.datapoints[i].get_unix_timestamp(),
                        "decoded_geohash_lon": 0,  # float(segment.datapoints[i].decoded_geohash_lon),
                        "decoded_geohash_lat": 0,  # float(segment.datapoints[i].decoded_geohash_lat),
                        "dx_euclidean": segment.datapoints[i].dx_euclidean,
                        "dx_manhatten": segment.datapoints[i].dx_manhatten,
                        "dt": segment.datapoints[i].dt,
                        "acceleration": segment.datapoints[i].acceleration,
                        "velocity": segment.datapoints[i].velocity,
                        "compass_bearing": segment.datapoints[
                            i
                        ].compass_bearing,
                    },
                    "regularized_data": {
                        "longitude": segment.datapoints[i].longitude - lon_min,
                        "latitude": segment.datapoints[i].latitude - lat_min,
                        # "time": str(segment.datapoints[i].time),
                        # "timestamp": segment.datapoints[i].get_unix_timestamp() - start_time,
                        "decoded_geohash_lon": 0,  # float(segment.datapoints[i].decoded_geohash_lon) - decoded_geohash_lon_min,
                        "decoded_geohash_lat": 0,  # float(segment.datapoints[i].decoded_geohash_lat) - decoded_geohash_lat_min,
                        "dx_euclidean": segment.datapoints[i].dx_euclidean
                        - dx_euclidean_min,
                        "dx_manhatten": segment.datapoints[i].dx_manhatten
                        - dx_manhatten_min,
                        "dt": segment.datapoints[i].dt - dt_min,
                        "acceleration": segment.datapoints[i].acceleration
                        - acceleration_min,
                        "velocity": segment.datapoints[i].velocity
                        - velocity_min,
                        "compass_bearing": segment.datapoints[i].compass_bearing
                        - compass_bearing_min,
                    },
                }
                v.append(datapoint_template_dict)
                for key in datapoint_template_dict["raw_data"]:
                    if key != "time":
                        datapoint_template_dict["raw_data"][key] = float(
                            datapoint_template_dict["raw_data"][key]
                        )
                for key in datapoint_template_dict["regularized_data"]:
                    if key != "time":
                        datapoint_template_dict["regularized_data"][
                            key
                        ] = float(
                            datapoint_template_dict["regularized_data"][key]
                        )
            segment_template_dict = {
                "meta": {
                    "color": "#" + str(x),
                    # "id": str(uuid.uuid4()),
                    "uuid": str(segment.uuid),
                    "min_lon": lon_min,
                    "max_lon": lon_max,
                    "min_lat": lat_min,
                    "max_lat": lat_max,
                },
                "segment": v,
            }
            feature_array_one.append(segment_template_dict)
        else:
            pass
    print("All segments: ", len(feature_array_one))
    return feature_array_one, feature_array_two


def render_polygons(points, labels, color="#FFFFFF", opacity=0.3):
    """Render convex hulls of points in each label-group.
    Parameters
    ----------
        color : str
            Color of stop location areas.
        opacity : float (in [0, 1])
            Opacity of stop location areas.
    """

    def _style_function(feature):
        return {
            "fillColor": feature["properties"]["color"],
            "color": feature["properties"]["color"],
            "weight": 1,
            "fillOpacity": opacity,
        }

    stop_hulls = []
    for stop_idx in set(labels) - {-1}:
        p = points[labels == stop_idx]
        stop_hulls.append(utils.convex_hull(p))

    features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"color": color},
                "geometry": {"type": "Polygon", "coordinates": []},
            }
        ],
    }

    for hull in stop_hulls:
        features["features"][0]["geometry"]["coordinates"].append(
            hull[:, ::-1].tolist()
        )
    return features
