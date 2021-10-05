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
# from infostop import compute_intervals
import os

import gpxpy
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import requests  # pip install requests
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from DataPoint import DataPoint
from DataPointStream import DataPointStream
from requests.auth import HTTPBasicAuth
from settings import EPOCH

filepath = "map.html"

# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
# logging.basicConfig(level = logging.DEBUG)


# ------------------------------------------------------------------------------#
#                                GLOBAL VARIABLES                              #
# ------------------------------------------------------------------------------#
PASSWORD = "pAssw0rd!"
USERNAME = "nsa@email.com"

# ENDPOINTS
# Base url for local testing
base_url = "http://127.0.0.1:8000/api/v1/"

# Compile all urls into a dict
urls = {
    "base": base_url,
    "users_url": base_url + "users/",
    "users_me_url": base_url + "users/me/",
    "datapoints_url": base_url + "users/me/datapoints/",
}


# ------------------------------------------------------------------------------#
#                            GLOBAL HELPER METHODS                             #
# ------------------------------------------------------------------------------#
def send_request(url, data, method, headers={}, auth=None):
    """A helper method for sending requests..."""
    include_headers = {
        "content-type": "application/json",
    }
    include_headers.update(headers)
    if method == "GET":
        if not auth is None:
            response = requests.get(url, headers=include_headers, auth=auth,)
        else:
            response = requests.get(url, headers=include_headers,)
    elif method == "POST":
        data = json.dumps(data)
        if not auth is None:
            response = requests.post(
                url, data=data, headers=include_headers, auth=auth,
            )
        else:
            response = requests.post(url, data=data, headers=include_headers,)
    elif method == "PATCH":
        data = json.dumps(data)
        if not auth is None:
            response = requests.patch(
                url, data=data, headers=include_headers, auth=auth,
            )
        else:
            response = requests.patch(url, data=data, headers=include_headers,)
    elif method == "OPTIONS":
        if not auth is None:
            response = requests.options(
                url, headers=include_headers, auth=auth,
            )
        else:
            response = requests.options(url, headers=include_headers,)
    elif method == "DELETE":
        if not auth is None:
            response = requests.delete(url, headers=include_headers, auth=auth,)
        else:
            response = requests.delete(url, headers=include_headers,)
    return response


# ------------------------------------------------------------------------------#
#                                  USER TESTS                                  #
# ------------------------------------------------------------------------------#
def create_user():
    """Test user creation endpoint..."""
    data = {
        "email": USERNAME,
        "password": PASSWORD,
    }
    response = send_request(url=urls["users_url"], data=data, method="POST",)
    return response


def get_user(user_data):
    """"""
    data = {}
    headers = {}
    response = send_request(
        url=urls["users_me_url"],
        data=data,
        method="GET",
        headers=headers,
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        # auth = HTTPBasicAuth(user_data["email"], user_data["password"]),
    )
    return response


# ------------------------------------------------------------------------------#
#                                 ACTIVITY TESTS                               #
# ------------------------------------------------------------------------------#
def create_datapoint(data):
    """"""
    headers = {}
    response = send_request(
        url=urls["datapoints_url"],
        data=data,
        method="POST",
        headers=headers,
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        # auth = HTTPBasicAuth(user_data["email"], user_data["password"]),
    )
    return response


def get_datapoint(datapoint_uid):
    """"""
    if datapoint_uid is None:  # Get all activites
        url = urls["datapoints_url"]
    else:  # Get a single activity
        url = urls["datapoints_url"] + f"{datapoint_uid}/"
    data = {}
    headers = {}
    response = send_request(
        url=url,
        data=data,
        method="GET",
        headers=headers,
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        # auth = HTTPBasicAuth(user_data["email"], user_data["password"]),
    )
    return response


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
    points = list()
    for item in sorted_data:
        p = DataPoint.from_gpx(item["point"])
        points.append(p)
    return points


# def compute_intervals(labels, times, max_time_between=86400):
#     """Compute stop and moves intervals from the list of labels.

#     Parameters
#     ----------
#         labels: 1d np.array of integers
#         times: 1d np.array of integers. `len(labels) == len(times)`.
#     Returns
#     -------
#         intervals : array-like (shape=(N_intervals, 3))
#             Columns are "label", "start_time", "end_time"

#     """
#     assert len(labels) == len(times), '`labels` and `times` must match in length'

#     # Input and output sequence
#     trajectory = np.hstack([labels.reshape(-1, 1), times.reshape(-1,1)])
#     final_trajectory = []

#     # Init values
#     loc_prev, t_start = trajectory[0]
#     t_end = t_start

#     # Loop through trajectory
#     for loc, time in trajectory[1:]:

#         # if the location name has not changed update the end of the interval
#         if (loc == loc_prev) and (time - t_end) < max_time_between:
#             t_end = time

#         # if the location name has changed build the interval
#         else:
#             final_trajectory.append([loc_prev, t_start,  t_end])

#             # and reset values
#             t_start = time
#             t_end = time

#         # update current values
#         loc_prev = loc

#     # Add last group
#     if loc_prev == -1:
#         final_trajectory.append([loc_prev, t_start,  t_end])

#     return final_trajectory


def main():
    counter = 0
    rel_path = "./gpsdata"
    datapoints = parse_gpx(rel_path)
    dps = DataPointStream(min_time=600, min_distance=300,)
    for point in datapoints:
        dps.update(point)
    print("Number of segments: ", len(dps.segments))
    fig, axs = plt.subplots(1, 1)
    y0 = []
    for segment in dps.segments:
        x0 = []
        y0 = []
        x1 = []
        y1 = []
        # x2 = []; y2 = []
        # times = []
        # test = []
        # tracker = Tracker()
        if counter in [i for i in range(len(dps.segments))]:
            # if counter in [i for i in range(67, 68)]:
            for k in range(len(segment)):
                # predicted_value = tracker.predict_and_update([point.latitude * 1000., point.longitude * 1000.], point.dt)
                # predicted_value = tracker.predict_and_update([float(point.decoded_geohash_lat) * 1000., float(point.decoded_geohash_lon) * 1000.], point.dt)

                # print("Predicted: ", predicted_value[0][0] / 1000., predicted_value[1][0] / 1000.)
                # print("    ---> : ", point.latitude, point.longitude)

                # y0.append([predicted_value[0][0] / 1000., predicted_value[1][0] / 1000., point.get_timestamp()])
                # test.append([predicted_value[0][0] / 1000., predicted_value[1][0] / 1000.])
                # times.append(point.get_timestamp())
                # x0.append(float(point.latitude))
                # y0.append(float(point.longitude))
                # x1.append(predicted_value[0][0] / 1000.)
                # y1.append(predicted_value[1][0] / 1000.)
                # x2.append(float(point.decoded_geohash_lat))
                # y2.append(float(point.decoded_geohash_lon))
                # print(k, len(segment))
                # if segment[k].corrected_dx_euclidean > 100:
                #     print(
                #         "Velocity               :", segment[k].velocity,
                #         "Corrected velocity     :", segment[k].corrected_velocity, "\n",
                #         "Acceleration           :", segment[k].acceleration,
                #         "Corrected acceleration :", segment[k].corrected_acceleration,
                #     )
                x0.append(float(segment[k].corrected_latitude))
                y0.append(float(segment[k].corrected_longitude))
                x1.append(float(segment[k].latitude))
                y1.append(float(segment[k].longitude))
                # x1.append(predicted_value[0][0] / 1000.)
                # y1.append(predicted_value[1][0] / 1000.)
            # break

            plt.plot(y0, x0)
            plt.scatter(y1, x1, s=2, color="red")

            # plt.scatter([y0[0]], [x0[0]], s = 2, color = "blue")
            # plt.scatter([y0[-1]], [x0[-1]], s = 2, color = "yellow")
            # plt.plot(y2, x2, color = "yellow")
            # plt.xlim([10.36, 10.44])
            # plt.ylim([55.34, 55.41])
        counter += 1
    # model = Infostop()
    # labels = model.fit_predict(np.array(test))
    # plt.scatter(y0, x0, s = 5, color = "red")
    # plt.xlim([10.36, 10.44])
    # plt.ylim([55.34, 55.41])
    # print("Segment length: ", len(test), "Labels:", labels.shape)
    # print("Labels        : ", labels)
    # medians = model.compute_label_medians()
    # print("Label medians : ", medians)
    # plt.scatter([medians[0][1], medians[1][1]], [medians[0][0], medians[1][0]], s = 5, color = "blue")
    plt.show()

    # traj = compute_intervals(labels = np.array(labels), times = np.array(times))
    # for item in traj:
    #     print(item)
    # print(labels)
    # folmap = plot_map(dps.model)
    # folmap.m.save(filepath)
    # webbrowser.open(filepath)

    # print(y0)


# def main():
#     counter = 0
#     rel_path = "./gpsdata"
#     datapoints = parse_gpx(rel_path)
#     dps = DataPointStream(
#         min_time = 600,
#         min_distance = 100,
#     )
#     for point in datapoints:
#         dps.update(point)
#     print("Number of segments: ", len(dps.segments))
#     fig, axs = plt.subplots(3, 1)
#     for segment in dps.segments:
#         x0 = []; y0 = []
#         x1 = []; y1 = []
#         x2 = []; y2 = []
#         t = []
#         tracker = Tracker()
#         if counter in [i for i in range(len(dps.segments))]:
#             for point in segment:
#                 []
#                 print(float(point.get_timestamp()))
#                 # self.decoded_geohash_lat, self.decoded_geohash_lon
#                 t.append(float(point.get_timestamp()))
#                 x0.append(float(point.latitude))
#                 y0.append(float(point.longitude))
#                 # x1.append(( float(point.latitude) - float(point.decoded_geohash_lat) )/ 1)
#                 # y1.append(( float(point.longitude) - float(point.decoded_geohash_lon) )/ 1)
#                 x1.append( float(point.avg_lat) )
#                 y1.append( float(point.avg_lon) )
#                 x2.append(float(point.decoded_geohash_lat))
#                 y2.append(float(point.decoded_geohash_lon))
#             # plt.plot(y1, x1)
#             # plt.scatter(y0, x0, s = 5)
#             # plt.plot([i for i in range(len(x1))], x1)
#             # plt.plot([i for i in range(len(y1))], y1)

#             # axs[0].plot([i for i in range(len(x1))], x1)
#             # axs[1].plot([i for i in range(len(y1))], y1)
#             axs[0].plot(y1, x1)
#             # axs[1].plot(t, y1)
#             axs[1].scatter(y0, x0, color = "blue", s = 2.5)

#             # axs[2].scatter(y0[0], x0[0], color = "yellow", label = "start")
#             # axs[2].scatter(y0[-1], x0[-1], color = "green", label = "end")
#             # axs[2].legend()


#             # axs[2].scatter(y2, x2, color="red", s = 5)
#             # axs[2].scatter(y2, x2, color="red", s = 5)
#             axs[1].set_ylim([55.34, 55.41])
#             axs[1].set_xlim([10.36, 10.44])

#             # plt.plot(y1, x1)
#             # plt.scatter(y0, x0, s = 5)
#             # break
#         counter += 1
#     # plt.xlim([10.36, 10.44])
#     # plt.ylim([55.34, 55.41])
#     plt.show()
#     # Create a user data
#     # response = create_user()
#     # for datapoint in datapoints:
#     #     data = {
#     #         "longitude": float(datapoint.longitude),
#     #         "latitude": float(datapoint.latitude),
#     #         "external_timestamp": str(datapoint.time)
#     #     }
#     #     # # Create activity data
#     #     respose = create_datapoint(data)
#     #     print("Response: ", respose)


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
