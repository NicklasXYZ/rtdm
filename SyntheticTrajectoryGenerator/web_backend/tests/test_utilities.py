# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import json
from datetime import datetime

import numpy as np
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
import requests  # pip install requests
from faker import Faker  # pip install faker
from requests.auth import HTTPBasicAuth

# ------------------------------------------------------------------------------#
#                                GLOBAL VARIABLES                              #
# ------------------------------------------------------------------------------#
# User password
PASSWORD = "pAssw0rd!"

# Base url for local testing
BASE_URL = "http://localhost:8000/api/v1/"
print(
    "#------------------------------------------------------------------------------#"
)
print("# NOTE: TESTINING ", BASE_URL)
print(
    "#------------------------------------------------------------------------------#"
)

# Compile all urls into a dict
urls = {
    "base": BASE_URL,
    "users": BASE_URL + "users/",
    "users_me": BASE_URL + "users/me/",
    "users_me_datapoints": BASE_URL + "users/me/datapoints/",
    "users_me_remove_datapoints": BASE_URL + "users/remove_datapoints/",
}


def send_request(url, data, method, headers={}, auth=None):
    """A helper method for sending requests..."""
    response = None
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


def create_datapoint_data_json():
    """"""
    timestamp = str(datetime.utcnow())
    return {
        "longitude": np.random.uniform(0.00, 5.00),
        "latitude": np.random.uniform(5.00, 10.00),
        "external_timestamp": timestamp,
    }


# ------------------------------------------------------------------------------#
# TEST: USER MODEL FUNCTIONALITY & METHODS                                     #
# ------------------------------------------------------------------------------#
class User:
    """"""

    def __init__(self):
        """"""
        self.user_uid = None

    def create_user(self):
        """"""
        # Create json user data
        faker = Faker()
        profile = faker.profile()
        self.user_data = {
            "email": profile["mail"],
            "password": PASSWORD,
        }
        headers = {}
        response = send_request(
            url=urls["users"],
            data=self.user_data,
            method="POST",
            headers=headers,
        )
        response_data = json.loads(response.content)
        print(
            response.status_code,
            "<-- Status code.",
            "Function:",
            self.create_user.__name__,
        )
        # Assertions
        assert response.status_code == 201
        # Set the state
        assert response_data["email"] == self.user_data["email"].lower().strip()
        # Save essential data
        self.user_uid = response_data["uid"]

    def get_user(self):
        """"""
        data = {}
        headers = {}
        response = send_request(
            url=urls["users_me_url"],
            data=data,
            method="GET",
            headers=headers,
            auth=HTTPBasicAuth(
                self.user_data["email"], self.user_data["password"]
            ),
        )
        response_data = json.loads(response.content)
        print(
            response.status_code,
            "<-- Status code.",
            "Function:",
            self.get_user.__name__,
        )
        # Assertions
        assert response.status_code == 200
        assert response_data["uid"] == self.user_uid
        return response_data

    def create_datapoint(self, lon, lat, timestamp):
        """"""
        data = {
            "longitude": lon,
            "latitude": lat,
            "external_timestamp": timestamp,
        }
        headers = {}
        response = send_request(
            url=urls["users_me_datapoints"],
            data=data,
            method="POST",
            headers=headers,
            auth=HTTPBasicAuth(
                self.user_data["email"], self.user_data["password"]
            ),
        )
        response_data = json.loads(response.content)
        print(
            response.status_code,
            "<-- Status code.",
            "Function:",
            self.create_datapoint.__name__,
        )
        return response_data
