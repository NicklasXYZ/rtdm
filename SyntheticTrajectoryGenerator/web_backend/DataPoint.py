# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
#                     Website    : Nicklas.xyz                                 #
#                     Github     : github.com/NicklasXYZ                       #
# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
import numpy as np
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from settings import EPOCH
from utils import (compass_bearing, euclidean_distance, geohash_decode,
                   geohash_encode, manhatten_distance, to_polar)

# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
# logging.basicConfig(level = logging.DEBUG)


# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
class DataPoint:
    def __init__(self, latitude: float, longitude: float, time) -> None:
        # Assume WGS84 input latitude and longitude coordinates
        # Longitude and latitude coordinates
        self.longitude = longitude
        self.latitude = latitude
        # Polar coordinates
        self.r, self.phi = to_polar(latitude, longitude)
        # Timestamp
        self.time = time
        # A corresponding GeoHash and decoded latitude and longitude coordinates
        encoded_geohash = geohash_encode(latitude, longitude)
        decoded_geohash = geohash_decode(encoded_geohash)
        self.geohash = encoded_geohash
        self.decoded_geohash_lat, self.decoded_geohash_lon = decoded_geohash
        # NOTE: We need 2 consecutive points to compute these components:
        # Change in euclidean distance
        self.dx_euclidean = 0
        # Change in manhatten distance
        self.dx_manhatten = 0
        # Change in time
        self.dt = 0
        # Acceleration
        self.acceleration = 0
        # Velocity
        self.velocity = 0
        # Compass bearing
        self.compass_bearing = 0

    def get_timestamp(self):
        return (self.time - EPOCH).total_seconds()

    def _euclidean_distance(self, previous_datapoint) -> float:
        return euclidean_distance(
            self.latitude,
            self.longitude,
            previous_datapoint.latitude,
            previous_datapoint.longitude,
        )

    def _manhatten_distance(self, previous_datapoint) -> float:
        return manhatten_distance(
            self.latitude,
            self.longitude,
            previous_datapoint.latitude,
            previous_datapoint.longitude,
        )

    def _time_difference(self, previous_datapoint) -> float:
        return np.abs((self.time - previous_datapoint.time).total_seconds())

    def _compute_compass_bearing(self, previous_datapoint) -> float:
        return compass_bearing(
            self.latitude,
            self.longitude,
            previous_datapoint.latitude,
            previous_datapoint.longitude,
        )

    def compute_metrics(self, previous_datapoint):
        delta_time = self._time_difference(previous_datapoint)
        delta_x_euclidean = self._euclidean_distance(previous_datapoint)
        delta_x_manhatten = self._manhatten_distance(previous_datapoint)
        compass_bearing = self._compute_compass_bearing(previous_datapoint)
        velocity = 0
        delta_velocity = 0
        acceleration = 0
        if delta_time != 0:
            velocity = delta_x_euclidean / delta_time
            delta_velocity = velocity - previous_datapoint.velocity
            acceleration = delta_velocity / delta_time
        self.dt = delta_time
        self.velocity = velocity
        self.acceleration = acceleration
        self.dx_euclidean = delta_x_euclidean
        self.dx_manhatten = delta_x_manhatten
        self.compass_bearing = compass_bearing
        return self

    @staticmethod
    def from_gpx(gpx_track_point):
        return DataPoint(
            latitude=gpx_track_point.latitude,
            longitude=gpx_track_point.longitude,
            time=gpx_track_point.time,
        )


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    """"""
