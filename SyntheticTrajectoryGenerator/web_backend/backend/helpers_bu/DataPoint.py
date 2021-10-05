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

# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
import numpy as np

# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from .settings import EPOCH
from .utils import (compass_bearing, euclidean_distance, geohash_decode,
                    geohash_encode, manhatten_distance)

# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
logging.basicConfig(level=logging.DEBUG)


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class GeohashedDataPoint:
    def __init__(
        self, latitude: float, longitude: float, precision: int
    ) -> None:
        # Set the precision of the Geohash
        self.precision = precision
        # A corresponding GeoHash and decoded latitude and longitude coordinates
        self.geohash = None  # Encoded coordinate pair as a geohash
        self.latitude = None  # Decoded geohash latitude coordinate
        self.longitude = None  # Decoded geohash longitude coordinate
        self.geohash_datapoint(latitude, longitude)

    def geohash_datapoint(self, latitude, longitude):
        self.geohash = geohash_encode(
            latitude=latitude, longitude=longitude, precision=self.precision
        )
        decoded_geohash = geohash_decode(self.geohash)
        self.latitude = float(decoded_geohash[0])
        self.longitude = float(decoded_geohash[1])


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class BaseDataPoint:
    # NOTE: Assume WGS84 input latitude and longitude coordinates
    def __init__(self, latitude: float, longitude: float, time=None,) -> None:
        # Set a unique identifier of the datapoint
        self.uuid = uuid.uuid4()
        # Longitude and latitude coordinates
        self.longitude = longitude
        self.latitude = latitude
        # Timestamp (should be a datetime object)
        self.time = time
        if not time is None:
            # Encode cyclical feature: Time of day
            total_seconds = self.time.hour * self.time.minute * self.time.second
            seconds_in_day = 24.0 * 60.0 * 60.0
            self.time_sin = np.sin(2.0 * np.pi * total_seconds / seconds_in_day)
            self.time_cos = np.cos(2.0 * np.pi * total_seconds / seconds_in_day)
        else:
            self.time_sin = None
            self.time_cos = None

    def get_geohashed_datapoint(self, precision):
        return GeohashedDataPoint(
            latitude=self.latitude,
            longitude=self.longitude,
            precision=precision,
        )

    def get_unix_timestamp(self):
        # Retrieve a unix timestamp (in seconds)
        if not self.time is None:
            return (self.time - EPOCH).total_seconds()
        else:
            return None

    @staticmethod
    def from_gpx(gpx_track_point):
        return BaseDataPoint(
            latitude=gpx_track_point.latitude,
            longitude=gpx_track_point.longitude,
            time=gpx_track_point.time,
        )


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class DataPoint(BaseDataPoint):
    # NOTE: Assume WGS84 input latitude and longitude coordinates
    def __init__(self, latitude: float, longitude: float, time=None,) -> None:
        # Compute basic attributes based on latitude, longitude and time alone...
        super().__init__(
            latitude=latitude, longitude=longitude, time=time,
        )
        # Compute additional attributes based on 2 consecutive data points..
        # Change in location: Euclidean distance
        self.dx_euclidean = 0
        # Change in location: Manhatten distance
        self.dx_manhatten = 0
        # Change in time
        self.dt = 0
        # Velocity
        self.velocity = 0
        # Acceleration
        self.acceleration = 0
        # Compass bearing
        self.compass_bearing = 0
        # Reserve a varibale for storing a corresponding filtered "DataPoint"
        # object
        self.filtered_datapoint = None

    def _euclidean_distance(self, previous_datapoint) -> float:
        # Wrapper method
        return euclidean_distance(
            self.latitude,
            self.longitude,
            previous_datapoint.latitude,
            previous_datapoint.longitude,
        )

    def _manhatten_distance(self, previous_datapoint) -> float:
        # Wrapper method
        return manhatten_distance(
            self.latitude,
            self.longitude,
            previous_datapoint.latitude,
            previous_datapoint.longitude,
        )

    def _time_difference(self, previous_datapoint) -> float:
        return np.abs((self.time - previous_datapoint.time).total_seconds())

    def _compute_compass_bearing(self, previous_datapoint) -> float:
        # Wrapper method
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
        if delta_time > 0:
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

    def create_filtered_datapoint(self, latitude, longitude):
        # print(latitude, longitude)
        self.filtered_datapoint = KalmanFilteredDataPoint(
            latitude=latitude, longitude=longitude, time=self.time,
        )

    @staticmethod
    def from_gpx(gpx_track_point):
        return DataPoint(
            latitude=gpx_track_point.latitude,
            longitude=gpx_track_point.longitude,
            time=gpx_track_point.time,
        )


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class KalmanFilteredDataPoint(BaseDataPoint):
    def __init__(self, latitude: float, longitude: float, time=None) -> None:
        super().__init__(
            latitude=latitude, longitude=longitude, time=time,
        )
        # "Corrected" attributes arising from the application of a Kalman filter
        # self.datapoint = BaseDataPoint(
        #     latitude = latitude,
        #     longitude = longitude,
        #     time = time,
        # )
        # Compute additional attributes based on 2 consecutive data points..
        # Change in location: Euclidean distance
        self.dx_euclidean = 0
        # Change in location: Manhatten distance
        self.dx_manhatten = 0
        # Change in time
        self.dt = 0
        # Velocity
        self.velocity = 0
        # Acceleration
        self.acceleration = 0
        # Compass bearing
        self.compass_bearing = 0

    def _euclidean_distance(self, previous_datapoint) -> float:
        # Wrapper method
        return euclidean_distance(
            self.latitude,
            self.longitude,
            previous_datapoint.latitude,
            previous_datapoint.longitude,
        )

    def _manhatten_distance(self, previous_datapoint) -> float:
        # Wrapper method
        return manhatten_distance(
            self.latitude,
            self.longitude,
            previous_datapoint.latitude,
            previous_datapoint.longitude,
        )

    def _time_difference(self, previous_datapoint) -> float:
        return np.abs((self.time - previous_datapoint.time).total_seconds())

    def _compute_compass_bearing(self, previous_datapoint) -> float:
        # Wrapper method
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
        if delta_time > 0:
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


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    """"""
