# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
#                     Website    : Nicklas.xyz                                 #
#                     Github     : github.com/NicklasXYZ                       #
# ------------------------------------------------------------------------------#
# Resources
#  - Distance between two (lat, lon) points:
#    https://www.movable-type.co.uk/scripts/latlong.html
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import logging
from itertools import tee

import geohash2 as gh
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
import numpy as np
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from settings import EARTH_RADIUS

# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
logging.basicConfig(level=logging.DEBUG)


def to_radians(number: float):
    """Convert degrees to radians."""
    return np.pi * number / 180.0


def to_polar(latitude: float, longitude: float):
    """Convert longitude and latitude coordinates to polar coordinates."""
    r = np.sqrt(latitude ** 2 + longitude ** 2)
    phi = np.arctan2(latitude, longitude)
    return r, phi


def distance(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
    haversine: bool = False,
) -> float:
    """"""
    # If the points are too distant, then compute the haversine distance, as it is more precise:
    if (haversine == True) or (
        abs(latitude_1 - latitude_2) > 0.2
        or abs(longitude_1 - longitude_2) > 0.2
    ):
        return haversine_distance(
            latitude_1, longitude_1, latitude_2, longitude_2
        )
    else:
        return euclidean_distance(
            latitude_1, longitude_1, latitude_2, longitude_2
        )


def haversine_distance(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Haversine distance between two points, expressed in meters."""
    d_lat = to_radians(latitude_1 - latitude_2)
    d_lon = to_radians(longitude_1 - longitude_2)
    lat1 = to_radians(latitude_1)
    lat2 = to_radians(latitude_2)
    a = np.sin(d_lat / 2) * np.sin(d_lat / 2) + np.sin(d_lon / 2) * np.sin(
        d_lon / 2
    ) * np.cos(lat1) * np.cos(lat2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = EARTH_RADIUS * c
    return d


def euclidean_distance(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """"""
    dx = to_radians(latitude_2 - latitude_1) * np.cos(
        to_radians(latitude_1 + latitude_2) / 2
    )
    dy = to_radians(longitude_2 - longitude_1)
    distance_2d = np.sqrt(dx * dx + dy * dy)
    return distance_2d * EARTH_RADIUS


def manhatten_distance(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    dx = to_radians(latitude_2 - latitude_1) * np.cos(
        to_radians(latitude_1 + latitude_2) / 2
    )
    dy = to_radians(longitude_2 - longitude_1)
    distance_2d = np.abs(dx) + np.abs(dy)
    return distance_2d * EARTH_RADIUS


def compass_bearing(
    latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float
):
    """"""
    d_longitude = to_radians(longitude_2 - longitude_1)
    longitude_1 = to_radians(longitude_1)
    longitude_2 = to_radians(longitude_2)
    latitude_1 = to_radians(latitude_1)
    latitude_2 = to_radians(latitude_2)
    y = np.sin(d_longitude) * np.cos(latitude_2)
    x = np.cos(latitude_1) * np.sin(latitude_2) - np.sin(latitude_1) * np.cos(
        latitude_2
    ) * np.cos(d_longitude)
    brng = np.rad2deg(np.arctan2(y, x))
    compass_bearing = (brng + 360) % 360
    return compass_bearing


def geohash_encode(latitude: float, longitude: float, precision: int) -> str:
    """
    Geohash length  |  Cell width   |    Cell height 1                  ≤ 5,000km    x    5000km 2                  ≤
    1,250km    x     625km 3                  ≤ 156km      x     156km 4                  ≤ 39.1km     x      19.5km 5.

    ≤ 4.89km     x       4.89km 6                  ≤ 1.22km     x       0.61km 7                  ≤ 153m       x
    153m 8                  ≤ 38.2m      x      19.1m 9                  ≤ 4.77m      x       4.77m 10                 ≤
    1.19m      x       0.596m 11                 ≤ 149mm      x     149mm 12                 ≤ 37.2mm     X
    18.6mm.
    """
    return gh.encode(latitude, longitude, precision)


def geohash_decode(geohash: str):
    return gh.decode(geohash)


def destination(
    latitude_1: float, longitude_1: float, bearing: float, distance: float
):
    # Predict the destination given current location (latitude_1, longitude_1),
    # bearing and a distance to a possible arrival location.
    latitude_1 = to_radians(latitude_1)
    longitude_1 = to_radians(longitude_1)
    latitude_2 = np.asin(
        np.sin(latitude_1) * np.cos(distance / EARTH_RADIUS)
        + np.cos(latitude_1) * np.sin(distance / EARTH_RADIUS) * np.cos(bearing)
    )
    longitude_2 = longitude_1 + np.atan2(
        np.sin(bearing) * np.sin(distance / EARTH_RADIUS) * np.cos(latitude_1),
        np.cos(distance / EARTH_RADIUS)
        - np.sin(latitude_1) * np.sin(latitude_2),
    )
    latitude_2 = np.rad2deg(latitude_2)
    longitude_2 = np.rad2deg(longitude_2)
    return latitude_2, longitude_2


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    now, nxt = tee(iterable)
    next(nxt, None)
    return zip(now, nxt)
