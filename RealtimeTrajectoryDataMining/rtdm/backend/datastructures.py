import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import backend.models as django_models
import backend.plotutils as plotutils
import backend.utils as utils
import folium
import geohash_hilbert as ghh
import gpxpy
import msgpack
import numpy as np
import pandas as pd
import utm
from backend.typealias import CoordinatePair
from django.conf import settings
from shapely.geometry import LineString


class BaseDataPoint:
    """A representation of a point with time and location-specific data."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        external_timestamp: Union[None, datetime] = None,
        user: Union[None, str] = None,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            latitude (float): The latitude coordinate of a location in degrees.
            longitude (float): The longitude coordinate of a location in \
                degrees.
            external_timestamp (Union[None, datetime], optional): The \
                timestamp of the observation. Defaults to None.
            user (Union[None, str], optional): A unique identifier of the \
                entity the datapoint belongs to. Defaults to None.
        """
        self.latitude = latitude
        self.longitude = longitude
        # Keep an identifier of the user that the datapoint belongs to
        self.user = user
        # Keep an identifier of the trajectory that the datapoint belongs to
        self.trajectory: Union[None, str] = None
        # A label assigned to the datapoint. For example:
        # - "1" for anomaly
        # - "0" for normal
        self.label: Union[None, str] = None
        # Set the timestamp of the datapoint (it should be a datetime object)
        self.external_timestamp = external_timestamp
        # The weight assigned to the datapoint. Defaults to 0.0
        self.weight: float = 0.0

    def get_geohashed_datapoint(
        self, precision: int, bits_per_char: int,
    ) -> "GeohashedDataPoint":
        """
        Create a corresponding datapoint containing geohashing information.

        Args:
            precision (int): Precision of the geohash.
            bits_per_char (int): Number of bits each character in the geohash \
                encodes.

        Returns:
            GohashedDataPoint:
        """
        return GeohashedDataPoint(
            latitude=self.latitude,
            longitude=self.longitude,
            precision=precision,
            bits_per_char=bits_per_char,
        )

    def get_unix_timestamp(self) -> Union[None, float]:
        """
        Return a unix timestamp of the timestamp associated with the datapoint.

        Returns:
            Union[None, float]: None if no timestamp is associated with the \
                datapoint, otherwise a unix timestamp is returned, i.e. the \
                time in seconds since January 1st, 1970 at UTC.
        """
        if self.external_timestamp is not None:
            # Retrieve a unix timestamp (in seconds)
            return (self.external_timestamp - settings.EPOCH).total_seconds()
        else:
            return None

    def plot(
        self,
        color: str,
        radius: float = 2.0,
        opacity: float = 1.0,
        center: Union[None, CoordinatePair] = None,
        map_: Union[None, folium.Map] = None,
    ) -> Union[None, folium.Map]:
        """
        Plot the datapoint on a map.

        Args:
            color (str): The color the datapoint is plotted with.
            radius (float, optional): The radius the datapoint is plotted \
                with. Defaults to 2.0.
            opacity (float, optional):  The opacity the datapoint is plotted \
                with. Defaults to 1.0.
            center (Union[None, CoordinatePair], optional): A \
                latitude/longitude coordinate pair that determines the \
                centering of the map that the datapoint should be plotted on.
            map_ (Union[None, folium.Map], optional): A folium map object. \
                Defaults to None.

        Raises:
            ValueError: If either a 'map_' or 'center' function argument is \
                missing.

        Returns:
            Union[None, folium.Map]: A 'folium.Map' object.
        """
        if map_ is None:
            if center is not None:
                map_ = plotutils.setup_map(center=center)
            else:
                raise ValueError("Missing 'center' function argument.")
        if map_ is not None:
            folium.CircleMarker(
                [self.latitude, self.longitude],
                radius=radius,
                color=color,
                opacity=opacity,
                popup=f"Weight   : {self.weight},\n"
                + f"Lat      : {self.latitude},\n"
                + f"Lon      : {self.longitude},\n"
                + f"Timestamp: {self.external_timestamp}",
            ).add_to(map_)
            return map_
        else:
            return None

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, BaseDataPoint):
            return False
        if not self.latitude == obj.latitude:
            return False
        if not self.longitude == obj.longitude:
            return False
        if not self.external_timestamp == obj.external_timestamp:
            return False
        return True


class DataPoint(BaseDataPoint):
    """An extended datapoint class with additional relevant methods."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        external_timestamp: Union[None, datetime] = None,
        speed: float = 0.0,
        accuracy: float = 0.0,
        user: Union[None, str] = None,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            latitude (float): The latitude coordinate of the location in \
                degrees.
            longitude (float): The longitude coordinate of the location in \
                degrees.
            external_timestamp (Union[None, datetime], optional): The \
                timestamp of the observation. Defaults to None.
            speed (float, optional): The average speed (in meters per second). \
                Defaults to 0.
            accuracy (float, optional): The horizontal accuracy of the
                location (in  meters). Defaults to 0.
            user (Union[None, str], optional): The unique identifier of the \
                entity the datapoint belongs to. Defaults to None.
        """
        # Compute basic attributes based on latitude, longitude and time alone
        super().__init__(
            latitude=latitude,
            longitude=longitude,
            external_timestamp=external_timestamp,
            user=user,
        )
        # If the datapoint represents an aggregate datapoint made up of several
        # other datapoints, then keep a list of all of these datapoints for
        # possible later use
        self.datapoints: List["DataPoint"] = []
        # Compute additional attributes based on consecutive datapoints:
        # - The change in space (in meters) from one datapoint to another
        self.dx: float = 0.0
        # - Change in time (in seconds) from one datapoint to another
        self.dt: float = 0.0
        # - Average speed (in meters per second)
        self.speed = speed
        # - Average acceleration (in meters per second squared)
        self.acceleration: float = 0.0
        # - The horizontal accuracy of the location (in meters)
        self.accuracy = accuracy

        # Additional data associated with the datapoint
        self.magnetometer_data: Union[None, "MagnetometerData"] = None
        self.device_motion_data: Union[None, "DeviceMotionData"] = None

    @classmethod
    def from_gpx(cls, gpx_track_point: gpxpy.gpx.GPXTrackPoint) -> "DataPoint":
        """
        Deserialize data into a DataPoint object.

        Args:
            gpx_track_point: An object defined by the 'gpxpy' library that \
                contains GPS location data.

        Returns:
            DataPoint: A 'DataPoint' object.
        """
        return DataPoint(
            latitude=gpx_track_point.latitude,
            longitude=gpx_track_point.longitude,
            external_timestamp=gpx_track_point.time,
        )

    @classmethod
    def from_json(cls, data: str) -> "DataPoint":
        """
        Deserialize data into a DataPoint object.

        Args:
            data (str): Json string data.

        Returns:
            DataPoint: A DataPoint object.
        """
        # Load data: str  -->  dict
        data_dict = json.loads(data)
        # Convert  : dict --> Python object
        return cls.from_dict(data_dict)

    @classmethod
    def from_msgpack(cls, data: bytes) -> "DataPoint":
        """
        Deserialize data into a DataPoint object.

        Args:
            data (bytes): msgpack byte data.

        Returns:
            DataPoint: A DataPoint object.
        """
        # Load data: bytes  -->  dict
        data_dict = msgpack.loads(data)
        # Convert  : dict --> Python object
        return cls.from_dict(data_dict)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "DataPoint":
        """
        Convert a Python dictionary into a DataPoint object.

        Args:
            data_dict (Dict[str, Any]): Dictionary data.

        Returns:
            DataPoint: A DataPoint object.
        """
        dp = DataPoint(
            latitude=data_dict["latitude"],
            longitude=data_dict["longitude"],
            external_timestamp=datetime.fromisoformat(
                data_dict["external_timestamp"],
            ),
        )

        # BaseDataPoint fields
        dp.user = data_dict["user"]
        dp.weight = data_dict["weight"]
        dp.label = data_dict["label"]
        try:
            dp.trajectory = data_dict["trajectory"]
        except KeyError:
            pass

        # DataPoint fields
        dp.datapoints = [cls.from_dict(_) for _ in data_dict["datapoints"]]
        dp.dx = data_dict["dx"]
        dp.dt = data_dict["dt"]
        dp.speed = data_dict["speed"]
        dp.acceleration = data_dict["acceleration"]
        dp.accuracy = data_dict["accuracy"]

        # Set MegnetometerData field
        try:
            if data_dict["magnetometer_data"] is not None:
                dp.magnetometer_data = MagnetometerData.from_dict(
                    data_dict=data_dict["magnetometer_data"],
                )
        except KeyError:
            # ... Else let it be None by default
            pass

        # Set DeviceMotionData field
        try:
            if data_dict["device_motion_data"] is not None:
                dp.device_motion_data = DeviceMotionData.from_dict(
                    data_dict=data_dict["device_motion_data"],
                )
        except KeyError:
            # ... Else let it be None by default
            pass
        return dp

    @classmethod
    def from_model(cls, datapoint: django_models.DataPoint) -> "DataPoint":
        """
        Convert a Django database model object to a Python DataPoint object.

        Args:
            datapoint (django_models.DataPoint): A Django database model \
                object.

        Returns:
            DataPoint: A Python DataPoint object.
        """
        dict_ = datapoint.__dict__.copy()
        dict_["external_timestamp"] = str(dict_["external_timestamp"])
        dict_["user"] = str(dict_["user"])
        try:
            dict_["datapoints"] = msgpack.loads(dict_["datapoints"])
        # If 'datapoints' are not a bytes-like object then just continue
        except TypeError:
            pass
        # TODO: What is the right key?
        # dict_["trajectory"] = dict_["trajectory_id"]
        # dict_["trajectory"] = dict_["trajectory"]
        return cls.from_dict(dict_)

    def compute_metrics(self, previous_datapoint: "DataPoint") -> None:
        """
        Compute and update different metrics of a datapoint w.r.t another.

        Note:
            Compute and update different metrics of a datapoint with respect \
            to another datapoint (most likely a previous datapoint). The \
            metrics computed include among others: The distance in time and \
            space between the points along with the speed and acceleration.

        Args:
            previous_datapoint (DataPoint): Another datapoint (most likely a \
                previously observed datapoint).
        """
        dx = self._distance(previous_datapoint)
        delta_time = 0.0
        speed = 0.0
        delta_speed = 0.0
        acceleration = 0.0
        try:
            delta_time = self._time_difference(previous_datapoint)
            if delta_time > 0.0:
                speed = dx / delta_time
                delta_speed = speed - previous_datapoint.speed
                acceleration = delta_speed / delta_time
        except ValueError:
            # If one of the timestamps is None, then we can not compute the time
            # difference and a 'ValueError' is thrown. In this case just
            # continue and set default values
            pass
        self.dx = dx
        self.dt = delta_time
        self.speed = speed
        self.acceleration = acceleration

    def to_model(self) -> django_models.DataPoint:
        """
        Convert a Python DataPoint object to a Django database model object.

        Returns:
            django_models.DataPoint:
        """
        if self.user is None:
            # A unique identifier is required when converting to a Django
            # database model object
            raise TypeError(
                f"Attribute 'user' has value {self.user} and is not a valid \
                UUID",
            )
        dp = django_models.DataPoint(
            latitude=self.latitude,
            longitude=self.longitude,
            user=uuid.UUID(self.user),
            label=self.label,
            external_timestamp=self.external_timestamp,
            weight=self.weight,
            # Nested datapoints are just saved as extra byte-like data
            datapoints=msgpack.dumps([_.to_dict() for _ in self.datapoints]),
            dx=self.dx,
            dt=self.dt,
            speed=self.speed,
            acceleration=self.acceleration,
            accuracy=self.accuracy,
        )

        # Associate "MagnetometerData" with DataPoint
        if self.magnetometer_data is not None:
            dp.magnetometer = self.magnetometer_data.to_model()

        # Associate "DeviceMotionData" with DataPoint
        if self.device_motion_data is not None:
            dp.devicemotion = self.device_motion_data.to_model()
        return dp

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert a DataPoint object to a Python dictionary.

        Returns:
            Dict[str, Any]: A dictionary representing a DataPoint object.
        """
        dict_ = self.__dict__.copy()
        # Convert special fields manually:
        dict_["external_timestamp"] = str(dict_["external_timestamp"])
        dict_["datapoints"] = [_.to_dict() for _ in dict_["datapoints"]]

        # Convert "MagnetometerData" object to dict
        if dict_["magnetometer_data"] is not None:
            dict_["magnetometer_data"] = dict_["magnetometer_data"].to_dict()

        # Convert "DeviceMotionData" object to dict
        if dict_["device_motion_data"] is not None:
            dict_["device_motion_data"] = dict_["device_motion_data"].to_dict()
        return dict_

    def to_json(self) -> str:
        """
        Serialize a DataPoint Python object.

        Returns:
            str: A serialized DataPoint object.
        """
        # Dump data: Python object --> dict
        data = self.to_dict()
        # Dump data: dict --> str
        return json.dumps(data)

    def to_msgpack(self) -> str:
        """
        Serialize a DataPoint Python object.

        Returns:
            str: A serialized DataPoint object.
        """
        # Dump data: Python object --> dict
        data = self.to_dict()
        # Dump data: dict --> str
        return msgpack.dumps(data)

    def _distance(self, previous_datapoint: "DataPoint") -> float:
        """
        Compute the great circle distance (in meters) between two datapoints.

        Args:
            previous_datapoint (DataPoint): A previously observed datapoint.

        Returns:
            float: The great circle distance (in meters) between two datapoints.
        """
        return haversine_distance(
            lat_1=self.latitude,
            lon_1=self.longitude,
            lat_2=previous_datapoint.latitude,
            lon_2=previous_datapoint.longitude,
        )

    def _time_difference(self, previous_datapoint: "DataPoint") -> float:
        """
        Compute the difference in time (in seconds) between two datapoints.

        Args:
            previous_datapoint (DataPoint): A previously observed datapoint.

        Raises:
            ValueError: If a timestamp of one of the datapoints is None.

        Returns:
            float: The difference in time (in seconds) between two datapoints.
        """
        if (self.external_timestamp is not None) and (
            previous_datapoint.external_timestamp is not None
        ):
            return np.abs(
                (
                    self.external_timestamp
                    - previous_datapoint.external_timestamp
                ).total_seconds()
            )
        else:
            raise ValueError(
                "The change in time could not be calculated as the timestamp \
                of a 'DataPoint' was None."
            )

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, DataPoint):
            return False
        if super().__eq__(obj) is False:
            return False
        if not self.dx == obj.dx:
            return False
        if not self.dt == obj.dt:
            return False
        if not self.accuracy == obj.accuracy:
            return False
        if not self.speed == obj.speed:
            return False
        if not self.acceleration == obj.acceleration:
            return False
        # Check equality of nested data
        if not len(obj.datapoints) == len(self.datapoints):
            return False
        else:
            for i in range(len(obj.datapoints)):
                if not self.datapoints[i] == obj.datapoints[i]:
                    return False
        return True


class GeohashedDataPoint:
    """A utility/intermediate class for representing a geohashed datapoint."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        precision: int,
        bits_per_char: int,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            latitude (float): The latitude coordinate of a location in degrees.
            longitude (float): The longitude coordinate of a location in \
                degrees.
            precision (int): Precision of the geohash.
            bits_per_char (int): Number of bits each character in the geohash \
                encodes.
        """
        # Set the precision of the Geohash
        self.precision = precision
        self.bits_per_char = bits_per_char
        # Encoded latitude/longitude coordinate pair as a geohash
        self.geohash: Union[None, str] = None
        # Decoded geohash as a latitude/longitude coordinate pair
        self.decoded_latitude: Union[None, float] = None
        self.decoded_longitude: Union[None, float] = None
        self.geohash_datapoint(latitude, longitude)

    def geohash_datapoint(self, latitude: float, longitude: float) -> None:
        """
        Encode a latitude/longitude coordinate pair as a geohash.

        Args:
            latitude (float): The latitude coordinate of a location in degrees.
            longitude (float): The longitude coordinate of a location in \
                degrees.
        """
        # Encode a latitude/longitude coordinate pair as a geohash
        self.geohash = geohash_encode(
            latitude=latitude,
            longitude=longitude,
            precision=self.precision,
            bits_per_char=self.bits_per_char,
        )
        # Also retrieve the corresponding decoded geohash as a
        # latitude/longitude coordinate pair
        decoded_geohash = geohash_decode(
            geohash=self.geohash, bits_per_char=self.bits_per_char,
        )
        self.decoded_latitude = float(decoded_geohash[1])
        self.decoded_longitude = float(decoded_geohash[0])

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, GeohashedDataPoint):
            return False
        elif not self.geohash == obj.geohash:
            return False
        return True


class DeviceMotionData:
    """A class for collecting accelerometer data."""

    def __init__(
        self,
        x: float,  # noqa
        y: float,  # noqa
        z: float,  # noqa
        external_timestamp: Union[None, datetime] = None,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Note:
            A 'DeviceMotionData' object contains additional data that can be \
            associated with a location represented by a 'DataPoint' object. A \
            'DeviceMotionData' object contains accelerometer data obtained \
            from a motion sensor that detects the change in movement relative \
            to the current device orientation, in three dimensions along the \
            x, y, and z axis.

        Args:
            x (float): The movement relative to the current device orientation \
                in along the X axis.
            y (float): The movement relative to the current device orientation \
                in along the Y axis.
            z (float): The movement relative to the current device orientation \
                in along the Z axis.
            external_timestamp (Union[None, datetime], optional): The \
                corresponding timestamp of when the values were obtained. \
                Defaults to None.
        """
        self.x = x
        self.y = y
        self.z = z
        self.external_timestamp = external_timestamp

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "DeviceMotionData":
        """
        Convert a Python dictionary to a DeviceMotionData object.

        Args:
            data_dict (Dict[str, Any]): Dictionary data.

        Returns:
            DataPoint: A DeviceMotionData object.
        """
        return DeviceMotionData(
            x=data_dict["x"],
            y=data_dict["y"],
            z=data_dict["z"],
            external_timestamp=datetime.fromisoformat(
                data_dict["external_timestamp"],
            ),
        )

    @classmethod
    def from_model(
        cls, device_motion_data: django_models.DeviceMotionData,
    ) -> "DeviceMotionData":
        """
        Convert Django database model to Python DeviceMotionData object.

        Args:
            device_motion_data (django_models.DeviceMotionData):

        Returns:
            DeviceMotionData: A Python DeviceMotionData object.
        """
        dict_ = device_motion_data.__dict__.copy()
        dict_["external_timestamp"] = str(dict_["external_timestamp"])
        return cls.from_dict(dict_)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert a DeviceMotionData object to a Python dictionary.

        Returns:
            Dict[str, Any]: A dictionary representing a DeviceMotionData object.
        """
        dict_ = self.__dict__.copy()
        # Convert special fields manually:
        dict_["external_timestamp"] = str(dict_["external_timestamp"])
        return dict_

    def to_model(self) -> django_models.DeviceMotionData:
        """
        Convert Python DeviceMotionData object to Django database model object.

        Returns:
            django_models.DeviceMotionData: A Django database model object.
        """
        return django_models.DeviceMotionData(
            x=self.x,
            y=self.y,
            z=self.z,
            external_timestamp=self.external_timestamp,
        )


class MagnetometerData:
    """A class for collecting magnetic field values and magnitude."""

    def __init__(
        self,
        x: float,  # noqa
        y: float,  # noqa
        z: float,  # noqa
        magnitude: float,
        internal_timestamp: Union[None, datetime] = None,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Note:
            A 'MagnetometerData' object contains additional data that can \
            be associated with a location represented by a 'DataPoint' object. \
            A 'MagnetometerData' object contains raw directional x, y, and z \
            magnetometer values as well as a computed magnitude of the \
            magnetic field and an internally given timestamp for \
            organizational purposes.

        Args:
            x (float): The magnetic field in the X direction.
            y (float): The magnetic field in the Y direction.
            z (float): The magnetic field in the Z direction.
            magnitude (float): The magnitude of the magnetic field computed by \
                a device.
            internal_timestamp (Union[None, datetime], optional): An \
                internally given timestamp for organizational purposes. \
                Defaults to None.
        """
        self.x = x
        self.y = y
        self.z = z
        self.magnitude = magnitude
        self.internal_timestamp = internal_timestamp

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "MagnetometerData":
        """
        Convert a Python dictionary to a MagnetometerData object.

        Args:
            data_dict (Dict[str, Any]): Dictionary data.

        Returns:
            DataPoint: A DataPoint object.
        """
        return MagnetometerData(
            x=data_dict["x"],
            y=data_dict["y"],
            z=data_dict["z"],
            magnitude=data_dict["magnitude"],
            internal_timestamp=datetime.fromisoformat(
                data_dict["internal_timestamp"],
            ),
        )

    @classmethod
    def from_model(
        cls, magnetometer_data: django_models.MagnetometerData
    ) -> "MagnetometerData":
        """
        Convert Django database model object to Python MagnetometerData object.

        Args:
            magnetometer_data (django_models.MagnetometerData):

        Returns:
            MagnetometerData: A Python MagnetometerData object.
        """
        dict_ = magnetometer_data.__dict__.copy()
        dict_["internal_timestamp"] = str(dict_["internal_timestamp"])
        return cls.from_dict(dict_)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert a MagnetometerData object to a Python dictionary.

        Returns:
            Dict[str, Any]: A dictionary representing a MagnetometerData object.
        """
        dict_ = self.__dict__.copy()
        # Convert special fields manually:
        dict_["internal_timestamp"] = str(dict_["internal_timestamp"])
        return dict_

    def to_model(self) -> django_models.MagnetometerData:
        """
        Convert Python MagnetometerData object to Django database model object.

        Returns:
            django_models.MagnetometerData:
        """
        return django_models.MagnetometerData(
            x=self.x,
            y=self.y,
            z=self.z,
            magnitude=self.magnitude,
            internal_timestamp=self.internal_timestamp,
        )


class Trajectory:
    """A class representing consecutive datapoints ordered by timestamp."""

    def __init__(
        self,
        datapoints: List[DataPoint],
        user: Union[None, str] = None,
        color: Union[None, str] = None,
        tag: Union[None, str] = None,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            datapoints (List[DataPoint]): One or more chronologically ordered \
                DataPoint objects.
            user (Union[None, str], optional): The user that the DataPoint \
                objects and thus the Trajectory object belongs to. Defaults \
                to None.
            color (Union[None, str], optional): The color that should be \
                associated with the Trajectory object (for visualization \
                purposes). Defaults to None.
            tag (Union[None, str], optional): A keyword associated with the \
                Trajectory object. For example 'raw' or 'filtered' describing \
                a certain stage of processing. Defaults to None.
        """
        # Set a unique identifier of the trajectory
        self.uid: Union[None, str, uuid.UUID] = None
        # A unique identifier of a trajectory
        self.trajectory: Union[None, str, uuid.UUID] = None
        # Set the Django database field name that is used to associate a Django
        # database Datapoint object with a Trajectory
        self.__django_model_field = "trajectory"
        # Set the color of the trajectory for visualization purposes
        if color is None:
            color = f"#{utils.color_hex()}"
        self.color = color
        # Give the trajectory a tag for categorization
        self.tag = tag
        # Keep an identifier of the user the trajectory belongs to
        self.user = user
        # Store all the datapoints in the trajectory
        self.datapoints = datapoints

    @classmethod
    def from_json(cls, data: str) -> "Trajectory":
        """
        Deserialize string data into a Python Trajectory object.

        Args:
            data (str): Json string data.

        Returns:
            Trajectory: A Python Trajectory object.
        """
        # Load data: str  -->  dict
        data_dict = json.loads(data)
        # Convert  : dict --> Python object
        return cls.from_dict(data_dict)

    @classmethod
    def from_msgpack(cls, data: bytes) -> "Trajectory":
        """
        Deserialize data into a Trajectory object.

        Args:
            data (bytes): msgpack byte data.

        Returns:
            Trajectory: A Python Trajectory object.
        """
        # Load data: bytes  -->  dict
        data_dict = msgpack.loads(data)
        # Convert  : dict --> Python object
        return cls.from_dict(data_dict)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "Trajectory":
        """
        Convert a Python dictionary to a Trajectory object.

        Args:
            data_dict (Dict[str, Any]): Dictionary data.

        Returns:
            Trajectory: A Python Trajectory object.
        """
        traj = Trajectory([])
        traj.datapoints = [
            DataPoint.from_dict(_) for _ in data_dict["datapoints"]
        ]
        traj.uid = data_dict["uid"]
        # traj.trajectory = data_dict["trajectory"]
        traj.color = data_dict["color"]
        traj.user = data_dict["user"]
        traj.tag = data_dict["tag"]
        return traj

    @classmethod
    def from_model(cls, trajectory: django_models.Trajectory) -> "Trajectory":
        """
        Convert a Python dictionary into a Python Trajectory object.

        Returns:
            Trajectory: A Python Trajectory object.
        """
        traj = Trajectory([])
        traj.datapoints = [
            DataPoint.from_model(_) for _ in trajectory.datapoints.all()
        ]
        traj.uid = str(trajectory.uid)
        traj.trajectory = str(trajectory.trajectory)
        traj.color = trajectory.color
        traj.user = trajectory.user
        traj.tag = trajectory.tag
        return traj

    def to_model(self, save: bool = False) -> django_models.Trajectory:
        """
        Convert a Python Trajectory object into a Django database model object.

        Args:
            save (bool, optional): Specify if the created Django Trajectory \
                database model object should saved. Defaults to False.

        Returns:
            django_models.Trajectory: A django Trajectory database model object.
        """
        traj_obj = django_models.Trajectory(
            start_timestamp=None,
            end_timestamp=None,
            user=self.user,
            tag=self.tag,
        )
        if len(self.datapoints) > 0:
            dps = [_.to_model() for _ in self.datapoints]
            traj_obj.start_timestamp = dps[0].external_timestamp
            traj_obj.end_timestamp = dps[-1].external_timestamp
            obj_list = []
            for dp in dps:
                # Associate each datapoint with the trajectory
                setattr(dp, self.__django_model_field, traj_obj)  # noqa
                obj_list.append(dp)
        # TODO: Make sure to save trajectory and datapoints to db if
        # save == True
        return traj_obj

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert a Trajectory object into a Python dictionary.

        Returns:
            Dict[str, Any]: A dictionary representing a Trajectory object.
        """
        return {
            "uid": self.uid,
            "datapoints": [_.to_dict() for _ in self.datapoints],
            "color": self.color,
            "user": self.user,
            "tag": self.tag,
        }

    def to_json(self) -> str:
        """
        Serialize a Trajectory Python object.

        Returns:
            str: A serialized Trajectory object.
        """
        # Dump data: Python object --> dict
        data = self.to_dict()
        # Dump data: dict --> str
        return json.dumps(data)

    def to_msgpack(self) -> bytes:
        """
        Serialize a Trajectory Python object.

        Returns:
            str: A serialized Trajectory object.
        """
        # Dump data: Python object --> dict
        data = self.to_dict()
        # Dump data: dict --> bytes
        return msgpack.dumps(data)

    def plot(
        self,
        weight: float = 1.0,
        opacity: float = 1.0,
        center: Union[None, CoordinatePair] = None,
        map_: Union[None, folium.Map] = None,
    ) -> Union[None, folium.Map]:
        """
        Plot the datapoint on a map.

        Args:
            center (CoordinatePair): A latitude/longitude coordinate pair that
                determines the centering of the map that the trajectory should \
                be plotted on.
            weight (float, optional): The size of the line the trajectory is
                plotted with. Defaults to 1.0.
            opacity (float, optional): The opacity the trajectory is plotted \
                with. Defaults to 1.0.
            map_ (Union[None, folium.Map], optional): A folium map object.
                Defaults to None.

        Raises:
            ValueError: If either a 'map_' or 'center' function argument is
                missing.

        Returns:
            Union[None, folium.Map]: A 'folium.Map' object.
        """
        if map_ is None:
            if center is not None:
                map_ = plotutils.setup_map(center=center)
            else:
                raise ValueError("Missing 'center' function argument.")
        if map_ is not None:
            datapoints = []
            for datapoint in self.datapoints:
                datapoints.append([datapoint.latitude, datapoint.longitude])
            folium.PolyLine(
                datapoints,
                color=self.color,
                weight=weight,
                opacity=opacity,
                popup=f"UUID: {self.uid}",
            ).add_to(map_)
            return map_
        else:
            return None

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, Trajectory):
            return False
        elif (not self.uid == obj.uid) or (
            not len(self.datapoints) == len(obj.datapoints)
        ):
            return False
        return True


# DataPoint and Trajectory helper methods


def geohash_encode(
    latitude: float, longitude: float, precision: int, bits_per_char: int,
) -> str:
    """
    Encode a longitude/latitude location as a geohash.

    The precision of the encoding is specified in the following table:

    lvl| bits | error             | base4          | base16       | base64
    --------------------------------------------------------------------------
    0  |    0 |    20015.087 km   |     prec  0    |    prec 0    |    prec 0
    1  |    2 |    10007.543 km   |     prec  1    |              |
    2  |    4 |     5003.772 km   |     prec  2    |    prec 1    |
    3  |    6 |     2501.886 km   |     prec  3    |              |    prec 1
    4  |    8 |     1250.943 km   |     prec  4    |    prec 2    |
    5  |   10 |      625.471 km   |     prec  5    |              |
    6  |   12 |      312.736 km   |     prec  6    |    prec 3    |    prec 2
    7  |   14 |      156.368 km   |     prec  7    |              |
    8  |   16 |       78.184 km   |     prec  8    |    prec 4    |
    9  |   18 |       39.092 km   |     prec  9    |              |    prec 3
    10 |   20 |       19.546 km   |     prec 10    |    prec 5    |
    11 |   22 |     9772.992  m   |     prec 11    |              |
    12 |   24 |     4886.496  m   |     prec 12    |    prec  6   |    prec 4
    13 |   26 |     2443.248  m   |     prec 13    |              |
    14 |   28 |     1221.624  m   |     prec 14    |    prec  7   |
    15 |   30 |      610.812  m   |     prec 15    |              |    prec 5
    16 |   32 |      305.406  m   |     prec 16    |    prec  8   |
    17 |   34 |      152.703  m   |     prec 17    |              |
    18 |   36 |       76.351  m   |     prec 18    |    prec  9   |    prec 6
    19 |   38 |       38.176  m   |     prec 19    |              |
    20 |   40 |       19.088  m   |     prec 20    |    prec 10   |
    21 |   42 |      954.394 cm   |     prec 21    |              |    prec 7
    22 |   44 |      477.197 cm   |     prec 22    |    prec 11   |
    23 |   46 |      238.598 cm   |     prec 23    |              |
    24 |   48 |      119.299 cm   |     prec 24    |    prec 12   |    prec 8
    25 |   50 |       59.650 cm   |     prec 25    |              |
    26 |   52 |       29.825 cm   |     prec 26    |    prec 13   |
    27 |   54 |       14.912 cm   |     prec 27    |              |    prec 9
    28 |   56 |        7.456 cm   |     prec 28    |    prec 14   |
    29 |   58 |        3.728 cm   |     prec 29    |              |
    30 |   60 |        1.864 cm   |     prec 30    |    prec 15   |    prec 10
    31 |   62 |        0.932 cm   |     prec 31    |              |
    32 |   64 |        0.466 cm   |     prec 32    |    prec 16   |
    --------------------------------------------------------------------------

    Args:
        lat (float): Latitude of a location in degrees.
        lon (float): Longitude of a location in degrees.
        precision (int): Encoding precision.
        bits_per_char (int, optional): Number of bits each character in the \
            geohash encodes.


    Returns:
        str: A geohash corresponding to the given latitude/longitude location.
    """
    return ghh.encode(
        lat=latitude,
        lng=longitude,
        precision=precision,
        bits_per_char=bits_per_char,
    )


def geohash_decode(geohash: str, bits_per_char: int) -> Tuple[float, float]:
    """
    Decode a geohash as a latitude/longitude coordinate point.

    Args:
        geohash (str): A geohash encoding a location.
        bits_per_char (int, optional): Number of bits each character in the \
            geohash encodes.

    Returns:
        Tuple[float, float]: A geographical location i.e., a (latitude, \
            longitude) point.
    """
    return ghh.decode(geohash, bits_per_char=bits_per_char)


def latlon_to_utm(datapoint: DataPoint) -> CoordinatePair:
    x, y, _, _ = utm.from_latlon(datapoint.latitude, datapoint.longitude)
    return x, y


def interpolation_distance_delta(
    precision: int,
    bits_per_char: int,
    location: Union[None, CoordinatePair] = None,
) -> float:
    """
    Determine the distance between the center of two geohash grid cells.

    Args:
        precision (int): Encoding precision.
        bits_per_char (int, optional): Number of bits each character in the \
            geohash encodes.
        location (Union[None, CoordinatePair], optional): The location of \
            where the distance delta should be calculated. Defaults to None.

    Returns:
        float: The distance between the center of two geohash grid cells.
    """
    if location is not None:
        tmp0 = ghh.encode(
            location[1],
            location[0],
            bits_per_char=bits_per_char,
            precision=precision,
        )
    else:
        tmp0 = ghh.encode(
            settings.PRIME_MERIDIAN_COORDS[1],
            settings.PRIME_MERIDIAN_COORDS[0],
            bits_per_char=bits_per_char,
            precision=precision,
        )
    tmp1 = ghh.neighbours(tmp0, bits_per_char=bits_per_char)["north"]
    tmp0_ = ghh.decode(tmp0, bits_per_char=bits_per_char)
    tmp1_ = ghh.decode(tmp1, bits_per_char=bits_per_char)
    tmp0_ = utm.from_latlon(tmp0_[1], tmp0_[0])
    tmp1_ = utm.from_latlon(tmp1_[1], tmp1_[0])
    ls = LineString([[tmp0_[0], tmp0_[1]], [tmp1_[0], tmp1_[1]]])
    return ls.length


# Data processing-specific helper methods


def haversine_distance(
    lat_1: float, lon_1: float, lat_2: float, lon_2: float,
) -> float:
    """
    Calculate the 'Haversine' (great-circle) distance in meters.

    Note:
        Given two latitude/longitude locations calculate the great-circle \
        distance between them.

    Args:
        lat_1 (float): Latitude in degrees of the first location.
        lon_1 (float): Longitude in degrees of the first location.
        lat_2 (float): Latitude in degrees of the second location.
        lon_2 (float): Longitude in degrees of the second location.

    Returns:
        float: The distance between the two locations in meters.
    """
    d_lat = np.radians(lat_1 - lat_2)
    d_lon = np.radians(lon_1 - lon_2)
    lat1 = np.radians(lat_1)
    lat2 = np.radians(lat_2)
    temp_a = np.sin(d_lat / 2) * np.sin(d_lat / 2) + np.sin(d_lon / 2) * np.sin(
        d_lon / 2
    ) * np.cos(lat1) * np.cos(lat2)
    temp_c = 2 * np.arctan2(np.sqrt(temp_a), np.sqrt(1 - temp_a))
    distance = settings.EARTH_RADIUS * temp_c
    return distance


def read_gpx_files(path: str) -> Union[None, List[Dict[str, Any]]]:
    """
    Read .gpx files containing location data into a list of dictionaries.

    Args:
        path (str): The filepath to a directory containing .gpx files that \
            contain timestamped location data.

    Returns:
        Union[None, List[Dict[str, Any]]]: A list of timstamped locations \
            sorted by time.
    """
    sorted_data = []
    for gpx_file in os.listdir(path):
        filepath = os.path.join(path, gpx_file)
        with open(filepath, "r") as f:
            gpx = gpxpy.parse(f)
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        # Make the timestamp a naive datetime object
                        point.time = point.time.replace(tzinfo=None)
                        dict_ = {
                            "timestamp": (
                                point.time - settings.EPOCH
                            ).total_seconds(),
                            "point": point,
                        }
                        sorted_data.append(dict_)
    if len(sorted_data) > 0:
        # Perform in-place sorting producing a list of dictionaries sorted by
        # timestamp
        sorted_data.sort(key=lambda x: x.get("timestamp"))
        return sorted_data
    else:
        return None


def gpx_to_dataframe(path: str) -> Union[None, pd.DataFrame]:
    """
    A method for reading GPS data (.gpx files) into a Pandas DataFrame.

    Args:
        relative_path (str): The relative path to a directory that contains \
            .gpx files.

    Returns:
        pd.DataFrame: A pandas DataFrame object.
    """
    sorted_data = read_gpx_files(path=path)
    if sorted_data is None:
        return None
    else:
        points = []
        for item in sorted_data:
            data = item["point"]
            point = {
                "latitude": data.latitude,
                "longitude": data.longitude,
                "external_timestamp": data.time,
            }
            points.append(point)
        return pd.DataFrame(data=points)


def gpx_to_datapoints(path: str) -> Union[None, List[DataPoint]]:
    """
    A method for reading GPS data (.gpx files) into 'DataPoint' objects.

    Args:
        relative_path (str): The relative path to a directory that contains \
            .gpx files.

    Returns:
        list: A list of DataPoint objects.
    """
    sorted_data = read_gpx_files(path=path)
    if sorted_data is None:
        return None
    else:
        points = []
        for item in sorted_data:
            data = item["point"]
            point = DataPoint.from_gpx(data)
            points.append(point)
        return points
