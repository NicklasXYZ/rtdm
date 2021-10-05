import json
import logging
import random
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

import folium
import numpy as np
import pandas as pd
import requests
import utm
from folium import plugins
from shapely import ops
from shapely.geometry import LineString, Point

# Type aliases
Coordinates = Union[Tuple[float, float], List[float]]
Label = Union[None, bool]


# A global variable that is used to keep track of whether
# Valhalla has been tested with a dummy routing request and is
# working properly
TEST_VALHALLA = True


def rand_24_bit() -> int:
    """Returns a random 24-bit integer."""
    return random.randrange(0, 16 ** 6)


def color_dec() -> int:
    """Alias of rand_24 bit()"""
    return rand_24_bit()


def color_hex(num: int = rand_24_bit()) -> str:
    """Returns a 24-bit int in hex."""
    return "%06x" % num


def color_rgb(num: int = rand_24_bit()) -> Tuple[int, int, int]:
    """Returns three 8-bit numbers, one for each channel in RGB."""
    hx = color_hex(num)
    barr = bytearray.fromhex(hx)
    return (barr[0], barr[1], barr[2])


def setup_map(
    center: Coordinates, zoom_start: int = 14, tiles: str = "cartodbdark_matter"
):
    map_ = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles,)
    plugins.Fullscreen(position="topleft").add_to(map_)
    plugins.Draw(
        filename="placeholder.geojson", export=True, position="topleft"
    ).add_to(map_)
    return map_


def plot_datapoint(
    datapoint: Coordinates,
    map_,
    color: str,
    radius: float = 5.0,
    opacity: float = 1.0,
):
    folium.CircleMarker(
        [datapoint[0], datapoint[1]],
        radius=radius,
        color=color,
        opacity=opacity,
        popup=f"...",
    ).add_to(map_)
    return map_


def plot_trajectory(
    trajectory: List[Coordinates],
    map_,
    color: str,
    weight: float = 2.0,
    opacity: float = 1,
):
    folium.PolyLine(
        trajectory, color=color, weight=weight, opacity=opacity, popup=f"...",
    ).add_to(map_)
    return map_


class DataPoint:
    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude


class WayPoint(DataPoint):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        duration: float = 0.0,
        std: float = 0.0,
    ):
        """
        [summary]

        Args:
            latitude (float): [description]
            longitude (float): [description]
            duration (float, optional): [description]. Defaults to 0..
            std (float, optional): [description]. Defaults to 0..

        Raises:
            ValueError: [description]
        """
        super().__init__(latitude=latitude, longitude=longitude)
        if not duration >= 0.0:
            raise ValueError(
                "An object of type 'WayPoint' can not have a negative duration. "
                + "Change it to a floating point value >= 0",
            )
        self.std = std
        self.duration = duration

    @property
    def is_stop(self) -> bool:
        return self.duration > 0.0


class ValhallaInterface:

    # Regex for determining if a url is valid
    URL_REGEX = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ... or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        url: str = "http://localhost:8002/optimized_route",
        precision: float = 1.0 / 1e6,
        headers={},
        test_valhalla: bool = False,
    ) -> None:
        """
        Initialize, check and set given class variables on class instantiation.

        Args:
            url (str, optional): Address to be used when sending HTTP routing requests
                to Valhalla. Defaults to "http://localhost:8002/optimized_route".
            precision (float, optional): Valhalla latitude/longitude routing precision.
                Defaults to 1.0/1e6.
            headers (dict, optional): Additional HTTP request headers. Defaults to {}.
            test_valhalla (bool, optional): Test with dummy data whether Valhalla is
                working properly. Defaults to True.
        """
        global TEST_VALHALLA
        # Set Valhalla latitude/longitude routing precision
        self.precision = precision
        # Set necessary request headers
        required_headers = {"Content-Type": "application/json"}
        required_headers.update(headers)
        self.headers = required_headers
        try:
            self.url_isvalid(url, test_valhalla)
            self.url = url
            TEST_VALHALLA = False
        except Exception as e:
            logging.debug(e)
            TEST_VALHALLA = True

    def _dummy_data(
        self,
        start: Coordinates = [55.39594, 10.38831],
        end: Coordinates = [55.39500, 10.38800],
    ) -> str:
        return json.dumps(
            {
                "locations": [
                    {"lat": start[0], "lon": start[1]},
                    {"lat": end[0], "lon": end[1]},
                ],
                "costing": "auto",
                "directions_options": {"units": "kilometers"},
            }
        )

    def url_isvalid(self, url: str, test_valhalla: bool) -> None:
        """
        Check whether a given url is valid or not. Then send a dummy routing request to Valhalla to check everything is
        working properly.

        Args:
            url (str): The URL where Valhalla receives and responds to 'optimized
                routing' requests.
            test_valhalla (bool): A value that signals whether additional dummy
                routing requests should be sent to Valhalla to everything works
                properly.

        Raises:
            ValueError: If the given URL is not valid.
        """
        if re.match(self.URL_REGEX, url) is not None:
            # Test if Valhalla is working by sending a dummy routing request
            if test_valhalla or TEST_VALHALLA:
                dummy_data = self._dummy_data()
                requests.post(
                    url=url, data=dummy_data, headers=self.headers,
                )
        else:
            raise ValueError(f"The given url: {url} is not valid.")

    def decode_polyline(self, polyline_string: str) -> List[Coordinates]:
        """
        Decode a string into a a sequence of latitude/longitude coordinate pairs.

        Args:
            polyline_string (str): A string representation of a sequence of
                latitude/longitude pairs.

        Returns:
            List[Coordinates]: A sequence of latitude/longitude coorindate pairs.
        """
        index = 0
        latitude = 0
        longitude = 0
        coordinates = []
        changes = {"latitude": 0, "longitude": 0}
        # Coordinates have variable length when encoded, so just keep
        # track of whether we have hit the end of the string. In each
        # while loop iteration a single coordinate is decoded.
        while index < len(polyline_string):
            # Gather latitude/longitude changes and store them in a dictionary
            # such that we can apply them later on...
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
                [self.precision * latitude, self.precision * longitude],
            )
        return coordinates

    def send_optimized_route_request(self, wp0, wp1) -> Union[None, Any]:
        """
        [summary]

        Args:
            wp0 ([type]): [description]
            wp1 ([type]): [description]

        Returns:
            Union[None, Any]: [description]
        """

        def build_optimized_route_request(wp0, wp1):
            return json.dumps(
                {
                    "locations": [
                        # Start location
                        {"lat": wp0.latitude, "lon": wp0.longitude},
                        # End location
                        {"lat": wp1.latitude, "lon": wp1.longitude},
                    ],
                    # TODO: Pass these settings as changeable/optional arguments
                    "costing": "pedestrian",
                    "directions_options": {"units": "kilometers"},
                }
            )

        data = build_optimized_route_request(wp0, wp1)
        response = requests.post(self.url, data=data, headers=self.headers,)
        if response.status_code == 200:
            content = json.loads(response.content)
        else:
            logging.debug(
                "The routing request to Valhalla was not processed properly."
                + f"Valhalla returned HTTP status code: {response.status_code}",
            )
            content = None
        return content


class TrajectoryGenerator:
    """[summary]"""

    def __init__(
        self,
        waypoints: List[WayPoint],
        mean_time_delta: float,
        std_time_delta: float,
        mean_speed: float,
        std_speed: float,
        std_datapoint: float,
        **kwargs,
    ) -> None:
        """
        [summary]

        Args:
            waypoints (List[WayPoint]): [description]
            mean_time_delta (float): [description]
            std_time_delta (float): [description]
            mean_speed (float): [description]
            std_speed (float): [description]
            std_datapoint (float): [description]
        """
        self.polylines = None  # TODO: Refactor
        # Give the generator a unique identifier
        self.uid = str(uuid.uuid4())
        # Store the UTM grid zone for converting between coordinates in the UTM
        # projected coordinate system and latitude/longitude
        self._utm_grid_zone = None
        self.waypoints = waypoints
        self.mean_time_delta = mean_time_delta
        self.std_time_delta = std_time_delta
        self.mean_speed = mean_speed
        self.std_speed = std_speed
        self.std_datapoint = std_datapoint
        # Pass on extra settings to Valhalla
        self.vi = ValhallaInterface(**kwargs)
        self.trajectory, self.timestamps = self._generate_trajectory()

    def _stop_point(
        self, x: float, y: float, duration_waypoint: float, std_waypoint: float,
    ) -> Tuple[List[Coordinates], List[float]]:
        """
        Generate a stop point by randomly sampling locations around a given location.

        Args:
            x (float): First coordinate in the UTM coordinate system. This coordinate
                corresponds to a latitude coordinate in the WSG84 coordinate reference
                system.
            y (float): Second coordinate in the UTM coordinate system. This coordinate
                corresponds to a longitude coordinate in the WSG84 coordinate reference
                system.
            duration_waypoint (float): [description]
            std_waypoint (float): [description]

        Returns:
            Tuple[List[Coordinates], List[float]]: [description]
        """
        _dts = 0.0
        dts = []
        counter = 0
        coordinates = []
        while True:
            dt = self.mean_time_delta + np.abs(
                np.random.normal(0, self.std_time_delta)
            )
            dt_next = _dts + dt
            if dt_next > duration_waypoint:
                break
            else:
                _dts += dt
                dts.append(dt)
                coordinates.append(
                    [
                        x + np.random.normal(0, std_waypoint),
                        y + np.random.normal(0, std_waypoint),
                    ]
                )
                counter += 1
        return coordinates, dts

    def _create_segment(
        self, wp0: WayPoint, wp1: WayPoint,
    ) -> Tuple[List[Coordinates], List[float], List[Coordinates]]:
        """
        [summary]

        Args:
            wp0 (WayPoint): [description]
            wp1 (WayPoint): [description]

        Returns:
            Tuple[List[Coordinates], List[float]]: [description]
        """
        # TODO: Error handling. If content is None.
        content = self.vi.send_optimized_route_request(wp0=wp0, wp1=wp1,)
        # print("content     : ", content)
        # print("wp0, lat/lon: ", wp0.latitude, wp0.longitude,)
        # print("wp1, lat/lon: ", wp1.latitude, wp1.longitude,)
        polyline = self.vi.decode_polyline(content["trip"]["legs"][0]["shape"],)
        coordinates = []
        for coord in polyline[1:-1]:
            x, y, _, _ = utm.from_latlon(coord[0], coord[1])
            coordinates.append([x, y],)
        linestring, dts = self._interpolate(linestring=LineString(coordinates),)
        return list(linestring.coords), dts, coordinates

    def _trajectory_start(self, wp0: WayPoint,) -> Dict[str, Any]:
        """
        [summary]

        Args:
            wp0 (WayPoint): [description]

        Returns:
            Dict[str, Any]: [description]
        """
        start_x, start_y, _, _ = utm.from_latlon(wp0.latitude, wp0.longitude,)
        start_coordinates, start_dts = self._stop_point(
            start_x, start_y, wp0.duration, wp0.std,
        )
        return {
            "start_x": start_x,
            "start_y": start_y,
            "start_coordinates": start_coordinates,
            "start_dts": start_dts,
        }

    def _trajectory_next(self, wp0: WayPoint, wp1: WayPoint,) -> Dict[str, Any]:
        """
        [summary]

        Args:
            wp0 (WayPoint): [description]
            wp1 (WayPoint): [description]

        Returns:
            Dict[str, Any]: [description]
        """
        middle_coordinates, middle_dts, polyline = self._create_segment(
            wp0=wp0, wp1=wp1,
        )
        end_x, end_y, _, _ = utm.from_latlon(wp1.latitude, wp1.longitude,)
        end_coordinates, end_dts = self._stop_point(
            end_x, end_y, wp1.duration, wp1.std,
        )
        return {
            "end_x": end_x,
            "end_y": end_y,
            "middle_coordinates": middle_coordinates,
            "middle_dts": middle_dts,
            "end_coordinates": end_coordinates,
            "end_dts": end_dts,
            "polyline": polyline,
        }

    def _flatten(self, segments, timestamps) -> Tuple[LineString, List[float]]:
        """
        [summary]

        Args:
            segments ([type]): [description]
            timestamps ([type]): [description]

        Returns:
            Tuple[LineString, List[float]]: [description]
        """
        _trajectory = []
        _timestamps = []
        for arr in timestamps:
            _timestamps.extend(arr)
        for arr in segments:
            _trajectory.extend(arr)
        return LineString(_trajectory), _timestamps

    def _set_utm_grid_zone(self) -> None:
        """[summary]"""
        # Use the very first waypoint to define the UTM zone
        if self._utm_grid_zone is None:
            _, _, zone, lat_band = utm.from_latlon(
                self.waypoints[0].latitude, self.waypoints[0].longitude,
            )
            self._utm_grid_zone = (zone, lat_band)

    def _generate_trajectory(self) -> Tuple[LineString, List[float]]:
        """
        [summary]

        Raises:
            ValueError: [description]

        Returns:
            Tuple[LineString, List[float]]: [description]
        """
        segments = []
        timestamps = []
        polylines = []
        if len(self.waypoints) >= 2:
            self._set_utm_grid_zone()
            wp0 = self.waypoints[0]
            self.waypoints[1]
            for i in range(1, len(self.waypoints)):
                dict_ = self._trajectory_next(
                    wp0=self.waypoints[i - 1], wp1=self.waypoints[i],
                )
                polyline = dict_["polyline"]
                middle_coordinates = dict_["middle_coordinates"]
                middle_dts = dict_["middle_dts"]
                end_coordinates = dict_["end_coordinates"]
                end_dts = dict_["end_dts"]
                end_x = dict_["end_x"]
                end_y = dict_["end_y"]
                end_anchor = [[end_x, end_y]]
                if i == 1:
                    dict_ = self._trajectory_start(wp0=wp0,)
                    start_coordinates = dict_["start_coordinates"]
                    start_dts = dict_["start_dts"]
                    start_x = dict_["start_x"]
                    start_y = dict_["start_y"]
                    start_anchor = [[start_x, start_y]]
                    segment = (
                        start_coordinates
                        + start_anchor
                        + middle_coordinates
                        + end_anchor
                        + end_coordinates
                    )
                    start_dts[0] = 0
                    middle_dts[0] = self.mean_time_delta + np.abs(
                        np.random.normal(0, self.std_time_delta)
                    )
                    dt0 = self.mean_time_delta + np.abs(
                        np.random.normal(0, self.std_time_delta)
                    )
                    dt1 = self.mean_time_delta + np.abs(
                        np.random.normal(0, self.std_time_delta)
                    )
                    dts = start_dts + [dt0] + middle_dts + [dt1] + end_dts
                else:
                    segment = middle_coordinates + end_anchor + end_coordinates
                    middle_dts[0] = self.mean_time_delta + np.abs(
                        np.random.normal(0, self.std_time_delta)
                    )
                    dt0 = self.mean_time_delta + np.abs(
                        np.random.normal(0, self.std_time_delta)
                    )
                    dts = middle_dts + [dt0] + end_dts
                if len(self.waypoints) > (i + 1):
                    if len(end_coordinates) > 0:
                        latitude, longitude = utm.to_latlon(
                            end_coordinates[-1][0],
                            end_coordinates[-1][1],
                            # The UTM grid zone here is for example: (32, "U")
                            # It is set automatically...
                            *self._utm_grid_zone,
                        )
                        wp0 = WayPoint(
                            latitude=latitude,
                            longitude=longitude,
                            std=self.waypoints[i].std,
                            duration=self.waypoints[i].duration,
                        )
                    else:
                        wp0 = self.waypoints[i]
                    # Use the next waypoint
                    self.waypoints[i + 1]
                segments.append(segment)
                timestamps.append(dts)
                polylines.append(polyline)
        else:
            raise ValueError(
                "At least two objects of type 'WayPoint' need to be provided."
            )
        # Set internally:
        _polylines = []
        for arr in polylines:
            _polylines.extend(arr)
        self.polylines = _polylines
        return self._flatten(segments, timestamps)

    def _interpolate(
        self, linestring: LineString,
    ) -> Tuple[LineString, List[float]]:
        """
        [summary]

        Args:
            linestring (LineString): [description]

        Returns:
            Tuple[LineString, List[float]]: [description]
        """
        dxs, dts = self._segment_subdivision(linestring=linestring)
        points = [
            linestring.interpolate(dxs[i], normalized=True)
            for i in range(len(dxs))
        ]
        rand_arr = np.random.normal(
            0, self.std_datapoint, size=(len(points), 2)
        )
        for i in range(rand_arr.shape[0]):
            rand_arr[i, 0] += points[i].x
            rand_arr[i, 1] += points[i].y
        return LineString(rand_arr.tolist()), dts

    def _segment_subdivision(
        self, linestring: LineString,
    ) -> Tuple[List[float], List[float]]:
        """
        [summary]

        Args:
            linestring (LineString): [description]

        Raises:
            ValueError: [description]

        Returns:
            Tuple[List[float], List[float]]: [description]
        """
        dxs = [0.0]
        dts = [0.0]
        counter = 0
        linestring_length = linestring.length
        if linestring_length > 0:
            while True:
                dt = self.mean_time_delta + np.abs(
                    np.random.normal(0.0, self.std_time_delta)
                )
                dx = (
                    self.mean_speed + np.random.normal(0.0, self.std_speed)
                ) * dt
                dx_next = dxs[counter] + dx
                if dx_next > linestring_length:
                    break
                else:
                    dts.append(dt)
                    dxs.append(dx_next)
                    counter += 1
            return [v / linestring_length for v in dxs], dts
        else:
            # Linestring length is zero. Start and end location must thus be the same.
            # Alert the user about this....
            raise ValueError("")

    def to_latlon(self) -> List[Coordinates]:
        """
        [summary]

        Raises:
            AttributeError: [description]

        Returns:
            List[Coordinates]: [description]
        """
        coordinates = []
        if self.trajectory is not None:
            for coord in self.trajectory.coords:
                lat, lon = utm.to_latlon(
                    coord[0], coord[1], *self._utm_grid_zone
                )
                coordinates.append([lat, lon])
        else:
            raise AttributeError(
                "A trjectory has not been generated. Coordinates can thus not be "
                + "converted.",
            )
        return coordinates


class TrajectoryGeneratorCollection:
    """Bulk operations on a collection of trajectories."""

    def __init__(
        self,
        start_datetime: datetime,
        trajectory_generators: List[Tuple[Label, TrajectoryGenerator]],
        gap=timedelta(days=1.0),
        filename="trajectories",
    ) -> None:
        """
        [summary]

        Args:
            start_datetime (datetime): [description]
            trajectory_generators (List[Tuple[Label, TrajectoryGenerator]]): [description]
            gap ([type], optional): [description]. Defaults to timedelta(days = 1.).
            filename (str, optional): [description]. Defaults to "trajectories".
        """
        # Give the trajectory collection a unique identifier so that we can keep track
        # of it...
        self.uid = "TrajectoryCollection:" + str(uuid.uuid4)
        self.start_datetime = start_datetime
        self.trajectory_generators = trajectory_generators
        self.gap = gap
        self.filename = filename

    # def to_dataframe(self) -> pd.DataFrame:
    #     """Convert a collection of 'TrajectoryGenerator' objects to a single Pandas
    #     dataframe.

    #     Returns:
    #         pd.DataFrame: A collection of 'TrajectoryGenerator' as a single Pandas
    #             dataframe.
    #     """
    #     main_data = []
    #     cumulative_gap = self.start_datetime
    #     for anomalous, trajectory_generator in self.trajectory_generators:
    #         timestamps = [
    #             cumulative_gap + timedelta(seconds = seconds)
    #             for seconds in np.cumsum(trajectory_generator.timestamps)
    #         ]
    #         data = [
    #             {
    #                 # Set datapoint location
    #                 "latitude": lat,
    #                 "longitude": lon,
    #                 # Set trajectory identifier
    #                 "uid": trajectory_generator.uid,
    #             }
    #             for lat, lon in trajectory_generator.to_latlon()
    #         ]
    #         # df = df.append(
    #         #     pd.DataFrame(data = data, index = timestamps)
    #         # )
    #         df = pd.DataFrame(data = data, index = timestamps)
    #         df.attrs["uid"] = trajectory_generator.uid
    #         main_data.append({
    #             # Store df as JSON object
    #             "df": df.to_json(orient = "split"),
    #             "uid": trajectory_generator.uid,
    #             "anomalous": anomalous,
    #         })
    #         # The start time of each new trajectory is 'self.gap' time apart
    #         cumulative_gap = timestamps[-1] + self.gap
    #     # Create main dataframe
    #     df = pd.DataFrame(data = main_data)
    #     # Add metadata
    #     df.attrs["class"] = self.__class__.__name__
    #     df.attrs["uid"] = self.uid
    #     return df

    # def get_first_anom_points(self, buffer = 12.5, consecutive_threshold = 5):
    #     # ARG 'buffer' should be adjusted the amount of noise in the generated
    #     # trajectories.
    #     #
    #     # Algorithm for finding the point at which a trajectory becomes anomalous:
    #     # - Get all trajectories termed normal between an origin and a destination
    #     # - Create a polygon out of these
    #     # - Find intersection of created polygon and anomalous trajectory
    #     # --> The first intersection point along the trajectory is when it becomes
    #     #     anomalous
    #     # Create separate lists for normal and amomalous trajectories
    #     norm_traj = [t for v, t in self.trajectory_generators if v == False]
    #     anom_traj = [t for v, t in self.trajectory_generators if v == True]
    #     # Create 'normal' polygon
    #     _polygons = [t.trajectory.buffer(buffer) for t in norm_traj]
    #     anom_points = {}
    #     for traj in anom_traj:
    #         polygon = ops.unary_union(
    #             _polygons + \
    #             [t.trajectory.buffer(buffer) for t in anom_traj if traj.uid != t.uid]
    #         ).buffer(buffer)
    #         inter = polygon.intersection(traj.trajectory.buffer(buffer))
    #         diff = traj.trajectory.difference(inter)
    #         # anom_point = None
    #         # Run along trajectory and find when the first
    #         # anomalous point occurs
    #         consecutive = []; counter = 0
    #         for coords in traj.trajectory.coords:
    #             if diff.contains(Point(coords)):
    #                 consecutive.append(coords)
    #                 # If 'consecutive_threshold' consecutive points are outside the
    #                 # 'normal' polygon then it must be deviating and the start of an
    #                 # anomalous trajectory
    #                 if len(consecutive) >= consecutive_threshold:
    #                     # anom_point = consecutive[0]
    #                     # Adjust counter such that we get corresponding index
    #                     # of the coordinate in the list of coordinates of the
    #                     # trajectory
    #                     counter -= consecutive_threshold - 1
    #                     break
    #             else:
    #                 consecutive = []
    #             counter += 1
    #         coord = list(traj.trajectory.coords)[counter]
    #         lat, lon = utm.to_latlon(coord[0], coord[1], *traj._utm_grid_zone)
    #         anom_points[traj.uid] = [lat, lon]
    #     # Return lat/lon coordinates at which anomalous trajectories actually
    #     # become anomalous
    #     return anom_points

    def get_first_anom_points(self, pixels=7.5, consecutive_threshold=3):
        # ARG 'buffer' should be adjusted the amount of noise in the generated
        # trajectories.
        #
        # Algorithm for finding the point at which a trajectory becomes anomalous:
        # - Get all trajectories termed normal between an origin and a destination
        # - Create a polygon out of these
        # - Find intersection of created polygon and anomalous trajectory
        # --> The first intersection point along the trajectory is when it becomes
        #     anomalous
        # Create separate lists for normal and amomalous trajectories
        norm_traj = [t for v, t in self.trajectory_generators if v == False]
        anom_traj = [t for v, t in self.trajectory_generators if v == True]
        # Create 'normal' polygon
        # _polygons = [LineString(t.polylines).buffer(buffer) for t in norm_traj]
        _polygons = [LineString(t.trajectory).buffer(pixels) for t in norm_traj]
        anom_points = {}
        for traj in anom_traj:
            # polygon = ops.unary_union(
            #     _polygons + \
            #     [LineString(t.polylines).buffer(buffer) for t in anom_traj if traj.uid != t.uid]
            # ).buffer(buffer)
            polygon = ops.unary_union(
                _polygons
                + [
                    LineString(t.trajectory).buffer(pixels)
                    for t in anom_traj
                    if traj.uid != t.uid
                ]
            )
            # inter = polygon.intersection(LineString(traj.polylines).buffer(buffer))
            # diff = LineString(traj.polylines).difference(inter).buffer(10)
            inter = polygon.intersection(
                LineString(traj.trajectory).buffer(pixels)
            )
            diff = LineString(traj.trajectory).difference(inter)
            # anom_point = None
            # Run along trajectory and find when the first
            # anomalous point occurs
            consecutive = []
            counter = 0
            for coords in traj.trajectory.coords:
                if diff.contains(Point(coords)):
                    consecutive.append(coords)
                    # If 'consecutive_threshold' consecutive points are outside the
                    # 'normal' polygon then it must be deviating and the start of an
                    # anomalous trajectory
                    if len(consecutive) >= consecutive_threshold:
                        # anom_point = consecutive[0]
                        # Adjust counter such that we get corresponding index
                        # of the coordinate in the list of coordinates of the
                        # trajectory
                        counter -= consecutive_threshold - 1
                        break
                else:
                    consecutive = []
                counter += 1
            coord = list(traj.trajectory.coords)[counter]
            lat, lon = utm.to_latlon(coord[0], coord[1], *traj._utm_grid_zone)
            anom_points[traj.uid] = [lat, lon]
        # Return lat/lon coordinates at which anomalous trajectories actually
        # become anomalous
        return anom_points

    def to_dataframe(self, anom_start=False) -> pd.DataFrame:
        if anom_start:
            anom_points = self.get_first_anom_points()
        else:
            anom_points = None
        main_data = []
        cumulative_gap = datetime.now()  # self.start_datetime
        for anomalous, trajectory_generator in self.trajectory_generators:
            timestamps = [
                cumulative_gap + timedelta(seconds=seconds)
                for seconds in np.cumsum(trajectory_generator.timestamps)
            ]
            data = [
                {
                    # Set datapoint location
                    "latitude": lat,
                    "longitude": lon,
                    # Set trajectory identifier
                    "uid": trajectory_generator.uid,
                }
                for lat, lon in trajectory_generator.to_latlon()
            ]
            # df = df.append(
            #     pd.DataFrame(data = data, index = timestamps)
            # )
            df = pd.DataFrame(data=data, index=timestamps)
            df["anom_start"] = False
            df.attrs["uid"] = trajectory_generator.uid
            if anom_points is not None:
                if trajectory_generator.uid in anom_points:
                    anom_point = anom_points[trajectory_generator.uid]
                    df.loc[
                        (df["latitude"] == anom_point[0])
                        & (df["longitude"] == anom_point[1]),
                        "anom_start",
                    ] = True
                else:
                    anom_point = None
            else:
                anom_point = None
            main_data.append(
                {
                    # Store df as JSON object
                    "df": df.to_json(orient="split"),
                    "uid": trajectory_generator.uid,
                    "anomalous": anomalous,
                }
            )
            # The start time of each new trajectory is 'self.gap' time apart
            cumulative_gap = timestamps[-1] + self.gap
        # Create main dataframe
        df = pd.DataFrame(data=main_data)
        # Add metadata
        df.attrs["class"] = self.__class__.__name__
        df.attrs["uid"] = self.uid
        return df

    def _to_gpx(self, gpxpy, filepath: str) -> None:
        """
        [summary]

        Args:
            gpxpy ([type]): [description]
            filepath (str): [description]
        """
        gpx = gpxpy.gpx.GPX()
        # Create first track in our GPX:
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)
        # Create first segment in our GPX track:
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        df = self.to_dataframe()
        for index, row in df.iterrows():
            # Create points:
            gpx_segment.points.append(
                gpxpy.gpx.GPXTrackPoint(
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    time=index.to_pydatetime(),
                )
            )
        logging.debug(f"Saving trajectories to .gpx file: {filepath}")
        with open(filepath, "w") as f:
            f.write(gpx.to_xml())

    def to_gpx(self, filepath: str) -> None:
        """
        [summary]

        Args:
            filepath (str): [description]
        """
        try:
            import gpxpy

            self._to_gpx(gpxpy, filepath)
        except ImportError:
            msg = (
                "Python library 'gpxpy' was not found. "
                + "Install the library through the pip package manager by running: "
                + "pip intall gpxpy"
            )
            print(msg)
            logging.debug(msg)

    def to_json(self) -> str:
        """
        Serialize a collection of 'TrajectoryGenerator' objects.

        Returns:
            str: A string representing a collection of 'TrajectoryGenerator' objects.
        """
        data = {
            # Add metadata
            "meta": {"class": self.__class__.__name__, "uid": self.uid,},
            "trajectory_generators": [
                # TODO: Expand into nested dictionary with added 'TrajectoryGenerator'
                #       data
                (anomaly, trajectory_generator.to_latlon())
                for anomaly, trajectory_generator in self.trajectory_generators
            ],
            "start_datetime": str(self.start_datetime),
            "gap": self.gap.total_seconds(),
        }
        return json.dumps(data)
