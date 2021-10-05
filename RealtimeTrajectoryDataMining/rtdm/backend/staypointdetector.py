import itertools
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Set, Tuple, Union

import backend.datastructures as ds
import backend.models as django_models
import backend.plotutils as plotutils
import backend.utils as utils
import community as community_louvain  # pip install python-louvain
import folium  # pip install folium
import matplotlib.pylab as plt  # pip install matplotlib
import networkx as nx  # pip install networkx
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas
import seaborn as sns  # pip install seaborn
from backend.basegeohelper import _GeoHelper, _RedisGeoHelper
from backend.typealias import CoordinatePair
from django.conf import settings
from scipy import stats  # pip install scipy
from shapely.geometry import MultiPoint, Point, Polygon
from sklearn.neighbors import BallTree  # pip install scikit-learn


# Internally used class for collecting common functions across classes
# - reducing duplicate code...
class _BaseStayPointDetector:
    def __init__(
        self,
        radius: float,
        reference_user: str,
        min_weight: int = 0,
        percentile: Union[None, int] = None,
        trajectory_uids: Union[None, List[str]] = None,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            percentile (int): A percentile threshold for filtering out \
                insignificant stay points. The value should be between 0 and \
                100.
            radius (float): The max distance at which another datapoint is \
                considered a neighbour. The radius is assumed to be given in \
                meters.
        """
        self.reference_user = reference_user
        if trajectory_uids is None:
            self.trajectory_uids = []
        else:
            self.trajectory_uids = trajectory_uids
        # Set a percentile threshold for filtering out insignificant stay
        # points
        self.percentile = percentile
        # Set the max distance at which another datapoint is considered a
        # neighbour
        self.radius = radius / settings.EARTH_RADIUS
        self.min_weight = min_weight
        # Dataframe containing significant stay points, i.e., locations where an
        # entity has spent a significant amount of time compared to other stay
        # points
        self._df: pd.DataFrame = None
        # Validate input arguments
        self._check_vars()

    @abstractmethod
    def fit(
        self,
        user: str,
        trajectories: List[ds.Trajectory],
        min_weight: float = 0.0,
    ) -> None:
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    @abstractmethod
    # TODO: Determine function return type
    def extract_geofences(
        self, datapoints: List[ds.DataPoint], **kwargs: Any,
    ) -> List[Polygon]:
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    @abstractmethod
    def retrieve_polygons(
        self, bbox: List[float], **kwargs: Any,
    ) -> Union[Tuple[None, None], Tuple[List[Polygon], List[int]]]:
        """
        Retrieve polygons from cache.

        Args:
            bbox (List[float]): The bounding box of the geospatial area that \
                was queried.

        Returns:
            Union[Tuple[None, None], Tuple[List[Polygon], List[int]]]:: A list \
                of requested polygons or None.
        """
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    def create_geofencing(
        self, multiplier: float = 25,
    ) -> Union[Tuple[None, None], Tuple[List[Polygon], List[int]]]:
        """
        Create geofenced reginos around certain locations.

        Args:
            multiplier (float): Increase the size of the geofences \
                proportionally to the datapoint aggregation radius, i.e., \
                'multipler' * 'radius'. Defaults to 25.

        Returns:
            Union[None, List[Polygon]]: Returns a list of polygons, \
                otherwise None.
        """
        if self._df is None:
            logging.warning(
                "Function: '"
                + self.create_geofencing.__name__
                + "'. Problem: "
                + "The '.fit()' method needs to be called first!"
            )
            return None, None
        else:
            geometries = []
            labels = []
            grouped_df = self._df.groupby("label")
            # Extract all datapoints and their corresponding latitude and
            # longitude coordinates from each cluster
            for key, _ in grouped_df:
                values = grouped_df.get_group(key)
                coordinates = values[["latitude", "longitude"]].values.tolist()
                shape = []
                for coordinate_pair in coordinates:
                    polygons = Point(coordinate_pair).buffer(
                        self.radius * multiplier
                    )
                    points = list(polygons.exterior.coords)
                    shape.extend(points)
                multipoint = MultiPoint(shape)
                geometries.append(multipoint.convex_hull)
                labels.append(key)
            return geometries, labels

    def plot(
        self,
        center: Union[None, CoordinatePair] = None,
        color: Union[None, str] = None,
        radius: float = 5.0,
        opacity: float = 1,
        map_: Union[None, folium.Map] = None,
    ) -> folium.Map:
        if self._df is None:
            logging.warning(
                f"Function: '{self.plot.__name__}'. Problem: The '.fit()' \
                method needs to be called first!"
            )
            return None
        else:
            if map_ is None:
                if center is not None:
                    map_ = plotutils.setup_map(center)
                else:
                    raise ValueError(
                        "No map 'center' argument given. The map can thus \
                        not be initialized."
                    )
            stay_points = self._df[
                ["label", "mid_lat", "mid_lon", "color"]
            ].drop_duplicates()
            for row in stay_points.values:
                if color is not None:
                    _color = color
                else:
                    _color = row[3]
                folium.CircleMarker(
                    # Latitude, longitude coordinates of the marker
                    [row[1], row[2]],
                    # The radius (in pixels!) of the circle marker
                    radius=radius,
                    # The hex color code of the circle marker
                    color=_color,
                ).add_to(map_)
            # Plot geofence polygon geometries on the map
            geometries, _ = self.create_geofencing()
            if geometries is not None:
                for geometry in geometries:
                    folium.PolyLine(
                        # Transform a shapely Polygon to a float coordinate pair
                        list(geometry.exterior.coords),
                        color="cyan",
                        opacity=opacity,
                    ).add_to(map_)
            return map_

    def plot_weight_histogram(
        self,
        stay_points: List[ds.DataPoint],
        bw_adjust: float = 0.75,
        boxcox: bool = False,
    ) -> None:
        df = self._create_stay_point_df(stay_points)
        if boxcox:
            df["weight"], _ = stats.boxcox(df["weight"].values.ravel())
        # Plot distribution
        sns.kdeplot(df[["weight"]].values.ravel(), bw_adjust=bw_adjust)
        # Calculate weight percentiles based on all stay points
        if self.percentile is not None:
            df_dict = df.describe([(i + 1) / 100 for i in range(99)]).to_dict()
            pc = df_dict["weight"][str(self.percentile) + "%"]
            plt.axvline(
                pc, color="red", label="Percentile = " + str(self.percentile)
            )
        plt.title(
            f"Stay point weight distribution. Box-Cox transformed = \
            {str(boxcox)}"
        )
        if not boxcox:
            plt.xlim([0, np.max(df["weight"])])
        logging.info("Min weight        : " + str(np.min(df["weight"])))
        logging.info("Max weight        : " + str(np.max(df["weight"])))
        logging.info("Average weight    : " + str(np.mean(df["weight"])))
        logging.info("Standard deviation: " + str(np.std(df["weight"])))
        logging.info("Median weight     : " + str(np.median(df["weight"])))

    def _check_vars(self) -> None:
        """Check and validate all class arguments on class instantiation."""
        # percentile
        if self.percentile is not None:
            if not isinstance(self.percentile, int):
                error = f"ARG 'percentile' is of type \
                    {type(self.percentile)} but should be of type 'int'!"
                try:
                    self.percentile = int(self.percentile)
                except Exception:
                    raise TypeError(error)
            else:
                if self.percentile <= 0 or self.percentile >= 100:
                    raise ValueError(
                        f"The given 'percentile' value is: {self.percentile}. \
                        It needs to be larger than 0 but smaller than 100"
                    )
        # min_weight
        if not isinstance(self.min_weight, float):
            error = f"ARG 'min_weight' is of type \
                {type(self.min_weight)} but should be of type 'float'!"
            try:
                self.min_weight = float(self.min_weight)
            except Exception:
                raise TypeError(error)
        else:
            if self.min_weight < 0.0:
                raise ValueError(
                    f"The given 'min_weight' value is: {self.min_weight}. \
                    It needs to be larger than 0.0"
                )

    def _merge_data(
        self,
        df: pd.DataFrame,
        network_partition: Dict[str, Any],
        community_midpoints: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Merge extra relevant data into a given dataframe.

        Args:
            df (pd.DataFrame): The dataframe we want to merge all the extra \
                data into.
            network_partition (Dict): Dictionary of nodes (stay points) and \
                their corresponding assigned label that is associated with a \
                community.
            community_midpoints (Dict): The midpoints of each of the \
                communities.

        Returns:
            pd.DataFrame: The given dataframe enriched with the extra data \
                that was also given as input to the method.
        """
        labels = []
        index = []
        for key, value in network_partition.items():
            labels.append({"label": value})
            index.append(key)
        df_labels = pd.DataFrame(data=labels, index=index)
        # Add community labels to dataframe
        df = df.merge(df_labels, left_index=True, right_index=True)
        midpoints = []
        keys = list(community_midpoints.keys())
        colors = {_: f"#{utils.color_hex()}" for _ in keys}
        for key, value in community_midpoints.items():
            midpoints.append(
                {
                    "label": key,
                    "mid_lat": value[0],
                    "mid_lon": value[1],
                    "color": colors[key],
                }
            )
        df_midpoints = pd.DataFrame(data=midpoints)
        # Add the community midpoints to the dataframe
        df = df.merge(df_midpoints, on="label")
        return df

    def _create_stay_point_df(
        self, stay_points: List[ds.DataPoint],
    ) -> pd.DataFrame:
        """
        Construct a dataframe from a list of stay points.

        Args:
            stay_points (List[DataPoint]): Stay points that should be placed \
                in a dataframe.

        Returns:
            pd.DataFrame: A dataframe containing the given stay points.
        """
        # Run through all stay points and collect "significant" stay points
        # into a pandas dataframe
        data = []
        for stay_point in stay_points:
            dict_ = {
                "latitude": stay_point.latitude,
                "longitude": stay_point.longitude,
                "weight": stay_point.weight,
            }
            data.append(dict_)
        df = pd.DataFrame(data=data).sort_values(
            by=["weight"], ascending=False,
        )
        return df

    def _extract_stay_points(
        self, stay_points: List[ds.DataPoint]
    ) -> pd.DataFrame:
        """
        Given a list of stay points extract the most significant stay points.

        Args:
            stay_points (List[DataPoint]): A list of stay points.

        Returns:
            pd.DataFrame: A dataframe of significant stay points.
        """
        df = self._create_stay_point_df(stay_points)
        if self.percentile is not None:
            # Calculate weight percentiles (1..99) based on all stay points
            df_dict = df.describe([(i + 1) / 100 for i in range(99)]).to_dict()
            pc = df_dict["weight"][str(self.percentile) + "%"]
            # Filter out insignificant stay points that have a low weight and
            # thus fall below a certain percentile
            return df[df["weight"] > pc].sort_values(
                by=["weight"], ascending=False
            )
        else:
            return df.sort_values(by=["weight"], ascending=False)

    def _query_pairs(
        self, df: pd.DataFrame, leaf_size: int = 10
    ) -> Set[Tuple[int, int]]:
        """
        Extract all pairs of stay points that are within distance of each other.

        Args:
            df (pd.DataFrame): A dataframe of stay points that we need to \
                filter through and run distance queries on.

        Returns:
            Set[Tuple[Any, Any]]: All pairs of stay points that are within \
                some distance of each other (identified by indicies in the \
                given dataframe).
        """
        # The haversine distance is computed using trigonometric functions,
        # so convert lat/lon degrees to radians so we can calculate correct
        # distances between locations
        locations = np.radians(df[["latitude", "longitude"]])
        btree = BallTree(locations, leaf_size=leaf_size, metric="haversine")
        indicies, _ = btree.query_radius(
            X=locations, r=self.radius, sort_results=True, return_distance=True,
        )
        # Collect all pairs (x, y) <--> (y, x)
        pairs: Set[Tuple[int, int]] = set()
        for i in range(len(indicies)):
            npairs = len(indicies[i][1:])
            neighbours = iter(indicies[i][1:])
            datapoint = itertools.repeat(i, npairs)
            iterator = zip(neighbours, datapoint)
            for pair in list(iterator):
                pairs.add(tuple(sorted(pair)))  # type: ignore # noqa
        return pairs

    def _retrieve_distances(
        self, pairs: Set[Tuple[int, int]], df: pd.DataFrame,
    ) -> Dict[Tuple[int, int], float]:
        distances = {}
        for pair in pairs:
            ind_i = pair[0]
            ind_j = pair[1]
            row_i = df.iloc[ind_i]
            row_j = df.iloc[ind_j]
            distance = ds.haversine_distance(
                lat_1=row_i["latitude"],
                lon_1=row_j["longitude"],
                lat_2=row_j["latitude"],
                lon_2=row_j["longitude"],
            )
            distances[(ind_i, ind_j)] = distance
        return distances

    def _build_network(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build a network where stay points are nodes.

        Args:
            df (pd.DataFrame): A dataframe of stay points.

        Returns:
            nx.Graph: A network where stay points are nodes and these have \
                link between them if they are within some distance of each \
                other.
        """
        pairs = self._query_pairs(df)
        network = nx.Graph(trajectory="Network")
        distances = self._retrieve_distances(pairs=pairs, df=df)
        mean_distance = np.mean(list(distances.values()))
        for pair in pairs:
            ind_i = pair[0]
            ind_j = pair[1]
            row_i = df.iloc[ind_i]
            row_j = df.iloc[ind_j]
            # Weight each edge between a pair of nodes by the inverse
            # normalized distance
            network.add_edge(
                row_i.name,
                row_j.name,
                weight=distances[(ind_i, ind_j)] / mean_distance,
            )
        return network

    def _community_midpoints(
        self, df: pd.DataFrame, network_partition: Dict,
    ) -> Dict:
        """
        Partition a collection of stay points.

        Args:
            df (pd.DataFrame): A dataframe of significant stay points.
            network_partition (Dict[int, Any]):  A prartition of the stay \
                points.

        Returns:
            Dict: A ditionary containing community labels and corresponding \
                community midpoints.
        """
        community_midpoints = {}
        for datapoint_uid, label in network_partition.items():
            row = df.loc[df.index == datapoint_uid]
            if label not in community_midpoints:
                community_midpoints[label] = [
                    [*row["latitude"].values, *row["longitude"].values],
                ]
            else:
                community_midpoints[label].append(
                    [*row["latitude"], *row["longitude"]],
                )
        community_medians = {}
        for label in community_midpoints:
            community_medians[label] = np.median(
                np.array(community_midpoints[label]), axis=0,
            ).tolist()
        return community_medians


class StayPointDetector(_BaseStayPointDetector):
    """A class for finding stay points in a collection of trajectories."""

    def __init__(
        self,
        radius: float,
        reference_user: str,
        min_weight: int = 0,
        percentile: Union[None, int] = None,
        trajectory_uids: Union[None, List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            percentile (int): A percentile threshold for filtering out \
                insignificant stay points. The value should be between 0 and \
                100.
            radius (float): The max distance at which another datapoint is \
                considered a neighbour. The radius is assumed to be given in \
                meters.
        """
        kwargs.setdefault("namespace", type(self).__name__)
        self._geohelper = _GeoHelper(**kwargs)
        super().__init__(
            radius=radius,
            reference_user=reference_user,
            min_weight=min_weight,
            percentile=percentile,
            trajectory_uids=trajectory_uids,
        )

    def extract_geofences(
        self, datapoints: List[ds.DataPoint], **kwargs: Any,
    ) -> List[Polygon]:
        return self.polygons

    def fit(
        self,
        user: str,
        trajectories: List[ds.Trajectory],
        min_weight: float = 0.0,
    ) -> None:
        """
        Extract stay points belonging to a certain user.

        Args:
            user (str): A unique identifier of a user.
            datapoints (List[DataPoint], optional): Given datapoints that we \
                need to extract stay points from. Defaults to None.
            min_weight (float, optional): The minimum weight a datapoint \
                should have before it is considered a stay point. Defaults to 0.

        Raises:
            ValueError: If no datapoints was found with a weight larger than \
                the given minimum weight.
        """
        # If no trajectory uids were provided...
        if len(self.trajectory_uids) == 0:
            # Fetch all aggregate datapoints from the database that has a weight
            # larger than a given value and that belong to a certain user
            stay_points = []
            if self.min_weight > min_weight:
                min_weight = self.min_weight
            for trajectory in trajectories:
                for datapoint in trajectory.datapoints:
                    if datapoint.weight > min_weight:
                        stay_points.append(datapoint)
        else:
            stay_points = []
            print(
                f"Trajectory uids not provided. Using {len(stay_points)} \
                stay points."
            )
        if len(stay_points) == 0:
            raise ValueError(
                f"The given list 'stay_points' had length 0. The analysis can \
                thus not proceed."
            )
        # Extract locations where a user has spent a significant amount of time
        # compared to other stay points belonging to the the user
        df = self._extract_stay_points(stay_points=stay_points)
        # Build a network where nodes are significant stay points
        network = self._build_network(df=df)
        # Find community structure in the network. Each community in the
        # network represents a cluster of significant stay points
        network_partition = community_louvain.best_partition(
            graph=network, weight="weight",
        )
        # Compute a representative location of each of the clusters. These
        # points are only used for plotting purposes
        community_midpoints = self._community_midpoints(
            df=df, network_partition=network_partition,
        )
        # Merge all the data into a single dataframe
        self._df = self._merge_data(
            df=df,
            network_partition=network_partition,
            community_midpoints=community_midpoints,
        )

        # TODO: TEMP
        # Plot geofence polygon geometries on the map
        polygons, labels = self.create_geofencing()
        if polygons is not None and labels is not None:
            self.polygons = list(zip(polygons, labels))
        else:
            logging.debug("")


# NOTE: Multiple inheritance. Attributes in parent classes are set based on the
#       order of listed classes - left to right
class RedisStayPointDetector(_BaseStayPointDetector, _RedisGeoHelper):
    """A class for finding stay points in a collection of trajectories."""

    def __init__(
        self,
        radius: float,
        reference_user: str,
        min_weight: int = 0,
        percentile: Union[None, int] = None,
        trajectory_uids: Union[None, List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            percentile (int): A percentile threshold for filtering out stay \
                points. The value should be between 0 and 100.
            radius (float): The max distance at which another datapoint is \
                considered a neighbour. The radius is assumed to be given \
                in meters.
        """
        kwargs.setdefault("namespace", type(self).__name__)
        self._geohelper = _RedisGeoHelper(**kwargs)
        super().__init__(
            radius=radius,
            reference_user=reference_user,
            min_weight=min_weight,
            percentile=percentile,
            trajectory_uids=trajectory_uids,
        )

    # Override base class method
    def retrieve_polygons(
        self, bbox: List[float], **kwargs: Any,
    ) -> Union[Tuple[None, None], Tuple[List[Polygon], List[int]]]:
        """
        Retrieve polygons from cache.

        Args:
            bbox (List[float]): [description]

        Raises:
            KeyError: If an identifier 'user' is not contained in 'kwargs' \
                function argument.

        Returns:
            Union[Tuple[None, None], Tuple[List[Polygon], List[int]]]: a list \
                of requested polygons or None.
        """
        # Method called by base class: BaseGeoHelper
        if "user" not in kwargs:
            raise KeyError(
                "The 'user' key is not present in the given 'kwargs' argument. \
                The 'user' key needs to be present, such that an appropriate \
                the appropriate data can be retrieved."
            )
        self.fit(
            user=self.reference_user,
            trajectories=[],
            min_weight=self.min_weight,
        )
        polygons, labels = self.create_geofencing()
        # Reset variable
        self._df = None
        if polygons is not None and labels is not None:
            return polygons, labels
        else:
            logging.debug("")
            return None, None

    def extract_geohashes(
        self, datapoints: List[ds.DataPoint]
    ) -> Set[Tuple[str, float, float]]:
        """
        Extract necessary geohashes for retrieving geofences in Redis.

        Note:
            Extract necessary geohashes for retrieving geofences, in Redis, \
            that are essentially geospatial features representing stay point \
            clusters belonging to a user. This is much faster if there are \
            only few, as only a small number of objetcs need to be \
            deserialized and queried (in comparison network calls to Redis \
            have relatively high overhead in this case). On the other hand, \
            if there are many geofences then it does not make sense to \
            retrieve all the corresponding geospatial features. Sequential \
            look-ups (w.r.t. distance) are faster as we can more selectively \
            determine which geospatial features we need to deserialize and \
            query.

        Args:
            TODO

        ReturnS:
            TODO
        """
        geohashes: Set[Tuple[str, float, float]] = set()
        for datapoint in datapoints:
            geohashed_datapoint = datapoint.get_geohashed_datapoint(
                precision=self.geohash_precision,
                bits_per_char=self.geohash_bits_per_char,
            )
            if (
                geohashed_datapoint.geohash is not None
                and geohashed_datapoint.decoded_latitude is not None
                and geohashed_datapoint.decoded_longitude is not None
            ):
                # Save: 0: geohash, 1: latitude, 2: longitude
                geohashes.add(
                    (
                        geohashed_datapoint.geohash,
                        geohashed_datapoint.decoded_latitude,
                        geohashed_datapoint.decoded_longitude,
                    )
                )
            else:
                logging.debug("")
        return geohashes

    def extract_geofences(
        self, datapoints: List[ds.DataPoint], **kwargs: Any
    ) -> List[Polygon]:
        geohash_data = self.extract_geohashes(datapoints=datapoints)
        polygons = []
        seen = set()
        for value in geohash_data:
            polygon = self.get_geoset(
                lat=value[1],
                lon=value[2],
                **{
                    "geohash": value[0],
                    "user": kwargs.get("user"),
                    "datapoints": datapoints,
                },
            )
            if polygon is not None:
                if len(polygon) > 0:
                    for poly in polygon:
                        if poly[-1] not in seen:
                            seen.add(poly[-1])
                            polygons.append(poly)
            else:
                logging.debug("")
        return polygons

    def fit(
        self,
        user: str,
        trajectories: List[ds.Trajectory],
        min_weight: float = 0.0,  # In seconds
    ) -> None:
        """
        Extract stay points belonging to a certain user.

        Args:
            user (str): A unique identifier of a user.
            datapoints (List[DataPoint], optional): Given datapoints that \
                we need to extract significant stay points from. Defaults \
                to None.
            min_weight (float, optional): The minimum weight a datapoint \
                should have before it is considered a stay point. Defaults to 0.

        Raises:
            ValueError: If no datapoints was found with a weight larger than \
                the given minimum weight.
        """
        # If no trajectory uids were provided...
        if len(self.trajectory_uids) == 0:
            # Fetch all aggregate datapoints from the database that has a
            # weight larger than a given value and that belong to a certain user
            stay_points = list(
                django_models.DataPoint.objects.filter(
                    weight__gt=min_weight, user=user,
                )
            )
            print(
                f"Trajectory uids not provided. Using {len(stay_points)} \
                stay points."
            )
        else:
            # Fetch all aggregate datapoints from the database that has a weight
            # larger than a given value and that belong to a certain user.
            # Futhermore, only use aggregate datapoints that belong to certain
            # trajectories
            stay_points = list(
                django_models.DataPoint.objects.filter(
                    weight__gt=min_weight,
                    user=user,
                    trajectory_id__in=self.trajectory_uids,
                )
            )
            print(
                f"Trajectory uids provided. Using {len(stay_points)} stay \
                points."
            )
        if len(stay_points) == 0:
            raise ValueError(
                "The given list 'stay_points' had length 0. The analysis can \
                thus not proceed."
            )
        # Extract locations where a user has spent a significant amount of time
        # compared to other stay points belonging to the the user
        df = self._extract_stay_points(stay_points=stay_points)
        # Build a network where nodes are significant stay points
        network = self._build_network(df=df)
        # Find community structure in the network. Each community in the
        # network represents a cluster of significant stay points
        network_partition = community_louvain.best_partition(network)
        # Compute a representative location of each of the clusters. These
        # points are only used for plotting purposes
        community_midpoints = self._community_midpoints(
            df=df, network_partition=network_partition,
        )
        # Merge all the data into a single dataframe
        self._df = self._merge_data(
            df=df,
            network_partition=network_partition,
            community_midpoints=community_midpoints,
        )
