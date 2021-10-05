import logging
import time
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any, Callable, List, Tuple, Union

import backend.xchanges as xchanges
import geohash_hilbert as ghh  # pip install geohash_hilbert
import msgpack  # pip install msgpack
import redis  # pip install redis
from shapely.geometry import Polygon  # pip install shapely


# Internally used class
class _BaseGeoHelper:
    def __init__(
        self,
        # Pass Redis client and connection from outer scope
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        precision: int = 16,
        bits_per_char: int = 2,
        namespace: str = None,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            exchange (.., optional): Redis client and connection. Defaults to 5.
            precision (int, optional): Geohash precision. Defaults to 5.
            bits_per_char (int, optional): Geohash bits per character. \
                Defaults to 6.
        """
        self._exchange = exchange
        if namespace is None:
            self.namespace = type(self).__name__
        else:
            self.namespace = namespace
        self.geohash_precision = precision
        self.geohash_bits_per_char = bits_per_char

    def get_key(self, **kwargs: Any) -> str:
        """
        Construct a proper key for accessing a certain datastructure in Redis.

        Args:
            kwargs (dict): Additional information that can be used to generate
                a key.

        Returns:
            str: A key to a Redis datastructure.
        """
        # Set kwargs defualts
        kwargs.setdefault("obj_type", "StayPointCluster")
        # Catch possibly missing data
        if "user" not in kwargs:
            raise KeyError(
                "The 'user' key is not present in the given 'kwargs' argument. \
                The 'user' key needs to be present, such that an appropriate \
                key can be generated and used for accessing data in Redis."
            )
        return f"{kwargs['user']}:{kwargs['obj_type']}"

    @abstractmethod
    def set_ttl(self, key: str) -> None:
        """
        Set the time to live (ttl) for a certain key in Redis.

        Args:
            key (str): The key in Redis that should be assigned a time to live \
                (ttl).
        """
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    @abstractmethod
    def enforce_ttl(self) -> None:
        """Enforce time to live (ttl) whenever data is accessed in Redis."""
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    def encode_geoset(
        self,
        centroid: Tuple[float, ...],
        exterior: List[Tuple[float, ...]],
        extra_data: List[Any],
    ) -> Tuple[Any, ...]:
        """
        Prepare the geospatial features to be inserted into Redis.

        Args:
            centroid (Tuple[float, ...]): The 'center' of a polygon that is \
                used as a representitive point.
            exterior (List[float]): The exterior boundry of a polygon as a \
                sequence of points (latitude and longitude tuples).

        Returns:
            Tuple[Any, ...]: A 3-tuple consisting of the latitude and \
                longitude coordinates of the centroid of a polygon, along with \
                the exterior boundry of the polygon supplied as extra data.
        """
        coords_centroid: Any = sum(list(centroid), ())
        coords_exterior = list(exterior)
        data = {
            "timestamp": str(datetime.utcnow()),
            "exterior": coords_exterior,
            "extra_data": extra_data,
        }
        # Return 3-tuple: (lon, lat, extra data)
        # NOTE: This order should be used else queries will not work with Redis!
        return coords_centroid[1], coords_centroid[0], msgpack.dumps(data)

    def decode_geoset(self, geoset: List[Any]) -> List[Polygon]:
        """
        Decode encoded geospatial data contained in the geoset.

        Args:
            geoset (List[Any]): A list with encoded geospatial data.

        Returns:
            List[Polygon]: Geospatial features.
        """
        decoded_geoset = []
        for item in geoset:
            decoded_dict = msgpack.loads(item[0])
            decoded_geoset.append(
                [Polygon(decoded_dict["exterior"]), decoded_dict["extra_data"]],
            )
        return decoded_geoset

    @abstractmethod
    def get_geoset(
        self,
        lat: float,
        lon: float,
        # Pass extra identifiers used to retrive the needed data!
        **kwargs: Any,
    ) -> Union[None, List[Polygon]]:
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    @abstractmethod
    def _add_geoset(self, key: str, geoset: Tuple[Any, ...]) -> int:
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")


# Internally used class
class _GeoHelper(_BaseGeoHelper):
    def __init__(
        self,
        # Pass Redis client and connection from outer scope
        exchange: xchanges.InMemoryMessageExchangeWrapper,
        precision: int = 16,
        bits_per_char: int = 2,
        namespace: str = None,
    ) -> None:
        super().__init__(
            exchange=exchange,
            precision=precision,
            bits_per_char=bits_per_char,
            namespace=namespace,
        )

    def set_ttl(self, key: str) -> None:
        # Do not set or enforce any ttl for now as values are currently updated
        # regularly and when needed
        return None

    def enforce_ttl(self) -> None:
        # Do not set or enforce any ttl for now as values are currently updated
        # regularly and when needed
        return None

    def get_geoset(
        self,
        lat: float,
        lon: float,
        # Pass extra identifiers used to retrive the needed data!
        **kwargs: Any,
    ) -> Union[None, List[Polygon]]:
        key = self.get_key(**kwargs)
        if self._exchange is not None:
            geoset = self._exchange.kv_get(key)
            if geoset is not None:
                return self.decode_geoset(geoset)
            else:
                logging.warning("")  # TODO: Error message
                return None
        else:
            logging.debug("")  # TODO: Debug message
            return None

    def _add_geoset(self, key: str, geoset: Tuple[Any, ...]) -> int:
        if isinstance(self._exchange, xchanges.InMemoryMessageExchangeWrapper):
            if isinstance(
                self._exchange.client, xchanges.InMemoryMessageExchange
            ):
                self._exchange.kv_set(data=geoset, key=key)
                return 1
            else:
                logging.warning("")
                raise ValueError("")
        else:
            logging.warning("")
            raise ValueError("")


# Internally used class
class _RedisGeoHelper(_BaseGeoHelper):
    """A helper class for working with shapely objects with Redis."""

    def __init__(
        self,
        # Pass Redis client and connection from outer scope
        exchange: xchanges.RedisMessageExchangeWrapper,
        precision: int = 16,
        bits_per_char: int = 2,
        namespace: str = None,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            exchange (RedisQueueMessageExchange, optional): Redis client and \
                connection. Defaults to 5.
            precision (int, optional): Geohash precision. Defaults to 5.
            bits_per_char (int, optional): Geohash bits per character. \
                Defaults to 6.
        """
        super().__init__(
            exchange=exchange,
            precision=precision,
            bits_per_char=bits_per_char,
            namespace=namespace,
        )

    def set_ttl(self, key: str) -> None:
        # Do not set or enforce any ttl for now as values are currently updated
        # regularly and when needed
        return None

    def enforce_ttl(self) -> None:
        # Do not set or enforce any ttl for now as values are currently updated
        # regularly and when needed
        return None

    def get_geoset(
        self,
        lat: float,
        lon: float,
        # Pass extra identifiers used to retrive the needed data!
        **kwargs: Any,
    ) -> Union[None, List[Polygon]]:
        """
        Given a location in lat/lon coordinates return polygons in the vicinity.

        Note: TODO: In the future we might want cache other types of polygons \
            or be more specific.
        Args:
            lat (float): The latitude coordinate of a location in decimal \
                degrees.
            lon (float): The longitude coordinate of a location in decimal \
                degrees.

        Returns:
            Union[None, bool]: If the query was successful, then all polygons \
                in the vicinity of the given point is returned.
        """
        # If the geohash has already been computed, then simply use the geohash
        # that was passed as an extra argument
        if "geohash" in kwargs:
            geohash = kwargs.pop("geohash")
        # Else compute an appropriate geohash based on the given location
        else:
            geohash = ghh.encode(
                # NOTE: Make sure that the first arg is longitude then latitude!
                lon,
                lat,
                precision=self.geohash_precision,
                bits_per_char=self.geohash_bits_per_char,
            )
        # Get the bounding box encoded by the geohash
        geohash_data = ghh.rectangle(geohash)
        geohash_bbox = geohash_data["bbox"]
        # Swap elements in bbox. We get elements (lon, lat), but we want
        # (lat, lon)
        geohash_bbox = [
            geohash_bbox[1],
            geohash_bbox[0],
            geohash_bbox[3],
            geohash_bbox[2],
        ]
        # Contruct the key of the datastructure in Redis that we want to access
        key = self.get_key(**kwargs)
        polygons = self._get_geoset(
            key=key, lat=lat, lon=lon, bbox=geohash_bbox, **kwargs,
        )
        return polygons

    def _add_geoset(self, key: str, geoset: Any) -> int:
        """
        Add geospatial data to Redis by supplying a 'key' and encoded 'geoset'.

        Args:
            message (dict): The data that should be added to the Redis \
                datastructure.
            key (str): The key to the Redis datastructure.
        """
        if isinstance(self._exchange, xchanges.RedisMessageExchangeWrapper):
            if isinstance(self._exchange.client, redis.Redis):
                start_time = datetime.utcnow() + timedelta(
                    seconds=self._exchange.timeout
                )
                geodata_added = False
                while True:
                    if datetime.utcnow() - start_time > timedelta(
                        seconds=self._exchange.timeout
                    ):
                        logging.warn(
                            f"Waited {self._exchange.timeout} seconds. Data \
                            could not be added!"
                        )
                        # Attempt was unsuccessful
                        return 0
                    try:
                        if geodata_added is False:
                            # Add the geospatial data to Redis datastructure
                            return_code = self._exchange.client.geoadd(  # type: ignore # noqa
                                key, *geoset
                            )
                            if return_code > 0:
                                geodata_added = True
                        if geodata_added is True:
                            break
                    except redis.exceptions.ConnectionError:
                        # Try to fix the connection
                        self._exchange.reset_connection()
                    time.sleep(self._exchange.wait_time)
                # Set the ttl for the Redis datastructure with the given input
                # key
                self.set_ttl(key=key)
                # Attempt was successful
                return 1
            else:
                logging.warning("")
                raise ValueError("")
        else:
            logging.warning("")
            raise ValueError("")

    def _get_geoset(
        self,
        key: str,
        lat: float,
        lon: float,
        bbox: List[float],
        neighbours: int = 10,
        radius: float = 750.0,
        unit: str = "m",
        **kwargs: Any,
    ) -> Union[None, List[Polygon]]:
        """
        Retrieve a set of geospatial features (a geoset).

        Note:
            The set of geospatial features (a geoset) is contained in a \
            bounding box encoded by a geohash and near a given location.

        Args:
            key (str): The key used to specify the datastructure in Redis that \
                should be queried.
            lat (float): Latitude coordinate in decimal degrees.
            lon (float): Longitude coordinate in decimal degrees.
            bbox (Tuple[float,...]): A bounding box.
            k (int, optional): The 'k' nearest geospatial features in the Redis
                datastructure to return. Defaults to 5.
            radius (float, optional): The radius in meters to look for polygon \
                centroids. Defaults to 100.

        Returns:
            Union[None, List[Polygon]]: Return requested polygons if possible, \
                otherwise None.
        """
        if isinstance(self._exchange, xchanges.RedisMessageExchangeWrapper):
            if isinstance(self._exchange.client, redis.Redis):
                # Evict data that have been been cached for too long
                self.enforce_ttl()
                # Check if the datastructure that contains the 'geoset' exists
                # in cache
                if self._exchange.client.exists(key) == 0:
                    # -> If the datastructure does not exist then retrieve the
                    # data and cache the it by loading it into a Redis
                    # datastructure
                    logging.debug(
                        f"Function: '{self._get_geoset.__name__}'. \
                        Information: Retrieving geospatial features to be \
                        cached as a geoset!"
                    )
                    func = kwargs.get("func", None)
                    if func is not None:
                        retrieve_polygons: Callable = func
                        polygons, extra_data = retrieve_polygons(bbox, **kwargs)
                        if polygons is not None and extra_data is not None:
                            for i, polygon in enumerate(polygons):
                                geoset0 = self.encode_geoset(
                                    centroid=polygon.centroid.coords,
                                    exterior=polygon.exterior.coords,
                                    extra_data=extra_data[i],
                                )
                                self._add_geoset(key=key, geoset=geoset0)
                            # TODO: Structure return data more elegantly!
                            return list(zip(polygons, extra_data))
                        else:
                            logging.warning(
                                f"Function: '{self._get_geoset.__name__}'. \
                                Problem: The response was {polygons}, \
                                {extra_data}! Could not retrieve geoset data!"
                            )
                            return None
                    else:
                        # TODO: Define appropriate error message.
                        #       Function not passed as arg in dict.
                        logging.warning("")
                        raise KeyError("")
                # -> Otherwise, retrieve the existing data from Redis
                else:
                    logging.debug(
                        f"Function: '{self.get_geoset.__name__}'. Information: \
                        Reading cached data from Redis!"
                    )
                    geoset1 = self._exchange.client.georadius(  # type: ignore # noqa
                        name=key,
                        longitude=lon,
                        latitude=lat,
                        # Specify that the result and query is in input units
                        # 'unit'
                        unit=unit,
                        # Specify the lookup radius in input units 'unit'
                        radius=radius,
                        withdist=True,
                        # Return the 'neighbours' closest polygon centroids
                        count=neighbours,
                        # Return polygon centroids in sorted order: Closest to
                        # farthest
                        sort="ASC",
                    )
                    return self.decode_geoset(geoset1)
            else:
                logging.warning("")
                raise ValueError("")
        else:
            logging.warning("")
            raise ValueError("")
