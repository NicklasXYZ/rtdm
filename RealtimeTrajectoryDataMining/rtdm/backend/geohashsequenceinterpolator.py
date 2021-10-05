from typing import List, Tuple

import backend.datastructures as ds
import geohash_hilbert as ghh
import utm
from backend.typealias import Sequence, Union
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


# Class for namespacing operations on a sequence of geohashes. None of the
# methods in this class should be used by themselves. The a proper control
# flow, i.e., the proper sequence of operations should be handled in an outer
# scope.
class BaseGeohashSequenceInterpolator:
    @staticmethod
    def _interpolate_geohash_sequence(
        sequence: Sequence,
        distance_delta: float,
        precision: int,
        bits_per_char: int,
    ) -> Tuple[Sequence, Union[None, LineString]]:
        raise NotImplementedError("Missing implementation in subclass!")

    @staticmethod
    def _interpolate(
        start_geohash: str,
        end_geohash: str,
        precision: int,
        bits_per_char: int,
    ) -> Tuple[Sequence, Union[None, LineString]]:
        raise NotImplementedError("Missing implementation in subclass!")

    @staticmethod
    def _next_geohash(
        geohash: str,
        linestring: LineString,
        precision: int,
        bits_per_char: int,
        buffer: float = 2.5,
    ) -> str:
        raise NotImplementedError("Missing implementation in subclass!")


class GeohashSequenceInterpolator(BaseGeohashSequenceInterpolator):
    """The most consistent method for interpolating geohash sequences."""

    @staticmethod
    def _interpolate_geohash_sequence(
        sequence: Sequence,
        distance_delta: float,
        precision: int,
        bits_per_char: int,
    ) -> Tuple[Sequence, Union[None, LineString]]:
        if len(sequence) > 1:
            new_sequence: List[str] = []
            for i in range(1, len(sequence)):
                (
                    interpolated_sequence,
                    _,
                ) = GeohashSequenceInterpolator._interpolate(
                    start_geohash=sequence[i - 1],
                    end_geohash=sequence[i],
                    precision=precision,
                    bits_per_char=bits_per_char,
                )
                new_sequence.extend(interpolated_sequence[:-1])
            new_sequence.extend([sequence[-1]])
            return new_sequence, None
        else:
            return sequence, None

    @staticmethod
    def _sequence_to_linestring(
        sequence: Sequence,
        distance_delta: float,
        precision: int,
        bits_per_char: int,
    ) -> Tuple[Union[None, LineString], None]:
        coordinates = []
        for geohash in sequence:
            # The variable 'decoded_geohash' is a (Longitude, Latitude) pair
            decoded_geohash = ghh.decode(geohash, bits_per_char=bits_per_char)
            # The method 'utm.from_latlon' expects a (Latitude, Longitude) pair
            decoded_geohash = utm.from_latlon(
                decoded_geohash[1], decoded_geohash[0]
            )
            # Save as UTM converted (Latitude, Longitude) pair
            coordinates.append([decoded_geohash[0], decoded_geohash[1]])
        if len(coordinates) > 1:
            return LineString(coordinates), None
        else:
            return None, None

    @staticmethod
    def _interpolate(
        start_geohash: str,
        end_geohash: str,
        precision: int,
        bits_per_char: int,
    ) -> Tuple[Sequence, Union[None, LineString]]:
        geohash_sequence = [start_geohash]
        while geohash_sequence[-1] != end_geohash:
            linestring, _ = GeohashSequenceInterpolator._sequence_to_linestring(
                sequence=[geohash_sequence[-1], end_geohash],
                distance_delta=0,
                precision=precision,
                bits_per_char=bits_per_char,
            )
            next_geohash = GeohashSequenceInterpolator._next_geohash(
                geohash=geohash_sequence[-1],
                linestring=linestring,
                precision=precision,
                bits_per_char=bits_per_char,
            )
            if next_geohash is not None:
                geohash_sequence.append(next_geohash)
            # TODO: Define a better 'break out' criteria based on the
            # 'distance_delta' value (approximate distance between neighbouring
            # geohash grid cells)
            if len(geohash_sequence) > 1000:
                print("Geohash sequence problem!")
                break
        # Return geohash sequence between start and end geohash values.
        # Start and end geohash values are included
        return geohash_sequence, None

    @staticmethod
    def _next_geohash(
        geohash: str,
        linestring: LineString,
        precision: int,
        bits_per_char: int,
        buffer: float = 2.5,
    ) -> str:
        neighbours = ghh.neighbours(geohash, bits_per_char=bits_per_char)
        neighbour_objects = {}
        for key in neighbours:
            rectangle = ghh.rectangle(
                neighbours[key], bits_per_char=bits_per_char
            )
            rectangle_object = rectangle["geometry"]["coordinates"][0]
            transformed_coords = [
                utm.from_latlon(v[1], v[0]) for v in rectangle_object
            ]
            neighbour_objects[key] = Polygon(
                [(x, y) for x, y, _, _ in transformed_coords]
            )
        # Shapely 'unary_union': Join all neighbour objects into a single
        # geometry
        geometry = unary_union(list(neighbour_objects.values()))
        intersecting_points = list(geometry.intersection(linestring).coords)
        next_geohash = geohash
        if len(intersecting_points) > 0:
            point = Point(intersecting_points[0])
            for key in neighbour_objects:
                if neighbour_objects[key].buffer(buffer).contains(point):
                    next_geohash = neighbours[key]
                    break
            return next_geohash
        else:
            return next_geohash


class NaiveGeohashSequenceInterpolator(BaseGeohashSequenceInterpolator):
    """A naive method for interpolating geohash sequences."""

    # The encapsulated methods work by contructing a straight line between
    # geohash cells that travels through intermediate geohash cells that are
    # then added to the given input geohash sequence.

    @staticmethod
    def interpolate(
        linestring: LineString, distance_delta: float
    ) -> LineString:
        npoints = int(linestring.length // distance_delta) + 1
        points = [
            linestring.interpolate(i / float(npoints - 1), normalized=True)
            for i in range(npoints)
        ]
        points_ = []
        for point in points:
            points_.append(
                # TODO: Automatically determine the correct UTM Zone
                #       Zone 32 is Denmark... and the target region for now...
                utm.to_latlon(
                    easting=point.x,
                    northing=point.y,
                    zone_number=32,
                    zone_letter="U",
                )
            )
        return LineString(points_)

    @staticmethod
    def _interpolate_geohash_sequence(
        sequence: Sequence,
        distance_delta: float,
        precision: int,
        bits_per_char: int,
    ) -> Tuple[Sequence, Union[None, LineString]]:
        if len(sequence) > 1:
            new_sequence = []
            for i in range(1, len(sequence)):
                (
                    interpolated_sequence,
                    _,
                ) = NaiveGeohashSequenceInterpolator._interpolate(
                    start_geohash=sequence[i - 1],
                    end_geohash=sequence[i],
                    precision=precision,
                    bits_per_char=bits_per_char,
                )
                new_sequence.extend(interpolated_sequence[:-1])
            new_sequence.extend([sequence[-1]])
            # Return whole sequence including start and end geohash values
            # that aligns with start and end geohash values of a given input
            # sequence "sequence"
            return new_sequence, None
        else:
            return sequence, None

    @staticmethod
    def _sequence_to_linestring(
        sequence: Sequence,
        distance_delta: float,
        precision: int,
        bits_per_char: int,
    ) -> Tuple[Union[None, LineString], None]:
        coordinates = []
        for geohash in sequence:
            # The variable 'decoded_geohash' is a (Longitude, Latitude) pair
            decoded_geohash = ghh.decode(geohash, bits_per_char=bits_per_char)
            # The method 'utm.from_latlon' expects a (Latitude, Longitude) pair
            decoded_geohash = utm.from_latlon(
                decoded_geohash[1], decoded_geohash[0]
            )
            # Save as UTM converted (Latitude, Longitude) pair
            coordinates.append([decoded_geohash[0], decoded_geohash[1]])
        if len(coordinates) > 1:
            return LineString(coordinates), None
        else:
            return None, None

    @staticmethod
    def _interpolate(
        start_geohash: str,
        end_geohash: str,
        precision: int,
        bits_per_char: int,
        distance_delta: Union[None, float] = None,
    ) -> Tuple[Sequence, Union[None, LineString]]:
        sequence_: Sequence = []
        if distance_delta is None:
            distance_delta = ds.interpolation_distance_delta(
                precision=precision, bits_per_char=bits_per_char,
            )
        (
            linestring,
            _,
        ) = NaiveGeohashSequenceInterpolator._sequence_to_linestring(
            sequence=[start_geohash, end_geohash],
            distance_delta=distance_delta,
            precision=precision,
            bits_per_char=bits_per_char,
        )
        if linestring is not None:
            linestring = NaiveGeohashSequenceInterpolator.interpolate(
                linestring=linestring, distance_delta=distance_delta,
            )
            for datapoint in linestring.coords:
                geohash = ghh.encode(
                    # The method 'ghh.encode' expects a (Longitude, Latitude)
                    # pair
                    datapoint[1],
                    datapoint[0],
                    precision=precision,
                    bits_per_char=bits_per_char,
                )
                if len(sequence_) >= 1:
                    if sequence_[-1] != geohash:
                        sequence_.append(geohash)
                    # Else skip, so we do not collect identical geohash values
                else:
                    sequence_.append(geohash)
            # TODO: Check if this is consistent (start/end included) with
            # class 'GeohashInterpolator'
            return sequence_, linestring
        else:
            # TODO: Check if this is consistent (start/end included) with
            # class 'GeohashInterpolator'
            return [start_geohash, end_geohash], None


# TODO:
# Add method for displaying linestring, points, sequence + interpolated cells
