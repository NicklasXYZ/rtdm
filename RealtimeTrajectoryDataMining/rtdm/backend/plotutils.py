from typing import Union

# import backend.datastructures as ds
import folium  # pip install folium
import geohash_hilbert as ghh
from backend.typealias import CoordinatePair, Sequence
from folium import plugins  # pip install folium

# Folium map-specific helper methods


def setup_map(
    center: CoordinatePair,
    zoom_start: float = 14,
    filename: str = "draw.geojson",
    tiles: str = "cartodbdark_matter",
) -> folium.Map:
    """Create a basic folium map object."""
    map_ = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)
    # Add drawing capabilities to the interface
    plugins.Fullscreen(position="topleft").add_to(map_)
    plugins.Draw(filename=filename, export=True, position="topleft").add_to(
        map_
    )
    return map_


# def plot_datapoint(
#     datapoint: Union[ds.DataPoint, Point],
#     color: str,  # Stroke color
#     opacity: float = 1.0,  # Stroke opacity
#     weight: float = 1.0,  # Stroke width in pixels
#     radius: float = 2.5,  # Radius of the circle marker, in pixels
#     center: Union[None, CoordinatePair] = None,
#     map_: Union[None, folium.Map] = None,
# ) -> folium.Map:
#     if map_ is None and center is not None:
#         map_ = setup_map(center=center)
#     else:
#         error = (
#             "No 'folium.Map' object was passed. The map 'center' argument "
#             + "is thus required to instantiate a new 'folium.Map' object."
#         )
#         raise ValueError(error)
#     if isinstance(datapoint, ds.DataPoint):
#         pass
#     elif isinstance(datapoint, Point):
#         pass
#     else:
#         error = (
#             f"Expected 'datapoint' argument to be of type "
#             + "'ds.DataPoint' or 'shapely.geometry.Point'"
#         )
#         raise ValueError(error)
#     folium.CircleMarker(
#         [datapoint.latitude, datapoint.longitude],
#         color=color,
#         opacity=opacity,
#         weight=weight,
#         radius=radius,
#         popup=f"Weight   : {datapoint.weight},\n"
#         + f"Lat      : {datapoint.latitude},\n"
#         + f"Lon      : {datapoint.longitude},\n"
#         + f"Timestamp: {datapoint.external_timestamp}",
#     ).add_to(map_)
#     return map_


# def plot_trajectory(
#     trajectory: Union[ds.Trajectory, LineString],
#     color: str,  # Stroke color
#     opacity: float = 1.0,  # Stroke opacity
#     weight: float = 1.0,  # Stroke width in pixels
#     center: Union[None, CoordinatePair] = None,
#     map_: Union[None, folium.Map] = None,
# ) -> folium.Map:
#     if map_ is None and center is not None:
#         map_ = setup_map(center=center)
#     else:
#         error = (
#             "No 'folium.Map' object was passed. The map 'center' argument "
#             + "is thus required to instantiate a new 'folium.Map' object."
#         )
#         raise ValueError(error)
#     if isinstance(trajectory, ds.Trajectory):
#         datapoints = trajectory.datapoints
#         identifier = trajectory.uid
#     elif isinstance(trajectory, LineString):
#         datapoints = list(trajectory.coords)
#         identifier = None
#     else:
#         error = f"Expected 'trajectory' argument to be of type \
#                 'ds.Trajectory' or 'shapely.geometry.LineString'"
#         raise ValueError(error)
#     folium.PolyLine(
#         datapoints,
#         color=color,
#         opacity=opacity,
#         weight=weight,
#         popup=f"Identifier: {identifier}",
#     ).add_to(map_)
#     return map_


def plot_sequence(
    sequence: Sequence,
    bits_per_char: int,
    color: str,  # Stroke color
    opacity: float = 1.0,  # Stroke opacity
    weight: float = 1.0,  # Stroke width in pixels
    center: Union[None, CoordinatePair] = None,
    map_: Union[None, folium.Map] = None,
) -> folium.Map:
    if map_ is None and center is not None:
        map_ = setup_map(center=center)
    else:
        error = "No 'folium.Map' object was passed. The map 'center' argument \
            is thus required to instantiate a new 'folium.Map' object."
        raise ValueError(error)
    for geohash in sequence:
        obj = ghh.rectangle(geohash, bits_per_char=bits_per_char)
        bbox_rectangle = obj["geometry"]["coordinates"][0]
        swapped_coords = [(v[1], v[0]) for v in bbox_rectangle]
        folium.PolyLine(
            swapped_coords, color=color, weight=weight, opacity=opacity,
        ).add_to(map_)
    return map_


# # TODO: _xPLOTTING_
# def plot_trajectory_anomalies(
#     self,
#     df: pd.DataFrame,
#     anomaly_indices,
#     center: CoordinatePair,
#     weight: float = 2.0,
#     opacity: float = 1,
#     map_: Union[None, folium.Map] = None,
#     ) -> folium.Map:
#     if map_ is None:
#         map_ = utils.setup_map(center)
#     anomalous_datapoints = []
#     datapoints = []
#     for _, row in df.iterrows():
#         dp = ds.DataPoint(
#             latitude = row["latitude"],
#             longitude = row["longitude"],
#             external_timestamp = str(row.index),
#         )
#         for anomaly_index in anomaly_indices:
#             if row.name == anomaly_index:
#                 anomalous_datapoints.append(dp)
#         datapoints.append(dp)
#     trajectory_ = ds.Trajectory(anomalous_datapoints)
#     trajectory = ds.Trajectory(datapoints)
#     trajectory.plot(
#         center = center,
#         map_ = map_,
#     )
#     for datapoint in trajectory_.datapoints:
#         color = "#" + str(utils.color_hex())
#         map_ = datapoint.plot(
#             center = center,
#             map_ = map_,
#             color = color,
#         )
#     return map_


# # TODO: _xPLOTTING_
# def plot_score(self, df: pd.DataFrame, anomalies: pd.DataFrame):
#     return plot(
#         df,
#         anomaly = anomalies,
#         ts_linewidth = 1,
#         ts_markersize = 5.,
#         anomaly_color = "red"
#     )


# # TODO: _xPLOTTING_
# def plot_polyline(\
#     datapoints,
#     uid,
#     center,
#     weight: float = 2.0,
#     opacity: float = 1,
#     map_ = None,
#     ):
#     # Assign a random color to the trajectory
#     color = "#" + str(utils.color_hex())
#     folium.PolyLine(
#         datapoints,
#         color = color,
#         weight = weight,
#         opacity = opacity,
#         popup = f"UUID: {uid}",
#     ).add_to(map_)
#     return map_
