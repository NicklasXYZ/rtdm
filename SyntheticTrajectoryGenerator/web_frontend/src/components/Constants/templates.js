// GeoJSON template dictionaries
export const FEATURE = {
    "type": "Feature",
    "properties": {
        "color": "",                   // Should be added before setting data
    },
    "geometry": {
        "type": "", "coordinates": [], // Should be added before setting data
    },
};
export const FEATURE_COLLECTION = {
    "type": "geojson",
    "data": {
        "type": "FeatureCollection",
        "features": [],                // Should be added before setting data
    }
};
export const POINT_LAYER = {
    "id": "",     // Should be added before adding as Mapbox layer!
    "type": "circle",
    "source": "", // Should be added
    "paint": {
        "circle-radius": {
            "base": 5.0,
            "stops": [[12, 2], [22, 180]],
        },
        "circle-color": ["get", "color"],
    },
};

// Mapbox layer template dictionaries
export const POINT_LAYER = {
    "id": "",     // Should be added before adding as Mapbox layer!
    "type": "circle",
    "source": "", // Should be added before adding as Mapbox layer!
    "layout": {}, // Leave empty
    "paint": {
        "circle-radius": {
            "base": 5.0,
            "stops": [[12, 2], [22, 180]],
        },
        "circle-color": ["get", "color"],
    },
};
export const LINE_LAYER = {
    "id": "",     // Should be added before adding as Mapbox layer!
    "type": "line",
    "source": "", // Should be added before adding as Mapbox layer!
    "layout": {
        "line-join": "round",
        "line-cap" : "round"
    },
    "paint": {
        "line-width": 1.00,
        "line-color": ["get", "color"],
    },
};
export const FILL_LAYER = {
    "id": "",     // Should be added before adding as Mapbox layer!
    "type": "fill",
    "source": "", // Should be added before adding as Mapbox layer!
    "layout": {}, // Leave empty
    "paint": {
        "paint": {
            "fill-color": ["get", "color"],
            "fill-opacity": 1.00,
        },
    },
};
