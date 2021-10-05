// NODE MODULES:
import React, { Component } from "react";
import MapGL, {
    NavigationControl,
    FullscreenControl,
    // Marker,
    // Popup,
} from "react-map-gl";
// LOCAL IMPORTS:
// -> Constants
import { MAPBOX_STYLE, MAPBOX_TOKEN } from "./components/Constants/constants"
// -> Styling
import "./map.css";


// Component styling
const navStyle = {
    position: "absolute",
    top: 0,
    left: 0,
    padding: "10px",
};
const controlStyle = {
    padding: "10px",
};


export default class Map extends Component {

    constructor(props) {
        super(props);
        this.state = {
            // Settings related to the initial Mapbox map view
            viewport: {
                latitude: 55.39594,
                longitude: 10.38831,
                zoom: 15,
                bearing: 0,
                pitch: 0
            },
            // Mapbox map element rendering settings
            popupInfo: null,
            isCreated: false,
            reRender: true,
        };
    }

    // showDetails = () => {
    //     this.setState({popupInfo: true});
    // }

    // hideDetails = () => {
    //     this.setState({popupInfo: null});
    // }

    // renderPopup(index) {
    //     return this.state.popupInfo && (
    //         <Popup tipSize={5}
    //         anchor="bottom-right"
    //         longitude={markerList[index].long}
    //         latitude={markerList[index].lat}
    //         onMouseLeave={() => this.setState({popupInfo: null})}
    //         closeOnClick={true}>
    //         <p> Available beds:{ markerList[index].info } </p>
    //         </Popup>
    //     )
    // }

    componentDidMount() {
        // Retrieve Mapbox map object
        const map = this.reactMap.getMap();
        map.on("load", () => {
            // Specify GeoJSON data sources

            // SEGMENTS
            map.addSource("segments", {
                type: "geojson",
                data: {
                    type: "FeatureCollection",
                    features: [{
                        type: "Feature",
                        properties: {
                            color: "#33C9EB"
                        },
                        // "geometry": { "type": "Point", "coordinates": [10.38831, 55.39594] },
                        // "geometry": { "type": "LineString", "coordinates": [] },
                    }],
                }
            });

            // Specify Mapbox layers
            map.addLayer({
                id    : "lines",
                type  : "line",
                source: "segments",
                "layout": {
                    "line-join": "round",
                    "line-cap" : "round"
                },
                "paint": {
                    "line-width": 2.00,
                    "line-color": ["get", "color"],
                },
            });

            // CONVEX HULL
            map.addSource("convex_hull", {
                type: "geojson",
                data: {
                    type: "FeatureCollection",
                    features: [{
                        type: "Feature",
                        // geometry: { "type": "Polygon", "coordinates": [] },
                        properties: {
                            color: "#FFFFFF"
                        },
                    }],
                }
            });
            map.addLayer({
                "type": "fill",
                "layout": {},
                "paint": {
                    "fill-color": "#FFFFFF",
                    "fill-opacity": 1.00,
                },
                id    : "lines2",
                // type  : "line",
                source: "convex_hull",
                // "layout": {
                //     "line-join": "round",
                //     "line-cap" : "round"
                // },
                // "paint": {
                //     "line-width": 5,
                //     "line-color": ["get", "color"],
                // },
            });
            this.setState({ isCreated: true })
        });
    }

    structureData = (data) => {
        const newFeaturesList = [];
        for (let i = 0; i < data.rows.length; i++) {
            if (data.rows[i].selected === true) {
                const coordList = [];
                for (let j = 0; j < data.rows[i].segment.length; j++) {
                    coordList.push([ // Order: [Lon, Lat]
                        data.rows[i].segment[j].raw_data.longitude,
                        data.rows[i].segment[j].raw_data.latitude,
                    ])
                }
                newFeaturesList.push({
                    type: "Feature",
                    geometry: {
                        type: "LineString",
                        coordinates: coordList,
                    },
                    "properties": {
                        "color": data.rows[i].color,
                    },
                    // properties: {
                    //     id,
                    //     name: `Random Point #${id}`,
                    //     description: `description for Random Point #${id}`,
                    // },
                });
            }
        }
        console.log("Structured data: ", newFeaturesList)
        return {
            type: "FeatureCollection",
            features: newFeaturesList,
        }
    };



    updateData = (isCreated, updateRenderState, data, heatmap_data) => {
        // console.log("isCreated: ", isCreated)
        // DATAPOINTS
        // if (isCreated) {
        //     const map = this.reactMap.getMap();
        //     map.getSource("points").setData(data);
        //     console.log("Setting new data: ", data)
        //     // this.setState(
        //     //     {isCreated: false}
        //     // )
        // }
        // SEGMENTS
        if (isCreated === true) {
            if (data) {
                // Retrieve Mapbox map object
                const map = this.reactMap.getMap();
                // Generate GeoJSON data so we can draw elements on the mapbox map
                var new_data = this.structureData(data);

                map.getSource("segments").setData(new_data);

                map.getSource("convex_hull").setData(heatmap_data);

                updateRenderState(false)
            }

            // map.on("click", "segments", function (e) {
            //     new mapboxgl.Popup()
            //     .setLngLat(e.lngLat)
            //     .setHTML(e.features[0].properties.name)
            //     .addTo(map);
            //     });
        }
    }

    // Dynamically change the width of the mapbox map whenever
    // the browser window is resized
    onViewportChange = viewport => {
        const { width, height, ...etc } = viewport
        this.setState({ viewport: etc })
    }

    render() {
        const { viewport, isCreated } = this.state;
        const { datapoints, segments, heatmap_data, updateRenderState, reRender } = this.props

        if (reRender === true) {
            this.updateData(isCreated, updateRenderState, segments, heatmap_data);
        }

        return (
            <MapGL
            ref={ (reactMap) => this.reactMap = reactMap }
            width="100wh"
            height="75vh"
            { ...viewport }
            onViewportChange={ viewport => this.onViewportChange(viewport) }
            mapStyle={ MAPBOX_STYLE }
            mapboxApiAccessToken={ MAPBOX_TOKEN }
            >

            <div className="nav" style={ navStyle }>
            <div className="control" style={ controlStyle }>
                    <FullscreenControl />
                </div>
                <div className="control" style={ controlStyle }>
                    <NavigationControl />
                </div>

                {/* { markerList.map( (marker, index) => {
                    return (
                        <div key={index} >
                            <Marker  longitude={marker.long} latitude={marker.lat}>
                                <Icon name="hospital" size="big" onMouseEnter={ () => this.setState({popupInfo: true}) } onMouseLeave={ () => this.setState({popupInfo: null}) }/>
                            </Marker> { this.renderPopup(index) }
                        </div>
                    );
                }) } */}

            </div>

            </MapGL>
        );
    }
}
