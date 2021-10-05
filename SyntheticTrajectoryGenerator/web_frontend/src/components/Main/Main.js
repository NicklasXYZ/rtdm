// NODE MODULES:
import React, { Component } from "react";
import { LineChart } from "@carbon/charts-react";
import { Button } from "carbon-components-react";
import "carbon-components/scss/globals/scss/styles.scss";
import "@carbon/charts/styles.css";
// LOCAL IMPORTS:
// -> Components
import Map from "../../Map";
import WebSocketConnection from "../WebSocketConnection";
import CustomDataTable from "../CustomDataTable";
// import Dashboard from "../Dashboard";
// -> Constants
import { WEBSOCKET_ADDRESS } from "../Constants/constants"
import { TABLE_SORT_DIRECTION } from "../../misc";


const sortInfo = {
    columnId: "name",
    direction: TABLE_SORT_DIRECTION.ASCENDING,
};

const collectData = (segments, key) => {
    const newFeaturesList = [];
    for (let i = 0; i < segments.length; i++) {
        for (let j = 0; j < segments[i].segment.length; j++) {
            newFeaturesList.push({
                group: segments[i].meta.id,
                key: segments[i].segment[j]["regularized_data"]["timestamp"],
                value: segments[i].segment[j]["regularized_data"][key]
            });
        }
    }
    return {
        data: newFeaturesList,
        options: {
            "experimental": true,
            "title": "Line (discrete)",
            "axes": {
                "bottom": {
                    "title": "timestamp",
                    "mapsTo": "key",
                    "scaleType": "time",
                },
                "left": {
                    "mapsTo": "value",
                    "title": key,
                    "scaleType": "linear"
                }
            },
            "zoomBar": {
                "top": {
                    "enabled": true
                }
            },
            "height": "600px",
        }
    }
};


class Main extends Component {

    // state = {
    //     data: [
    //         {
    //             "group": "Dataset 1",
    //             "key": "Qty",
    //             "value": 34200
    //         },
    //         {
    //             "group": "Dataset 1",
    //             "key": "More",
    //             "value": 23500
    //         },
    //         {
    //             "group": "Dataset 1",
    //             "key": "Sold",
    //             "value": 53100
    //         },
    //         {
    //             "group": "Dataset 1",
    //             "key": "Restocking",
    //             "value": 42300
    //         },
    //         {
    //             "group": "Dataset 1",
    //             "key": "Misc",
    //             "value": 12300
    //         },
    //         {
    //             "group": "Dataset 2",
    //             "key": "Qty",
    //             "value": 34200
    //         },
    //         {
    //             "group": "Dataset 2",
    //             "key": "More",
    //             "value": 53200
    //         },
    //         {
    //             "group": "Dataset 2",
    //             "key": "Sold",
    //             "value": 42300
    //         },
    //         {
    //             "group": "Dataset 2",
    //             "key": "Restocking",
    //             "value": 21400
    //         },
    //         {
    //             "group": "Dataset 2",
    //             "key": "Misc",
    //             "value": 0
    //         },
    //         {
    //             "group": "Dataset 3",
    //             "key": "Qty",
    //             "value": 41200
    //         },
    //         {
    //             "group": "Dataset 3",
    //             "key": "More",
    //             "value": 18400
    //         },
    //         {
    //             "group": "Dataset 3",
    //             "key": "Sold",
    //             "value": 34210
    //         },
    //         {
    //             "group": "Dataset 3",
    //             "key": "Restocking",
    //             "value": 1400
    //         },
    //         {
    //             "group": "Dataset 3",
    //             "key": "Misc",
    //             "value": 42100
    //         },
    //         {
    //             "group": "Dataset 4",
    //             "key": "Qty",
    //             "value": 22000
    //         },
    //         {
    //             "group": "Dataset 4",
    //             "key": "More",
    //             "value": 1200
    //         },
    //         {
    //             "group": "Dataset 4",
    //             "key": "Sold",
    //             "value": 9000
    //         },
    //         {
    //             "group": "Dataset 4",
    //             "key": "Restocking",
    //             "value": 24000,
    //             "audienceSize": 10
    //         },
    //         {
    //             "group": "Dataset 4",
    //             "key": "Misc",
    //             "value": 3000,
    //             "audienceSize": 10
    //         }
    //     ],
    //     options: {
    //         "title": "Line (discrete)",
    //         "axes": {
    //             "bottom": {
    //                 "title": "2019 Annual Sales Figures",
    //                 "mapsTo": "key",
    //                 "scaleType": "labels"
    //             },
    //             "left": {
    //                 "mapsTo": "value",
    //                 "title": "Conversion rate",
    //                 "scaleType": "linear"
    //             }
    //         },
    //         "height": "400px"
    //     }
    // };

    constructor(props) {
        super(props);
        this.socket = new WebSocket(WEBSOCKET_ADDRESS)
        this.state = {
            datapoints: null,
            segments: null,
            heatmap_data: null,
            reRender: false,
        }
    }

    handleDataPoints = (data) => {
        this.setState({
            datapoints: data.data
        })
        console.log("State: ", this.state)
    }

    retrieveDataPoints = () => {
        const message = {
            type: "retrieve-datapoints",
            data: {
                id: "0123"
            },
        }
        // this.socket.send(
        //     JSON.stringify(message)
        // );
        this.setState({ // TMP
            datapoints: null,
        })
        console.log("Retrieving datapoints...")
    }

    updateRowState = (rowId, checked) => {
        const { segments } = this.state;
        // Adjust index. Row numbering starts from 1, but array index starts from 0.
        segments.rows[rowId - 1].selected = checked;
        console.log("row: ", segments.rows[rowId])
        console.log("row id", rowId, "checked", checked)
        this.setState({
            segments: segments,
            reRender: true,
        })
    }

    updateAllRowState = (checked) => {
        console.log(checked);
        const { segments } = this.state;
        // Adjust index. Row numbering starts from 1, but index starts from 0.
        for (let index = 0; index < segments.rows.length; index++) {
            segments.rows[index].selected = checked;

        }
        this.setState({
            segments: segments,
            reRender: true,
        })
    }


    handleSegments = (data) => {
        const segment_data = this.getTableData(data.data)
        const heatmap_data = data.extra_data
        this.setState({
            // segments: data.data
            segments: segment_data,
            heatmap_data: heatmap_data,
        })
        console.log("State: ", this.state)
    }

    retrieveSegments = () => {
        const message = {
            type: "retrieve-segments",
            data: {
                id: "0123"
            },
        }
        this.socket.send(
            JSON.stringify(message)
        );
        console.log("Retrieving segements...")
    }

    getTableData = (segments) => {
        const headerList = [
            {
                title: "Identifier",
                key: "uuid",
                id: "uuid",
            },
            {
                title: "Segment length",
                key: "length",
                id: "length",
            },
            {
                title: "Minimum longitude",
                key: "min_lon",
                id: "min_lon",
            },
            {
                title: "Maximum longitude",
                key: "max_lon",
                id: "max_lon",
            },
            {
                title: "Minimum latitude",
                key: "min_lat",
                id: "min_lat",
            },
            {
                title: "Maximum latitude",
                key: "max_lat",
                id: "max_lat",
            },
        ];
        const rowList = [];
        for (let i = 0; i < segments.length; i++) {
            rowList.push({
                id: Number(i + 1),
                uuid: segments[i].meta.uuid,
                name: segments[i].meta.id,
                length: segments[i].segment.length,
                min_lon: segments[i].meta.min_lon,
                max_lon: segments[i].meta.max_lon,
                min_lat: segments[i].meta.min_lat,
                max_lat: segments[i].meta.max_lat,
                selected: false,
                segment: segments[i].segment,
                color: segments[i].meta.color,
            });
        }
        const tableData = {
            headers: headerList,
            rows: rowList,
        }
        console.log(tableData);
        this.setState(
            {
                reRender: true,
            }
        )
        return tableData;
    }

    updateRenderState = (reRender) => {
        this.setState(
            {
                reRender: reRender,
            }
        )
    }

    render() {
        const { datapoints, segments, heatmap_data, testDemoRows, testDemoColumns, reRender } = this.state;
        var rows = null;
        var headers = null;
        console.log("Render state", reRender)
        if (segments) {
            console.log("Segments ", segments)
            rows = segments.rows;
            headers = segments.headers;
        }

        return (
            <>
            <WebSocketConnection
            socket={this.socket}
            handleDataPoints={this.handleDataPoints}
            handleSegments={this.handleSegments}
            />
            <Map
            datapoints={ datapoints }
            segments={ segments }
            heatmap_data={heatmap_data}
            updateRenderState={ this.updateRenderState }
            reRender={reRender}
            />

            <Button kind="tertiary" onClick={ this.retrieveDataPoints }>
                Retrieve Data
            </Button>
            <Button kind="tertiary" onClick={ this.retrieveSegments }>
                Retrieve Segements
            </Button>

            {/* { rows && headers && <CustomDataTable updateRowState={this.updateRowState} updateAllRowState={this.updateAllRowState} columns={headers} rows={rows} sortInfo={demoSortInfo} hasSelection={true} pageSize={10} start={0}/> } */}

            { rows && headers && <CustomDataTable updateRowState={this.updateRowState} updateAllRowState={this.updateAllRowState} columns={headers} rows={rows} sortInfo={sortInfo} hasSelection={true} pageSize={10} start={0}/> }

            {/* { data && <LineChart data={data.data} options={data.options}></LineChart> } */}

            {/* <Dashboard socket={this.socket}/> */}
            </>
        );
    }
}

export default Main;
