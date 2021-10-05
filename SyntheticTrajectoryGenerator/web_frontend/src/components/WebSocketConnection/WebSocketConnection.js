import React, { Component } from 'react';
import { RETRIEVE_DATAPOINTS, RETURN_DATAPOINTS, RETURN_SEGMENTS } from '../Constants/constants';

class WebSocketConnection extends Component {

    constructor(props) {
        super(props);
    }

    waitForSocketConnection = (socket) => {
        setTimeout(() => {
            if (socket.readyState === 1) {
                console.log("Connection is made")
                // const message = {
                //     type: "test-message",
                //     data: {
                //         id: "0123"
                //     },
                // }
                // socket.send(
                //     JSON.stringify(message)
                // );
            } else {
                console.log("Wait for a connection to be established...")
                this.waitForSocketConnection(socket);
            }
        }, 500);
    }

    setupConnection = () => {
        const {
            socket,
            handleDataPoints,
            handleSegments,
        } = this.props;

        socket.onopen = () => {
            console.log("Websocket connection established!")
        }

        socket.onmessage = (event) => {
            console.log("Message: ", event);
            const message = JSON.parse(event.data)
            // this.setState({dataFromServer: message})
            switch (message.type) {
                case RETURN_DATAPOINTS:
                  handleDataPoints(message);
                  break;
                case RETURN_SEGMENTS:
                  handleSegments(message);
                  break;
                default:
                  console.log("Recieving message failed");
            }
        }

        socket.onclose = () => {
            console.log("Websocket connection closed!")
        }

        socket.onclose = (event) => {
          console.log("Websocket closed: ", event);
        }

        socket.onerror = (error) => {
          console.error("Websocket error: ", error);
        }

        this.waitForSocketConnection(socket);
    }

    componentDidMount() {
        this.setupConnection();
    }

    render() {
        return (
            <></>
        );
    }
}

export default WebSocketConnection;
