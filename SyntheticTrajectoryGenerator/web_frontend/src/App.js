// NODE MODULES:
import React, { Component } from "react";
import { Content } from "carbon-components-react";
import { Route, Switch } from "react-router-dom";
// LOCAL IMPORTS:
// -> Components:
import Main from "./components/Main";
import MainHeader from './components/MainHeader';


class App extends Component {
    render() {
        return (
        <>
        <MainHeader />
            <Content>
                <Switch>
                    <Route exact path="/" component={Main} />
                </Switch>
            </Content>
        </>
        );
    }
}

export default App;
