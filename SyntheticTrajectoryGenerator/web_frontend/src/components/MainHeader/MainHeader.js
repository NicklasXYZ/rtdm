import React from "react";
import { Link } from "react-router-dom";
import {
    Header,
    HeaderContainer,
    HeaderName,
    HeaderNavigation,
    HeaderMenuButton,
    HeaderMenuItem,
    SkipToContent,
    SideNav,
    SideNavItems,
    HeaderSideNavItems,
} from "carbon-components-react";


const MainHeader = () => (
    <HeaderContainer
    render={({ isSideNavExpanded, onClickSideNavExpand }) => (
        <Header aria-label="Header">
            <SkipToContent />
            <HeaderMenuButton
            aria-label="Open or close the menu"
            onClick={ onClickSideNavExpand }
            isActive={ isSideNavExpanded }
            />
            <HeaderName element={ Link } to="/" prefix="">
                Home
            </HeaderName>
            <HeaderNavigation aria-label="Header navigation">
                <HeaderMenuItem element={ Link } to="/info">
                    Information
                </HeaderMenuItem>
            </HeaderNavigation>
            <SideNav
            aria-label="Side menu navigation"
            expanded={ isSideNavExpanded }
            isPersistent={ false }>
                <SideNavItems>
                    <HeaderSideNavItems>
                        <HeaderMenuItem element={ Link } to="/info">
                            Information
                        </HeaderMenuItem>
                    </HeaderSideNavItems>
                </SideNavItems>
            </SideNav>
        </Header>
    )}
    />
);

export default MainHeader;
