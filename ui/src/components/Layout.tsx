import React, {useEffect, useState} from 'react';
import {Link} from 'react-router-dom';
import {useAppContext} from "../contexts/AppContext";

interface LayoutProps {
    children: React.ReactNode;
}

const Layout = ({children}: LayoutProps) => {
    const {connectId, contactId} = useAppContext();
    const [localConnectId, setLocalConnectId] = useState(connectId);
    const [localContactId, setLocalContactId] = useState(contactId);

    useEffect(() => {
        setLocalConnectId(connectId);
        setLocalContactId(contactId);
        console.log("connectId:", connectId, "contactId:", contactId);
    }, [connectId, contactId]);

    return (
        <div className="layout">
            <nav className="sidebar">
                <ul>
                    <li><Link to="/chatbot">Chatbot</Link></li>
                    <li><Link to="/contact-history">Contact History</Link></li>
                    <li><Link to="/knowledge-base">Knowledge Base</Link></li>
                    {/*<li><Link to="https://Amazon.com">Amazon.com</Link></li>*/}
                </ul>
            </nav>
            <main className="content">
                {children}
            </main>
        </div>
    );
};

export default Layout;
