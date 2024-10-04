import React, {useEffect} from 'react';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';
import Layout from './components/Layout';
import ContactHistory from './components/ContactHistory';
import ContactDetail from './components/ContactDetail';
import KnowledgeBase from './components/KnowledgeBase';
import {useAppContext} from './contexts/AppContext';
import ChatUI from "./components/ChatUI";
import {AmazonConnectApp} from "@amazon-connect/app";
import {
    ContactClient,
    ContactConnected,
    ContactConnectedHandler, ContactDestroyedHandler,
    ContactStartingAcwHandler
} from "@amazon-connect/contact";

const App = () => {
    const {
        setConnectId,
        setConnectApp,
        setContactClient,
        setContactId,
        handleConnectEvent
    } = useAppContext();

    useEffect(() => {
        const initializeConnect = async () => {
            try {
                const {provider} = AmazonConnectApp.init({
                    onCreate: async (event) => {
                        console.log('connectInstanceId: ', event.context.appConfig.id);
                        setConnectId(event.context.appConfig.id);
                    },
                    onDestroy: async (event) => {
                        console.log('App being destroyed');
                        setConnectId(null);
                        setContactClient(null);
                        setContactId(null);
                    },
                });

                if (!provider) {
                    console.error('Amazon Connect not found');
                    return;
                }

                setConnectApp(provider);

                const contactClient = new ContactClient({provider});
                contactClient.onConnected(contactConnectedHandler);
                contactClient.onStartingAcw(contactStartingAcwHandler);
                contactClient.onDestroyed(contactDestroyedHandler);
                setContactClient(contactClient);

                console.log('Amazon Connect initialized successfully');
            } catch (error) {
                console.error('Failed to initialize Amazon Connect:', error);
            }
        };

        const contactConnectedHandler: ContactConnectedHandler = async (data: ContactConnected) => {
            console.log(data.contactId);
            setContactId(data.contactId);
        };

        const contactStartingAcwHandler: ContactStartingAcwHandler = async (data: ContactConnected) => {
            console.log(data.contactId);
        };

        const contactDestroyedHandler: ContactDestroyedHandler = async (data: ContactConnected) => {
            console.log(data.contactId)
            setContactId(null);
        };

        initializeConnect();
    }, [setConnectId, setConnectApp, setContactClient, setContactId, handleConnectEvent]);

    return (
        <Router>
            <Layout>
                <Routes>
                    <Route path="/chatbot" element={<ChatUI/>}/>
                    <Route path="/contact-history" element={<ContactHistory/>}/>
                    <Route path="/contact-history/contact/:id" element={<ContactDetail/>}/>
                    <Route path="/knowledge-base" element={<KnowledgeBase/>}/>
                    <Route path="*" element={<ChatUI/>}/>
                </Routes>
            </Layout>
        </Router>
    );
};

export default App