import React from 'react';
import ReactDOM from 'react-dom/client';
import {AmazonConnectApp} from "@amazon-connect/app";
import {ContactClient, ContactConnected, ContactConnectedHandler} from "@amazon-connect/contact";
import App from './App';
import {AppProvider} from './contexts/AppContext';
import './styles/main.css';

const root = ReactDOM.createRoot(
    document.getElementById('root') as HTMLElement
);

// let connectInstanceId = ""
// const {provider} = AmazonConnectApp.init({
//     onCreate: async (event) => {
//         console.log('onCreate event: ', event);
//         const {appInstanceId} = event.context;
//         console.log('appInstanceId: ', appInstanceId);
//         if (connectInstanceId == "") {
//             connectInstanceId = event.context.appConfig.id;
//             console.log('connectInstanceId: ', connectInstanceId);
//         }
//     },
//     onDestroy: async (event) => {
//         connectInstanceId = ""
//         console.log('App being destroyed');
//     },
// });

root.render(
    <React.StrictMode>
        <AppProvider>
            <App/>
        </AppProvider>
    </React.StrictMode>
);