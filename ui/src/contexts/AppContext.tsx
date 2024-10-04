import React, {createContext, useState, useContext, useCallback} from 'react';
import {ContactListResponse} from '../components/ContactHistory';
import {ContactClient} from "@amazon-connect/contact";
import {AmazonConnectApp} from "@amazon-connect/app";

interface AppState {
    chatUI: any[];
    contactHistory: ContactListResponse | null;
    knowledgeBase: any[]
}

interface AppContextType {
    state: AppState;
    setState: React.Dispatch<React.SetStateAction<AppState>>;
    connectId: string;
    setConnectId: (id: string) => void;
    connectApp: AmazonConnectApp;
    setConnectApp: (app: AmazonConnectApp) => void;
    contactClient: ContactClient;
    setContactClient: (client: ContactClient) => void;
    contactId: string;
    setContactId: (id: string) => void;
    handleConnectEvent: (eventType: string, data: any) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

interface AppProviderProps {
    children: React.ReactNode;
}

export const AppProvider = ({children}: AppProviderProps) => {
    const [state, setState] = useState<AppState>({
        chatUI: [],
        contactHistory: null,
        knowledgeBase: [],
    });

    const [connectId, setConnectId] = useState<any>(null);
    const [connectApp, setConnectApp] = useState<any>(null);
    const [contactClient, setContactClient] = useState<ContactClient>(new ContactClient());
    const [contactId, setContactId] = useState<any>(null);
    const handleConnectEvent = useCallback((eventType: string, data: any) => {
        console.log(`Received ${eventType} event:`, data);
        // 여기에 이벤트 처리 로직 추가
    }, []);

    return (
        <AppContext.Provider value={{
            state,
            setState,
            connectId,
            setConnectId,
            connectApp,
            setConnectApp,
            contactClient,
            setContactClient,
            contactId,
            setContactId,
            handleConnectEvent
        }}>
            {children}
        </AppContext.Provider>
    );
};

export const useAppContext = () => {
    const context = useContext(AppContext);
    if (context === undefined) {
        throw new Error('useAppContext must be used within an AppProvider');
    }
    return context;
};