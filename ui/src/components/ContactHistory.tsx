import React, {useState, useEffect, useCallback} from 'react';
import {Link} from 'react-router-dom';
import {formatTimestamp, getContactHistory} from '../services/api';
import RefreshButton from "./RefreshButton";
import {useAppContext} from '../contexts/AppContext';

interface Contact {
    Arn: string;
    Id: string;
    InitialContactId: string;
    PreviousContactId: string;
    InitiationMethod: 'INBOUND' | 'OUTBOUND' | 'TRANSFER' | 'QUEUE_TRANSFER' | 'CALLBACK' | 'API' | 'DISCONNECT' | 'MONITOR' | 'EXTERNAL_OUTBOUND';
    Channel: 'VOICE' | 'CHAT' | 'TASK';
    QueueInfo: {
        Id: string;
        EnqueueTimestamp: Date;
    };
    AgentInfo: {
        Id: string;
        ConnectedToAgentTimestamp: Date;
    };
    InitiationTimestamp: Date;
    DisconnectTimestamp: Date;
    ScheduledTimestamp: Date;
}

export interface ContactListResponse {
    Contacts: Contact[];
    NextToken: string;
    TotalCount: number;
}

const ContactHistory = () => {
    const {state, setState, contactId} = useAppContext();
    const [isLoading, setIsLoading] = useState(false);

    const fetchContacts = useCallback(async () => {
        setIsLoading(true);
        try {
            const data = await getContactHistory();
            setState(prevState => ({
                ...prevState,
                contactHistory: data
            }));
        } catch (error) {
            console.error('Error fetching contact history:', error);
        } finally {
            setIsLoading(false);
        }
    }, [setState]);

    useEffect(() => {
        if (!state.contactHistory || state.contactHistory.Contacts.length === 0) {
            fetchContacts();
        }
    }, [state.contactHistory, fetchContacts, contactId]);

    const handleRefresh = () => {
        fetchContacts();
    };

    return (
        <div className="contact-history">
            <div className="header-container">
                <h2>
                    Contact History
                    <RefreshButton onRefresh={handleRefresh}/>
                </h2>
                <span className="header-desc">{contactId ? '(ContactId: ' + contactId + ')' : ''}</span>
            </div>
            {isLoading ? (
                <p className="loading">
                    <div className="loading-spinner"></div>
                </p>
            ) : state.contactHistory && state.contactHistory.Contacts.length > 0 ? (
                <table className="contact-table">
                    <thead>
                    <tr>
                        <th>Contact ID</th>
                        <th>Channel</th>
                        <th>Initiation Time</th>
                        <th>Action</th>
                    </tr>
                    </thead>
                    <tbody>
                    {state.contactHistory.Contacts.map(contact => (
                        <tr key={contact.Id}>
                            <td>
                                <Link to={`/contact-history/contact/${contact.Id}`} className="review-link">
                                    {contact.Id}
                                </Link>
                            </td>
                            <td>{contact.Channel}</td>
                            <td>{formatTimestamp(new Date(contact.InitiationTimestamp))}</td>
                            <td>
                                <Link to={`/contact-history/contact/${contact.Id}`} className="review-link">
                                    Review Details
                                </Link>
                            </td>
                        </tr>
                    ))}
                    </tbody>
                </table>
            ) : (
                <p>No contacts found.</p>
            )}
        </div>
    );
};

export default ContactHistory;