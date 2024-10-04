import React, {useState, useEffect, useRef} from 'react';
import {useParams, Link} from 'react-router-dom';
import {
    formatTimestamp,
    getContactDetail,
    getContactDetailAnalysis, postExportSummaryDbToS3,
    postGetSummary,
    postPurgeSummaryDb
} from '../services/api';
import {CallTranscript, Participant, TranscriptItem} from "./ContactDetailChat";
import ReactMarkdown from "react-markdown";
import MessageBubble from "./ChatUIBubble";
import ContactDetailBubble from "./ContactDetailBubble";
import ChatUIBubble from "./ChatUIBubble";
import {Message} from "./ChatUI";
import {Simulate} from "react-dom/test-utils";
import load = Simulate.load;
import axios from "axios";
import Markdown from "react-markdown"; // API 함수를 구현해야 합니다
// import '../styles/ContactDetail.css'; // CSS 파일을 생성해야 합니다

interface HierarchyGroup {
    Arn?: string;
}

interface DeviceInfo {
    PlatformName?: string;
    PlatformVersion?: string;
    OperatingSystem?: string;
}

interface Capabilities {
    Video?: 'SEND';
}

interface QueueInfo {
    Id?: string;
    EnqueueTimestamp?: Date;
}

interface AgentInfo {
    Id?: string;
    ConnectedToAgentTimestamp?: Date;
    AgentPauseDurationInSeconds?: number;
    HierarchyGroups?: {
        Level1?: HierarchyGroup;
        Level2?: HierarchyGroup;
        Level3?: HierarchyGroup;
        Level4?: HierarchyGroup;
        Level5?: HierarchyGroup;
    };
    DeviceInfo?: DeviceInfo;
    Capabilities?: Capabilities;
}

interface WisdomInfo {
    SessionArn?: string;
}

interface Expiry {
    DurationInSeconds?: number;
    ExpiryTimestamp?: Date;
}

interface MatchCriteria {
    AgentsCriteria?: {
        AgentIds?: string[];
    };
}

interface AttributeCondition {
    Name?: string;
    Value?: string;
    ProficiencyLevel?: any; // You might want to define a more specific type here
    MatchCriteria?: MatchCriteria;
    ComparisonOperator?: string;
}

interface Expression {
    AttributeCondition?: AttributeCondition;
    AndExpression?: Expression[];
    OrExpression?: Expression[];
}

interface RoutingStep {
    Expiry?: Expiry;
    Expression?: Expression;
    Status?: 'ACTIVE' | 'INACTIVE' | 'JOINED' | 'EXPIRED';
}

interface RoutingCriteria {
    Steps?: RoutingStep[];
    ActivationTimestamp?: Date;
    Index?: number;
}

interface Customer {
    DeviceInfo?: DeviceInfo;
    Capabilities?: Capabilities;
}

interface Campaign {
    CampaignId?: string;
}

interface CustomerVoiceActivity {
    GreetingStartTimestamp?: Date;
    GreetingEndTimestamp?: Date;
}

interface AudioQualityMetrics {
    QualityScore?: any; // You might want to define a more specific type here
    PotentialQualityIssues?: string[];
}

interface QualityMetrics {
    Agent?: {
        Audio?: AudioQualityMetrics;
    };
    Customer?: {
        Audio?: AudioQualityMetrics;
    };
}

interface DisconnectDetails {
    PotentialDisconnectIssue?: string;
}

interface SegmentAttribute {
    ValueString?: string;
}

interface Contact {
    Arn: string;
    Id: string;
    InitialContactId?: string;
    PreviousContactId?: string;
    InitiationMethod?: 'INBOUND' | 'OUTBOUND' | 'TRANSFER' | 'QUEUE_TRANSFER' | 'CALLBACK' | 'API' | 'DISCONNECT' | 'MONITOR' | 'EXTERNAL_OUTBOUND';
    Name?: string;
    Description?: string;
    Channel?: 'VOICE' | 'CHAT' | 'TASK';
    QueueInfo?: QueueInfo;
    AgentInfo?: AgentInfo;
    InitiationTimestamp?: Date;
    DisconnectTimestamp?: Date;
    LastUpdateTimestamp?: Date;
    LastPausedTimestamp?: Date;
    LastResumedTimestamp?: Date;
    TotalPauseCount?: number;
    TotalPauseDurationInSeconds?: number;
    ScheduledTimestamp?: Date;
    RelatedContactId?: string;
    WisdomInfo?: WisdomInfo;
    QueueTimeAdjustmentSeconds?: number;
    QueuePriority?: number;
    Tags?: { [key: string]: string };
    ConnectedToSystemTimestamp?: Date;
    RoutingCriteria?: RoutingCriteria;
    Customer?: Customer;
    Campaign?: Campaign;
    AnsweringMachineDetectionStatus?: 'ANSWERED' | 'UNDETECTED' | 'ERROR' | 'HUMAN_ANSWERED' | 'SIT_TONE_DETECTED' | 'SIT_TONE_BUSY' | 'SIT_TONE_INVALID_NUMBER' | 'FAX_MACHINE_DETECTED' | 'VOICEMAIL_BEEP' | 'VOICEMAIL_NO_BEEP' | 'AMD_UNRESOLVED' | 'AMD_UNANSWERED' | 'AMD_ERROR' | 'AMD_NOT_APPLICABLE';
    CustomerVoiceActivity?: CustomerVoiceActivity;
    QualityMetrics?: QualityMetrics;
    DisconnectDetails?: DisconnectDetails;
    SegmentAttributes?: { [key: string]: SegmentAttribute };
}

interface ContactResponse {
    Contact: Contact;
}

interface PopupContent {
    title: string;
    content: string;
}


// props가 없으므로 빈 객체 타입을 사용합니다
type ContactDetailProps = Record<string, never>;

// const ContactDetail: React.FC = () => {
const ContactDetail = ({}: ContactDetailProps) => {
    const {id} = useParams<{ id: string }>();
    const [contact, setContact] = useState<ContactResponse | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const [contactAnalysis, setContactAnalysis] = useState<CallTranscript | null>(null);
    const [loadingAnalysis, setLoadingAnalysis] = useState<boolean>(true);
    const [errorAnalysis, setErrorAnalysis] = useState<string | null>(null);

    const [loadingRunbook, setLoadingRunbook] = useState(false);
    const [resultRunbook, setResultRunbook] = useState<string | null>(null);
    const messagesEndRef = useRef<null | HTMLDivElement>(null);


    useEffect(() => {
        const fetchContactDetail = async () => {
            setLoading(false);
            setLoadingAnalysis(false);
            try {
                setLoading(true);
                const data = await getContactDetail(id!);
                setContact(data);
                console.log("Success load Contact Data ========================");
                console.log(data);

                try {
                    setLoadingAnalysis(true);
                    const dataAnalysis = await getContactDetailAnalysis(id!);
                    if (dataAnalysis == null)
                        throw new Error("Failed to fetch Analysis data");

                    console.log("Analysis Data ========================");
                    console.log(dataAnalysis);
                    setContactAnalysis(dataAnalysis);
                } catch (err) {
                    setErrorAnalysis('Failed to fetch Analysis data');
                    // console.error(err);
                } finally {
                    setLoadingAnalysis(false);
                }
            } catch (err) {
                setError('Failed to fetch contact details');
                // console.error(err);
            } finally {
                setLoading(false);
            }

        };
        fetchContactDetail();
    }, [id]);

    if (loading) return <div><div className="loading-spinner"></div></div>;
    if (error) return <div>Error: {error}</div>;
    if (!contact) return <div>No contact found</div>;

    const convertTranscriptItemToMessageProp = (item: TranscriptItem): Message => {
        return {
            id: item.Id,
            text: item.Content,
            sender: item.ParticipantId,
            timestamp: (new Date(item.AbsoluteTime)).toLocaleTimeString('ko-KR', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            })
        };
    }

    const handleButtonClick = async () => {
        setLoadingRunbook(true);
        scrollToBottomNow()
        try {
            const response = await postGetSummary(contact.Contact.Id, JSON.stringify(contactAnalysis.Transcript))
            console.log("ContactId: " + contact.Contact.Id);
            setResultRunbook(response);
        } catch (error) {
            console.error('Error fetching data:', error);
            setResultRunbook('Error occurred while fetching data');
        } finally {
            setLoadingRunbook(false);
        }
    };

    const handlePurgeButtonClick = async () => {
        setLoadingRunbook(true);
        scrollToBottomNow()
        try {
            const response = await postPurgeSummaryDb(contact.Contact.Id)
            console.log("ContactId : " + contact.Contact.Id);
        } catch (error) {
            console.error('Error fetching data:', error);
            setResultRunbook('Error occurred while fetching data');
        } finally {
            setResultRunbook(null)
            setLoadingRunbook(false);
        }
    };

    const handleExportButtonClick = async () => {
        console.log("ContactId: " + contact.Contact.Id);
        const response = await postExportSummaryDbToS3(contact.Contact.Id)
        alert("Call Summary has beed deployed to S3.\nPlease resync your Knowledge Base.\n\n - " + response)
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({behavior: "smooth"});
    };

    const scrollToBottomNow = () => {
        messagesEndRef.current?.scrollIntoView();
    };

    return (
        <div className="contact-detail">
            <h2>
                Contact Detail
            </h2>
            <Link to="/contact-history" className="back-link">Back to Contact History</Link>
            <div className="contact-info">
                <table className="contact-table">
                    <tbody>
                    <tr>
                        <td>
                            <strong>ID:</strong>
                            {contact.Contact.Id}
                        </td>
                        <td>
                            <strong>Channel:</strong> {contact.Contact.Channel}
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Initiation
                                Timestamp:</strong> {formatTimestamp(new Date(contact.Contact.InitiationTimestamp))}
                        </td>
                        <td>
                            <strong>Disconnect
                                Timestamp:</strong> {formatTimestamp(new Date(contact.Contact.DisconnectTimestamp))}
                        </td>
                    </tr>
                    {
                        contact.Contact.AgentInfo ? (
                            <tr>
                                <td>
                                    <strong>Agent ID:</strong> {contact.Contact.AgentInfo.Id}
                                </td>
                                <td>
                                    <strong>Connected to
                                        Agent:</strong> {formatTimestamp(new Date(contact.Contact.AgentInfo.ConnectedToAgentTimestamp))}
                                </td>
                            </tr>
                        ) : (
                            <tr>
                                <td>
                                    <strong>Agent ID:</strong> No Agent Data
                                </td>
                                <td>
                                    <strong>Connected to
                                        Agent:</strong> No Agent Data
                                </td>
                            </tr>
                        )
                    }
                    </tbody>
                </table>

                {
                    loadingAnalysis ? (
                        <div className="loading">
                            <div className="loading-spinner"></div>
                        </div>
                    ) : errorAnalysis ? (
                        <div className="loading">No Analysis Data: {errorAnalysis}</div>
                    ) : (
                        <div className="agent-info">
                            <table className="contact-table">
                                <tbody>
                                <tr>
                                    <td>
                                        <strong>Participants Sentiment</strong>
                                        {contactAnalysis.Participants.map((p: Participant) => (
                                            <div>
                                                <strong>{p.ParticipantRole}:</strong> {contactAnalysis.Sentiment.OverallSentiment[p.ParticipantRole]}
                                            </div>
                                        ))}
                                    </td>
                                </tr>
                                {/*<tr>*/}
                                {/*    <td>*/}
                                {/*        {JSON.stringify(contactAnalysis)}*/}
                                {/*    </td>*/}
                                {/*</tr>*/}
                                </tbody>
                            </table>
                        </div>
                    )
                }
                {
                    loadingAnalysis ? (
                        <div className="loading">
                            <div className="loading-spinner"></div>
                        </div>
                    ) : errorAnalysis ? (
                        <div className="loading">No Analysis Data: {errorAnalysis}</div>
                    ) : (
                        <div className="chat-ui-after">
                            <div className="chat-messages">
                                {contactAnalysis.Transcript.map((trans) => (
                                    <ChatUIBubble key={trans.Id} message={convertTranscriptItemToMessageProp(trans)}/>
                                ))}
                            </div>
                        </div>
                    )
                }
                <div ref={messagesEndRef}/>
                <div className="button-container">
                    <button className="centered-button" onClick={handleButtonClick} disabled={loading}>
                        {loading ? 'Loading...' : 'AI Summarize'}
                    </button>
                    {resultRunbook && (
                        <button className="centered-button" onClick={handlePurgeButtonClick} disabled={loading}>
                            {loading ? 'Purging...' : 'Purge Summary Db'}
                        </button>
                    )}
                    {resultRunbook && (
                        <button className="centered-button" onClick={handleExportButtonClick}>
                            Export Summary to S3
                        </button>
                    )}
                </div>
                {loadingRunbook && <div className="loading-spinner"></div>}
                {resultRunbook && (
                    <div className="result-container">
                        <ReactMarkdown>{resultRunbook}</ReactMarkdown>
                    </div>

                )}
            </div>
        </div>
    );
};

export default ContactDetail;