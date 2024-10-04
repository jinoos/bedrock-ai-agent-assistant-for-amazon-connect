import React, {useCallback, useEffect, useRef, useState} from "react";
import {useAppContext} from '../contexts/AppContext';
import MessageBubble from './ChatUIBubble';
import axios from "axios";
import {v4 as uuidv4} from 'uuid';
import {acAgentGetARN} from "../services/api";
import RefreshButton from "./RefreshButton";

export interface Message {
    id: string;
    text: string;
    sender: string;
    timestamp: string;
}

type ChatUIProps = Record<string, never>;

const ChatUI = ({}: ChatUIProps) => {
    const {state, setState, connectId, contactId} = useAppContext();
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<null | HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const fetchDataSources = useCallback(async () => {
        // Implement fetchDataSources if needed
    }, [setState]);

    useEffect(() => {
        scrollToBottomNow();
        if (textareaRef.current) {
            adjustTextareaHeight();
        }
        focusTextarea();
    }, [connectId, connectId, message, state.chatUI]);

    const adjustTextareaHeight = () => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = `${textarea.scrollHeight - 20}px`;
        }
    };

    function newAgentMessage(text: string) {
        return newMessage(text, 'AGENT');
    }

    function newAIMessage(text: string) {
        return newMessage(text, 'AI');
    }

    function newMessage(test: string, sender: 'AGENT' | 'CUSTOMER' | 'AI' | 'SYSTEM'): Message {
        return {
            id: uuidv4(),
            text: test,
            sender: sender,
            timestamp: new Date().toLocaleTimeString('ko-KR', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            })
        };
    }

    const handleSendMessage = async (text: string) => {
        if (text.trim() && !isLoading) {
            setIsLoading(true);
            let msg = newAgentMessage(text);
            appendMessage(msg);
            setMessage('');


            try {
                const url = 'https://b0srts9afl.execute-api.ap-northeast-2.amazonaws.com/chatbot/query';
                const response = await axios.post(
                    url,
                    {query: text, contactId: contactId},
                    {
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    }
                );
                console.log('AI Response', response.data);
                let msgAI = newAIMessage(response.data.response)
                appendMessage(msgAI);
                // Handle AI response if needed
            } catch (error) {
                console.error('Error sending message:', error);
            } finally {
                setIsLoading(false);
                focusTextarea();
            }
        }
    };

    const appendMessage = (message: Message) => {
        setState(prevState => ({
            ...prevState,
            chatUI: [...prevState.chatUI, message]
        }));
        scrollToBottom();
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({behavior: "smooth"});
    };

    const scrollToBottomNow = () => {
        messagesEndRef.current?.scrollIntoView();
    };

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        handleSendMessage(message);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage(message);
        }
    };
    const focusTextarea = () => {
        if (textareaRef.current) {
            textareaRef.current.focus();
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setMessage(e.target.value);
    };

    return (
        <div className="chatUI">
            <div className="header-container">
                <h2>
                    Agent Assistant
                    <RefreshButton onRefresh={fetchDataSources}/>
                </h2>
                <span className="header-desc">{contactId ? '(ContactId: ' + contactId + ')' : ''}</span>
            </div>
            <div className="chat-ui">
                <div className="chat-messages">
                    {state.chatUI.map((message) => (
                        <MessageBubble key={message.id} message={message}/>
                    ))}
                    <div ref={messagesEndRef}/>
                </div>
                <form className="chat-input" onSubmit={handleSubmit}>
                    <textarea
                        disabled={isLoading}
                        ref={textareaRef}
                        value={message}
                        onChange={handleChange}
                        onKeyUp={handleKeyDown}
                        placeholder="Type a message"
                        rows={1}
                    />
                    <button type="submit" disabled={isLoading || !message.trim()}>
                        {isLoading ? 'AI thinking...' : 'Send'}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatUI;