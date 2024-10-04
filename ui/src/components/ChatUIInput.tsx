import React, {useState, KeyboardEvent, useRef, useEffect} from 'react';
import axios from "axios";

interface ChatInputProps {
    onSendMessage: (message: string) => void;
}

const ChatInput = ({onSendMessage}: ChatInputProps) => {
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        if (textareaRef.current) {
            adjustTextareaHeight();
        }
    }, [message]);

    const adjustTextareaHeight = () => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = `${textarea.scrollHeight - 20}px`;
        }
    };


    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (message.trim() && !isLoading) {
            setIsLoading(true);
            try {
                onSendMessage(message);
                setMessage('');
                console.log('message:', message);

                // Backend API 호출
                const url = 'https://b0srts9afl.execute-api.ap-northeast-2.amazonaws.com/chatbot/query';
                const response = await axios.post(
                    url,
                    {query: message},
                    {
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    }
                );
                // API 응답을 처리하고 싶다면 여기서 처리합니다.
                console.log('AI Response', response.data);
            } catch (error) {
                console.error('Error sending message:', error);
                // 오류가 발생해도 메시지를 보냅니다.
            } finally {
                setIsLoading(false);
            }
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e as unknown as React.FormEvent<HTMLFormElement>);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setMessage(e.target.value);
    };

    return (
        <form className="chat-input" onSubmit={handleSubmit}><textarea
            ref={textareaRef}
            value={message}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Type a message"
            rows={1}
        />
            <button type="submit" disabled={isLoading || !message.trim()}>{isLoading ? 'Sending...' : 'Send'}</button>
        </form>
    );
};

export default ChatInput;