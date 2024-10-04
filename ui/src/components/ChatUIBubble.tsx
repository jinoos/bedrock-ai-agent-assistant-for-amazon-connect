import React from 'react';

interface MessageProps {
  message: {
    id: string;
    text: string;
    sender: string;
    timestamp: string;
  };
}

const MessageBubble = ({ message }: MessageProps) => {
  const textLines = message.text.split('\n');
  return (
    <div className={`message-bubble ${message.sender}`}>
      <p>{message.text}</p>
      <span  className={`message-timestamp ${message.sender}`}>{message.timestamp}</span>
    </div>
  );
};

export default MessageBubble;