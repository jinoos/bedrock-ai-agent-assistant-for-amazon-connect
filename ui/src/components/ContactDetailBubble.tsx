import React from 'react';

interface TranscriptMessage {
    message:
        {
            Id: string;
            Content: string;
            ParticipantId: string;
            AbsoluteTime: string;
        }
}

const ContactDetailBubble = ({message}: TranscriptMessage) => {
    const textLines = message.Content.split('\n');
    return (
        <div className={`message-bubble ${message.ParticipantId}`}>
            <p>{message.Content}</p>
            <span className={`message-timestamp ${message.ParticipantId}`}>{message.AbsoluteTime}</span>
        </div>
    );
};

export default ContactDetailBubble;