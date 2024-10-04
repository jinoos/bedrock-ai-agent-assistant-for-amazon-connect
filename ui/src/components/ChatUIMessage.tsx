export interface Message {
    id: string;
    text: string;
    sender: 'AGENT' | 'CUSTOMER' | 'AI';
    timestamp: string;
}