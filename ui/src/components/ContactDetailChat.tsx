

export interface Category {
  MatchedCategories: string[];
  MatchedDetails: Record<string, unknown>;
}

export interface Participant {
  ParticipantId: string;
  ParticipantRole: "AGENT" | "CUSTOMER";
}

export interface Sentiment {
  OverallSentiment: Record<string, number>;
}

export interface TranscriptItem {
  Content: string;
  ParticipantId: string;
  Id: string;
  AbsoluteTime: string;
}

export interface CallTranscript {
  Categories: Category;
  Channel: string;
  LanguageCode: string;
  Participants: Participant[];
  Sentiment: Sentiment;
  Transcript: TranscriptItem[];
}