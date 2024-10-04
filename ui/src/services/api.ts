import axios from 'axios';
import {format} from "date-fns";
import {AgentClient} from "@amazon-connect/contact";
import {KB_API_BASE_URL} from "../Configure";


// const API_BASE_URL = 'https://b0srts9afl.execute-api.ap-northeast-2.amazonaws.com';
// const KB_API_BASE_URL = 'https://3riwt9alt0.execute-api.ap-northeast-2.amazonaws.com/prod';

export const postChatbotQuery = async (contactId: string, query: string) => {
    const url = `${KB_API_BASE_URL}/chatbot/query`;
    const response = await axios.post(`${KB_API_BASE_URL}/chatbot/query`, {'query': query, 'contactId': contactId});
    return response.data;
};

export const postGetSummary = async (contactId: string, query: string) => {
    const url = `${KB_API_BASE_URL}/llm-call/summary/` + contactId;
    const response = await axios.post(url, {'query': query});
    return response.data;
};

export const postPurgeSummaryDb = async (contactId: string) => {
    const url = `${KB_API_BASE_URL}/llm-call/summary/` + contactId + '/purge';
    const response = await axios.post(url);
    return response.data;
};

export const postExportSummaryDbToS3 = async (contactId: string) => {
    const url = `${KB_API_BASE_URL}/llm-call/summary/` + contactId + '/export';
    const response = await axios.post(url);
    return response.data;
};


export const getContactHistory = async () => {
    const url = `${KB_API_BASE_URL}/contact-history`;
    const response = await axios.get(`${KB_API_BASE_URL}/contact-history`);
    console.log(response.data);
    return response.data;
};


export const getContactDetail = async (contactId: string) => {
    const url = `${KB_API_BASE_URL}/contact-history/contact/${contactId}`;
    const response = await axios.get(`${KB_API_BASE_URL}/contact-history/contact/${contactId}`);
    console.log(response.data);
    return response.data;
};


export const getContactDetailAnalysis = async (contactId: string) => {
    const url = `${KB_API_BASE_URL}/contact-history/contact/${contactId}/analysis`;
    const response = await axios.get(`${KB_API_BASE_URL}/contact-history/contact/${contactId}/analysis`);
    console.log(response.data);
    return response.data;
};

export const getChatDetail = async (contactId: string) => {
    const url = `${KB_API_BASE_URL}/contact-history/${contactId}`;
    const response = await axios.get(`${KB_API_BASE_URL}/contact-history/${contactId}`);
    console.log(response.data);
    return response.data;
};

export const getKnowledgeBaseDS = async () => {
    const url = `${KB_API_BASE_URL}/knowledgebase/data_sources`;
    const response = await axios.get(`${KB_API_BASE_URL}/knowledgebase/data_sources`);
    console.log(response.data);
    return response.data;
};

export const syncDataSource = async (dataSourceId: string) => {
    const url = `${KB_API_BASE_URL}/knowledgebase/data_sources/${dataSourceId}/ingestion_job/start`;
    const response = await axios.post(`${KB_API_BASE_URL}/knowledgebase/data_sources/${dataSourceId}/ingestion_job/start`);
    console.log(response.data);
    return response.data;
};

export const checkDSStatus = async (dataSourceId: string, ingestionJobId: string) => {
    const url = `${KB_API_BASE_URL}/knowledgebase/data_sources/${dataSourceId}/ingestion_job/${ingestionJobId}/stats`;
    const response = await axios.get(`${KB_API_BASE_URL}/knowledgebase/data_sources/${dataSourceId}/ingestion_job/${ingestionJobId}/stats`);
    console.log(response.data);
    return response.data;
};

export const getLastDSStatus = async (dataSourceId: string, ingestionJobId: string) => {
    const url = `${KB_API_BASE_URL}/knowledgebase/data_sources/${dataSourceId}/ingestion_job/${ingestionJobId}/stats`;
    const response = await axios.get(`${KB_API_BASE_URL}/knowledgebase/data_sources/${dataSourceId}/ingestion_job/${ingestionJobId}/stats`);
    console.log(response.data);
    return response.data;
};


export const postKnowledgeCachePurge = async () => {
    const url = `${KB_API_BASE_URL}/knowledgebase/caches/purge`;
    const response = await axios.post(url);
    console.log(response.data);
    return response.data;
};

export const formatTimestamp = (timestamp: Date): string => {
    return format(new Date(timestamp), 'yyyy-MM-dd HH:mm:ss');
};

export const acAgentGetARN = async () => {
    const client = new AgentClient();
    return client.getARN()
}
