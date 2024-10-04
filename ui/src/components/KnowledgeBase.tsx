import React, {useState, useEffect, useCallback} from 'react';
import {useAppContext} from '../contexts/AppContext';
import {getKnowledgeBaseDS, syncDataSource, checkDSStatus, postKnowledgeCachePurge} from '../services/api';
import RefreshButton from "./RefreshButton";

interface KBDataSource {
    dataSourceId: string;
    name: string;
    status: string;
    startedAt: string;
}

interface Ingestion {
    ingestionJob: IngestionJob;
}

interface IngestionJob {
    dataSourceId: string;
    ingestionJobId: string;
    knowledgeBaseId: string;
    startedAt: string;
    status: string;
    updatedAt: string;
}

const KnowledgeBase = () => {
    const {state, setState, contactId} = useAppContext();
    const [isLoading, setIsLoading] = useState(false);
    const [isPurging, setIsPurging] = useState(false);

    const fetchDataSources = useCallback(async () => {
        setIsLoading(true);
        try {
            const data = await getKnowledgeBaseDS();
            setState(prevState => ({
                ...prevState,
                knowledgeBase: data.dataSourceSummaries
            }));
        } catch (error) {
            console.error('Error fetching knowledge base sources:', error);
        } finally {
            setIsLoading(false);
        }
    }, [setState]);

    useEffect(() => {
        setIsPurging(false)

        if (state.knowledgeBase.length === 0) {
            fetchDataSources();
        }
    }, [state.knowledgeBase.length, fetchDataSources, contactId]);

    const handleRefresh = () => {
        fetchDataSources();
    };


    const handleSync = async (dataSourceId: string) => {
        try {
            const ingestion: Ingestion = await syncDataSource(dataSourceId);
            setState(prevState => ({
                ...prevState,
                knowledgeBase: prevState.knowledgeBase.map(ds =>
                    ds.dataSourceId === dataSourceId ? {...ds, status: 'SYNCING', ingestion} : ds
                )
            }));

            const checkStatus = async () => {
                const ingestionLoop: Ingestion = await checkDSStatus(dataSourceId, ingestion.ingestionJob.ingestionJobId);
                setState(prevState => ({
                    ...prevState,
                    knowledgeBase: prevState.knowledgeBase.map(ds =>
                        ds.dataSourceId === dataSourceId ? {...ds, status: ingestionLoop.ingestionJob.status} : ds
                    )
                }));

                if (ingestionLoop.ingestionJob.status !== 'COMPLETE' && ingestionLoop.ingestionJob.status !== 'FAILED') {
                    setTimeout(checkStatus, 2000);
                }
            };

            setTimeout(checkStatus, 2000);
        } catch (error) {
            console.error('Error syncing knowledge base:', error);
        }
    };
    const handlePurge = async () => {
        try {
            setIsPurging(true)
            console.log("Purge start.")
            const ingestion: Ingestion = await postKnowledgeCachePurge();
            console.log("Purged end - " + ingestion)
        } catch (error) {
            console.error('Error syncing knowledge base:', error);
        }
        setIsPurging(false)
    };

    return (
        <div className="knowledge-base">
            <div className="header-container">
                <h2>
                    Knowledge Base
                    <RefreshButton onRefresh={handleRefresh}/>
                </h2>
                <span className="header-desc">{contactId ? '(ContactId:' + contactId + ')' : ''}</span>
            </div>
            {isLoading ? (
                <p className="loading">
                    <div className="loading-spinner"></div>
                </p>
            ) : (
                <table className="kb-table">
                    <thead>
                    <tr>
                        <th>Name</th>
                        <th>Status</th>
                        <th>Action</th>
                    </tr>
                    </thead>
                    <tbody>
                    {state.knowledgeBase.map((source: KBDataSource) => (
                        <tr key={source.dataSourceId}>
                            <td>{source.name}</td>
                            <td>
                                <span className={`status-badge ${source.status.toLowerCase()}`}>
                                    {source.status}
                                </span>
                            </td>
                            <td>
                                <button
                                    className="sync-button"
                                    onClick={() => handleSync(source.dataSourceId)}
                                    disabled={source.status !== 'AVAILABLE' && source.status !== 'COMPLETE'}
                                >
                                    {source.status === 'AVAILABLE' ? 'Sync' : 'In Progress'}
                                </button>
                            </td>
                            <td>
                                <button
                                    className="sync-button"
                                    onClick={() => handlePurge()}
                                    disabled={isPurging}
                                >
                                    {isPurging ? 'Purging...' : 'Cache Purge'}
                                </button>
                            </td>
                        </tr>
                    ))}
                    </tbody>
                </table>
            )}
        </div>
    );
};

export default KnowledgeBase;
