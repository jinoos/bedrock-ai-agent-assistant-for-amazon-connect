import json
import os
import uuid
from datetime import datetime, timedelta

import boto3
from aws_lambda_powertools import Tracer, Logger
from aws_lambda_powertools.event_handler import APIGatewayRestResolver, CORSConfig
from aws_lambda_powertools.logging import correlation_paths
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
from redis import Redis

from utils.ParameterStore import ParameterStore
from utils.ssm import parameter_store

from time import sleep


def get_param(key):
    ps = ParameterStore()
    return ps.get_param(key)


def get_param_(key, default):
    ps = ParameterStore()
    res = ParameterStore().get_param(key)
    if not res:
        return default
    return res


def custom_serializer(obj) -> str:
    return json.dumps(obj, default=str)


tracer = Tracer()
logger = Logger()
region = boto3.Session().region_name
cors_config = CORSConfig(allow_origin="*")
app = APIGatewayRestResolver(serializer=custom_serializer, cors=cors_config)


@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
@tracer.capture_lambda_handler
def lambda_handler(event, context) -> dict:
    print(f"Event: {event}")
    res = app.resolve(event, context)
    print(f"Res: {res}")
    return res


@app.get("/knowledgebase")
def main():
    return list_data_sources()


@app.get("/knowledgebase/data_sources")
def list_data_sources():
    knowledgeBaseId = get_knowledge_base_id()
    print(f"knowledgeBaseId: {knowledgeBaseId}")
    client = boto3.client('bedrock-agent', region_name=get_bedrock_region())
    res = client.list_data_sources(knowledgeBaseId=knowledgeBaseId)
    print(f"Res: {res}")
    return res


@app.get("/knowledgebase/data_sources/<dataSourceId>/ingestion_job/last")
def last_ingestion_job(dataSourceId: str):
    knowledgeBaseId = get_knowledge_base_id()
    client = boto3.client('bedrock-agent', region_name=get_bedrock_region())

    res = client.list_ingestion_jobs(
        knowledgeBaseId=knowledgeBaseId,
        dataSourceId=dataSourceId,
        sortBy={
            'attribute': 'STARTED_AT',
            'order': 'DESCENDING',
        }
    )
    return res['ingestionJobSummaries'][0]


@app.post("/knowledgebase/data_sources/<dataSourceId>/ingestion_job/start")
def start_ingestion_job(dataSourceId: str):
    knowledgeBaseId = get_knowledge_base_id()
    client = boto3.client('bedrock-agent', region_name=get_bedrock_region())
    res = client.start_ingestion_job(
        knowledgeBaseId=knowledgeBaseId,
        dataSourceId=dataSourceId
    )
    return res


@app.get("/knowledgebase/data_sources/<dataSourceId>/ingestion_job/<ingestionJobId>/stats")
def start_ingestion_job(dataSourceId: str, ingestionJobId: str):
    knowledgeBaseId = get_knowledge_base_id()
    client = boto3.client('bedrock-agent', region_name=get_bedrock_region())
    res = client.get_ingestion_job(
        knowledgeBaseId=knowledgeBaseId,
        dataSourceId=dataSourceId,
        ingestionJobId=ingestionJobId
    )
    return res


@app.post("/knowledgebase/caches/purge")
def purge_cache():
    print(f"Purge all cache")
    res = {'region': purge_region_cache(), 'semantic': purge_semantic_cache()}
    return res


@app.post("/knowledgebase/caches/region/purge")
def purge_region_cache():
    print(f"Purge region cache")
    redis = get_region_redis_client()
    res = redis.delete(get_region_cache_key())
    return res


@app.post("/knowledgebase/caches/semantic/purge")
def purge_semantic_cache():
    print(f"Purge opensearch indexes")
    credentials = boto3.Session(region_name=get_bedrock_region()).get_credentials()
    auth = AWSV4SignerAuth(credentials, get_bedrock_region(), service="aoss")
    osUri = get_semantic_cache_endpoint()
    print(f"Host: {host_from_uri(osUri)}")
    osClient = OpenSearch(
        hosts=[{"host": host_from_uri(osUri), "port": 443}],
        http_auth=auth,
        use_ssl=True,
        connection_class=RequestsHttpConnection,
    )
    res = osClient.indices.get_alias(index="*")
    print(f"Index list: {res}")
    deleteRes = []
    for k in list(res.keys()):
        if k.startswith("cache_"):
            # flush is not implemented on serverless
            # therefore, will delete+create instead
            # print(f"Flush index name: {k}")
            idx = osClient.indices.get(index=k)
            idxForCreate = cleanup_index_for_create(idx[k])
            osClient.indices.delete(index=k)
            sleep(1)
            res = osClient.indices.create(index=k, body=idxForCreate)
            deleteRes.append(res)

    print(f"Flush indices: {deleteRes}")
    return deleteRes


@app.post("/knowledgebase/llm/history/<historyKey>/contact_id/<contactId>")
def setContactIdToLlmHistory(historyKey: str, contactId: str):
    if historyKey == "":
        return {"error": "historyKey is required"}

    if contactId == "":
        return {"error": "contactId is required"}

    table = get_ddb_llm_history()
    ddbResource = boto3.resource('dynamodb')
    res = ddbResource.Table(table).get_item(
        Key={
            'Id': historyKey
        }
    )
    if not res or not res['Item']:
        return {"error": "historyKey or Item not found"}

    if res['Item']['ContactId'] == contactId:
        return {"error": "Same contactId is already set"}

    res['Item']['ContactId'] = contactId
    res['Item']['Id'] = str(uuid.uuid4())

    res = ddbResource.Table(table).put_item(Item=res)
    print(f"Insert LLM History: {res}")
    return res


def insert_llm_history(contactId: str, query: str, answer: str, instruction: str = "", context: str = "",
                       queryDate=None, answerDate=None):
    if contactId is None:
        contactId = 'None'
    if context is None:
        context = ''
    if instruction is None:
        instruction = ''

    if not queryDate:
        queryDate = datetime.utcnow()
    if not answerDate:
        answerDate = datetime.utcnow()

    ddbResource = boto3.resource('dynamodb')
    table = get_ddb_llm_history()
    lq = {
        'Id': str(uuid.uuid4()),
        'ContactId': contactId.strip(),
        'Query': query.strip(),
        'Answer': answer.strip(),
        'QueriedDate': queryDate.isoformat(),
        'AnsweredDate': answerDate.isoformat(),
        'Instruction': instruction.strip(),
        'Context': context.strip(),
    }
    print(f"Input: {lq}")
    res = ddbResource.Table(table).put_item(Item=lq)
    print(f"Insert LLM History: {res}")
    return res


def cleanup_index_for_create(idx: dict):
    del idx['settings']['index']['provided_name']
    del idx['settings']['index']['creation_date']
    del idx['settings']['index']['uuid']
    del idx['settings']['index']['version']
    return idx


def get_region_redis_client() -> Redis:
    return Redis(host=host_from_uri(get_region_cache_endpoint()),
                 port=port_from_uri(get_region_cache_endpoint()),
                 ssl=ssl_from_redis_uri(get_region_cache_endpoint()))


def get_region_cache_key():
    return os.getenv('region_cache_key', 'bedrock')


def get_region_cache_ssl():
    con_str = get_region_cache_endpoint()
    return ssl_from_redis_uri(con_str)


def host_from_uri(uri: str):
    return uri.split('://')[1].split(':')[0]


def port_from_uri(uri: str):
    print(f"URL: {uri}")
    return uri.split('://')[1].split(':')[1]


def protocol_from_uri(uri: str):
    return uri.split('://')[0]


def get_bedrock_region():
    return get_param('/aaa/bedrock_region')


def get_sematic_cache_score(event):
    return get_config(event, '/aaa/sematic_cache_score', 0.5)


def get_semantic_cache_endpoint():
    return get_param('/aaa/semantic_cache_endpoint')


def get_region_cache_endpoint():
    return get_param('/aaa/region_cache_endpoint')


def get_knowledge_base_id():
    return get_param('/aaa/knowledge-base-id')


def get_ddb_llm_history():
    return get_param('/aaa/dynamodb_llm_history')


def get_ddb_contact_summary():
    return get_param('/aaa/dynamodb_contact_summary')


def ssl_from_redis_uri(uri):
    if protocol_from_uri(uri).split('://')[0] == 'rediss':
        return True
    else:
        return False


def get_config(event, key, default=None):
    cache = event.get(key)
    cache_env = get_param(key)
    if cache:
        return cache
    if cache_env:
        return cache_env
    return default
