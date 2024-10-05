from dataclasses import dataclass
import uuid
import datetime
from typing import Dict, Any, Optional, List

from aws_lambda_powertools import Tracer, Logger
from aws_lambda_powertools.event_handler import CORSConfig, APIGatewayRestResolver
from aws_lambda_powertools.logging import correlation_paths

import boto3
import json
import pickle
import hashlib

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from langchain_community.cache import OpenSearchSemanticCache
from opensearchpy import AWSV4SignerAsyncAuth
from redis import Redis

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.globals import set_llm_cache
from langchain_community.embeddings import BedrockEmbeddings

from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever

from utils.ParameterStore import ParameterStore


def custom_serializer(obj) -> str:
    return json.dumps(obj, default=str)


tracer = Tracer()
logger = Logger()
cors_config = CORSConfig(
    allow_origin="*"
)
app = APIGatewayRestResolver(serializer=custom_serializer, cors=cors_config)


@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
@tracer.capture_lambda_handler
def lambda_handler(event, context) -> dict:
    print(f"Event: {event}")
    if event['httpMethod'] == "OPTIONS":
        return {
            'headers': {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Method": "GET,POST,OPTIONS",
            },
            'statusCode': 200,
            'body': {}
        }

    res = app.resolve(event, context)
    if "headers" not in res:
        res["headers"] = {"Access-Control-Allow-Origin": "*"}
    else:
        res["headers"]["Access-Control-Allow-Origin"] = "*"

    print(f"Response: {res}")
    return res


@app.post("/llm-call/test-query")
def llm_chat_bot():
    body = app.current_event.json_body
    print(f"Http Body: {body}")
    instruction = get_instruction_template()
    param = LlmParam(query=body.get("query"), instruction=instruction)
    res = llmCall(param)
    return res


@app.post("/llm-call/summary/<contactId>", cors=False)
def getSummary(contactId: str):
    body = app.current_event.json_body
    query = body['query'].strip()

    if query == "":
        return {"error": "query is empty"}

    if contactId == "":
        return {"error": "contactId is empty"}

    dbRes = getLlmSummaryDb(contactId)
    if dbRes:
        return dbRes['Answer']

    instruction = generateSummaryLlmInstruction()
    query = "아래 대화 json을 분석해서 정리해줘\n\n<json>\n" + query + "\n</json>"
    param = LlmParam(
        query=query,
        instruction=instruction
    )
    res = llmCall(param)
    print(f"Response: {res.response}")
    if res:
        insertLlmSummaryDb(contactId, query, res.response, instruction)
        return res.response
    else:
        return {"error": "Failed to generate summary"}


@app.post("/llm-call/summary/<contactId>/purge", cors=False)
def purgeSummaryDb(contactId: str):
    if contactId == "":
        return {"error": "contactId is empty"}
    res = removeLlmSummary(contactId)
    return res


# S3에 Summary 데이터를 Export 한다.
@app.post("/llm-call/summary/<contactId>/export", cors=False)
def exportSummaryDbToS3(contactId: str):
    res = exportLlmSummaryToS3(contactId)
    return res


def generateSummaryLlmInstruction() -> str:
    instruction = get_summary_instruction()
    return instruction


@dataclass
class LlmParam:
    query: str = ""
    instruction: str = ""
    context: str = ""


@dataclass
class LlmResponse:
    parameter: LlmParam = None
    response: str = ""


# @todo : Context 구현하기
def llmCall(param: LlmParam) -> LlmResponse:
    human = get_human_template()
    prompt = get_prompt(human, param.instruction)
    bedrock_runtime = get_bedrock_runtime()

    retriever = get_bedrock_kb_retriever()
    retriever.get_relevant_documents(param.query)
    model = get_bedrock_model(bedrock_runtime)

    chain = get_chain(model, prompt, retriever)

    res = None
    if get_region_cache_enabled():
        print(f"Region cache enabled")
        region_cache = get_region_cache_instance()
        region_cache_key = get_region_cache_key()

        cache_key = get_hash(param.query)
        return_obj = region_cache.hget(region_cache_key, cache_key)

        if not return_obj:
            res = chain.invoke(param.query)
            serialized = pickle.dumps(res)
            region_cache.hset(region_cache_key, cache_key, serialized)
        else:
            res = pickle.loads(return_obj)
    else:
        res = chain.invoke(param.query)

    result = LlmResponse()
    result.parameter = param
    result.response = res
    return result


def removeLlmSummary(contactId: str):
    client = boto3.resource('dynamodb')
    tableName = get_ddb_contact_summary()
    table = client.Table(tableName)
    response = table.delete_item(Key={'ContactId': contactId})
    print(f"LLM Summary deleted by contactId:{contactId}")
    return response


def getLlmSummaryDb(contactId: str) -> Optional[dict]:
    client = boto3.resource('dynamodb')
    tableName = get_ddb_contact_summary()
    table = client.Table(tableName)
    response = table.get_item(Key={'ContactId': contactId})
    print(f"LLM Summary DB response: {response}")
    if 'Item' not in response:
        return None
    print(f"LLM Summary queried by contactId:{contactId}, list: {response['Item']}")
    return response['Item']


def insertLlmSummaryDb(contactId: str, query: str, answer: str, instruction: str = "", context: str = "",
                       queryDate=None, answerDate=None):
    if contactId is None:
        contactId = 'None'
    if context is None:
        context = ''
    if instruction is None:
        instruction = ''

    if not queryDate:
        queryDate = datetime.datetime.utcnow()
    if not answerDate:
        answerDate = datetime.datetime.utcnow()

    ddbResource = boto3.resource('dynamodb')
    table = get_ddb_contact_summary()
    lq = {
        'ContactId': contactId.strip(),
        'Query': query.strip(),
        'Answer': answer.strip(),
        'QueriedDate': queryDate.isoformat(),
        'AnsweredDate': answerDate.isoformat(),
        'Instruction': instruction.strip(),
        'Context': context.strip(),
    }
    print(f"Insert LLM Summary: {lq}")
    res = ddbResource.Table(table).put_item(Item=lq)
    print(f"Insert LLM Summary DB response : {res}")
    return res


def exportLlmSummaryToS3(contactId: str):
    bucket = get_summary_bucket_name()
    prefix = get_summary_bucket_prefix()
    if contactId is None:
        raise Exception("No contentId")

    summaryData = getLlmSummaryDb(contactId)
    if summaryData is None:
        raise Exception("No summary data")

    key = prefix + contactId + "_summary.md"

    s3 = boto3.client('s3')
    try:
        # 파일 존재 여부 확인
        s3.head_object(Bucket=bucket, Key=key)
        # 파일이 존재하면 삭제
        s3.delete_object(Bucket=bucket, Key=key)
        print(f"기존 파일 '{key}'를 삭제했습니다.")
    except ClientError as e:
        # 파일이 존재하지 않으면 (404 에러) 그냥 넘어갑니다.
        if e.response['Error']['Code'] != '404':
            raise
    # 새로운 데이터 업로드
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=summaryData['Answer'])
        print(f"Upload file success s3://{bucket}/{key}")
    except ClientError as e:
        raise

    return "s3://" + bucket + "/" + key


def getLlmHistory(contactId: str):
    client = boto3.resource('dynamodb')
    tableName = get_ddb_llm_history()
    table = client.Table(tableName)

    response = table.query(
        IndexName='ContactId-AnsweredDate-index',
        KeyConditionExpression=Key('ContactId').eq(contactId),
        ScanIndexForward=True  # AnsweredDate의 역순으로 정렬
    )
    print(f"LLM Query history by contactId:{contactId}, list: {response['Items']}")
    return response['Items']


def insertLlmHistory(contactId: str, query: str, answer: str, instruction: str = "", context: str = "",
                     queryDate=None, answerDate=None):
    if contactId is None:
        contactId = 'None'
    if context is None:
        context = ''
    if instruction is None:
        instruction = ''

    if not queryDate:
        queryDate = datetime.datetime.utcnow()
    if not answerDate:
        answerDate = datetime.datetime.utcnow()

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
    print(f"Insert LLM History: {lq}")
    res = ddbResource.Table(table).put_item(Item=lq)
    return res


def ps_get(key, enc=False):
    defaultRegion = "ap-northeast-2"
    pm = ParameterStore(defaultRegion)
    return pm.get_param(key, enc)


def ps_get_int(key):
    return int(ps_get(key, enc=False))


def ps_get_float(key):
    return float(ps_get(key, enc=False))


def ps_get_bool(key):
    ret = ps_get(key, enc=False)
    if ret in (True, False):
        return ret
    if ret.lower() in ('true', '1', 't'):
        return True
    return False


def ps_get_(key, default):
    res = ps_get(key, enc=False)
    if not res:
        return default
    return res


def get_agent_query(event):
    body = event.get("body", None)
    # body = get_config(event, "body", None)
    if body is None:
        return ps_get("/aaa/query")
    body_json = json.loads(body)
    return body_json.get('query', None)


def get_contact_id(event):
    body = event.get("body", None)
    if body is None:
        return ps_get("/aaa/query")
    body_json = json.loads(body)
    return body_json.get('contactId"', None)


def get_bedrock_region():
    return ps_get('/aaa/bedrock_region')


def get_bedrock_kb_id():
    return ps_get('/aaa/bedrock_kb_id')


def get_bedrock_kb_result_count():
    return ps_get_int('/aaa/bedrock_kb_result_count')


def get_bedrock_model_id():
    return ps_get('/aaa/bedrock_model_id')


def get_semantic_cache_embedding_model_id():
    return ps_get('/aaa/semantic_cache_embedding_model_id')


def get_region_cache_endpoint():
    return ps_get('/aaa/region_cache_endpoint')


def get_region_cache_ssl():
    con_str = get_region_cache_endpoint()
    if con_str.split('://')[0] == 'rediss':
        return True
    else:
        return False


def get_host_from_uri(uri):
    return uri.split('://')[1].split(':')[0]


def get_port_from_uri(uri):
    return uri.split('://')[1].split(':')[1]


def get_ssl_from_redis_uri(uri):
    if uri.split('://')[0] == 'rediss':
        return True
    else:
        return False


def get_temperature():
    return ps_get_float('/aaa/temperature')


def get_top_k():
    return ps_get_int('/aaa/top_k')


def get_top_p():
    return ps_get_int('/aaa/top_p')


def get_max_tokens():
    return ps_get_int('/aaa/max_tokens')


def get_region_cache_key():
    return ps_get('/aaa/region_cache_key')


def get_summary_bucket_name():
    return ps_get('/aaa/summary_bucket_name')


def get_summary_bucket_prefix():
    return ps_get('/aaa/summary_bucket_prefix')


def get_hash(input_string):
    encoded_str = input_string.encode()
    sha256 = hashlib.sha256()
    sha256.update(encoded_str)
    return sha256.hexdigest()


def serialize(input_json):
    import base64
    json_str = json.dumps(input_json, default=str)
    code_bytes = json_str.encode('UTF-8')
    result = base64.b64encode(code_bytes)
    return result.decode('ascii')


def deserialize(input_string):
    import base64
    code_bytes = input_string.encode('ascii')
    decoded = base64.b64decode(code_bytes)
    return json.loads(decoded.decode('UTF-8'))


def get_semantic_cache_enabled():
    return ps_get_bool('/aaa/semantic_cache_enabled')


def get_semantic_cache_score():
    return ps_get_float('/aaa/semantic_cache_score')


def get_semantic_cache_endpoint():
    return ps_get('/aaa/semantic_cache_endpoint')


def get_region_cache_enabled():
    return ps_get_bool('/aaa/region_cache_enabled')


def get_ddb_llm_history():
    return ps_get('/aaa/dynamodb_llm_history')


def get_ddb_contact_summary():
    return ps_get('/aaa/dynamodb_contact_summary')


def get_summary_instruction():
    return ps_get('/aaa/summary_instruction')


def get_query_instruction():
    return ps_get('/aaa/query_instruction')


def get_chain(model, prompt, retriever):
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    return chain


def get_region_cache_instance():
    region_cache = Redis(host=get_host_from_uri(get_region_cache_endpoint()),
                         port=get_port_from_uri(get_region_cache_endpoint()),
                         ssl=get_ssl_from_redis_uri(get_region_cache_endpoint()))
    return region_cache


def enable_semantic_cache(bedrock_runtime):
    bedrock_embeddings = get_bedrock_embedding(bedrock_runtime)
    auth = get_semantic_cache_auth()
    cache = get_semantic_cache_instance(auth, bedrock_embeddings)
    set_llm_cache(cache)


def get_bedrock_model(bedrock_runtime):
    model = ChatBedrock(
        client=bedrock_runtime,
        model_id=get_bedrock_model_id(),
        model_kwargs=get_bedrock_model_kwargs(),
    )
    return model


def get_bedrock_model_kwargs():
    model_kwargs = {
        "max_tokens": get_max_tokens(),
        "temperature": get_temperature(),
        "top_k": get_top_k(),
        "top_p": get_top_p(),
        # "stop_sequences": ["\n\nHuman"],
    }
    return model_kwargs


def get_bedrock_kb_retriever():
    retriever = AmazonKnowledgeBasesRetriever(
        region_name=get_bedrock_region(),
        knowledge_base_id=get_bedrock_kb_id(),
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": get_bedrock_kb_result_count()}},
    )
    return retriever


def get_semantic_cache_instance(auth, bedrock_embeddings):
    return OpenSearchSemanticCache(
        http_auth=auth,
        opensearch_url=get_semantic_cache_endpoint(),
        embedding=bedrock_embeddings,
        score_threshold=float(get_semantic_cache_score())
    )


def get_semantic_cache_auth():
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAsyncAuth(credentials, get_bedrock_region(), service="aoss")
    return auth


def get_bedrock_embedding(bedrock_runtime):
    bedrock_embeddings = BedrockEmbeddings(
        model_id=get_semantic_cache_embedding_model_id(),
        client=bedrock_runtime,
        credentials_profile_name="default"
    )
    return bedrock_embeddings


def get_bedrock_runtime():
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=get_bedrock_region(),
    )
    return bedrock_runtime


def get_prompt(human, instruction):
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(instruction),
            HumanMessagePromptTemplate.from_template(human)
        ],
        input_variables=['context', 'question']
    )
    return prompt


def get_human_template():
    human = """
        제시된 맥락: {context}
        새로운 질문: {question}
    """
    return human


def get_instruction_template():
    instruction = get_query_instruction()
    return instruction
