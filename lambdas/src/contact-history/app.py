import dateutil
from aws_lambda_powertools import Tracer, Logger
from aws_lambda_powertools.event_handler import APIGatewayRestResolver, CORSConfig
from aws_lambda_powertools.logging import correlation_paths
from boto3.dynamodb.conditions import Key
from dateutil.parser import parser

from dataclasses import dataclass
import uuid
import datetime

import boto3
import json
import pickle
import hashlib

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


def get_param(key):
    ps = ParameterStore()
    return ps.get_param(key)


def custom_serializer(obj) -> str:
    return json.dumps(obj, default=str)


tracer = Tracer()
logger = Logger()
region = boto3.Session().region_name
cors_config = CORSConfig(allow_origin="*")
app = APIGatewayRestResolver(serializer=custom_serializer, cors=cors_config)

region_name = boto3.Session().region_name
client = boto3.client('connect', region_name=region_name)


@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
@tracer.capture_lambda_handler
def lambda_handler(event, context) -> dict:
    print(f"Event: {event}")
    res = app.resolve(event, context)
    print(f"Res: {res}")
    return res


@app.get("/contact-history")
def main():
    return contact_history()

@app.get("/contact-history/contact/<contactId>")
def contact_info(contactId: str):
    res = client.describe_contact(
        InstanceId=get_connect_id(),
        ContactId=contactId
    )
    return res


@app.get("/contact-history/llm_history/<contactId>/list")
def get_contact_llm_history(contactId: str):
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


@app.get("/contact-history/contact/<contactId>/analysis")
# @app.get("/contact-history/contact/<contactId>/analysis/")
def get_analysis_data(contactId: str):
    contactInfo = contact_info(contactId)
    if not nested_key_exists(contactInfo, ['Contact', 'AgentInfo', 'ConnectedToAgentTimestamp']):
        return None

    s3Data = get_s3_analysis_file(contactInfo)
    print(f"S3 Data: {s3Data}")
    if s3Data is None:
        return None

    s3DataObj = json.loads(s3Data)

    analysisData = None
    if contactInfo['Contact']['Channel'] == 'VOICE':
        analysisData = get_analysis_data_for_voice(contactInfo, s3DataObj)
    elif contactInfo['Contact']['Channel'] == 'CHAT':
        analysisData = get_analysis_data_for_chat(contactInfo, s3DataObj)
    else:
        return None

    aiList = get_contact_llm_history(contactId)
    newTranscript = []
    for trans in analysisData['Transcript']:
        if len(aiList) == 0:
            newTranscript.append(trans)
            continue

        tranDate = dateutil.parser.parse(trans['AbsoluteTime']).replace(tzinfo=None)
        aiDate = dateutil.parser.parse(aiList[0]['AnsweredDate']).replace(tzinfo=None)

        if aiDate > tranDate:
            newTranscript.append(trans)
            continue

        while len(aiList) > 0 and aiDate < tranDate:
            newTranscript.append(convert_ai_response_to_transcript(aiList.pop(0)))
            if len(aiList) == 0:
                break
            aiDate = dateutil.parser.parse(aiList[0]['AnsweredDate'])
    for ai in aiList:
        newTranscript.append(convert_ai_response_to_transcript(ai))

    print(f"Analysis Data: {newTranscript}")
    analysisData['Transcript'] = newTranscript
    return analysisData


def convert_ai_response_to_transcript(aiResponse: dict):
    return {
        'Content': 'Question: ' + aiResponse['Query']+'\n\nAI Answer:\n'+aiResponse['Answer'],
        'ParticipantId': 'AI',
        'Id': str(uuid.uuid4()),
        'AbsoluteTime': dateutil.parser.parse(aiResponse['AnsweredDate'])+datetime.timedelta(hours=9)
    }


def get_analysis_data_for_voice(contactInfo: dict, s3DataObj: dict):
    ao = {}
    ao['Categories'] = s3DataObj['Categories']
    ao['Channel'] = s3DataObj['Channel']
    ao['LanguageCode'] = s3DataObj['LanguageCode']
    ao['Participants'] = s3DataObj['Participants']
    ao['Sentiment'] = {}
    ao['Sentiment']['OverallSentiment'] = s3DataObj['ConversationCharacteristics']['Sentiment']['OverallSentiment']
    ao['Transcript'] = []
    for seg in s3DataObj['Transcript']:
        ao['Transcript'].append(convert_tran_segment_from_voice(contactInfo, seg))

    return ao


def get_analysis_data_for_chat(contactInfo: dict, s3DataObj: dict):
    ao = {}
    ao['Categories'] = s3DataObj['Categories']
    ao['Channel'] = s3DataObj['Channel']
    ao['LanguageCode'] = s3DataObj['LanguageCode']
    ao['Participants'] = s3DataObj['Participants']
    pIdx = {}
    for t in s3DataObj['Participants']:
        pIdx[t['ParticipantId']] = t['ParticipantRole']

    ao['Sentiment'] = {}
    ao['Sentiment']['OverallSentiment'] = s3DataObj['ConversationCharacteristics']['Sentiment']['OverallSentiment'][
        'DetailsByParticipantRole']
    ao['Transcript'] = []
    for seg in s3DataObj['Transcript']:
        if seg['Type'] == 'EVENT':
            continue
        seg['ParticipantId'] = pIdx[seg['ParticipantId']]
        ao['Transcript'].append(convert_tran_segment_from_chat(contactInfo, seg))

    return ao


def convert_tran_segment_from_voice(contactInfo: dict, seg: dict):
    connectedTime = contactInfo['Contact']['AgentInfo']['ConnectedToAgentTimestamp']
    newSeg = {
        'Content': seg['Content'],
        'ParticipantId': seg['ParticipantId'],
        'Id': seg['Id'],
        'AbsoluteTime': connectedTime + datetime.timedelta(milliseconds=seg['BeginOffsetMillis'])
    }
    return newSeg


def convert_tran_segment_from_chat(contactInfo: dict, seg: dict):
    newSeg = {
        'Content': seg['Content'],
        'ParticipantId': seg['ParticipantId'],
        'Id': seg['Id'],
        'AbsoluteTime': seg['AbsoluteTime']
    }
    return newSeg


def get_connect_s3_bucket_name():
    # Amazon Connect 클라이언트 생성
    connect = boto3.client('connect')

    # 함수 호출 및 응답 받기
    response = connect.list_instance_storage_configs(
        InstanceId=get_connect_id(),
        ResourceType='CALL_RECORDINGS'
    )
    # 응답에서 BucketName 추출
    storage_config = response['StorageConfigs'][0]
    bucket_name = storage_config['StorageConfig']['S3Config'].get('BucketName')
    return bucket_name


def get_s3_analysis_file(contactInfo):
    s3Bucket = get_connect_s3_bucket_name()
    channel = contactInfo['Contact']['Channel']

    contactId = contactInfo['Contact']['Id']
    print(f"ContactInfo: {contactInfo}")
    connectedTime = contactInfo['Contact']['AgentInfo']['ConnectedToAgentTimestamp']
    s3Path = ""
    if channel == 'VOICE':
        s3Path = connectedTime.strftime("Analysis/Voice/%Y/%m/%d")
    else:  # CHAT
        s3Path = connectedTime.strftime("Analysis/Chat/%Y/%m/%d")

    print(f"FilePath: {s3Path}")
    s3 = boto3.client('s3', region_name=boto3.Session().region_name)
    objectList = s3.list_objects_v2(Bucket=s3Bucket, Prefix=s3Path)
    print(f"ObjectList: {objectList}")
    for obj in objectList['Contents']:
        print(obj)
        if "/" + contactId in obj['Key'] and obj['Key'].endswith(".json"):
            f = s3.get_object(Bucket=s3Bucket, Key=obj['Key'])
            return f['Body'].read().decode('utf-8')

    return None


def nested_key_exists(d, keys):
    if keys and d:
        return nested_key_exists(d.get(keys[0]), keys[1:])
    return not keys and d is not None


def contact_history():
    endTime = datetime.datetime.utcnow()
    startTime = endTime - datetime.timedelta(days=30)
    contactList = client.search_contacts(
        InstanceId=get_connect_id(),
        TimeRange={
            'Type': 'INITIATION_TIMESTAMP',
            'StartTime': startTime.isoformat(),
            'EndTime': endTime.isoformat()
        },
        Sort={
            'FieldName': 'INITIATION_TIMESTAMP',
            'Order': 'DESCENDING'
        },
        MaxResults=100
    )

    return contactList


def insert_llm_history(contactId: str, query: str, answer: str, instruction: str = "", context: str = "",
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
    res = ddbResource.Table(table).put_item(Item=lq)
    print(f"Insert LLM History: {res}")
    return res



@app.post("/contact-history/test-query")
def llm_chat_bot():
    body = app.current_event.json_body
    print(f"Http Body: {body}")
    instruction = get_instruction_template()
    param = LlmParam(query=body.get("query"), instruction=instruction)
    res = llmCall(param)
    return res


@app.post("/contact-history/generate-summary", cors=False)
def generateSummary():
    body = app.current_event.json_body
    print(f"Http Body: {body}")
    query = body["query"]
    print(f"Http Body: {query}")
    if query.strip() == "":
        return {"error": "query is empty"}

    instruction = get_summary_instruction()

    param = LlmParam(
        query=body.get("query"),
        instruction=instruction
    )
    res = llmCall(param)
    return res.response


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

    if get_semantic_cache_enabled():
        print(f"Semantic Cache enabled")
        enable_semantic_cache(bedrock_runtime)

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
    return ps_get('/aaa/connect_id')


def get_summary_instruction():
    return ps_get('/aaa/summary_instruction')


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


def get_connect_id():
    return ps_get_bool('/aaa/connect_id')


def get_semantic_cache_enabled():
    return ps_get_bool('/aaa/semantic_cache_enabled')


def get_semantic_cache_score():
    return ps_get_float('/aaa/semantic_cache_score')


def get_semantic_cache_endpoint():
    return ps_get('/aaa/semantic_cache_endpoint')


def get_region_cache_enabled():
    return ps_get_bool('/aaa/region_cache_enabled')


def get_ddb_llm_history():
    return get_param('/aaa/dynamodb_llm_history')


def get_ddb_contact_summary():
    return get_param('/aaa/dynamodb_contact_summary')


def get_query_instruction():
    return get_param('/aaa/query_instruction')


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


def insert_llm_history(contactId: str, query: str, answer: str, instruction: str = "", context: str = "",
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
