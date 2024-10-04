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
instanceId = '79609c92-0c7b-4f3c-9845-3b8676198539'

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
        InstanceId=instanceId,
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
            print(f"AI Message: -----------------")
            newTranscript.append(convert_ai_response_to_transcript(aiList.pop(0)))
            if len(aiList) == 0:
                break
            aiDate = dateutil.parser.parse(aiList[0]['AnsweredDate'])
    for ai in aiList:
        print(f"AI Message: -----------------")
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


def get_s3_analysis_file(contactInfo):
    s3Bucket = "anytelecom-connect-data"
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
        InstanceId=instanceId,
        TimeRange={
            'Type': 'INITIATION_TIMESTAMP',
            'StartTime': startTime.isoformat(),
            'EndTime': endTime.isoformat()
        },
        Sort={
            'FieldName': 'INITIATION_TIMESTAMP',
            'Order': 'DESCENDING'
        },
        # Channels=[
        #     'VOICE',
        #     'CHAT'
        # ],
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

    instruction = """
        당신은 전화 상담원의 통화 기록을 분석하는 전문 분석가 입니다.
        제공된 문의는 고객('CUSTOMER'), 상담원('AGENT'), 그리고 상담원을 돕는 AI('AI')가 나눈 대화 내용입니다.
        
        대화 내용은 JSON 형태로 제공됩니다. 
        - Content: 대화내용
        - ParticipantId: 대화자 구분
        - Id: Json 고유키
        - AbsoluteTime: 대화 발생 시간

        아래 기준으로 대화 내용을 분석하고, "처리 기준", "개인 정보 처리 기준"을 바탕으로 정보를 정리
        "응답 포멧" 대로 정보를 구성하여 Markdown 형식으로 응답
        분석할 고객의 대화 내용이 없다면, 간략하 분석할 내용 없음으로 표시
        고객이 제공한 개인정보 노출 금지!
        고객이 요청한 변경 정보 상세 노출 금지!

        처리 기준
        - 언어: 한국어로 답변
        - 객관성: 당신의 개인적인 해석이나 판단을 배제
        - 개인정보보안: 이름, 나이, 주소, 전화번호, 카드번호, CVS코드, 비밀번호 등 개인 정보는 제거
        - 정확성: 날짜, 시간, 이름, 숫자, 장소 등의 구체적인 정보를 정확하게 발견하고 전달
        - 완전성: 중요한 정보가 누락되지 않도록 주의
        - 가독성: 모든 출력에서 적절한 띄어쓰기를 사용
        - 단순함: 존칭은 생략하고 정보만 전달
        - 순서가 있는 동작이 포함된 답변은, 단계별로 라인을 구분하고, 순서를 표시 해 주세요. 
        - 여러가지의 문제를 해결 했다면, 문제별로 조치내용을 구분지어 정리

        개인 정보 처리 기준
        - 대화 에서 개인 정보 제거
        - 이름, 주소, 나이, 성별, 전화번호, 카드번호, CVC코드, 비밀번호는 절대 노출 불가
        - 대화에서 이름 제거
        - 대화에서 성명 제거
        - 대화에서 주소 제거
        - 대화에서 나이 제거
        - 대화에서 성별 제거
        - 대화에서 전화번호 제거
        - 대화에서 전화 번호 제거
        - 대화에서 카드번호 제거
        - 대화에서 카드 번호 제거
        - 대화에서 CVC 제거
        - 대화에서 비밀번호 제거

        정리 기준
        - 고객과 상담원의 대화 내용을 중심으로 맥락을 파악합니다.
        - AI의 대화내용은 상담원의 응답을 확장해 주는 용도입니다. 주된 내용에서는 빼주세요. 
        - 순서있는 목록, 순서없는 목록, 문서 링크등 활용 가능 

        <응답포멧>
         ## 제목 (제목을 20자 이내로 요약)
         ### 요약
         (상담 내용을 100자 이내로 요약)

         ### 고객의 문의 또는 문제
         (고객이 현재 알고 싶어하는 내용이나 문제를 200자 이내로 요약)

         ### 문제 해결 방법
         (상담원과 고객의 대화를 바탕으로 문제 해결을 위한 설명이나 방법을 요약합니다. 500자 이내)

         ### 해결되지 않은 문제
         (고객의 처한 상황이나 문의 중 대화중에 해결되지 않은 궁금증이나 문제를 300자 이내로 정리)

         ### 최종 고객 만족도
         (상담이 끝나는 시점에 고객의 만족도와 상태를 30자 이내로 분석)

         ### 키워드 리스트
         (본 대외에서 키워드, Tag를 최소 2개~5개로 나열)
        </응답포멧>
    """

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
    instruction = """
        당신의 임무는 제공된 문서를 바탕으로, 문의에 정확하고 간결하게 답변.
        제공된 문서나 정보가 없을 경우 아래의 규칙에 의해 답변

        답변을 할때는 아래의 기준을 사용.
        - 언어: 한국어로 답변
        - 객관성: 당신의 개인적인 해석이나 판단을 배제
        - 정확성: 날짜, 시간, 이름, 숫자, 장소 등의 구체적인 정보를 정확하게 발견하고 전달
        - 정보보안: 이름, 성별, 나이, 개인의 주소와 같은 개인정보는 배제
        - 완전성: 중요한 정보가 누락되지 않도록 주의
        - 가독성: 모든 출력에서 적절한 띄어쓰기를 사용하
        - 단순함: 존칭은 생략하고 정보만 전달
        - 순서가 있는 동작이 포함된 답변은, 단계별로 라인을 구분하고, 순서를 표시 해 주세요. 

        제공된 정보가 없을 경우 아래 기준으로 답변.
        - 제공된 정보가 없을 경우 '정보 없음' 으로만 답변
        - 존칭, 사과, 상세 또는 부가 설명 금지
    """
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
