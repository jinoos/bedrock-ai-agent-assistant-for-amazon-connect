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


@app.get("/knowledgebase/l2")
def get_knowledge_base_id():
    llm_history = [
        {
            "contactId": "db344a49-4618-4277-b62f-844f800e4c65",
            "query": "멤버십 카드 발급",
            "answer": """
                    멤버십 카드 발급에 대해 요약하면 다음과 같습니다.

        1. SK텔레콤 고객센터(114 또는 1599-0011)나 SK텔레콤 대리점을 방문하여 멤버십 카드 재발급을 요청할 수 있습니다.
        2. 대리점 방문 시 필요한 서류:
           - 개인: 이동전화 명의자 신분증, 멤버십 명의자 신분증(다를 경우)
           - 법인/공공기관: 사업자등록증(또는 고유번호증) 사본, 멤버십 명의자 신분증
        3. 멤버십 카드 재발급은 월 4회까지만 가능합니다.
        4. 재발급 신청 후 7-10일 내에 우편으로 새 카드를 받을 수 있습니다.
        5. 기존 카드는 새 카드 수령 전까지 사용 가능하며, 새 카드 신청 시 자동으로 이전 카드가 해지됩니다.
                    """,
            "instruction": """
                    """,
            "context": None,
            "queryDate": (datetime.strptime("2024-08-24 18:38:54", '%Y-%m-%d %H:%M:%S') - timedelta(hours=9)),
            "answerDate": (datetime.strptime("2024-08-24 18:38:54", '%Y-%m-%d %H:%M:%S') - timedelta(hours=9)),
        },
        {
            "contactId": "db344a49-4618-4277-b62f-844f800e4c65",
            "query": "멤버십 카드 발급 대리점, 필요 서류",
            "answer": """
            멤버십 카드 재발급 방법은 다음과 같습니다.

1. SK텔레콤 고객센터(SK텔레콤 휴대폰에서 지역번호 없이 114 : 무료, 일반전화로 1599-0011 : 유료)로 전화하여 재발급 요청할 수 있습니다.

2. SK텔레콤 대리점/지점에 방문하여 재발급을 받을 수 있습니다. 이 경우 아래의 구비서류가 필요합니다.

- 개인: 
  - 이동전화 명의자 = 멤버십 명의자: 이동전화 명의자 신분증
  - 이동전화 명의자 ≠ 멤버십 명의자: 이동전화 명의자 신분증, 멤버십명의자 신분증
  - 대리인: 명의자 신분증, 멤버십명의자 신분증, 대리인 신분증, 멤버십명의자와 동의통화

- 법인/공공기관 등:
  - 멤버십 명의자: 사업자등록증(or 고유번호증) 사본, 멤버십명의자 신분증
  - 대리인: 사업자등록증(or 고유번호증) 사본, 멤버십명의자 신분증, 대리인 신분증, 멤버십 명의자 동의통화

참고로 멤버십 카드 재발급 신청 시 7일~10일 이내에 우편으로 받으실 수 있습니다.
                    """,
            "instruction": "",
            "context": None,
            "queryDate": (datetime.strptime("2024-08-24 18:46:41", '%Y-%m-%d %H:%M:%S') - timedelta(hours=9)),
            "answerDate": (datetime.strptime("2024-08-24 18:46:41", '%Y-%m-%d %H:%M:%S') - timedelta(hours=9)),
        },
        {
            "contactId": "db344a49-4618-4277-b62f-844f800e4c65",
            "query": "VIP 멤버십 혜택",
            "answer": """
            VIP 멤버십 혜택에 대해 요약하면 다음과 같습니다:

1. VIP PICK: 연 12회(월 1회) 원하는 혜택을 직접 선택할 수 있습니다. 선택 가능한 제휴혜택은 VIP PICK에서 확인할 수 있습니다.
2. 영화 혜택: VIP 회원은 CGV에서 무료 관람권 1개(연 3회)와 평일/주말 관람권 1+1(연 9회) 혜택을 받을 수 있습니다. 또한 전 등급 회원은 영화 예매 시 최대 4,000원 할인 혜택을 받을 수 있습니다.
3. 기타 혜택: 외식, 공연 등 다양한 제휴사에서 VIP 멤버십 혜택을 이용할 수 있습니다. 자세한 내용은 T world 웹사이트나 T 멤버십 앱에서 확인할 수 있습니다.
            """,
            "instruction": """""",
            "context": None,
            "queryDate": (datetime.strptime("2024-08-24 18:49:34", '%Y-%m-%d %H:%M:%S') - timedelta(hours=9)),
            "answerDate": (datetime.strptime("2024-08-24 18:49:34", '%Y-%m-%d %H:%M:%S') - timedelta(hours=9)),
        },
        {
            "contactId": "db344a49-4618-4277-b62f-844f800e4c65",
            "query": "멤버십 카드 분실시 재발급",
            "answer": """
            제공된 문서를 바탕으로 답변드리겠습니다.

1. SK텔레콤 멤버십 카드를 분실한 경우 다음과 같은 방법으로 재발급을 받으실 수 있습니다.
   - SK텔레콤 고객센터(114 또는 1599-0011)로 전화하여 재발급 요청
   - SK텔레콤 대리점/지점 방문하여 재발급 요청

2. SK텔레콤 대리점/지점 방문 시 필요한 구비서류는 다음과 같습니다.
   - 개인: 이동전화 명의자 신분증, 멤버십 명의자 신분증(명의자가 다른 경우)
   - 법인/공공기관: 사업자등록증(또는 고유번호증) 사본, 멤버십 명의자 신분증, 대리인 신분증

3. 멤버십 카드 재발급 신청 시 7-10일 내에 우편으로 새로운 카드를 받으실 수 있습니다.
4. 재발급 신청 시 기존 카드는 새 카드 수령 전까지 사용 가능하며, 새 카드 신청 시 기존 카드는 자동 해지됩니다.
5. 다만 재발급 사유를 분실로 선택한 경우 기존 카드는 즉시 사용이 불가합니다.
6. 멤버십 카드 재발급은 월 4회까지만 가능합니다.
                """,
            "instruction": """
                """,
            "context": None,
            "queryDate": (datetime.strptime("2024-08-24 18:52:45", '%Y-%m-%d %H:%M:%S') - timedelta(hours=9)),
            "answerDate": (datetime.strptime("2024-08-24 18:52:45", '%Y-%m-%d %H:%M:%S') - timedelta(hours=9)),
        },
    ]
    # return (llm_history[0].get("queryDate").isoformat())
    # return llm_history

    res = []
    for q in llm_history:
        r = insert_llm_history(q['contactId'], q['query'], q['answer'], q['instruction'], q['context'], q['queryDate'],
                               q['answerDate'])
        res.append(r)

    return res


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

    table = 'llm_history'
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
    table = 'llm_history'
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
    return get_param_('/abp/bedrock_region', 'us-west-2')


def get_sematic_cache_score(event):
    return get_config(event, '/abp/sematic_cache_score', 0.5)


def get_semantic_cache_endpoint():
    return get_param_('/abp/semantic_cache_endpoint', 'https://gl4t5xr8g4ch9shxorh3.us-west-2.aoss.amazonaws.com')


def get_region_cache_endpoint():
    return get_param_('/abp/region_cache_endpoint',
                      'rediss://anytel-cache-1bwhhn.serverless.apn2.cache.amazonaws.com:6379')


def get_knowledge_base_id():
    return get_param_('/abp/knowledge-base-id', 'DGWRG5M6CE')


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
