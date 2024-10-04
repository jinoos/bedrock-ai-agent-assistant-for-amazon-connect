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
from langchain_core.caches import BaseCache, RETURN_VAL_TYPE
from langchain_core.embeddings import Embeddings
from langchain_core.load import loads, dumps
from langchain_core.outputs import Generation
from opensearchpy import AWSV4SignerAsyncAuth
from redis import Redis

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.globals import set_llm_cache
from langchain_community.vectorstores import (
    OpenSearchVectorSearch as OpenSearchVectorStore,
)
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
    instruction = """
        당신은 전화 상담원의 통화 기록을 분석 하는 전문 분석가 입니다.
        입력된 JSON 문서를 바탕으로 최종 결과물을 응답하세요
        제공된 문의는 고객('CUSTOMER'), 상담원('AGENT'), 그리고 상담원을 돕는 AI('AI')가 나눈 대화 내용
        "~하겠습니다", "~하려고합니다" 와 같은 표현 금지
        
        제공된 질문은 JSON 형태입니다. 대화내용과 대화자 구분을 활용하여 분석하세요.
        - Content: 대화내용
        - ParticipantId: 대화자 구분
        - Id: Json 고유키
        - AbsoluteTime: 대화 발생 시간

        아래 처리 기준을 바탕으로 응답합니다.

        처리 기준
        - 언어: 한국어로 답변
        - 객관성: 당신의 개인적인 해석이나 판단을 배제
        - 정확성: 날짜, 시간, 이름, 숫자, 장소 등의 구체적인 정보를 정확하게 발견하고 전달
        - 완전성: 중요한 정보가 누락되지 않도록 주의
        - 가독성: 모든 출력에서 적절한 띄어쓰기를 사용
        - 단순함: 존칭은 생략하고 정보만 전달
        - 순서가 있는 동작이 포함된 답변은, 단계별로 라인을 구분하고, 순서를 표시 해 주세요. 
        - 여러가지의 문제를 해결 했다면, 문제별로 조치내용을 구분지어 정리
        - 개인 정보 처리는 아래 기준으로 처리
        - 고객과 상담원의 대화 내용을 중심으로 맥락을 파악합니다.
        - AI의 대화내용은 상담원의 응답을 확장해 주는 용도입니다. 주된 내용에서는 빼주세요. 
        - 순서있는 목록, 순서없는 목록, 문서 링크등 활용 가능 

        응답 기준
         - 응답은 아래 <응답포멧>을 참고
         - <format>의 괄호는 작성에 대한 설명입니다. 응답에선 형식을 제외
         - Markdown 문서 포멧으로 작성
         - 응답에서 <개인 정보>는 "XXXXX" 형태로 변경
         - 응답에서 <개인 정보>는 노출 금지
         - 구체적인 번호는 생략
         - 구체적인 개인 정보는 생략
         
        <개인 정보>
         - 주민번호
         - 주민 번호
         - 전화번호
         - 전화 번호
         - 성명
         - 이름
         - 성별
         - 주소
         - 이메일
         - 여권번호
         - 나이
         - 연령

        <응답포멧>
         ## 제목 (제목을 20자 이내로 요약)
         (상담 내용을 100자 이내로 요약, 어떤 문의에서 필요한 문서인지 인지할 수 있어야 함)

         ### 고객의 문의 또는 문제
         (고객이 현재 알고 싶어하는 내용이나 문제를 자세히 기술 이내로 요약)

         ### 문제 해결 방법
         (상담원과 고객의 대화를 바탕으로 문제 해결을 위한 설명이나 방법을 요약합니다. 500자 이내)

         ### 해결되지 않은 문제
         (고객의 처한 상황이나 문의 중 대화중에 해결되지 않은 궁금증이나 문제를 300자 이내로 정리)

         ## 키워드 리스트
         (본 대외에서 키워드, Tag를 최소 2개~5개로 나열형( - list) 항목으로 제작)
         
         ## 상담 관련 정리
         ### 상담 일시
         YYYY-mm-dd HH:ii ~ HH:ii (총 XX분) (상담 일시와 상담 시간을 정리. 시간 표현은 분까지만 표현할 것. 데이터의 일시는 UTC 시간대 임. KST 시간대로 변환해서 정리. 정확히 고객과 상담원의 대화 시간 만 계산할 것, AI 메시지 시간은 제외)
         ### 최종 정리
         (제 3자의 입장에서 상담 과정을 보고, 전체적으로 정리)
        </응답포멧>
    """
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


# contactId = get_contact_id(event)
# res = insert_llm_history(contactId, input_query, response, instruction)


# @todo : Context 구현하기
def llmCall(param: LlmParam) -> LlmResponse:
    human = get_human_template()
    prompt = get_prompt(human, param.instruction)
    bedrock_runtime = get_bedrock_runtime()

    # if get_semantic_cache_enabled():
    #     print(f"Semantic Cache enabled")
    #     enable_semantic_cache(bedrock_runtime)

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
    tableName = 'llm_summary'
    table = client.Table(tableName)
    response = table.delete_item(Key={'ContactId': contactId})
    print(f"LLM Summary deleted by contactId:{contactId}")
    return response


def getLlmSummaryDb(contactId: str) -> Optional[dict]:
    client = boto3.resource('dynamodb')
    tableName = 'llm_summary'
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
    table = 'llm_summary'
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
    bucket = "anytelecom"
    bucketOregon = "anytelecom-oregon"
    prefix = "callcenter-contents/call-summary/"
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
        s3.head_object(Bucket=bucketOregon, Key=key)
        # 파일이 존재하면 삭제
        s3.delete_object(Bucket=bucket, Key=key)
        s3.delete_object(Bucket=bucketOregon, Key=key)
        print(f"기존 파일 '{key}'를 삭제했습니다.")
    except ClientError as e:
        # 파일이 존재하지 않으면 (404 에러) 그냥 넘어갑니다.
        if e.response['Error']['Code'] != '404':
            raise

    # 새로운 데이터 업로드
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=summaryData['Answer'])
        s3.put_object(Bucket=bucketOregon, Key=key, Body=summaryData['Answer'])
        print(f"Upload file success s3://{bucket}/{key}")
    except ClientError as e:
        raise

    return "s3://" + bucket + "/" + key



def getLlmHistory(contactId: str, indexForward: bool = True):
    client = boto3.resource('dynamodb')
    tableName = 'llm_history'
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
    return OpenSearchSemanticCacheCustom(
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


class OpenSearchSemanticCacheCustom(BaseCache):
    """Cache that uses OpenSearch vector store backend"""

    def __init__(
            self, opensearch_url: str, embedding: Embeddings, score_threshold: float = 0.2, **kwargs: Any
    ):
        """
        Args:
            opensearch_url (str): URL to connect to OpenSearch.
            embedding (Embedding): Embedding provider for semantic encoding and search.
            score_threshold (float, 0.2):
        Example:
        .. code-block:: python
            import langchain
            from langchain.cache import OpenSearchSemanticCache
            from langchain.embeddings import OpenAIEmbeddings
            langchain.llm_cache = OpenSearchSemanticCache(
                opensearch_url="http//localhost:9200",
                embedding=OpenAIEmbeddings()
            )
        """
        self._cache_dict: Dict[str, OpenSearchVectorStore] = {}
        self.opensearch_url = opensearch_url
        self.embedding = embedding
        self.score_threshold = score_threshold
        self.kwargs = kwargs

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"cache_{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> OpenSearchVectorStore:
        index_name = self._index_name(llm_string)

        # return vectorstore client for the specific llm string
        if index_name in self._cache_dict:
            return self._cache_dict[index_name]

        # create new vectorstore client for the specific llm string
        self._cache_dict[index_name] = OpenSearchVectorStore(
            opensearch_url=self.opensearch_url,
            index_name=index_name,
            embedding_function=self.embedding,
            **self.kwargs
        )

        # create index for the vectorstore
        vectorstore = self._cache_dict[index_name]
        if not vectorstore.index_exists():
            _embedding = self.embedding.embed_query(text="test")
            vectorstore.create_index(len(_embedding), index_name)
        return vectorstore

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search(
            query=prompt,
            k=1,
            score_threshold=self.score_threshold,
        )
        if results:
            for document in results:
                try:
                    generations.extend(loads(document.metadata["return_val"]))
                except Exception:
                    print(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )

                    generations.extend(
                        _load_generations_from_json(document.metadata["return_val"])
                    )
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "OpenSearchSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs["llm_string"])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].delete_index(index_name=index_name)
            del self._cache_dict[index_name]


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


def _load_generations_from_json(generations_json: str) -> RETURN_VAL_TYPE:
    try:
        results = json.loads(generations_json)
        return [Generation(**generation_dict) for generation_dict in results]
    except json.JSONDecodeError:
        raise ValueError(
            f"Could not decode json to list of generations: {generations_json}"
        )
