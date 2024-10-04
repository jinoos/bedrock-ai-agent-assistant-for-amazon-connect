import logging
import uuid
import datetime
from typing import Dict, Any, Optional, List
from utils.ssm import parameter_store

import boto3
import json
import pickle
import hashlib

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

_region = boto3.Session().region_name
_pm = parameter_store(_region)


def get_param(key):
    return _pm.get_params(key, enc=False)


def get_param_int(key):
    return int(_pm.get_params(key, enc=False))


def get_param_float(key):
    return float(_pm.get_params(key, enc=False))


def get_param_(key, default):
    res = _pm.get_params(key, enc=False)
    if not res:
        return default
    return res


def get_agent_query(event):
    body = event.get("body", None)
    # body = get_config(event, "body", None)
    if body is None:
        return get_param("/abp/query")
    body_json = json.loads(body)
    return body_json.get('query', None)


def get_contact_id(event):
    body = event.get("body", None)
    if body is None:
        return get_param("/abp/query")
    body_json = json.loads(body)
    return body_json.get('contactId', None)


def get_bedrock_region():
    return get_param('/abp/bedrock_region')


def get_bedrock_kb_id():
    return get_param('/abp/bedrock_kb_id')


def get_bedrock_kb_result_count():
    return get_param_int('/abp/bedrock_kb_result_count')


def get_bedrock_model_id():
    return "anthropic.claude-3-haiku-20240307-v1:0"
    return get_param('/abp/bedrock_model_id')



def get_semantic_cache_embedding_model_id():
    # return "amazon.titan-embed-text-v2:0"
    return get_param('/abp/semantic_cache_embedding_model_id')


def get_region_cache_endpoint():
    return get_param('/abp/region_cache_endpoint')


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
    return get_param_float('/abp/temperature')


def get_top_k():
    return get_param_int('/abp/top_k')


def get_top_p():
    return get_param_int('/abp/top_p')


def get_max_tokens():
    print(f"Max Token: {get_param_int('/abp/max_tokens')}")
    return get_param_int('/abp/max_tokens')


def get_region_cache_key():
    return get_param('/abp/region_cache_key')


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


def get_semantic_cache_enabled(event):
    return get_config_bool(event, '/abp/semantic_cache_enabled')


def get_semantic_cache_score():
    return get_param('/abp/semantic_cache_score')


def get_semantic_cache_endpoint():
    return get_param('/abp/semantic_cache_endpoint')


def get_region_cache_enabled(event):
    return get_config_bool(event, '/abp/region_cache_enabled', False)


def get_config_bool(event, key, default=False):
    ret = get_config(event, key, default)
    if ret in (True, False):
        return ret
    if ret.lower() in ('true', '1', 't'):
        return True
    return default


def get_config(event, key, default=None):
    cache = event.get(key)
    cache_env = get_param(key)
    if cache:
        return cache
    if cache_env:
        return cache_env
    return default


def get_prompt_query(event):
    return get_agent_query(event)


def main2(event):
    pass
    # instruction = """
    #     당신의 임무는 제공된 문서를 바탕으로, 문의에 정확하고 간결하게 답변.
    #     제공된 문서나 정보가 없을 경우 아래의 규칙에 의해 답변
    #
    #     답변을 할때는 아래의 기준을 사용.
    #     - 언어: 한국어로 답변
    #     - 객관성: 당신의 개인적인 해석이나 판단을 배제
    #     - 정확성: 날짜, 시간, 이름, 숫자, 장소 등의 구체적인 정보를 정확하게 발견하고 전달
    #     - 정보보안: 이름, 성별, 나이, 개인의 주소와 같은 개인정보는 배제
    #     - 완전성: 중요한 정보가 누락되지 않도록 주의
    #     - 가독성: 모든 출력에서 적절한 띄어쓰기를 사용하
    #     - 단순함: 존칭은 생략하고 정보만 전달
    #     - 순서가 있는 동작이 포함된 답변은, 단계별로 라인을 구분하고, 순서를 표시 해 주세요.
    #     제공된 정보가 없을 경우 아래 기준으로 답변.
    #     - 제공된 정보가 없을 경우 '정보 없음' 으로만 답변
    #     - 존칭, 사과, 상세 또는 부가 설명 금지
    # """
    # human = """
    #     제시된 맥락: {context}
    #     새로운 질문: {question}
    # """
    #
    # prompt = ChatPromptTemplate(
    #     messages=[
    #         SystemMessagePromptTemplate.from_template(instruction),
    #         HumanMessagePromptTemplate.from_template(human)
    #     ],
    #     input_variables=['context', 'question']
    # )
    #
    # bedrock_runtime = boto3.client(
    #     service_name="bedrock-runtime",
    #     region_name=get_bedrock_region(),
    # )
    # input_query = get_agent_query(event)
    # print(f"Query: {input_query}")
    #
    # retriever = AmazonKnowledgeBasesRetriever(
    #     region_name=get_bedrock_region(),
    #     knowledge_base_id=get_bedrock_kb_id(),
    #     retrieval_config={"vectorSearchConfiguration": {"numberOfResults": get_bedrock_kb_result_count()}},
    # )
    # # retriever.invoke(input_query)
    #
    # print(f"Invoke model: {get_bedrock_model_id()}")
    # model_kwargs = {
    #     "max_tokens": get_max_tokens(),
    #     "temperature": get_temperature(),
    #     "top_k": get_top_k(),
    #     "top_p": get_top_p(),
    #     "stop_sequences": ["\n\nHuman"],
    # }
    # model = ChatBedrock(
    #     client=bedrock_runtime,
    #     model_id=get_bedrock_model_id(),
    #     model_kwargs=model_kwargs
    # )
    # print(f"model_id: {model_kwargs}")
    # # model = ChatBedrock(
    # #     client=bedrock_runtime,
    # #     model_id="anthropic.claude-3-haiku-20240307-v1:0",
    # #     model_kwargs={"temperature": 0},
    # # )
    # # response = model.invoke("세종대왕")
    #
    # chain = (
    #         {"context": retriever, "question": RunnablePassthrough()}
    #         | prompt
    #         | model
    #         | StrOutputParser()
    # )
    # response = chain.invoke(input_query)
    #
    # res = {
    #     "response": response,
    #     "query": input_query,
    #     "instruction": instruction,
    # }
    # print(res)
    # return res


def main(event):
    print(f"Main start :{event}")
    input_query = get_prompt_query(event)
    print(f"Query: {input_query}")

    instruction = get_instruction_template()
    human = get_human_template()
    prompt = get_prompt(human, instruction)
    bedrock_runtime = get_bedrock_runtime()

    if get_semantic_cache_enabled(event):
        print(f"Semantic Cache enabled")
        enable_semantic_cache(bedrock_runtime)

    # Amazon Bedrock - KnowledgeBase Retriever
    retriever = get_bedrock_kb_retriever()
    retriever.get_relevant_documents(input_query)
    model = get_bedrock_model(bedrock_runtime)

    chain = get_chain(model, prompt, retriever)

    response = None
    if get_region_cache_enabled(event):
        print(f"Region cache enabled")
        region_cache = get_region_cache_instance()
        region_cache_key = get_region_cache_key()

        cache_key = get_hash(input_query)
        return_obj = region_cache.hget(region_cache_key, cache_key)

        if not return_obj:
            response = chain.invoke(input_query)
            serialized = pickle.dumps(response)
            region_cache.hset(region_cache_key, cache_key, serialized)
        else:
            response = pickle.loads(return_obj)
    else:
        response = chain.invoke(input_query)

    contactId = get_contact_id(event)
    res = insert_llm_history(contactId, input_query, response, instruction)

    res = {
        "response": response,
        "query": input_query,
        "instruction": instruction,
    }
    print(res)
    # response['query'] = input_query
    return res


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
    cache = get_sementic_cache_instance(auth, bedrock_embeddings)
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


def get_sementic_cache_instance(auth, bedrock_embeddings):
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
    print(f"Embedding Id: {get_semantic_cache_embedding_model_id()}")
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


if __name__ == "__main__":
    response = main({"cache_enabled": False})


def lambda_handler(event, context):
    print(f"Event: {event}")
    req = event.get("requestContext")
    http = req.get("http")
    method = http.get("method")
    path = http.get("path")

    resp = {}
    if method == "POST" and path == "/chatbot/query":
        resp = main(event)

    return {
        'headers': {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": True,
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Method": "GET,POST,OPTIONS",
        },
        'statusCode': 200,
        'body': json.dumps(resp, default=str, ensure_ascii=False)
    }


logger = logging.getLogger(__file__)


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
                    logger.warning(
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
    """Load generations from json.

    Args:
        generations_json (str): A string of json representing a list of generations.

    Raises:
        ValueError: Could not decode json string to list of generations.

    Returns:
        RETURN_VAL_TYPE: A list of generations.

    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    try:
        results = json.loads(generations_json)
        return [Generation(**generation_dict) for generation_dict in results]
    except json.JSONDecodeError:
        raise ValueError(
            f"Could not decode json to list of generations: {generations_json}"
        )


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
