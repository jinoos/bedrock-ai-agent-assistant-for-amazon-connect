import logging
import uuid
import datetime

from langchain_community.cache import OpenSearchSemanticCache

from utils.ssm import parameter_store

import boto3
import json
import pickle
import hashlib

from opensearchpy import AWSV4SignerAsyncAuth
from redis import Redis

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.globals import set_llm_cache
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
        return get_param("/aaa/query")
    body_json = json.loads(body)
    return body_json.get('query', None)


def get_contact_id(event):
    body = event.get("body", None)
    if body is None:
        return get_param("/aaa/query")
    body_json = json.loads(body)
    return body_json.get('contactId', None)


def get_bedrock_region():
    return get_param('/aaa/bedrock_region')


def get_bedrock_kb_id():
    return get_param('/aaa/bedrock_kb_id')


def get_bedrock_kb_result_count():
    return get_param_int('/aaa/bedrock_kb_result_count')


def get_bedrock_model_id():
    return get_param('/aaa/bedrock_model_id')



def get_semantic_cache_embedding_model_id():
    return get_param('/aaa/semantic_cache_embedding_model_id')


def get_region_cache_endpoint():
    return get_param('/aaa/region_cache_endpoint')


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
    return get_param_float('/aaa/temperature')


def get_top_k():
    return get_param_int('/aaa/top_k')


def get_top_p():
    return get_param_int('/aaa/top_p')


def get_max_tokens():
    return get_param_int('/aaa/max_tokens')


def get_region_cache_key():
    return get_param('/aaa/region_cache_key')


def get_ddb_llm_history():
    return get_param('/aaa/dynamodb_llm_history')


def get_ddb_contact_summary():
    return get_param('/aaa/dynamodb_contact_summary')


def get_query_instruction():
    return get_param('/aaa/query_instruction')


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
    return get_config_bool(event, '/aaa/semantic_cache_enabled')


def get_semantic_cache_score():
    return get_param('/aaa/semantic_cache_score')


def get_semantic_cache_endpoint():
    return get_param('/aaa/semantic_cache_endpoint')


def get_region_cache_enabled(event):
    return get_config_bool(event, '/aaa/region_cache_enabled', False)


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
    instruction = get_query_instruction()
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
