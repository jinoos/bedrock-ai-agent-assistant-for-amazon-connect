import os
import json
import boto3


# region = os.environ['bedrock_region']
# model_id = os.environ['bedrock_kb_id']
# model_arn = os.environ['bedrock_model_arn']

def lambda_handler(event, context):
    bedrock = boto3.client('bedrock-agent-runtime', region_name=get_bedrock_region())

    session_id = event.get('sessionId', None)

    prompt = generate_prompt(get_agent_query(event))
    print(f"Prompt: {prompt}")
    input_data = {
        'input': {
            'text': prompt
        },
        'retrieveAndGenerateConfiguration': {
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': get_bedrock_kb_id(),
                'modelArn': get_bedrock_model_arn()
            },
            'type': 'KNOWLEDGE_BASE'
        }
    }

    # SessionId가 있는 경우 추가
    if session_id:
        input_data['sessionId'] = session_id

    print(f"InputData: {input_data}")
    response = bedrock.retrieve_and_generate(**input_data)
    print(f"Response: {response}")
    return response


def generate_prompt(query):
    return f"""
당신은 고도로 훈련된 AI 어시스턴트 입니다.
당신의 임무는 상담원이 문의한 내용을 바탕으로, 상담원 인식하고, 고객응답에 활용되기 쉽도록 간결하고 명확한 답변을 생성 합니다.
순서가 있는 동작이 포함된 답변은, 단계별로 라인을 구분하고 순서를 표시 해 주세요. 

답변을 할때는 아래의 기준을 사용하세요.
    언어: 한국어로 답변하세요.
    객관성: 개인적인 해석이나 판단을 배제하고, Knowledge Base의 내용을 참고하여 답변합니다.
    정확성: 날짜, 시간, 이름, 숫자, 장소 등의 구체적인 정보를 정확하게 발견하고 전달합니다.
    체계성: 정보를 논리적이고 구조화된 방식으로 제시하세요. 
    완전성: 중요한 정보가 누락되지 않도록 주의하세요. 
    가독성: 모든 출력에서 적절한 띄어쓰기를 사용하여 가독성을 높이세요.

상담원의 문의 내용은 다음과 같습니다.
---------------------
{query}
    """


def get_agent_query(event):
    env_query = os.environ['query']
    if env_query:
        return env_query
    else:
        return event['query']


def get_bedrock_region():
    default_bedrock_region = 'us-west-2'
    region = os.environ['bedrock_region']
    if not region:
        region = default_bedrock_region

    return region


def get_bedrock_kb_id():
    default_bedrock_kb_id = 'DGWRG5M6CE'
    model_id = os.environ['bedrock_kb_id']
    if not model_id:
        model_id = default_bedrock_kb_id

    return model_id


def get_bedrock_model_arn():
    default_bedrock_model_arn = 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
    model_arn = os.environ['bedrock_model_arn']
    if not model_arn:
        model_arn = default_bedrock_model_arn

    return model_arn
