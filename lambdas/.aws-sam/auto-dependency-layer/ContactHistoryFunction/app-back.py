import datetime
import json
import uuid

import boto3
import dateutil
from aws_lambda_powertools import Tracer, Logger
from aws_lambda_powertools.event_handler import APIGatewayRestResolver, CORSConfig
from aws_lambda_powertools.logging import correlation_paths
from boto3.dynamodb.conditions import Key
from dateutil.parser import parser

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


# datetime.datetime.now(datetime.timezone.utc)


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
    # chatData = get_s3_analysis_file(res)
    # if chatData:
    #     res['AnalysisData'] = json.loads(chatData)
    # print(f"S3 Data: {res}")

    return res


@app.get("/contact-history/llm_history/<contactId>/list")
def get_contact_llm_history(contactId: str):
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
        'AbsoluteTime': dateutil.parser.parse(aiResponse['AnsweredDate']).replace(tzinfo=None)
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
    res = ddbResource.Table(table).put_item(Item=lq)
    print(f"Insert LLM History: {res}")
    return res
