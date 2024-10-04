import datetime
import uuid

import boto3


class LLmDdbHistoryKey:
    def __init__(self, client, key: str):
        self.client = client
        self.key = key

# Status = QUERIED, ANSWERED

class LlmDdbHistory:
    def __init__(self, endpoint_url: str, session: boto3.Session = boto3.Session()):
        self.context = None
        self.instruction = None
        self.query = ""
        self.endpoint_url = endpoint_url
        self.session = session
        self.ddbClient = self.session.client('dynamodb')

    def query(self, contactId:str, query: str, instuction: str = "", context: str = "") -> LLmDdbHistoryKey:
        self.query = query
        self.instruction = instuction
        self.context = context

        lq = {
            'Id': uuid.uuid4(),
            'ContactId': contactId,
            'QueriedDate': datetime.datetime.utcnow(),
            'Query': query,
            'Instruction': instuction,
            'Context': context,
            'AnsweredDate': "",
            'Answer': "",
        }

        key = None
        return LLmDdbHistoryKey(key, self.ddbClient)

    def get_history(self):
        return self.history

    def set_llm(self, llm):
        self.llm = llm

    def set_history(self, history):
        self.history = history

    def __str__(self):
        return "LlmHistory(llm={}, history={})".format(self.llm, self.history)
