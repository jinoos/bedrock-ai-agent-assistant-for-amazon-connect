import boto3


class ParameterStore(object):
    _clients = {}

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(ParameterStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, region_name="ap-northeast-2", **kwargs):
        self.region_name = region_name
        self.region(region_name, **kwargs)

    def _get_client(self):
        return self._clients[self.region_name]

    def region(self, region_name, **kwargs):
        self.region_name = region_name
        if region_name not in self._clients:
            self._clients[region_name] = boto3.client('ssm', region_name=region_name, **kwargs)
        return self

    def put_param(self, key, value, dtype="String", overwrite=False, enc=False):
        # Specify the parameter name, value, and type
        if enc: dtype = "SecureString"

        # Put the parameter
        return self._get_client().put_parameter(
            Name=key,
            Value=value,
            Type=dtype,
            Overwrite=overwrite  # Set to True if you want to overwrite an existing parameter
        )

    def get_param(self, key, enc=False):
        try:
            response = self._get_client().get_parameters(
                Names=[key, ],
                WithDecryption=enc
            )
            return response['Parameters'][0]['Value']
        except Exception as e:
            return None

    def get_all_params(self, ):
        response = self._get_client().describe_parameters(MaxResults=50)
        return [dicParam["Name"] for dicParam in response["Parameters"]]

    def delete_param(self, listParams):
        return self._get_client().delete_parameters(
            Names=listParams
        )


def __repr__(self):
    return f"ParameterStore(name={self.name}, value={self.value})"
