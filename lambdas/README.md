# Lambda functions
> 정리 중

## Prerequisites
### CFN Packaging
```shell
aws cloudformation package \
   --template-file template.yaml \
   --s3-bucket <S3-FOR-CFN-PACKAGE> \
   --s3-prefix <S3-PREFIX-FOR-CFN-PACKAGE> \
   --output-template-file template-out.yaml
```
### Deploy
```shell
aws cloudformation deploy \
   --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_NAMED_IAM \
   --template-file template-out.yaml \
   --stack-name <STACK-NAME> \
   --parameter-overrides <PARAMETERS>
# example
aws cloudformation deploy \
   --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_NAMED_IAM \
   --template-file template-out.yaml \
   --stack-name aaa-lambdas \
   --parameter-overrides \
   "BedrockKnowledgeBaseId=abcd" \
   "LambdaSGs=sg-04e05abbb2fcbd13e" \
   "LambdaSubnets=subnet-0fd4dc571bc0fd7b9,subnet-02bdaf284cd3018cf"
```
Parameters:
  - LambdaSubnets: `,` seperated Subnet Ids
    - Required
    - Example: `subnet-0fd4dc571bc0fd7b9,subnet-02bdaf284cd3018cf`
  - LambdaSGs: `,` Seperated Security Groups
    - Required
    - Example: `sg-04e05abbb2fcbd13e`
  - BedrockRegion:
    - Default: `us-west-2`
  - BedrockModelId:
    - Default: `anthropic.claude-3-haiku-20240307-v1:0`
  - BedrockModelIdSummary:
    - Default: `anthropic.claude-3-5-sonnet-20240620-v1:0`
  - BedrockKnowledgeBaseId:
  - SemanticCacheEnable:
    - Default: `False`
  - SemanticCacheEmbeddingModelId:
    - Default: `amazon.titan-embed-text-v2:0`
  - SemanticCacheVectorDBEndpoint:
    - Default: `''`
