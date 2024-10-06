import boto3
import json
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-west-2")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

def format_request(prompt):
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }

def invoke_model(request):
    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(request))
        return json.loads(response["body"].read())
    except ClientError as e:
        return {'error': f"ClientError: {str(e)}"}
    except Exception as e:
        return {'error': str(e)}

def lambda_handler(event, context):
    text_prompt = event.get('text_prompt', '')
    intro = "You are an technician giving suggestions on how to treat electronics, give your suggestions in bullet points and keep it concise.\n Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.\n"
    
    if not text_prompt:
        return {
            'message': 'error: No text prompt provided'
        }
        
    text_prompt = intro + text_prompt
    request = format_request(text_prompt)
    model_response = invoke_model(request)

    if 'error' in model_response:
        return {
            'message': model_response
        }

    response_text = model_response["content"][0]["text"]
    return {'message': response_text}