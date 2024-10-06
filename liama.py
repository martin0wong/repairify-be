import json
import boto3
import re
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-west-2")
model_id = "us.meta.llama3-2-90b-instruct-v1:0"

def is_base64_image(data):
    """Check if the provided data is a base64 encoded image."""
    return bool(re.match(r"^data:image/.+;base64,", data))

def decode_base64_image(data):
    """Decode a base64 image and return the binary data."""
    header, encoded = data.split(",", 1)
    return base64.b64decode(encoded)

def lambda_handler(event, context):
    prompt = event.get('text_prompt', '')
    intro = "You are an technician giving suggestions on how to treat electronics, give your suggestions in bullet points and keep it concise.\n Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.\n"
    
    if not prompt:
        return {
            'message': {'error': 'No prompt provided'}
        }

    # Handle base64 image input
    if is_base64_image(prompt):
        image_data = decode_base64_image(prompt)
        # You may want to process the image data here, depending on your model's requirements
        image_data = intro + "The image data is given below, read the data and give feedback accordingly.\n" + image_data
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{image_data}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        prompt = intro + prompt
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 256,
        "temperature": 0.2
    }

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(native_request))
        model_response = json.loads(response['body'].read())

        # Extract and clean the message
        response_text = model_response.get("generation", "No response generated.").strip()
        return {'message': response_text}

    except ClientError as e:
        return {'message': f"Can't invoke '{model_id}'. Reason: {str(e)}"}
    except Exception as e:
        return {'message': str(e)}
