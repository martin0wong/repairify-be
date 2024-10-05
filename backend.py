from flask import Flask, request, jsonify
import config
import re
import base64
from langchain_aws import ChatBedrock

# Create a Flask app
app = Flask(__name__)

# Set up the Bedrock model using LangChain
llama_model = ChatBedrock(
    model_name="us.meta.llama3-2-90b-instruct-v1:0",
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_DEFAULT_REGION,
)

def is_base64_image(data):
    """Check if the provided data is a base64 encoded image."""
    return bool(re.match(r"^data:image/.+;base64,", data))

def decode_base64_image(data):
    """Decode a base64 image and return the binary data."""
    header, encoded = data.split(",", 1)
    return base64.b64decode(encoded)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        if is_base64_image(prompt):
            # Decode the base64 image
            image_data = decode_base64_image(prompt)
            # Process the image through Bedrock
            message = llama_model(image_data)
        else:
            # Process the text prompt through Bedrock
            message = llama_model(prompt)

        return jsonify({'message': message}), 200

    except Exception as e:
        return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
