
import os
from flask import Flask, request
from utils import prompt_to_img
from io import BytesIO
import json
import boto3
import uuid

app = Flask(__name__)

@app.route('/api/prompt2image', methods=['POST'])
def get_image():
    request_data = request.get_json()
    plt_img = prompt_to_img(prompts=request_data['prompt'], height=512, weight=512, num_inference_steps=30)[0]
    buffered = BytesIO()
    plt_img.save(buffered, format="JPEG")
    buffered.seek(0)

    session = boto3.Session()
    s3 = session.client("s3")
    image_title = uuid.uuid4().hex.upper()[0:6]
    s3.put_object(Body=buffered, Bucket='pil-test-image', Key=f"{image_title}.jpeg")

    response_body = json.dumps({
        "data": {
            "img_title": image_title,
            "img_url": f"https://pil-test-image.s3.us-east-2.amazonaws.com/{image_title}.jpeg",
            "width": 512,
            "height" : 512,
        }
    })

    return response_body, 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
