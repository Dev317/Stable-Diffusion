import os
from flask import Flask, request
from flask_cors import CORS
from utils import prompt_to_img, predict
from io import BytesIO
import json
import boto3
import uuid
import tensorflow as tf

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

BUCKET_NAME = os.environ.get('BUCKET_NAME', 'pil-test-image')


@app.route('/api/audio2image', methods=['POST'])
def get_image():
    dirname = 'upload'
    save_path = os.path.join(dirname, "temp.wav")
    request.files['wav_file'].save(save_path)

    model = tf.keras.models.load_model('./weights.best.basic_cnn.hdf5')
    category = predict(save_path, model)

    prompt = f"Create an image of the following category: {category}"

    plt_img = prompt_to_img(prompts=prompt, height=512, width=512, num_inference_steps=30)[0]
    buffered = BytesIO()
    plt_img.save(buffered, format="JPEG")
    buffered.seek(0)

    session = boto3.Session()
    s3 = session.client("s3")
    image_title = uuid.uuid4().hex.upper()[0:6]
    s3.put_object(Body=buffered, Bucket=BUCKET_NAME, Key=f"{image_title}.jpeg")

    response_body = json.dumps({
        "data": {
            "image": f"https://{BUCKET_NAME}.s3.us-east-2.amazonaws.com/{image_title}.jpeg",
            "image_title": image_title,
            "category": category,
            "width": 512,
            "height" : 512,
        }
    })

    return response_body, 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
