from flask import Blueprint,Flask, jsonify, request, current_app
import subprocess
import os

from http.client import HTTPException
import io

from io import BytesIO
import json
import requests
import base64

import random
import string

UPLOAD_FOLDER = '/workspace/uploads'
MODEL_OUTPUT_FOLDER = '/workspace/model_output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
bp_index = Blueprint(name="index", import_name=__name__, url_prefix="")

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'cu':'dd'}), 200

#이미지 파일 저장 후 customized model 생성
@app.route('/generateModel', methods=['POST'])
def upload_images():

    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    else:
        print("file exist")

    files = request.files.getlist('files')

    uploaded_filenames = []
    file_key = generate_random_string()
    print(file_key)
    for index, file in enumerate(files, start=1):
        if file and allowed_file(file.filename):
            print(index,"번째 파일")
            # Save the uploaded file to the UPLOAD_FOLDER with the key as a subdirectory
            #print(app.config['UPLOAD_FOLDER'])
            upload_folder_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file_key)
            print(upload_folder_path)
            os.makedirs(upload_folder_path, exist_ok=True)
            print("생성")
            
            # Save the file with a numerical filename (1, 2, 3, ...)
            filename = f"{index}.png"
            filepath = os.path.join(upload_folder_path, filename)
            file.save(filepath)

            uploaded_filenames.append(f"uploads/{file_key}/{filename}")

    # asyncio.create_task(send_get_request(file_key))
    requests.get("http://train:4000/train/"+file_key)
    # # Generate a download URL for the user
    # download_url = f"http://203.252.161.106/output/{file_key}"

    return jsonify({'result': "성공"}), 200

def generate_random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/train/<key>', methods=['GET'])
def train_model(key):
    # 특정 디렉토리에 대해 이미지 태그 생성
    catption_res = gen_lora(key)
    # # Call entrypoint.sh script from /sd-scripts/
    #script_path = '/sd-scripts/entrypoint.sh'
    script_path = '/workspace/train/entrypoint.sh'
    subprocess.run(['/bin/bash', script_path, "stabilityai/stable-diffusion-2-1",key,key])
    # return f'Training for key {key} started.'
    return f'Image tagging started {catption_res}'


def gen_lora(folder_name : str):    
    #사용가능한 모델확인
    # response = requests.get("http://127.0.0.1:7860/tagger/v1/interrogators")
    # print(response.status_code)
    
    # sd_url = 'http://203.252.161.105:7860/tagger/v1/interrogate'
    sd_url = 'http://203.252.161.105:7860/tagger/v1/interrogate'
    # sd_url = 'https://92dba0dbfb47e03d96.gradio.live/tagger/v1/interrogate'
    model = 'wd14-convnext'
    threshold = 0.35
    #base_path = '/workspace/workspace/images/'+ folder_name #train/images/asdfasdf/01.png
    base_path = '/workspace/uploads/'+ folder_name #train/images/asdfasdf/01.png


    # print(os.listdir('/workspace/workspace/images'))
    print(os.listdir('/workspace/uploads'))
    print(base_path)
    print(os.listdir(base_path))
    for file in os.listdir(base_path):
        file_path =  f"{base_path}/{file}"
        
        if not is_image_file(file):  
            continue #  if not image file, continue for loop
        
        # base64 encode
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            encoded_image = base64.b64encode(image_data).decode("utf-8")

        data={
            "image": encoded_image,
            "model": model,
            "threshold": threshold,
        }
        print('response sent' +  file )
        print('data :  '+  data['model'])
        response = requests.post(sd_url, json=data)

        json_data=response.text
        tagger_infor = json.loads(json_data)

        txt_path  = f"{base_path}/{file.split('.')[0]}.txt"
        print('============')
        print(tagger_infor)
        print('============')
        with open(txt_path, 'w') as f:
            for key in tagger_infor['caption'].keys():
                print(key)
                f.write(f'{key}, ')
    return True
                
                

def is_image_file(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    extension = os.path.splitext(file_path)[1].lower()
    return extension in image_extensions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug= True)
