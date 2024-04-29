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

import uuid


UPLOAD_FOLDER = '/workspace/uploads'
OUTPUT_FOLDER = '/workspace/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/test', methods=['GET'])
def test():
    # token1=create_token()
    # print(token1)
    # valid=validate_token(token1)
    # print(valid)

    return jsonify({'cu':'dd'}), 200

#이미지 파일 저장 후 customized model 생성
@app.route('/generateModel', methods=['POST'])
def upload_images():

    request_id = request.form.get('id')
    if not request_id:
        return jsonify({'error': 'No ID provided in the request'}), 400
    print(request_id)

    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    else:
        print("image file exist")

    files = request.files.getlist('files')

    #파일 패쓰 생성
    uploaded_filenames = []
    file_key = str(uuid.uuid4())
    print("uuid path 생성:",file_key)

    #이미지 파일 저장
    for index, file in enumerate(files, start=1):
        if file and allowed_file(file.filename):
            upload_folder_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file_key)
            os.makedirs(upload_folder_path, exist_ok=True)
            
            # Save the file with a numerical filename (1, 2, 3, ...)
            filename = f"{index}.png"
            filepath = os.path.join(upload_folder_path, filename)
            file.save(filepath)

            uploaded_filenames.append(f"uploads/{file_key}/{filename}")
    
    output_folder_path = os.path.join(current_app.config['OUTPUT_FOLDER'],file_key)
    

    # #이미지에 대한 txt파일 생성
    # catption_res = gen_tagger(file_key)
    # if catption_res == False:
    #     print("태그 생성 실패")
    #     return jsonify({'error': 'Tag creation failed'}), 400
    # else:
    #     print("태그 생성 성공")
    
    
    # #customized 모델 생성
    # train_res = train_model(file_key)
    # if(train_res==False):
    #     print("모델 생성 실패")
    #     return jsonify({'error': 'customized model creation failed'}), 400

    return jsonify({'user_id':request_id,'modelName':file_key,'modelPath': output_folder_path, 'result': "Customized Model Creation Completed"}), 200
    


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#@app.route('/train/<key>', methods=['GET'])
# 특정 디렉토리에 대해 이미지 태그 생성
def train_model(key):
    script_path = '/workspace/train/entrypoint.sh'
    try:
        #result=subprocess.run(['/bin/bash', script_path, "stabilityai/stable-diffusion-2-1",key,key])
        result=subprocess.run(['/bin/bash', script_path, "/workspace/workspace/model/magic.safetensors",key,key])
        if result.returncode != 0:
            return False
    except subprocess.CalledProcessError as e:
        return False
    return True


def gen_tagger(folder_name : str):    
    #사용가능한 모델확인
    # response = requests.get("http://127.0.0.1:7860/tagger/v1/interrogators")
    # print(response.status_code)
    
    sd_url = 'http://203.252.161.105:7860/tagger/v1/interrogate'
    # sd_url = 'https://92dba0dbfb47e03d96.gradio.live/tagger/v1/interrogate'
    model = 'wd14-convnext'
    threshold = 0.35
    base_path = '/workspace/uploads/'+ folder_name


    # print(os.listdir('/workspace/uploads'))
    # print(base_path)
    # print(os.listdir(base_path))
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
        # print('response sent' +  file )
        # print('data :  '+  data['model'])
        
        try:
            response = requests.post(sd_url, json=data)
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}")
            return False
        
        json_data=response.text
        tagger_infor = json.loads(json_data)

        txt_path  = f"{base_path}/{file.split('.')[0]}.txt"
        # print('============')
        # print(tagger_infor)
        # print('============')
        with open(txt_path, 'w') as f:
            for key in tagger_infor['caption'].keys():
                # print(key)
                f.write(f'{key}, ')
    return True
                
                

def is_image_file(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    extension = os.path.splitext(file_path)[1].lower()
    return extension in image_extensions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug= True)
