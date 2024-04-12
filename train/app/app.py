from flask import Flask
import subprocess
import os

from http.client import HTTPException
import io

from io import BytesIO
import json
import requests
import base64



app = Flask(__name__)


@app.route('/train/<key>', methods=['GET'])
def train_model(key):
    # 특정 디렉토리에 대해 이미지 태그 생성
    catption_res = gen_lora(key)
    # # Call entrypoint.sh script from /sd-scripts/
    # script_path = '/sd-scripts/entrypoint.sh'
    # subprocess.run(['/bin/bash', script_path, "stabilityai/stable-diffusion-2-1",key,key])
    # return f'Training for key {key} started.'
    return f'Image tagging started {key}'


def gen_lora(folder_name : str):    
    #사용가능한 모델확인
    # response = requests.get("http://127.0.0.1:7860/tagger/v1/interrogators")
    # print(response.status_code)
    
    # sd_url = 'http://203.252.161.105:7860/tagger/v1/interrogate'
    sd_url = 'https://92dba0dbfb47e03d96.gradio.live/tagger/v1/interrogate'
    model = 'wd14-convnext'
    threshold = 0.35
    base_path = '/workspace/workspace/images/'+ folder_name #train/images/asdfasdf/01.png

    # print(os.getcwd())
    # print(os.listdir(os.getcwd()))
    # print(os.listdir('../'+os.getcwd()))
    print(os.listdir('/workspace/workspace/images'))
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
