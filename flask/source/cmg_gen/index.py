import os
import random
import string
from flask import Blueprint, jsonify, request, send_from_directory, current_app
import logging 
import asyncio
import requests


bp_index = Blueprint(name="index", import_name=__name__, url_prefix="")


UPLOAD_FOLDER = '/workspace/uploads'
MODEL_OUTPUT_FOLDER = '/workspace/model_output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# [".png", ".jpg", ".jpeg", ".webp", ".bmp"] 


@bp_index.route('/test', methods=['GET'])
def test():
    return jsonify({'cu':'dd'})

@bp_index.route('/upload', methods=['POST'])
async def upload_images():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')

    uploaded_filenames = []
    file_key = generate_random_string()
    for index, file in enumerate(files, start=1):
        if file and allowed_file(file.filename):

            # Save the uploaded file to the UPLOAD_FOLDER with the key as a subdirectory
            upload_folder_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file_key)
            os.makedirs(upload_folder_path, exist_ok=True)
            
            # Save the file with a numerical filename (1, 2, 3, ...)
            filename = f"{index}.png"
            filepath = os.path.join(upload_folder_path, filename)
            file.save(filepath)

            uploaded_filenames.append(f"uploads/{file_key}/{filename}")

    # asyncio.create_task(send_get_request(file_key))
    requests.get("http://train:4000/train/"+file_key)
    # Generate a download URL for the user
    download_url = f"http://203.252.166.213/output/{file_key}"

    return jsonify({'download_url': download_url})

@bp_index.route('/output/<model_key>', methods=['GET'])
def download_model(model_key):
    model_path = os.path.join(current_app.config['MODEL_OUTPUT_FOLDER'], f'{model_key}/{model_key}.safetensors')
    model_folder_path = os.path.join(current_app.config['MODEL_OUTPUT_FOLDER'], model_key)


    if os.path.exists(model_path):
        return send_from_directory(current_app.config['MODEL_OUTPUT_FOLDER'], f'{model_key}/{model_key}.safetensors', as_attachment=True)
    elif os.path.exists(model_folder_path):
        return jsonify({'message': 'Generating'})
    else:
        return jsonify({'error': 'Model not found'}), 404



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

async def send_get_request(key):
    # 외부 서버의 URL 설정
    external_url = f''

    try:
        # GET 요청 보내기
        logging.info(external_url)
        response = requests.get("http://train:4000/train/"+key)

        # 응답 확인
        if response.status_code == 200:
            logging.info(response.text)
            return f'Success! Response: {response.text}'
        else:
            logging.info(response.status_code)
            return f'Error! Status code: {response.status_code}'

    except Exception as e:
        return f'An error occurred: {str(e)}'