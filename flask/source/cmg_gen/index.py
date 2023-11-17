import os
import random
import string
from flask import Blueprint, jsonify, request, send_from_directory

bp_index = Blueprint(name="index", import_name=__name__, url_prefix="")


UPLOAD_FOLDER = '/workspace/uploads'
MODEL_OUTPUT_FOLDER = '/workspace/model_output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}



@bp_index.route(rule="/", methods=["GET"])
def index():
    return "Hello Flask"


@bp_index.route('/output/<model_key>', methods=['GET'])
def download_model(model_key):
    model_path = os.path.join(bp_index.config['MODEL_OUTPUT_FOLDER'], f'model_{model_key}.h5')

    if os.path.exists(model_path):
        return send_from_directory(bp_index.config['MODEL_OUTPUT_FOLDER'], f'model_{model_key}.h5', as_attachment=True)
    else:
        return jsonify({'error': 'Model not found'}), 404


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


@bp_index.route('/upload', methods=['POST'])
def upload_images():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')

    uploaded_filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = generate_random_string() + '.png'
            filepath = os.path.join(bp_index.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_filenames.bp_indexend(filename)

    # Call the script inside the Docker container to train the model
    train_model_command = "docker exec easy-lora-train python /workspace/train_model.py"
    os.system(train_model_command)

    # Generate a random key for the trained model
    model_key = generate_random_string()
    model_path = os.path.join(bp_index.config['MODEL_OUTPUT_FOLDER'], f'model_{model_key}.h5')

   
    # Generate a download URL for the user
    download_url = f"http://localhost:5000/output/{model_key}"

    return jsonify({'download_url': download_url})


