from flask import Flask
import subprocess
import os

app = Flask(__name__)


@app.route('/train/<key>', methods=['GET'])
def train_model(key):
    # Call entrypoint.sh script from /app/sd-scripts/
    script_path = '/app/sd-scripts/entrypoint.sh'
    subprocess.run(['/bin/bash', script_path, "runwayml/stable-diffusion-v1-5",key,key])
    return f'Training for key {key} started.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
