from flask import Blueprint, Flask, jsonify, request, current_app
import subprocess
import os

from http.client import HTTPException
import io

import json
import requests
import base64




from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
import mysql.connector


UPLOAD_FOLDER = "/workspace/uploads"
OUTPUT_FOLDER = "/workspace/output"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

db = {"user": "redhare", "password": "redhare!", "host": "host.docker.internal", "port": 3306, "database": "cmg"}

DB_URL = (
    f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8"
)
print(DB_URL)
engine = create_engine(DB_URL, pool_pre_ping=True)

try:
    with engine.connect() as connection:
        print("DB연결 완료")
except Exception as e:
    print(f"DB 실패: {str(e)}")

Session = sessionmaker(bind=engine)
Base = declarative_base()


# User 모델
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)



# CM 모델
class customized_model(Base):
    __tablename__ = "customized_model"

    cid = Column(Integer, primary_key=True)
    uid = Column(Integer)
    independentKey = Column(String)


# processing
class cm_processing(Base):
    __tablename__ = "cm_processing"

    pid = Column(Integer, primary_key=True)
    independentKey = Column(String)
    status = Column(String)


@app.route("/DB_check", methods=["GET"])
def connect_DB():
    with Session() as session:
        try:
            users = session.query(User).all()
            user_data = [{"id": user.id,  "email": user.email} for user in users]

            # CM 모델 테이블 조회
            cm_models = session.query(customized_model).all()
            model_data = [{"cid": model.cid, "uid": model.id, "independentKey": model.independentKey} for model in cm_models]

            # Session 모델 테이블 조회
            cm_sessions = session.query(cm_processing).all()
            session_data = [
                {"pid": session.pid, "independentKey": session.independentKey, "status": session.status} for session in cm_sessions
            ]

            response_data = {
                "users": user_data,
                "customized_models": model_data,
                "cm_processing_sessions": session_data,
            }
            return jsonify(response_data), 200

        except Exception as e:
            print("실패", e)



# 사용자별 로라 조회
@app.route("/show_models", methods=["GET"])
def show_models():
    uid = request.form.get("uid")
    print("username", uid)
    if not uid:
        return jsonify({"error": "No Name provided in the request"}), 400

    with Session() as session:
        try:
            # 요청받은 ID와 일치하는 independentKey를 가진 customized_model 테이블의 행들을 조회
            all_models = session.query(customized_model).filter_by(uid=uid).all()

            if all_models:
                model_info = []
                for customizedModels in all_models:
                    print(customizedModels.independentKey)
                    model_info.append({"model_name": f"{uid}_{customizedModels.independentKey}"})
                return jsonify({"models": model_info}), 200
            else:
                print("모델 없음")
                return jsonify({"error": "No models"}), 400

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({"error": e}), 400


# 로라 삭제
@app.route("/delete_model", methods=["DELETE"])
def delete_file():
    independentKey = request.form.get("independentKey")
    print("username", independentKey)
    if not independentKey:
        return jsonify({"error": "No Name provided in the request"}), 400

    with Session() as session:
        try:
            find_model = session.query(customized_model).filter_by(independentKey=independentKey).first()
            find_process = session.query(cm_processing).filter_by(independentKey=independentKey).first()

            if find_model:
                flie_name = f"{find_model.id}_{independentKey}"
                print(flie_name)

                session.delete(find_process)
                session.delete(find_model)
                session.commit()

                directory_path = "/workspace/output/"
                file_path = os.path.join(directory_path, flie_name + ".safetensors")
                print(file_path)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    return jsonify({"message": f"Model {independentKey} and associated file deleted successfully"}), 200
                else:
                    return jsonify({"error": f"File {independentKey} not found in directory"}), 404
            else:
                return jsonify({"error": f"Model with independentKey {independentKey} not found in database"}), 404

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({"error": e}), 400


# 이미지 파일 저장 후 customized model 생성
@app.route("/generateModel", methods=["POST"])
def upload_images():

    request_id = request.form.get("id")
    if not request_id:
        return jsonify({"error": "No ID provided in the request"}), 400
    print(request_id)

    # id 존재여부 검사
    check_id = exist_id(request_id)
    if not check_id:
        return jsonify({"error": "ID not exist"}), 400

    request_independentKey = request.form.get("independentKey")
    if not request_independentKey:
        return jsonify({"error": "No independentKey provided in the request"}), 400
    print(request_independentKey)

    # independentKey 중복검사
    duplicate_check = duplicate_independentKey(request_independentKey)
    if duplicate_check:
        return jsonify({"error": "The independentKey is duplicated"}), 400

    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    else:
        print("image file exist")

    files = request.files.getlist("files")

    # 파일 패쓰 생성
    uploaded_filenames = []

    # 모델 이름 생성
    file_key = request_id + "_" + request_independentKey

    print("independentKey path 생성:", file_key)

    # 이미지 파일 저장
    for index, file in enumerate(files, start=1):
        if file and allowed_file(file.filename):
            upload_folder_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file_key)
            os.makedirs(upload_folder_path, exist_ok=True)

            # Save the file with a numerical filename (1, 2, 3, ...)
            filename = f"{index}.png"
            filepath = os.path.join(upload_folder_path, filename)
            file.save(filepath)

            uploaded_filenames.append(f"uploads/{file_key}/{filename}")

    output_folder_path = os.path.join(current_app.config["OUTPUT_FOLDER"], file_key)

    # 이미지에 대한 txt파일 생성
    catption_res = gen_tagger(file_key)
    if catption_res == False:
        print("태그 생성 실패")
        return jsonify({"error": "Tag creation failed"}), 400
    else:
        print("태그 생성 성공")

    # 모델 생성중
    cm_processing_status(request_independentKey, 0)

    # customized 모델 생성
    train_res = train_model(file_key)
    if train_res == False:
        cm_processing_status(request_independentKey, 2)  # 모델 생성 실패
        print("모델 생성 실패")
        return jsonify({"error": "customized model creation failed"}), 400

    cm_processing_status(request_independentKey, 1)  # 모델 생성 완료
    save_model_db(request_id, request_independentKey)

    model_name = file_key + ".safetensors"
    return (
        jsonify(
            {
                "uid": request_id,
                "model_name": model_name,
                "model_path": output_folder_path,
                "result": "Customized Model Creation Completed",
            }
        ),
        200,
    )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/train/<key>', methods=['GET'])
# 특정 디렉토리에 대해 이미지 태그 생성
def train_model(key):
    script_path = "/workspace/train/entrypoint.sh"
    try:
        # result=subprocess.run(['/bin/bash', script_path, "stabilityai/stable-diffusion-2-1",key,key])
        result = subprocess.run(["/bin/bash", script_path, "/workspace/workspace/model/magic.safetensors", key, key])
        if result.returncode != 0:
            return False
    except subprocess.CalledProcessError as e:
        return False
    return True


def gen_tagger(folder_name: str):
    # 사용가능한 모델확인
    # response = requests.get("http://127.0.0.1:7860/tagger/v1/interrogators")
    # print(response.status_code)

    sd_url = "http://203.252.161.105:7860/tagger/v1/interrogate"
    # sd_url = 'https://92dba0dbfb47e03d96.gradio.live/tagger/v1/interrogate'
    model = "wd14-convnext.v1"
    threshold = 0.35
    base_path = "/workspace/uploads/" + folder_name

    # print(os.listdir('/workspace/uploads'))
    # print(base_path)
    # print(os.listdir(base_path))
    for file in os.listdir(base_path):
        file_path = f"{base_path}/{file}"

        if not is_image_file(file):
            continue  #  if not image file, continue for loop

        # base64 encode
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            encoded_image = base64.b64encode(image_data).decode("utf-8")

        data = {
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

        json_data = response.text
        tagger_infor = json.loads(json_data)

        txt_path = f"{base_path}/{file.split('.')[0]}.txt"
        # print('============')
        # print(tagger_infor)
        # print('============')
        with open(txt_path, "w") as f:
            for key in tagger_infor["caption"].keys():
                print(key)
                f.write(f"{key}, ")
    return True


def is_image_file(file_path):
    image_extensions = [".jpg", ".jpeg", ".png"]
    extension = os.path.splitext(file_path)[1].lower()
    return extension in image_extensions


# 파일 이름 가져오기
def find_files_with_username(directory, username):
    files_with_username = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and username in filename:
            files_with_username.append(filepath)

    return files_with_username


# 모델 상태 업데이트
def cm_processing_status(request_independentKey, status_num):
    with Session() as session:
        try:
            if status_num == 0:  # 생성중
                change_processing = cm_processing(independentKey=request_independentKey, status="모델 생성중")
                session.add(change_processing)
                print("모델 생성중 업데이트")
            else:
                change_processing = session.query(cm_processing).filter_by(independentKey=request_independentKey).first()
                if status_num == 1:
                    change_processing.status = "모델 생성 완료"
                    print("모델 생성 완료 업데이트")
                else:
                    change_processing.status = "모델 생성 실패"
                    print("모델 생성 실패 업데이트")
            session.commit()

        except Exception as e:
            print(f"레코드 추가 중 오류 발생: {str(e)}")
            session.rollback()


# 모델 DB 저장
def save_model_db(request_id, request_independentKey):
    with Session() as session:
        try:
            save_model = customized_model(user_id=request_id, independentKey=request_independentKey)
            session.add(save_model)
            session.commit()
        except Exception as e:
            print("실패", e)


# independentKey 중복검사
def duplicate_independentKey(request_independentKey):
    with Session() as session:
        try:
            existing_model = session.query(customized_model).filter_by(independentKey=request_independentKey).first()
            print("중복검사", request_independentKey)

            if existing_model:
                print(f"Error: independentKey '{request_independentKey}' already exists in the database.")
                return True
            else:
                return False
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({"error": e}), 400


# id 존재여부 검사
def exist_id(request_id):
    with Session() as session:
        try:
            existing_model = session.query(User).filter_by(fullName=request_id).first()
            print("존재검사", request_id)

            if existing_model:
                print(f"유저 존재")
                return True
            else:
                return False
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({"error": e}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
