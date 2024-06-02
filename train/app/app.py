from flask import Blueprint, Flask, jsonify, request, current_app
import subprocess
import os
import cv2

from http.client import HTTPException
import io

import json
import requests
import base64
import threading


import uuid
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
import mysql.connector

from enum import Enum, auto

import time

UPLOAD_FOLDER = "/workspace/uploads"
OUTPUT_FOLDER = "/workspace/output"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

db = {"user": "redhare", "password": "redhare!", "host": "host.docker.internal", "port": 3306, "database": "cmg"}

DB_URL = f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8"
engine = create_engine(DB_URL, pool_pre_ping=True)

class cm_status(Enum):
    generating = "Model Generating"
    complete = "Model Generation Complete"
    fail = "Model Generation Failed"
    removed = "removed Model"


# try:
#     with engine.connect() as connection:
#         print("DB연결 완료")
# except Exception as e:
#     print(f"DB 실패: {str(e)}")

Session = sessionmaker(bind=engine)
Base = declarative_base()

def create_tables():
    Base.metadata.create_all(engine)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    refreshToken = Column(String(255), nullable=True)
    createdAt = Column(DateTime,nullable=True)
    updatedAt = Column(DateTime,nullable=True)


    def __repr__(self):
        return f"<User(email='{self.email}'>"
    
class CustomizedModel(Base):
    __tablename__ = 'Customized_Models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    independent_key = Column(String(255), nullable=False)
    thumbnail_image = Column(String(255), nullable=True)
    cm_nickname = Column(String(255), nullable=True)
    createdAt = Column(DateTime,nullable=True)
    updatedAt = Column(DateTime,nullable=True)

    def __repr__(self):
        return f'<CustomizedModel {self.independent_key}, {self.user_id}, {self.thumbnail_image}, {self.cm_nickname}>'

class CMProcessing(Base):
    __tablename__ = 'CMProcessings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    independent_key = Column(String(255), nullable=False)
    thumbnail_image = Column(String(255), nullable=True)
    file_number = Column(Integer, nullable=True)
    cm_nickname = Column(String(255), nullable=True)
    status = Column(String(255), nullable=False)
    createdAt = Column(DateTime,nullable=True)
    updatedAt = Column(DateTime,nullable=True)

    def __repr__(self):
        return f"<CMProcessing(id={self.id}, independent_key='{self.independent_key}', status='{self.status}')>"

@app.route("/ttt", methods=["GET"])
def test():
    #os.system('python new_trainning.py')
    command = ['python3','-u','new_trainning.py','-r']
    #result = subprocess.run(command,check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = subprocess.Popen(command,stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while result.poll() == None:
        out = result.stdout.readline()
        print(out, end='')

    # 프로세스 종료 후 남은 출력 읽기
    for out in result.stdout.readlines():
        print(out, end='')

@app.route("/DB_check", methods=["GET"])
def connect_DB():
    with Session() as session:
        try:
            users = session.query(User).all()
            user_data = [{'id': user.id, 'email': user.email} for user in users]

            cm_models = session.query(CustomizedModel).all()
            model_data = [{'id': model.id, 'user_id': model.user_id, 'independent_key': model.independent_key} for model in cm_models]

            cm_sessions = session.query(CMProcessing).all()
            session_data = [{'id': cm_session.id, 'independent_key': cm_session.independent_key , 'status': cm_session.status} for cm_session in cm_sessions]

            response_data = {
                'users': user_data,
                'customized_models': model_data,
                'cm_processing_sessions': session_data
            }
            return jsonify(response_data), 200

        except Exception as e:
            print("실패", e)
            return jsonify({'error': str(e)}), 500




# 사용자별 로라 조회
@app.route("/model", methods=["GET"])
def show_models():
    target_user_id = request.args.get("user_id")
    print(target_user_id)
    with Session() as session:
        try:
            # 요청받은 ID와 일치하는 uuid를 가진 customized_model 테이블의 행들을 조회
            all_models = session.query(CustomizedModel).filter_by(user_id=target_user_id).all()
            all_modeel_processings = session.query(CMProcessing).filter_by(user_id=target_user_id).all()
            print(all_modeel_processings)
            model_info = []
            for processing in all_modeel_processings:
                model_info.append({
                    "user_id" : target_user_id,
                    "status": processing.status,
                    "independent_key" : processing.independent_key,
                    "thumbnail_image" : processing.thumbnail_image,
                    "cm_nickname" : processing.cm_nickname,
                    "file_number" : processing.file_number,
                    "model_name" :  f"{target_user_id}_{processing.independent_key}",
                    "createdAt" : processing.createdAt.strftime('%Y.%m.%d %H:%M'),
                    "updateAt" : processing.updatedAt.strftime('%Y.%m.%d %H:%M'),
                }
                )
            return jsonify({"models": model_info}), 200
                # model_info.append({"processing": f"{target_user_id}-{processing.independent_key}-{processing.status}"})
            # else:
            #     for processing in all_modeel_processings:
            #          model_info.append({
            #             "user_id" : target_user_id,
            #             "status": processing.status,
            #             "independent_key" : processing.independent_key,
            #             "model_name" :  f"{target_user_id}_{processing.independent_key}"   
            #         }
            #         )
            #         # model_info.append({"processing": f"{target_user_id}-{processing.independent_key}-{processing.status}"})
                # return jsonify({"models": model_info}), 200

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({"error": e}), 400


# 로라 삭제
@app.route("/model", methods=["DELETE"])
def delete_file():
    independentKey = request.form.get("independent_key")
    userId = request.form.get("user_id")

    print("username", independentKey)
    if not independentKey:
        return jsonify({"error": "No Name provided in the request"}), 400
    
    cm_processing_status(userId,independentKey,3)

    return jsonify({'independentKey':independentKey,'userId':userId}), 200

# 이미지 파일 저장 후 customized model 생성
@app.route("/model", methods=["POST"])
def model_generation():
    data=request.form
    user_id = data.get("user_id")
    request_independentKey = data.get("independent_key")
    thumbnail_image = data.get("thumbnail_image")
    file_number = data.get("file_number") # 추가됨
    
    cm_nickname = data.get("cm_nickname")
    
    check_id = exist_id(user_id)
    if not check_id:
        return jsonify({"error": "ID not exist"}), 400

    
    if not request_independentKey:
        return jsonify({"error": "No independentKey provided in the request"}), 400
    print(request_independentKey)

    # independentKey 중복검사
    duplicate_check = duplicate_independentKey(request_independentKey)
    if duplicate_check:
        print("duplicate_independentKey")
        return jsonify({"error": "The independentKey is duplicated"}), 400

    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    else:
        print("image file exist")

    files = request.files.getlist("files")
    
    # 파일 패쓰 생성
    uploaded_filenames = []

    # 모델 이름 생성
    file_key = user_id + "_" + request_independentKey

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
    

    
    thread = threading.Thread(target=upload_images,args = (user_id,request_independentKey,file_key,thumbnail_image,cm_nickname,file_number))
    thread.start()
    return jsonify({'message': "result"}), 200



@app.route('/model/video', methods=['POST'])
def upload_video():
    data=request.form
    user_id = data.get("user_id")
    request_independentKey = data.get("independent_key")
    thumbnail_image = data.get("thumbnail_image")
    cm_nickname = data.get("cm_nickname")
    interval_str = data.get("interval")
    
    
    check_id = exist_id(user_id)
    if not check_id:
        return jsonify({"error": "ID not exist"}), 400

    
    if not request_independentKey:
        return jsonify({"error": "No independentKey provided in the request"}), 400
    print(request_independentKey)

    # independentKey 중복검사
    duplicate_check = duplicate_independentKey(request_independentKey)
    if duplicate_check:
        print("duplicate_independentKey")
        return jsonify({"error": "The independentKey is duplicated"}), 400


    
    # 파일 패쓰 생성
    uploaded_filenames = []

    # 모델 이름 생성
    file_key = user_id + "_" + request_independentKey

    print("independentKey path 생성:", file_key)

    # 비디오 저장장
    # video 변수는 FileStorage 객체입니다.
    video = request.files['video']
    # FileStorage 객체에서 안전한 파일 이름을 추출합니다.
    filename = "video" + file_key

    # 추출된 파일 이름과 업로드 폴더 경로를 결합합니다.
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # 결합된 경로에 파일을 저장합니다.
    video.save(video_path)
    
    # 동영상 파일을 열기
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    interval = int(interval_str)
    uploaded_filenames = []

    # 프레임을 추출하고 파일로 저장
    while success:
        if count % interval == 0:
            # 업로드 폴더 경로 설정
            upload_folder_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file_key)
            os.makedirs(upload_folder_path, exist_ok=True)

            # 파일 이름 설정 (1, 2, 3, ...)
            filename = f"{count // interval + 1}.png"
            filepath = os.path.join(upload_folder_path, filename)

            # 이미지 파일로 저장
            cv2.imwrite(filepath, image)
            uploaded_filenames.append(f"uploads/{file_key}/{filename}")

        success, image = vidcap.read()
        count += 1

    output_folder_path = os.path.join(current_app.config["OUTPUT_FOLDER"], file_key)

    # 이미지에 대한 txt파일 생성
    

    
    thread = threading.Thread(target=upload_images,args = (user_id,request_independentKey,file_key,thumbnail_image,cm_nickname,0))
    thread.start()
    return jsonify({'message': "result"}), 200



    
def upload_images(user_id,request_independentKey,file_key,thumbnail_image,cm_nickname,file_number):
    
    start_time = time.time()
    # 모델 생성중
    cm_processing_status(user_id,request_independentKey, 0,thumbnail_image,cm_nickname,file_number)

    with app.app_context():
        catption_res = gen_tagger(file_key)
        if catption_res == False:
            cm_processing_status(user_id,request_independentKey, 2,thumbnail_image,cm_nickname)  # 모델 생성 실패
            print("태그 생성 실패")
            return jsonify({"error": "Tag creation failed"}), 400
        else:
            print("태그 생성 성공")


        # customized 모델 생성
        train_res = train_model(file_key)
        if train_res == False:
            cm_processing_status(user_id,request_independentKey, 2,thumbnail_image,cm_nickname)  # 모델 생성 실패
            print("모델 생성 실패")
            return jsonify({"error": "customized model creation failed"}), 400

        cm_processing_status(user_id,request_independentKey,1,thumbnail_image,cm_nickname)#모델 생성 완료
        save_model_db(user_id,request_independentKey,thumbnail_image,cm_nickname)

        model_name=file_key+'.safetensors'

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"실행 시간: {elapsed_time}초")
    
    # return jsonify({'user_id':user_id,'model_name':model_name,'model_path': output_folder_path, 'result': "Customized Model Creation Completed"}), 200
    

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/train/<key>', methods=['GET'])
# 특정 디렉토리에 대해 이미지 태그 생성
def train_model(key):
    script_path = "/workspace/train/entrypoint.sh"
    #command = ['python3','new_trainning.py',key]
    try:
        # result=subprocess.run(['/bin/bash', script_path, "stabilityai/stable-diffusion-2-1",key,key])
        result = subprocess.run(["/bin/bash", script_path, "/workspace/workspace/model/magic.safetensors", key, key])
        #result = subprocess.run(command,check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        if result.returncode != 0:
            return False
    except subprocess.CalledProcessError as e:
        print("Error",e)
        return False
    return True


def gen_tagger(folder_name: str):
    # 사용가능한 모델확인
    # response = requests.get("http://127.0.0.1:7860/tagger/v1/interrogators")
    # print(response.status_code)

    sd_url = "http://redhare.ddns.net:6786/tagger/v1/interrogate"
    # sd_url = 'https://92dba0dbfb47e03d96.gradio.live/tagger/v1/interrogate'
    model = 'wd14-vit.v1'
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

        txt_path  = f"{base_path}/{file.split('.')[0]}.txt"
        print('============')
        print(tagger_infor)
        print('============')
        with open(txt_path, 'w') as f:
            for key in tagger_infor['caption']['tag'].keys():
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
def cm_processing_status(user_id,request_independentKey, status_num,request_thumbnail_image = None,request_cm_nickname = None, file_number = 0 ):
    with Session() as session:
        try:
            if status_num==0: #생성중
                change_processing = CMProcessing(independent_key=request_independentKey,
                                  user_id=user_id,
                                  status=cm_status.generating.name,
                                  thumbnail_image=request_thumbnail_image,
                                  cm_nickname=request_cm_nickname,
                                  file_number = file_number,
                                  createdAt=datetime.now(),
                                  updatedAt=datetime.now())
                session.add(change_processing)
                print("모델 생성중 업데이트")
            else:
                change_processing = session.query(CMProcessing).filter_by(independent_key=request_independentKey).first()
                if(status_num==1):
                    change_processing.status = cm_status.complete.name
                    print("모델 생성 완료 업데이트")
                if(status_num==2):
                    change_processing.status = cm_status.fail.name
                    print("모델 생성 실패 업데이트")
                if(status_num==3):
                    change_processing.status = cm_status.removed.name
            session.commit()

        except Exception as e:
            print(f"레코드 추가 중 오류 발생: {str(e)}")
            session.rollback()


# 모델 DB 저장
def save_model_db(request_id, request_independentKey,request_thumbnail_image,request_cm_nickname):
    with Session() as session:
        try:
            save_model = CustomizedModel(user_id=request_id, independent_key=request_independentKey,
                                thumbnail_image = request_thumbnail_image, cm_nickname = request_cm_nickname,
                                createdAt=datetime.now(),
                                updatedAt=datetime.now())
            session.add(save_model)
            session.commit()
        except Exception as e:
            print("실패", e)


# independentKey 중복검사
def duplicate_independentKey(request_independentKey):
    with Session() as session:
        try:
            existing_model = session.query(CustomizedModel).filter_by(independent_key=request_independentKey).first()
            print("중복검사",request_independentKey)

            if existing_model:
                print(f"Error: independentKey '{request_independentKey}' already exists in the database.")
                return True
            else:
                return False
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({"error": e}), 400


# id 존재여부 검사
def exist_id(user_id):
    with Session() as session:
        try:
            existing_model = session.query(User).filter_by(id=user_id).first()
            print("존재검사", user_id)

            if existing_model:
                print(f"유저 존재")
                return True
            else:
                return False
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({"error": e}), 400


if __name__ == "__main__":
    create_tables()
    app.run(host="0.0.0.0", port=4000, debug=True)
 