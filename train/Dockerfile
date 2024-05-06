
# step 1
FROM continuumio/miniconda3:latest AS compile-image

ENV PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y curl git wget software-properties-common

# conda
RUN conda create --name conda python=${PYTHON_VERSION} pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# step 2
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
# user
ARG USERNAME=user
ARG USER_UID=1000
#수정됨 1000->1000
ARG USER_GID=$USER_UID

# root
RUN apt-get update && \
  apt-get install -y curl git wget software-properties-common && \
  apt-get install -y libgl1 gnupg2 moreutils tk libglib2.0-0 libaio-dev

# create user 그룹 생성 후 유저 넣기
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# set user 유저로 실행
USER $USERNAME
#user에게 conda 사용권 부여 
COPY --from=compile-image --chown=$USERNAME /opt/conda /opt/conda 
ENV PATH /opt/conda/bin:$PATH
#캐시 데이터 기본경로를 사용자 홈 디렉터리로 지정 / 사용자 별로 데이터 관리
ENV HF_HOME=/home/$USERNAME/.cache/huggingface

# shell 배시쉘 사용
SHELL ["/bin/bash", "-c"]

# app
WORKDIR /app/
RUN git clone https://github.com/kohya-ss/sd-scripts.git
WORKDIR /app/sd-scripts
RUN git reset --hard 16e5981d3153ba02c34445089b998c5002a60abc

# config파일, 도커 컨테이너의 복사된 파일의 소유권
COPY --chown=$USERNAME default_config.yaml $HF_HOME/accelerate/default_config.yaml

# install
WORKDIR /app/sd-scripts

RUN source activate conda && \
  pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116 && \
  pip install --upgrade -r requirements.txt && \
  pip install --no-cache-dir xformers==0.0.17
  # pip install Flask==2.3.2 \
#수정: no-cach 끝에 &&추가


# Python sql패키지 설치
RUN pip3 install numpy tensorflow sqlalchemy
RUN pip install mysql-connector-python

# install triton
# https://github.com/P2Enjoy/kohya_ss-docker/blob/736c0116b00904e700f3bad825a19c873f1a85be/docker-compose.yml
RUN source activate conda && \
  pip install triton==2.0.0

WORKDIR /app/
COPY ./app/app.py /app/sd-scripts/
RUN source activate conda
RUN pip install Flask==2.3.2
# Flask가 사용할 포트
EXPOSE 4000

#docker 컨테이너가 시작될 때 기본 명령 - Flask 실행
CMD ["python", "app.py"]



# additional libraries
# USER root
# RUN apt-get install -y unzip
# USER $USERNAME

# # additional scripts 스크립트 복사 및 권한 설정
# COPY --chown=$USERNAME download_images.sh .
# RUN chmod +x download_images.sh 스크립트 파일에 권한 부여

# entrypoint
WORKDIR /app/sd-scripts
COPY --chown=$USERNAME entrypoint.sh .

RUN chmod +x entrypoint.sh

# RUN pip install --no-cache-dir flask


# ENTRYPOINT ["./entrypoint.sh"]
