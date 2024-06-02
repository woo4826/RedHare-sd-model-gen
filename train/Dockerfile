
# step 1
FROM continuumio/miniconda3:latest AS compile-image

ENV PYTHON_VERSION=3.10
#ENV PYTHON_VERSION=3.10.6
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

#LoRA 트레이닝 설치 환경 변수 설정
# ENV TF_CPP_MIN_LOG_LEVEL=3 \
#     BITSANDBYTES_NOWELCOME=1 \
#     SAFETENSORS_FAST_GPU=1

# app
WORKDIR /app/
RUN git clone https://github.com/kohya-ss/sd-scripts.git
WORKDIR /app/sd-scripts/
RUN git reset --hard 16e5981d3153ba02c34445089b998c5002a60abc

#LoRA 트레이닝 설치
# RUN git reset --hard 9a67e0df390033a89f17e70df5131393692c2a55
# RUN wget https://raw.githubusercontent.com/hollowstrawberry/kohya-colab/main/train_network_wrapper.py -q -O train_network_wrapper.py && \
#     wget https://raw.githubusercontent.com/hollowstrawberry/kohya-colab/main/dracula.py -q -O dracula.py



# config파일, 도커 컨테이너의 복사된 파일의 소유권
COPY --chown=$USERNAME default_config.yaml $HF_HOME/accelerate/default_config.yaml

# install
WORKDIR /app/sd-scripts

RUN source activate conda && \
  pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116 && \
  pip install --upgrade -r requirements.txt && \
  pip install --no-cache-dir xformers==0.0.17
 #pip install Flask==2.3.2 \
#수정: no-cach 끝에 &&추가

#LoRA 트레이닝 세팅
# RUN source activate conda && \
#     pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 && \
#     pip install --upgrade -r requirements.txt && \
#     pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121


# Python sql패키지 설치
RUN pip3 install numpy tensorflow sqlalchemy
RUN pip install mysql-connector-python

#LoRA 트레이닝 Python 라이브러리 설치
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
#     && echo 'source $HOME/.cargo/env' >> $HOME/.bashrc \
#     && source $HOME/.cargo/env
# RUN pip install transformers

# USER root
# RUN apt-get update -qq && \
#     apt-get install -y aria2 -qq && \
#     pip install --upgrade pip
# RUN pip install accelerate==0.15.0 diffusers==0.10.2 bitsandbytes==0.41.3.post2 && \
#     pip install opencv-python==4.8.0.76 
# RUN pip install torchtext==0.16.2 torchaudio==2.2.0 && \
#     pip install jax==0.4.23 \
#     jaxlib==0.4.23 \
#     toml==0.10.2 \
#     ftfy==6.1.1 \
#     einops==0.6.0 \
#     timm==0.6.12 \
#     fairscale==0.4.13 \
#     albumentations==1.3.1 \
#     voluptuous==0.13.1 \
#     requests==2.31.0 \
#     safetensors \
#     lion_pytorch==0.0.6 \
#     dadaptation==3.1 \
#     prodigyopt==1.0 \
#     pygments
# RUN pip install IPython


#pytorch-lightning==1.9.0 \
# LoRA 트레이닝 파일 설정
# RUN sed -i "s@cpu@cuda@" library/model_util.py
# RUN sed -i 's/from PIL import Image/from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True/g' library/train_util.py
# RUN sed -i 's/{:06d}/{:02d}/g' library/train_util.py && sed -i 's/"." + args.save_model_as)/"-{:02d}.".format(num_train_epochs) + args.save_model_as)/g' train_network.py


# install triton
# https://github.com/P2Enjoy/kohya_ss-docker/blob/736c0116b00904e700f3bad825a19c873f1a85be/docker-compose.yml
RUN source activate conda && \
  pip install triton==2.0.0

WORKDIR /app/
COPY ./app/app.py /app/sd-scripts/

#LoRA 트레이닝 파일 복사
COPY ./new_trainning.py /app/sd-scripts/

RUN source activate conda
RUN pip install Flask==2.3.2
RUN pip install opencv-python-headless
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