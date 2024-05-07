#!/bin/bash

# Usage
# ./entrypoint.sh <model_name> <output_name>

export BUILD_HF_HOME=$HF_HOME
export HF_HOME=/workspace/.cache/huggingface

# install default_config.yaml
mkdir -p "$HF_HOME/accelerate"
cp "$BUILD_HF_HOME/accelerate/default_config.yaml" "$HF_HOME/accelerate"

# the output directory for a LoRA model 로라모델 디렉토리
mkdir -p /workspace/output

# model name such as `runwayml/stable-diffusion-v1-5`
# also accepts a path to model file
MODEL_NAME_OR_PATH="$1"
echo "$MODEL_NAME_OR_PATH"
# try to use `model/*.safetensors` if the argument not specified
#모델 이름 or path 할당
#만약 변수가 비어 있다면 .safetensors 파일찾아서 저장
#만약 파일이 없다면, 에러 메시지를 출력
if [ -z "$MODEL_NAME_OR_PATH" ]; then
  ls -l /workspace/workspace/model/*.safetensors
  MODEL_PATH=`echo /workspace/workspace/model/magic.safetensors`
  if [ ! -f "$MODEL_PATH" ]; then
    echo -e "Error: model/*.safetensors does not exist"
    exit 1
  fi
  MODEL_NAME_OR_PATH="$MODEL_PATH"  
fi

echo "사용된 모델 이름: $MODEL_NAME_OR_PATH"

# output name without file extension
# the filename defaults to `output/lora.safetensors`
#출력 파일의 이름
OUTPUT_NAME="$2"
if [ -z "$OUTPUT_NAME" ]; then
  OUTPUT_NAME="lora"
fi

# the path to the uploaded file
#업로드 파일의 경로
FOLDER_KEY="$3"
if [ -z "$FOLDER_KEY" ]; then
  echo -e "Error: Path to the uploaded file is not provided"
  exit 1
fi


# Create dataset_config_{FOLDER_KEY}.toml
#toml(데이터 구조 나타내는 형식) 생성
#
DATASET_CONFIG_FILE="/workspace/train/dc_$FOLDER_KEY.toml"
echo "[general]" > "$DATASET_CONFIG_FILE"
echo "[[datasets]]" >> "$DATASET_CONFIG_FILE"
echo "[[datasets.subsets]]" >> "$DATASET_CONFIG_FILE"
echo "image_dir = '/workspace/uploads/$FOLDER_KEY'" >> "$DATASET_CONFIG_FILE"
echo "caption_extension = '.txt'" >> "$DATASET_CONFIG_FILE"
echo "num_repeats = 20" >> "$DATASET_CONFIG_FILE"

source activate conda

accelerate launch --num_cpu_threads_per_process 4 train_network.py \
  --pretrained_model_name_or_path="$MODEL_NAME_OR_PATH" \
  --output_dir="/workspace/output/" \
  --output_name="$OUTPUT_NAME" \
  --dataset_config="/workspace/train/dc_$FOLDER_KEY.toml" \
  --train_batch_size=2 \
  --max_train_epochs=20 \
  --resolution="512,512" \
  --optimizer_type="AdamW8bit" \
  --learning_rate=1e-4 \
  --network_dim=32 \
  --network_alpha=16 \
  --enable_bucket \
  --bucket_no_upscale \
  --lr_scheduler=cosine_with_restarts \
  --lr_scheduler_num_cycles=3 \
  --lr_warmup_steps=0 \
  --keep_tokens=0 \
  --shuffle_caption \
  --caption_dropout_rate=0.05 \
  --save_model_as=safetensors \
  --clip_skip=2 \
  --seed=42 \
  --color_aug \
  --xformers \
  --mixed_precision=fp16 \
  --network_module=networks.lora \
  --persistent_data_loader_workers

# accelerate launch --num_cpu_threads_per_process 1 train_network.py \
#   --pretrained_model_name_or_path="$MODEL_NAME_OR_PATH" \
#   --output_dir="/workspace/output/$FOLDER_KEY" \
#   --output_name="$OUTPUT_NAME" \
#   --dataset_config="/workspace/train/dc_$FOLDER_KEY.toml" \
#   --train_batch_size=2 \
#   --max_train_epochs=5 \
#   --resolution="512,512" \
#   --optimizer_type="AdamW8bit" \
#   --learning_rate=1e-4 \
#   --network_dim=128 \
#   --network_alpha=64 \
#   --enable_bucket \
#   --bucket_no_upscale \
#   --lr_scheduler=cosine_with_restarts \
#   --lr_scheduler_num_cycles=4 \
#   --lr_warmup_steps=50 \
#   --keep_tokens=1 \
#   --shuffle_caption \
#   --caption_dropout_rate=0.05 \
#   --save_model_as=safetensors \
#   --clip_skip=2 \
#   --seed=42 \
#   --color_aug \
#   --xformers \
#   --mixed_precision=fp16 \
#   --network_module=networks.lora \
#   --persistent_data_loader_workers