import os
import re
import toml
from time import time
from IPython.display import Markdown, display
import sys
import subprocess


model_file = "/workspace/workspace/model/magic.safetensors"
current_directory = os.getcwd()
print("현재 작업 디렉토리:", current_directory)
# These may be set by other cells, some are legacy
if "custom_dataset" not in globals(): #config파일 있는 경우 
  custom_dataset = None
if "override_dataset_config_file" not in globals(): #dataset_config 파일 전달하는 경우 
  override_dataset_config_file = None
if "override_config_file" not in globals():  #config파일 있는 경우 
  override_config_file = None
if "optimizer" not in globals():
  optimizer = "AdamW8bit"
if "optimizer_args" not in globals():
  optimizer_args = None
if "continue_from_lora" not in globals(): #""
  continue_from_lora = ""
if "weighted_captions" not in globals(): #false
  weighted_captions = False
if "adjust_tags" not in globals(): #false
  adjust_tags = False
if "keep_tokens_weight" not in globals(): #1.0
  keep_tokens_weight = 1.0
# if "model_file" not in globals():
#   model_file = None
if "model_url" in globals():
  old_model_url = model_url
else:
  old_model_url = None

COLAB = True
XFORMERS = True

# if len(sys.argv) > 1:
#     #project_name = sys.argv[1]
# else:
#     #project_name = "" #@param {type:"string"}  #input, output 파일 이름 전달
#     return "파일전달안됨"
project_name="1_cha_new"
print("파일이름",project_name)

model_url = "https://huggingface.co/hollowstrawberry/stable-diffusion-guide/resolve/main/models/sd-v1-5-pruned-noema-fp16.safetensors"


custom_model_is_based_on_sd2 = False #false

resolution = 512 #@param {type:"slider", min:512, max:1024, step:128}
flip_aug = False #false
caption_extension = ".txt" #param {type:"string"}
shuffle_tags = True 
shuffle_caption = shuffle_tags #True
activation_tags = "1" #@param [0,1,2,3]
keep_tokens = int(activation_tags) #1

num_repeats = 10 #@param {type:"number"}
preferred_unit = "Epochs" #@param ["Epochs", "Steps"]
how_many = 10 #@param {type:"number"}
max_train_epochs = how_many if preferred_unit == "Epochs" else None #10
max_train_steps = how_many if preferred_unit == "Steps" else None #None
save_every_n_epochs = 1 #@param {type:"number"}
keep_only_last_n_epochs = 10 #@param {type:"number"}
train_batch_size = 2 #@param {type:"slider", min:1, max:8, step:1}

unet_lr = 5e-4 #@param {type:"number"}
text_encoder_lr = 1e-4 #@param {type:"number"}
#@markdown The scheduler is the algorithm that guides the learning rate. If you're not sure, pick `constant` and ignore the number. I personally recommend `cosine_with_restarts` with 3 restarts.
lr_scheduler = "cosine_with_restarts" #@param ["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"]
lr_scheduler_number = 3 #@param {type:"number"}
lr_scheduler_num_cycles = lr_scheduler_number if lr_scheduler == "cosine_with_restarts" else 0 #3
lr_scheduler_power = lr_scheduler_number if lr_scheduler == "polynomial" else 0 #0
#@markdown Steps spent "warming up" the learning rate during training for efficiency. I recommend leaving it at 5%.
lr_warmup_ratio = 0.05 #@param {type:"slider", min:0.0, max:0.5, step:0.01}
lr_warmup_steps = 0
#@markdown New feature that adjusts loss over time, makes learning much more efficient, and training can be done with about half as many epochs. Uses a value of 5.0 as recommended by [the paper](https://arxiv.org/abs/2303.09556).
min_snr_gamma = True #@param {type:"boolean"}
min_snr_gamma_value = 5.0 if min_snr_gamma else None #5.0

#@markdown ### ▶️ Structure
#@markdown LoRA is the classic type and good for a variety of purposes. LoCon is good with artstyles as it has more layers to learn more aspects of the dataset.
#lora_type = "LoRA" #@param ["LoRA", "LoCon"]
network_dim = 16 #@param {type:"slider", min:1, max:128, step:1}
network_alpha = 8 #@param {type:"slider", min:1, max:128, step:1}
#@markdown The following two values only apply to the additional layers of LoCon.
conv_dim = 8 #@param {type:"slider", min:1, max:64, step:1}
conv_alpha = 4 #@param {type:"slider", min:1, max:64, step:1}

network_module = "networks.lora"
network_args = None

root_dir = "/workspace"#"/content" if COLAB else "~/Loras" #/content
deps_dir = os.path.join(root_dir, "deps") 
#repo_dir = os.path.join(root_dir, "kohya-trainer") 
repo_dir = os.path.join("/app/sd-scripts") 

#folder_structure 이미지셋 상위 경로
#main_dir      = os.path.join(root_dir, "uploads") if COLAB else root_dir
images_folder = os.path.join(root_dir, "uploads", project_name)
output_folder = os.path.join(root_dir, "output", project_name)
config_folder = os.path.join(root_dir, "config", project_name)
log_folder    = os.path.join(root_dir, "log")

config_file = os.path.join(config_folder, "training_config.toml")
dataset_config_file = os.path.join(config_folder, "dataset_config.toml")
accelerate_config_file = os.path.join(repo_dir, "accelerate_config/config.yaml")

from accelerate.utils import write_basic_config
if not os.path.exists(accelerate_config_file):
  write_basic_config(save_location=accelerate_config_file)

def validate_dataset():

  global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, keep_tokens_weight, weighted_captions, adjust_tags
  supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

  print("\n💿 Checking dataset...") #프로젝트 이름 존재 검사
  #strip 양쪽 문자열 제거, any 특수문자 포함 여부 확인
  if not project_name.strip() or any(c in project_name for c in " .()\"'\\/"):
    print("💥 Error: Please choose a valid project name.")
    return "project 없음"

  # TOML 파일 예시
  # [datasets]
  # [[datasets.subsets]]
  # image_dir = "dataset/images/folder1"
  # num_repeats = 3
  # is_reg = true
  
  if custom_dataset: #toml 파일 여부 확인
    try:
      datconf = toml.loads(custom_dataset) #toml형식 파일 딕셔너리 형태로 파싱
      datasets = [d for d in datconf["datasets"][0]["subsets"]] #딕셔너리 datasets 첫번째 값 리스트 생성
    except:
      print(f"💥 Error: Your custom dataset is invalid or contains an error! Please check the original template.")
      return "toml파일 에러"
    reg = [d.get("image_dir") for d in datasets if d.get("is_reg", False)] #image_dir
    datasets_dict = {d["image_dir"]: d["num_repeats"] for d in datasets} #dict={"image폴더 경로" : "num_repeats"} 형태로 저장
    folders = datasets_dict.keys() #image폴더
    files = [f for folder in folders for f in os.listdir(folder)] #image폴더에서 사진저장
    #images_repeats={"image폴더 경로":(folders 리스트의 사진파일의 개수, 반복 횟수)} 저장
    images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets_dict[folder]) for folder in folders}
  else: #없으면 toml 임의로 저장
    reg = []
    folders = [images_folder]
    files = os.listdir(images_folder)
    images_repeats = {images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), num_repeats)}

  for folder in folders: #폴더 존재확인
    if not os.path.exists(folder):
      print(f"💥 Error: The folder {folder.replace('/content/drive/', '')} doesn't exist.")
      return "폴더 존재하지 않음"
  for folder, (img, rep) in images_repeats.items(): #폴더 내용 검사
    if not img:
      print(f"💥 Error: Your {folder.replace('/content/drive/', '')} folder is empty.")
      return "이미지 없음"
  for f in files: #파일 확장자 검사 
    if not f.lower().endswith((".txt", ".npz")) and not f.lower().endswith(supported_types):
      print(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")
      return "txt파일 없음"

  #태그 확인
  if not [txt for txt in files if txt.lower().endswith(".txt")]:
    caption_extension = ""
  #continue_from_lora 사용 안함
  if continue_from_lora and not (continue_from_lora.endswith(".safetensors") and os.path.exists(continue_from_lora)):
    print(f"💥 Error: Invalid path to existing Lora. Example: /content/drive/MyDrive/Loras/example.safetensors")
    return "continue from lora error"

  #images_repeats.values
  pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values()) #(사진개수=20, num_repeats=10) 
  steps_per_epoch = pre_steps_per_epoch/train_batch_size
  total_steps = max_train_steps or int(max_train_epochs*steps_per_epoch) #max_train_epochs =10, 
  estimated_epochs = int(total_steps/steps_per_epoch)
  lr_warmup_steps = int(total_steps*lr_warmup_ratio) #lr_warmup_ratio=0.05

  #잘들어갔는지 검사
  for folder, (img, rep) in images_repeats.items():
    print("📁"+folder.replace("/content/drive/", "") + (" (Regularization)" if folder in reg else ""))
    print(f"📈 Found {img} images with {rep} repeats, equaling {img*rep} steps.")
  print(f"📉 Divide {pre_steps_per_epoch} steps by {train_batch_size} batch size to get {steps_per_epoch} steps per epoch.")
  if max_train_epochs:
    print(f"🔮 There will be {max_train_epochs} epochs, for around {total_steps} total training steps.")
  else:
    print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

  if total_steps > 10000:
    print("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...")
    return "steps 너무 많음"

  return True

def create_config():
  global dataset_config_file, config_file, model_file

  if override_config_file: #training_config.toml파일
    config_file = override_config_file
    print(f"\n⭕ Using custom config file {config_file}")
  else:
    config_dict = {
      "additional_network_arguments": {
        "unet_lr": unet_lr, #5e-4
        "text_encoder_lr": text_encoder_lr, #1e-4
        "network_dim": network_dim, #16
        "network_alpha": network_alpha, #8
        "network_module": network_module, #"networks.lora"
        "network_args": network_args, #None
        "network_train_unet_only": True if text_encoder_lr == 0 else None, #None
        "network_weights": continue_from_lora if continue_from_lora else None #None
      },
      "optimizer_arguments": {
        "learning_rate": unet_lr, #5e-4
        "lr_scheduler": lr_scheduler, #"cosine_with_restarts"
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,#3
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,#None
        "lr_warmup_steps": lr_warmup_steps if lr_scheduler != "constant" else None,#None
        "optimizer_type": optimizer,
        "optimizer_args": optimizer_args if optimizer_args else None, #None
      },
      "training_arguments": {
        "max_train_steps": max_train_steps,
        "max_train_epochs": max_train_epochs,
        "save_every_n_epochs": save_every_n_epochs,
        "save_last_n_epochs": keep_only_last_n_epochs,
        "train_batch_size": train_batch_size,
        "noise_offset": None,
        "clip_skip": 2,
        "min_snr_gamma": min_snr_gamma_value,
        "weighted_captions": weighted_captions,
        "seed": 42,
        "max_token_length": 225,
        "xformers": XFORMERS,
        "lowram": COLAB,
        "max_data_loader_n_workers": 8,
        "persistent_data_loader_workers": True,
        "save_precision": "fp16",
        "mixed_precision": "fp16",
        "output_dir": output_folder,
        "logging_dir": log_folder,
        "output_name": project_name,
        "log_prefix": project_name,
      },
      "model_arguments": {
        "pretrained_model_name_or_path": model_file,
        "v2": custom_model_is_based_on_sd2, #false
        "v_parameterization": True if custom_model_is_based_on_sd2 else None, #None
      },
      "saving_arguments": {
        "save_model_as": "safetensors",
      },
      "dreambooth_arguments": {
        "prior_loss_weight": 1.0,
      },
      "dataset_arguments": {
        "cache_latents": True,
      },
    }

    for key in config_dict: #none이 아닌 항목만 남김
      if isinstance(config_dict[key], dict):
        config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

    with open(config_file, "w") as f: #config_file에 toml 형식으로 작성
      f.write(toml.dumps(config_dict))
    print(f"\n📄 Config saved to {config_file}")

  if override_dataset_config_file: 
    dataset_config_file = override_dataset_config_file
    print(f"⭕ Using custom dataset config file {dataset_config_file}")
  else:#toml 파일 경로+toml파일 이름
    dataset_config_dict = {
      "general": {
        "resolution": resolution, #512
        "shuffle_caption": shuffle_caption, #True
        "keep_tokens": keep_tokens, #1
        "flip_aug": flip_aug,
        "caption_extension": caption_extension, #사용안함
        "enable_bucket": True,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
        "min_bucket_reso": 320 if resolution > 640 else 256,
        "max_bucket_reso": 1280 if resolution > 640 else 1024,
      },
      "datasets": toml.loads(custom_dataset)["datasets"] if custom_dataset else [
        {
          "subsets": [
            {
              "num_repeats": num_repeats,
              "image_dir": images_folder,
              "class_tokens": None if caption_extension else project_name
            }
          ]
        }
      ]
    }

    for key in dataset_config_dict: #none이 아닌 항목만 남김
      if isinstance(dataset_config_dict[key], dict):
        dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

    with open(dataset_config_file, "w") as f: #toml형식 생성
      f.write(toml.dumps(dataset_config_dict))
    print(f"📄 Dataset config saved to {dataset_config_file}")

# def download_model():
#   global old_model_url, model_url, model_file
#   real_model_url = model_url.strip()  # pretrained model 기입

#   # 모델 파일 이름 결정
#   if real_model_url.lower().endswith((".ckpt", ".safetensors")):
#       model_file = f"/content{real_model_url[real_model_url.rfind('/'):]}"
#   else:
#       model_file = "/content/downloaded_model.safetensors"
#       if os.path.exists(model_file):
#           os.remove(model_file)  # !rm "{model_file}" 을 파이썬 코드로 변환

#   # 모델 URL 업데이트
#   if m := re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", model_url):
#       real_model_url = real_model_url.replace("blob", "resolve")
#   elif m := re.search(r"(?:https?://)?(?:www\\.)?civitai\.com/models/([0-9]+)(/[A-Za-z0-9-_]+)?", model_url):
#       if m.group(2):
#           model_file = f"/content{m.group(2)}.safetensors"
#       if m := re.search(r"modelVersionId=([0-9]+)", model_url):
#           real_model_url = f"https://civitai.com/api/download/models/{m.group(1)}"
#       else:
#           raise ValueError("URL doesn't include a modelVersionId.")

#   # 모델 다운로드
#   subprocess.run(["aria2c", real_model_url, "--console-log-level=warn", "-c", "-s", "16", "-x", "16", "-k", "10M", "-d", "/", "-o", model_file], check=True)

#   # .safetensors 파일 처리
#   if model_file.lower().endswith(".safetensors"):
#       from safetensors.torch import load_file as load_safetensors
#       try:
#           test = load_safetensors(model_file)
#           del test
#       except:
#           new_model_file = os.path.splitext(model_file)[0] + ".ckpt"
#           os.rename(model_file, new_model_file)  # !mv "{model_file}" "{new_model_file}" 을 파이썬 코드로 변환
#           model_file = new_model_file
#           print(f"Renamed model to {os.path.splitext(model_file)[0]}.ckpt")

#   # .ckpt 파일 처리
#   if model_file.lower().endswith(".ckpt"):
#       from torch import load as load_ckpt
#       try:
#           test = load_ckpt(model_file)
#           del test
#       except:
#           return False

#   return True


def main():
  # if COLAB and not os.path.exists('/content/drive'):
  #   from google.colab import drive
  #   print("📂 Connecting to Google Drive...")
  #   drive.mount('/content/drive')
  print("실행")

  for dir in (deps_dir, repo_dir, log_folder, images_folder, output_folder, config_folder):
    os.makedirs(dir, exist_ok=True)

  if not validate_dataset():
    return "파일 유효하지 않음"



  print("\n⭐ Starting trainer...\n")
  os.chdir(repo_dir) #작업디렉토리 변경

  #train_network_wrapper.py실행
  #!accelerate launch --config_file={accelerate_config_file} --num_cpu_threads_per_process=1 train_network_wrapper.py --dataset_config={dataset_config_file} --config_file={config_file}
  
  
  # if old_model_url != model_url or not model_file or not os.path.exists(model_file):
  #   print("\n🔄 Downloading model...")
  #   if not download_model():
  #     print("\n💥 Error: The model you selected is invalid or corrupted, or couldn't be downloaded. You can use a civitai or huggingface link, or any direct download link.")
  #     return
  #   print()
  # else:
  #   print("\n🔄 Model already downloaded.\n")

  create_config()

  result = subprocess.run([
    "accelerate", "launch",
    "--config_file", accelerate_config_file,
    "--num_cpu_threads_per_process", "1",
    "train_network_wrapper.py",
    "--dataset_config", dataset_config_file,
    "--config_file", config_file
  ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

  # 실행 결과와 에러 메시지 출력
  print("STDOUT:", result.stdout)
  print("STDERR:", result.stderr)

main()