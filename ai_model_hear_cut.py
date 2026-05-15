# Data retrievel
import gdown
import os

file_id = '1vhz1KngnsvK4BxHgv86OYvkx_iZqG3BZ'
url = f'https://drive.google.com/uc?id={file_id}'

print("جاري تحميل ملف الداتا سيت")
gdown.download(url, '/content/train.rar', quiet=False)

print("جاري فك الضغط")
!apt-get install unrar 
!mkdir -p /content/dataset
!unrar x /content/train.rar /content/dataset/

print("تم التحميل وفك الضغط بنجاح في فولدر /content/dataset/")

# install library
!pip install --upgrade gdown

!apt-get update
!apt-get install unrar



""" Data modification"""

import os

dataset_path = "/content/dataset/"
# Choose a unique keyword for your style (e.g., modern_cut_style)
style_keyword = "modern_cut_style"

for filename in os.listdir(dataset_path):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        base_name = os.path.splitext(filename)[0]
        with open(os.path.join(dataset_path, f"{base_name}.txt"), "w") as f:
            # This description is what the model will learn and associate with the image
            f.write(f"a professional photo of a person with {style_keyword} haircut")

print(f"Caption files prepared successfully with keyword: {style_keyword}")



# install library and script in github 
!pip install -U torchao peft diffusers transformers accelerate -q
!pip install git+https://github.com/huggingface/diffusers.git -U -q
!pip install -U accelerate transformers peft -q
!rm train_dreambooth_lora.py
!wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py

   

""" creaate fine tuning using lora"""

import os


INSTANCE_DIR = "/content/dataset/train" 

   
if not os.path.exists(INSTANCE_DIR):
     
    INSTANCE_DIR = "/content/dataset"

print(f"المسار المستخدم حالياً: {INSTANCE_DIR}")

  
!accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="{INSTANCE_DIR}" \
  --output_dir="/content/haircut_lora_model" \
  --instance_prompt="a photo of a person with modern_cut_style haircut" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --validation_prompt="a professional photo of a man with modern_cut_style haircut" \
  --seed="42" \
  --mixed_precision="fp16"



