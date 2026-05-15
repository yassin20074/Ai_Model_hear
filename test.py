""" Model Experience"""

import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from PIL import Image, ImageFilter
import numpy as np

# bring the model
model_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")


pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

#  install a lora file
pipe.load_lora_weights("/content/haircut_lora_model/pytorch_lora_weights.safetensors")
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# image preparation
input_image_path = "/content/photo_2026-04-01_21-56-22.jpg"
init_image = Image.open(input_image_path).convert("RGB").resize((512, 512))


# creating a mask
mask_array = np.zeros((512, 512), dtype=np.uint8)
mask_array[0:180, :] = 255 # حددنا منطقة أصغر لتقليل الارتفاع
mask_image = Image.fromarray(mask_array).convert("L")
mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=15)) # تمويه قوي للحواف

# create a prompt
prompt = "a natural photo of a man with modern_cut_style haircut, realistic texture, perfectly blended with skin, soft natural hair, matching skin tone"
negative_prompt = "face, eyes, eyebrows, nose, mouth, extra faces, distorted skin, high hair, unrealistic hair, cartoon, illustration, blurry, glitch"

#creating settings

with torch.autocast("cuda"):
    final_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        strength=0.5,          
        guidance_scale=10.0,     
        num_inference_steps=40      
    ).images[0]
  
#save and display the result

final_image.save("haircut_on_face_natural1.png")
final_image.show()
