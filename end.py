from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from PIL import Image, ImageFilter
import io
import base64
import numpy as np

app = FastAPI()

model_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


lora_path = "حط المسار"
pipe.load_lora_weights(lora_path)


pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

class HaircutRequest(BaseModel):
    image_base64: str

def decode_image(data):
    img_data = base64.b64decode(data)
    return Image.open(io.BytesIO(img_data)).convert("RGB").resize((512, 512))

@app.post("/v1/generate-haircut")
async def generate_haircut(request: HaircutRequest):
    try:
        init_image = decode_image(request.image_base64)
        
    
        mask_array = np.zeros((512, 512), dtype=np.uint8)
        mask_array[0:180, :] = 255 
        mask_image = Image.fromarray(mask_array).convert("L")
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=15))

        
        prompt = "a natural photo of a man with modern_cut_style haircut, realistic texture, perfectly blended with skin, soft natural hair, matching skin tone"
        negative_prompt = "face, eyes, eyebrows, nose, mouth, extra faces, distorted skin, high hair, unrealistic hair, cartoon, illustration, blurry, glitch"

        with torch.autocast("cuda"):
            result_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                strength=0.5,           
                guidance_scale=10.0,
                num_inference_steps=40  
            ).images[0]
 
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {"status": "success", "image": img_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)