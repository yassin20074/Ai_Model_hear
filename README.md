​## 💇‍♂️ AI-Powered Personalized Haircut Stylist

​Stable Diffusion + LoRA Custom Training & Inpainting
​This project provides an advanced AI solution for realistic haircut transformations. By combining Generative AI with Computer Vision techniques, the system allows users to try on specific hairstyles while maintaining their original facial features with high precision.

​## 🚀 Key Features
​**Custom Style Training:** Utilized LoRA (Low-Rank Adaptation) to train the model on the specific modern_cut_style.
​**Precision Inpainting:** Leveraged Stable Diffusion Inpainting to modify only the hair region, keeping eyes, nose, and mouth 100% intact.
​**Seamless Edge Blending:** Implemented advanced Gaussian Blur Masking to ensure the new hair blends naturally with the user's forehead, eliminating "jitter" or harsh edges.
​**Production-Ready API:** A fully functional FastAPI backend designed to integrate with Web and Mobile front-ends.

## ​🛠 Technical Stack

​**Base Model:** Stable Diffusion v1-5
**​Fine-tuning:** DreamBooth LoRA
​Libraries: Hugging Face diffusers, PyTorch, NumPy
​**Image Processing:** Pillow (PIL) for soft-masking and refinement.
​**Backend:** FastAPI with Uvicorn for high-performance inference.
​**Scheduler:** DDIMScheduler for enhanced edge smoothness.

​## Project Architecture
​**Dataset:** Curated high-quality images of professional haircuts.
​**Training Phase:** Fine-tuned the model to produce a .safetensors weight file.
​**Inference Pipeline:**  Load Base Model + LoRA weights.
​Generate a Soft Mask (0-180px) with 15px Blur.
​Run Inpainting with strength=0.5 for a natural look.
​**Deployment:** Containerized API ready for Docker/Railway deployment

## 📈 Future Roadmap
​**Auto-Segmentation:** Integrate SAM (Segment Anything Model) to detect hair boundaries automatically.
​**Higher Resolution:** Upgrade to SDXL (1024x1024) for ultra-sharp textures.
​**Database Integration:** Connect to PostgreSQL to store user transformation history (linked to Edu QR or similar systems).

​**Developed by:** Yassin
AI Systems & Machine Learning Engineer
