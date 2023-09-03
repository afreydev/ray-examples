import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


model_id = "stabilityai/stable-diffusion-2-1"
model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
model.save_pretrained("gradio/models/stable-diffusion-2-1")
