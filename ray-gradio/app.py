import ray
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from ray import serve
from ray.serve.gradio_integrations import GradioServer


@ray.remote(num_gpus=1)
def image(prompt):

    model_id = "/mnt/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to("cuda")

    negative_prompt = "(((deformed))), (extra_limb), (long body :1.3), (mutated hands and fingers:1.5), (mutation poorly drawn :1.2), (poorly drawn hands), (ugly), Images cut out at the top, anatomical nonsense, bad anatomy, bad anatomy, bad breasts, bad composition, bad ears, bad hands, bad proportions, bad shadow, blurred, blurry, blurry imag, bottom, broken legs, cloned face, colorless, cropped, deformed, deformed body feature, dehydrated, disappearing arms, disappearing calf, disappearing legs, disappearing thigh, disfigure, disfigured, duplicate, error, extra arms, extra breasts, extra ears, extra fingers, extra legs, extra limbs, fused ears, fused fingers, fused hand, gross proportions, heavy breasts, heavy ears, left, liquid body, liquid breasts, liquid ears, liquid tongue, long neck, low quality, low res, low resolution, lowers, malformed, malformed hands, malformed limbs, messy drawing, missing arms, missing breasts, missing ears, missing hand, missing legs, morbid, mutated, mutated body part, mutated hands, mutation, mutilated, old photo, out of frame, oversaturate, poor facial detail, poorly Rendered fac, poorly drawn fac, poorly drawn face, poorly drawn hand, poorly drawn hands, poorly rendered hand, right, signature, text font ui, too many fingers, ugly, uncoordinated body, unnatural body, username, watermark, worst quality"
    r_image = pipe(
            prompt,
            width=768,
            height=768,
            num_inference_steps=50,
            negative_prompt=negative_prompt
        ).images[0]
    return r_image

example_input = "a small lion fighting with a t-rex"

def gradio_stable_diff_builder():

    def image_sd(prompt):
        sd_image = image.remote(prompt)
        r_image = ray.get(sd_image)
        return r_image

    return gr.Interface(
        fn=image_sd,
        inputs=[gr.Textbox(value=example_input, label="Input prompt")],
        outputs=[gr.Image(label="Image", width=250)],
    )

app = GradioServer.options(ray_actor_options={"num_cpus": 2}).bind(
    gradio_stable_diff_builder
)
