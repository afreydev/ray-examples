import ray
import torch
import io
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ray.util.state import summarize_tasks
from ray import serve

app = FastAPI()

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


@serve.deployment(num_replicas=5, route_prefix="/")
@serve.ingress(app)
class APIIngress:
    def __init__(self) -> None:
        print("Initializing")
        self.task_types = [
            "PENDING_OBJ_STORE_MEM_AVAIL",
            "PENDING_NODE_ASSIGNMENT",
            "SUBMITTED_TO_WORKER",
            "PENDING_ARGS_FETCH",
            "SUBMITTED_TO_WORKER"
        ]

    @app.get("/imagine")
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"
        future = image.remote(prompt)
        result = ray.get(future)
        buf = io.BytesIO()
        result.save(buf, format='JPEG', quality=100)
        buf.seek(0) # important here!
        return StreamingResponse(buf, media_type="image/jpeg")

    @app.get("/pending-tasks")
    async def pending_tasks(self):
        summary = summarize_tasks()
        pending = 0
        if "cluster" in summary and "summary" in summary["cluster"]:
            tasks = summary["cluster"]["summary"]
            for key in tasks:
                task = tasks[key]
                if task["type"] == "NORMAL_TASK":
                    for task_type in task["state_counts"]:
                        if task_type in self.task_types:
                            pending += task["state_counts"][task_type]
        return pending

deployment = APIIngress.bind()
