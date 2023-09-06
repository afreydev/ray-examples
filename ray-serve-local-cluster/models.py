import ray
import uuid
import time
from fastapi import FastAPI
from fastapi.responses import Response
from ray.util.state import summarize_tasks
from ray import serve

app = FastAPI()

@ray.remote(resources={"rtx4090": 1})
def special_device(prompt):
    print(f"Running in special device: {prompt}")
    time.sleep(60)
    return "finished!"

@ray.remote(num_gpus=1)
def running_in_gpu(prompt):
    print(f"Running in the gpu: {prompt}")
    result = ray.get(special_device.remote(prompt))
    print(result) 

@serve.deployment(num_replicas=1, route_prefix="/")
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
        running_in_gpu.remote(prompt)
        return Response(content="ok")

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
