# Based on: 
# https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html#submitting-a-job
# script.py
import ray
import time
import os

@ray.remote(num_gpus=2)
def hello_world(value):
    print(f"This is the env var: {value}")
    time.sleep(30)
    print("I'm ready")
    return "hello world"

# Automatically connect to the running Ray cluster.
ray.init()
ray.get(hello_world.remote(os.environ.get("MY_ENV_VALUE")))

# https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html