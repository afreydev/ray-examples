# Based on: 
# https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html#submitting-a-job
# script.py
import ray
import time

@ray.remote(num_gpus=2)
def hello_world():
    print("I'm going to sleep")
    time.sleep(30)
    print("I'm ready")
    return "hello world"

# Automatically connect to the running Ray cluster.
ray.init()
print(ray.get(hello_world.remote()))

# https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html