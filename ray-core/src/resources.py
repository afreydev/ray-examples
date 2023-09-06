# Based on: 
# https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html#submitting-a-job
# script.py
import ray

@ray.remote(num_gpus=2)
def hello_world():
    print("I'm working right now")
    return "hello world"

# Automatically connect to the running Ray cluster.
ray.init()
print(ray.get(hello_world.remote()))

# ray job submit --working-dir /mnt -- python script2.py