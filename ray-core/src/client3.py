from ray.job_submission import JobSubmissionClient

# If using a remote cluster, replace 127.0.0.1 with the head node's IP address.
client = JobSubmissionClient("http://127.0.0.1:8265")
job_id = client.submit_job(
    # Entrypoint shell command to execute
    entrypoint="python env_vars.py",
    runtime_env={"working_dir": "/mnt", "env_vars": {"MY_ENV_VALUE": "This is an example"}}
)
print(job_id)