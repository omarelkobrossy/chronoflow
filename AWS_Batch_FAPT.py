import pandas as pd
import math
import os
import subprocess
# from utils import preprocess_data

def generate_window_indices(data_length, window_size, step_size):
    indices = []
    i = 0
    while i + window_size <= data_length:
        indices.append((i, i + window_size))
        i += step_size
    return indices

def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def main():
    # User-editable parameters
    data_path = "DB/TSLA_15min_indicators.csv"
    window_size = 2350
    step_size = 310
    num_instances = 20
    job_queue = "MLTraderQueue"  # <-- EDIT THIS
    job_definition = "arn:aws:batch:us-east-1:671583585806:job-definition/ML-Trader-Job-Definition:4"  # <-- EDIT THIS
    job_name_prefix = "fapt"

    df = pd.read_csv(data_path)
    data_length = len(df)

    window_indices = generate_window_indices(data_length, window_size, step_size)
    chunks = chunkify(window_indices, num_instances)
    os.makedirs("AWS_Batch_Jobs/windows", exist_ok=True)
    for idx, chunk in enumerate(chunks):
        chunk_start = chunk[0][0]
        chunk_end = chunk[-1][1]
        job_name = f"{job_name_prefix}-{idx}"
        command = [
            "aws", "batch", "submit-job",
            "--job-name", job_name,
            "--job-queue", job_queue,
            "--job-definition", job_definition,
            "--container-overrides",
            f'command=[\"python\",\"Optimization/FAPT_Optimization.py\",\"--start\",\"{chunk_start}\",\"--end\",\"{chunk_end}\",\"--s3\"]'
        ]
        print(f"Submitting AWS Batch job for chunk {idx}: start={chunk_start}, end={chunk_end}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"Job submitted: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job for chunk {idx}: {e.stderr}")

if __name__ == "__main__":
    main()
