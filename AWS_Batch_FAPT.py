import pandas as pd
import numpy as np
import os
import json
import subprocess
from datetime import datetime
import argparse
from utils import preprocess_data

def parse_args():
    parser = argparse.ArgumentParser(description='Submit AWS Batch array job for FAPT optimization')
    parser.add_argument('--job-queue', type=str, required=True, help='AWS Batch job queue name')
    parser.add_argument('--job-definition', type=str, required=True, help='AWS Batch job definition name')
    parser.add_argument('--data-path', type=str, default='DB/TSLA_15min_indicators.csv', help='Path to the dataset')
    parser.add_argument('--window-size', type=int, default=2350, help='Window size for optimization')
    parser.add_argument('--step-size', type=int, default=310, help='Step size between windows')
    parser.add_argument('--array-size', type=int, default=32, help='Number of parallel instances to run')
    parser.add_argument('--symbol', type=str, default='TSLA', help='Trading symbol')
    return parser.parse_args()

def calculate_total_windows(df_length, window_size, step_size):
    """Calculate total number of windows that will be processed"""
    return (df_length - window_size) // step_size + 1

def submit_array_job(job_name, job_queue, job_definition, array_size, total_windows, window_size, step_size):
    """Submit a single AWS Batch array job"""
    # Create a JSON file with window information that will be used by the container
    window_info = {
        'total_windows': total_windows,
        'window_size': window_size,
        'step_size': step_size,
        'array_size': array_size
    }
    
    os.makedirs('AWS_Batch_Jobs', exist_ok=True)
    window_info_file = f'AWS_Batch_Jobs/{job_name}_window_info.json'
    with open(window_info_file, 'w') as f:
        json.dump(window_info, f, indent=4)
    
    # Submit the array job
    command = [
        "aws", "batch", "submit-job",
        "--job-name", job_name,
        "--job-queue", job_queue,
        "--job-definition", job_definition,
        "--array-properties", f"size={array_size}",
        "--container-overrides", f'command=["python","Optimization/FAPT_Optimization.py","--array-index","$AWS_BATCH_JOB_ARRAY_INDEX","--window-info","{window_info_file}"]'
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Successfully submitted array job {job_name} with {array_size} parallel instances")
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting array job {job_name}:")
        print(f"Error: {e.stderr}")
        return None

def main():
    args = parse_args()
    
    # Load and preprocess data
    print(f"Loading data from {args.data_path}...")
    df, _, _ = preprocess_data(pd.read_csv(args.data_path))
    
    # Calculate total number of windows
    total_windows = calculate_total_windows(len(df), args.window_size, args.step_size)
    print(f"Total windows to process: {total_windows}")
    
    # Generate timestamp for this batch run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_name = f"fapt-{args.symbol}-{timestamp}"
    
    # Submit array job
    job_result = submit_array_job(
        job_name=job_name,
        job_queue=args.job_queue,
        job_definition=args.job_definition,
        array_size=args.array_size,
        total_windows=total_windows,
        window_size=args.window_size,
        step_size=args.step_size
    )
    
    if job_result:
        # Save job information
        batch_info = {
            'timestamp': timestamp,
            'symbol': args.symbol,
            'total_windows': total_windows,
            'window_size': args.window_size,
            'step_size': args.step_size,
            'array_size': args.array_size,
            'job_name': job_name,
            'job_id': job_result.get('jobId')
        }
        
        batch_info_file = f'AWS_Batch_Jobs/{args.symbol}_batch_{timestamp}.json'
        with open(batch_info_file, 'w') as f:
            json.dump(batch_info, f, indent=4)
        
        print(f"\nBatch information saved to {batch_info_file}")
        print(f"Submitted array job with {args.array_size} parallel instances for processing {total_windows} windows")

if __name__ == "__main__":
    main() 