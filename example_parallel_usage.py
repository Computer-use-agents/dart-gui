#!/usr/bin/env python3
"""
Example script demonstrating how to use the parallel trajectory splitting functionality.
"""

import torch
from transformers import AutoProcessor
from verl.trainer.ppo.trajectory_splitter import TrajectorySplitter, StepwiseTrajectorySplitter

def main():
    # Initialize the processor (you'll need to replace with your actual model)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
    
    # Initialize the trajectory splitter
    splitter = TrajectorySplitter(
        processor=processor,
        root_dir="/path/to/your/dataset/root",
        window_size=5,
        stride_size=5,
        max_prompt_length=32048,
        max_response_length=32000,
        truncation="error",
        limit_images=5
    )
    
    # Example dataset IDs
    dataset_ids = ["dataset_001", "dataset_002", "dataset_003", "dataset_004"]
    
    # Example reward tensor
    reward_tensor = torch.tensor([0.8, 0.6, 0.9, 0.7])
    
    print("Processing with sequential method...")
    # Sequential processing (original method)
    result_sequential = splitter.split(dataset_ids, reward_tensor)
    
    print("Processing with parallel method...")
    # Parallel processing (new method)
    result_parallel = splitter.split_parallel(
        dataset_ids, 
        reward_tensor, 
        num_cpus=4
    )
    
    print("Results should be identical!")
    print(f"Sequential result keys: {list(result_sequential.keys())}")
    print(f"Parallel result keys: {list(result_parallel.keys())}")
    
    # Example with StepwiseTrajectorySplitter
    print("\n--- Using StepwiseTrajectorySplitter ---")
    stepwise_splitter = StepwiseTrajectorySplitter(
        processor=processor,
        root_dir="/path/to/your/dataset/root",
        window_size=5,
        stride_size=1,
        max_prompt_length=32048,
        max_response_length=32000,
        truncation="error",
        limit_images=5
    )
    
    # Parallel processing with stepwise splitter
    result_stepwise_parallel = stepwise_splitter.split_parallel(
        dataset_ids, 
        reward_tensor, 
        num_cpus=4
    )
    
    print("Stepwise parallel processing completed!")

if __name__ == "__main__":
    main() 