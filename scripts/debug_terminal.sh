#!/bin/bash

# Script to continuously sleep and log messages
# This script will run indefinitely, sleeping and logging messages
pip install backoff
pip install swanlab
pip install nvitop
cd /app/data/arpo_workspace/verl

ray start --head --dashboard-host=0.0.0.0

# CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 -m pytest -v -s tests/osworld/test_agent.py::test_agent

while true; do
    # Get current timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log message with timestamp
    echo "[$timestamp] Debug terminal is running..."
    
    # Sleep for 5 seconds
    sleep 5
done
