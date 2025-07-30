# Parallel Trajectory Splitting with Ray

This document explains how to use the new parallel trajectory splitting functionality that has been added to both `TrajectorySplitter` and `StepwiseTrajectorySplitter` classes.

## Overview

The parallel implementation uses [Ray](https://www.ray.io/) to distribute the processing of multiple dataset IDs across multiple CPU cores, significantly speeding up the trajectory splitting process when dealing with large numbers of datasets.

## New Methods

Both classes now have a new method called `split_parallel()` that provides the same functionality as the original `split()` method but with parallel processing.

### Method Signature

```python
def split_parallel(
    self, 
    dataset_ids: list[str], 
    reward_tensor: torch.Tensor | None = None,
    num_cpus: int = 4
) -> DataProto:
```

### Parameters

- `dataset_ids`: List of dataset IDs to process (same as original)
- `reward_tensor`: Optional tensor of rewards for each dataset (same as original)
- `num_cpus`: Number of CPU cores to use for parallel processing (new parameter)

### Returns

Returns the same `DataProto` object as the original `split()` method.

## Usage Examples

### Basic Usage

```python
from verl.trainer.ppo.trajectory_splitter import TrajectorySplitter
from transformers import AutoProcessor

# Initialize processor and splitter
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
splitter = TrajectorySplitter(
    processor=processor,
    root_dir="/path/to/dataset",
    window_size=5,
    stride_size=5
)

# Dataset IDs to process
dataset_ids = ["dataset_001", "dataset_002", "dataset_003"]
reward_tensor = torch.tensor([0.8, 0.6, 0.9])

# Parallel processing
result = splitter.split_parallel(
    dataset_ids, 
    reward_tensor, 
    num_cpus=4
)
```

### StepwiseTrajectorySplitter

```python
from verl.trainer.ppo.trajectory_splitter import StepwiseTrajectorySplitter

stepwise_splitter = StepwiseTrajectorySplitter(
    processor=processor,
    root_dir="/path/to/dataset",
    window_size=5,
    stride_size=1
)

# Parallel processing with stepwise splitter
result = stepwise_splitter.split_parallel(
    dataset_ids, 
    reward_tensor, 
    num_cpus=4
)
```

## Performance Benefits

The parallel implementation can provide significant speedup when processing multiple datasets:

- **Sequential**: Processes one dataset at a time
- **Parallel**: Processes multiple datasets simultaneously across CPU cores

The actual speedup depends on:
- Number of CPU cores available
- Number of datasets to process
- I/O bottlenecks (file reading)
- Memory availability

## Requirements

- Ray must be installed: `pip install ray`
- Sufficient memory to hold multiple datasets in memory simultaneously
- Multiple CPU cores for optimal performance

## Ray Initialization

The method automatically initializes Ray if it hasn't been initialized yet. You can also initialize Ray manually before calling the method:

```python
import ray

# Initialize Ray with specific resources
ray.init(num_cpus=8, num_gpus=1)

# Then use the parallel method
result = splitter.split_parallel(dataset_ids, reward_tensor, num_cpus=8)
```

## Error Handling

The parallel implementation maintains the same error handling as the original sequential version. If any dataset processing fails, the error will be propagated back to the caller.

## Memory Considerations

Since multiple datasets are processed simultaneously, memory usage will be higher than the sequential version. Monitor your system's memory usage and adjust `num_cpus` accordingly if you encounter memory issues.

## Compatibility

The parallel method produces identical results to the sequential method, so you can safely switch between them without affecting your training pipeline. 