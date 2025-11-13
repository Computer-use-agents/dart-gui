CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m pytest -v -s tests/osworld/test_vllm_spmd.py::test_vllm_spmd_with_rollout

CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 -m pytest -v -s tests/osworld/test_agent.py::test_agent


python -m pytest -v -s tests/osworld/test_dataset.py::test_list_env