import torch
import aiofiles, json
from pathlib import Path
import ray, os
from typing import List
import asyncio
from trajectory_splitter import TrajectorySplitter

@ray.remote
class StorageActor:
    def __init__(self, storage_cfg):
        self.root = Path(storage_cfg.root)
        self.root.mkdir(parents=True, exist_ok=True)
        
        # init splitter
        self.splitter = TrajectorySplitter(
            self.root,
            storage_cfg.splitter.window_size,
            storage_cfg.splitter.stride_size,
            storage_cfg.splitter.max_texts - storage_cfg.splitter.max_images
        )
        self.splitted_root = storage_cfg.splitter.output_dir

    # ---- save screenshot ----
    async def save_frame(self, task_root: str, step: int, png_bytes: bytes) -> str:
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn   = save_dir / f"image_{step:04d}.png"
        async with aiofiles.open(fn, "wb") as f:
            await f.write(png_bytes)
        return str(fn.relative_to(self.root / task_root))
    
    # ---- save partial trajectory json ----
    async def save_partial_traj(self, task_root: str, step: int, partial_traj: list[dict]):
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"msg_for_prompt_{step}.json"
        async with aiofiles.open(fn, "w") as f:
            await f.write(json.dumps(partial_traj, ensure_ascii=False, indent=2))
            
    # ---- save vllm logp ----
    async def save_partial_vllm_logp(self, task_root: str, step: int, logp: List[float]):
        
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"vllm_logp_for_step_{step}.pt"
        if logp is None:
            logp_cpu = torch.tensor([]).cpu()
        else:
            logp_cpu = torch.tensor(logp).cpu()
        # torch.save(logp_cpu, fn)
        await asyncio.to_thread(torch.save, logp_cpu, fn)

    # ---- save full trajectory json ----
    async def save_episode(self, task_root: str, episode_summary: list[dict]):
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"final_messages.json"
        async with aiofiles.open(fn, "w") as f:
            await f.write(json.dumps(episode_summary, ensure_ascii=False, indent=2))

    # ---- save reward txt ----
    async def save_reward(self, task_root: str, reward: float):
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        with open(save_dir / "reward.txt", "w") as f:
            f.write(str(reward))
        with open(save_dir / "reward_from_env.txt", "w") as f:
            f.write(str(reward))
            
    # ---- save task config(task info) json ----
    async def save_task_config(self, task_root: str, task_config: dict):
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"task_config.json"
        async with aiofiles.open(fn, "w") as f:
            await f.write(json.dumps(task_config, ensure_ascii=False, indent=2))
        return str(self.root)
            
    # ---- split and save full trajectory json ----
    async def split_episode(self, 
                            task_root: str,
                            full_messages: list[dict],
                            task_config: dict,
                            reward: float
                            ) -> tuple[str, int]:

        dataset_id = task_root
        out_dir = os.path.join(self.root, dataset_id, self.splitted_root)

        split_meta = self.splitter.split_and_save(
            dataset_id=dataset_id,
            output_dir=out_dir,
            full_messages=full_messages,
            task_config=task_config,
            reward=reward
        )
        return str(self.root), self.splitted_root, split_meta