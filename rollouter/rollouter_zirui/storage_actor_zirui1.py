import torch
import aiofiles, json
from pathlib import Path
import ray, os
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
    async def save_partial_pt(self, task_root: str, step: int, logp: list[float], token_ids: list[int] = None, prompt_token_ids: list[int] = None):
        
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"data_for_step_{step}.pt"

        data_to_save = {
            "logp": torch.tensor(logp).cpu() if logp is not None else torch.tensor([]).cpu(),
            "token_ids": torch.tensor(token_ids).cpu() if token_ids is not None else torch.tensor([]).cpu(),
            "prompt_token_ids": torch.tensor(prompt_token_ids).cpu() if prompt_token_ids is not None else torch.tensor([]).cpu(),
        }

        await asyncio.to_thread(torch.save, data_to_save, fn)

    async def save_img_pt(self, task_root: str, images: list, image_grid_thw: torch.Tensor, num_patches_list: list[int], pixel_values):
        """
        保存图像 tensor 和 patch 信息为 .pt 文件

        Args:
            task_root: 保存根目录
            images: 原始 PIL.Image.Image 列表
            image_grid_thw: 模型处理后的 tensor
            num_patches_list: 每张图的 patch 数量
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True, parents=True)
        fn = save_dir / "images_data.pt"

        data_to_save = {
            "images": images,
            "image_grid_thw": image_grid_thw.cpu(),
            "num_patches_list": num_patches_list,
            "pixel_values": pixel_values.cpu()
        }

        await asyncio.to_thread(torch.save, data_to_save, fn)

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