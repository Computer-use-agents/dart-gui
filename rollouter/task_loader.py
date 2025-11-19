import asyncio
from typing import List, Optional, Union, Dict, Any
import json
import os
import hashlib
from pathlib import Path
from omegaconf import DictConfig
from dataclasses import dataclass, asdict
import copy
import logging
import random
from prompts import COMPUTER_USE_PROMPT, COMPUTER_USE_PROMPT_WITH_CALL_USER
from log_config import setup_logging

# 添加 MySQL ORM 导入
from mysql_rollout import MySQLRolloutORM, DB_CONFIG

# 设置统一的日志系统
setup_logging()
logger = logging.getLogger(__name__)

class TaskLoader:
    def __init__(self, task_cfg: DictConfig, storage_root):
        self.task_file = Path(task_cfg.task_file)
        #self.task_root = Path(task_cfg.task_root)
        self.osworld_root = Path(task_cfg.osworld_root)
        
        self._latest_sha: Optional[str] = None
        self.storage_root = storage_root
        self.resume = task_cfg.resume
        # 动态 rollout_n 相关参数
        self.success_rate_threshold = getattr(task_cfg, 'success_rate_threshold', 1.0)  # 成功率阈值
        self.min_rollout_n = getattr(task_cfg, 'min_rollout_n', 1)  # 最小 rollout_n
        self.run_id = getattr(task_cfg, 'run_id', None)  # 运行ID，用于查询数据库
        self.db_enabled = getattr(task_cfg, 'db_enabled', False)  # 是否启用数据库功能
        
        # 初始化数据库连接
        self.mysql_orm = None
        if self.db_enabled and self.run_id:
            try:
                self.mysql_orm = MySQLRolloutORM(DB_CONFIG)
                logger.info("MySQL ORM 初始化成功")
            except Exception as e:
                logger.error(f"MySQL ORM 初始化失败: {e}")
                self.mysql_orm = None
        
        # # 任务采样计数字典
        # self.task_sample_count: Dict[str, int] = {}
        # import pdb; pdb.set_trace()
        # 任务成功率映射
        self.mapping_task_success: Dict[str, float] = {}
        # 任务轨迹数映射
        self.mapping_task_counts: Dict[str, float] = {}
        self.mapping_task_dynamic_n: Dict[str, float] = {}

        
        # 定期更新任务成功率映射
        self._update_mapping_task_success()

    # def poll_for_tasks(self) -> List[Dict]:
    #     """find new tasks json file
    #     return list of TaskInfo dict if there is new json
    #     else return []
    #     """
    #     # updated_flag = self._maybe_refresh_dataset()
    #     # if not updated_flag:
    #     #     return []
    #     self._maybe_refresh_dataset()

    #     return [task.to_dict() for task in self._tasks]
    def poll_for_tasks(self) -> List[Dict]:
        """find new tasks json file
        return list of TaskInfo dict if there is new json
        else return []
        """
        self._maybe_refresh_dataset()
        
        # 更新任务成功率映射
        self._update_mapping_task_success()
        
        tasks_list = [task.to_dict() for task in self._tasks]
        random.shuffle(tasks_list)

        return tasks_list 
    
    def _maybe_refresh_dataset_bak(self):
        
        # check new json
        latest_json = self._find_latest_json()

        if latest_json is None:
            return False # no json file
        
        sha = self._calc_sha1(latest_json)
        if sha == self._latest_sha:
            return False # no change
        
        with open(latest_json) as f:
            data = json.load(f)
            
        raw_tasks = [
            {"task_type": task_type, "task_id": task_id}
            for task_type, task_ids in data.items()
            for task_id in task_ids
        ]
        
        self._tasks = [build_task(raw, self.osworld_root) for raw in raw_tasks]
        self._latest_sha = sha

        logger.info(f"当前任务文件: {str(latest_json)}")
        logger.info(f"任务总数: {len(raw_tasks)}")
        
        return True
    
    def _maybe_refresh_dataset(self):
        
        latest_json = self.task_file
        print("Current tasks file: ", str(latest_json))
        
        with open(latest_json) as f:
            data = json.load(f)
            
        raw_tasks = [
            {"task_type": task_type, "task_id": task_id}
            for task_type, task_ids in data.items()
            for task_id in task_ids
        ]
        
        if self.resume:
            # 过滤已完成或类型不匹配的任务
            filtered_tasks = []
            storage_root = Path(self.storage_root)

            for raw in raw_tasks:
                task_id = str(raw["task_id"])
                task_type_expected = raw["task_type"]

                # 找到所有以 task_id 开头的子目录（允许有多个版本）
                candidate_dirs = [
                    d for d in storage_root.iterdir()
                    if d.is_dir() and d.name.startswith(task_id)
                ]

                # 默认认为任务未完成
                task_finished = False

                for d in candidate_dirs:
                    cfg_path = d / "task_config.json"
                    if not cfg_path.exists():
                        print("找不到config文件")
                        continue

                    try:
                        with cfg_path.open("r", encoding="utf-8") as cf:
                            cfg = json.load(cf)
                    except Exception:
                        print("配置损坏，忽略此目录")
                        continue

                    # 3.1 task_type 不同 => 不是同一个任务，直接跳过这目录
                    if cfg.get("raw", {}).get("task_type") != task_type_expected:
                        continue

                    # 3.2 task_type 相同，检查 reward.txt
                    if (d / "reward.txt").exists():
                        task_finished = True
                        break  # 已找到完成记录，无需再看其他目录
                if not task_finished:
                    filtered_tasks.append(raw)
            self._tasks = [build_task(raw, self.osworld_root) for raw in filtered_tasks]
            print(f"Total number of tasks: {len(raw_tasks)}, Remained:{len(filtered_tasks)}")

        else:
            self._tasks = [build_task(raw, self.osworld_root) for raw in raw_tasks]
            print(f"Total number of tasks: {len(raw_tasks)}")

        return True
        
    def _find_latest_json(self) -> Optional[Path]:
        files = list(self.task_root.glob("*.json"))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None
    
    @staticmethod
    def _calc_sha1(fp: Path, chunk_size=2<<20) -> str:
        h = hashlib.sha1()
        with fp.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _update_mapping_task_success(self):
        """更新任务成功率映射"""
        if not self.mysql_orm or not self.run_id:
            logger.info("数据库未启用或未配置 run_id，跳过更新任务成功率映射")
            return
        
        try:
            # 获取最新的模型版本的成功率数据
            # avg_nonneg, count_all, distinct_task_cnt, mapping_task_success = \
            #     self.mysql_orm.get_nth_newest_model_success(self.run_id, 1)
            avg_nonneg, count_all, distinct_task_cnt, mapping_task_success, mapping_task_counts = self.mysql_orm.get_latest_success_for_each_task(self.run_id, self.mapping_task_dynamic_n)
            self.mapping_task_success = mapping_task_success
            self.mapping_task_counts = mapping_task_counts
            logger.info(f"更新任务成功率和轨迹数映射完成，共 {len(mapping_task_success)} 个任务")
        except Exception as e:
            logger.error(f"更新任务成功率和轨迹数映射失败: {e}")

    def get_dynamic_rollout_n(self, task_id: str, rollout_n) -> int:
        """
        根据任务成功率动态计算 rollout_n 值
        - 成功率低于阈值的任务保持最大采样次数
        - 成功率高于阈值的任务采用反比例函数调整采样次数
        """
        # 获取任务成功率，默认为0
        success_rate = self.mapping_task_success.get(task_id, 0.0)
        traj_counts = self.mapping_task_counts.get(task_id, 0.0)
        
        # # 获取该任务已被采样的次数
        # sample_count = self.task_sample_count.get(task_id, 0)
        
        # 如果成功率低于阈值，保持最大采样次数
        if success_rate < self.success_rate_threshold:
            dynamic_rollout_n = rollout_n
            logger.info(f"任务 {task_id} 轨迹数 {traj_counts} 成功率 {success_rate:.2%} 低于阈值 {self.success_rate_threshold:.2%}，使用默认采样次数: {dynamic_rollout_n}")
        else:
            # weight = 1.0 / (success_rate * (sample_count + 1))
            # 为了避免除零和过小值，增加一个小常数
            # weight = 1.0 / (success_rate * (sample_count + 1) + 1e-8)
            dynamic_rollout_n = int(9 / success_rate - 7)
            dynamic_rollout_n = max(self.min_rollout_n, dynamic_rollout_n)
            logger.info(f"任务 {task_id} 轨迹数 {traj_counts} 成功率 {success_rate:.2%} 高于阈值 {self.success_rate_threshold:.2%}，使用动态采样次数: {dynamic_rollout_n} (计算值: {int(9 / success_rate - 7)})")

        
        # # 更新采样计数
        # self.task_sample_count[task_id] = sample_count + 1
        self.mapping_task_dynamic_n[task_id] = dynamic_rollout_n
        logger.info(f"任务 {task_id} 的动态 rollout_n: {dynamic_rollout_n} 成功率: {success_rate:.2%}")
        return traj_counts, success_rate, dynamic_rollout_n

    def sort_tasks_by_success_rate(self, tasks: List[Dict]) -> List[Dict]:
        """
        根据任务成功率对任务进行排序
        - 未在映射中的任务排在前面
        - 映射中的任务按成功率升序排列
        """
        def sort_key(task):
            task_id = task.get('task_config', {}).get('raw', {}).get('task_id', '')
            # 如果任务不在成功率映射中，排在前面（返回-1）
            if task_id not in self.mapping_task_success:
                return (-1, 0)  # (优先级, 成功率)
            # 如果任务在映射中，按成功率排序
            return (0, self.mapping_task_success[task_id])
        
        # 使用稳定排序
        return sorted(tasks, key=sort_key)

@dataclass
class TaskInfo:
    messages: List
    instruction: str
    task_config: Dict

    def to_dict(self):
        return asdict(self)


def build_task(raw: Dict, osworld_root: Path, use_call_user: bool = False) -> TaskInfo:

    task_type = raw["task_type"]
    task_id = raw["task_id"]
    task_path = os.path.join(osworld_root, task_type, task_id + ".json")
    with open(task_path) as f:
        task_data = json.load(f)

    task_data["raw"] = {
        "task_type": task_type,
        "task_id": task_id
    }

    instruction = task_data["instruction"]

    # if "human-ground-truth" in task_data and "single-action" in task_data["human-ground-truth"]:
    #     plan = task_data["human-ground-truth"]["single-action"]
    #     plan_text = "\n".join(plan)
    #     instruction = instruction.strip() + "\nHere is an instruction to help you complete the task: \n" + plan_text
    if "human-ground-truth" in task_data and "single-action" in task_data["human-ground-truth"]:
        plan = task_data["human-ground-truth"]["single-action"]
        plan_text = "\n".join(plan)

        plan_mode = "always"
        use_plan = False
        if plan_mode == "always":
            use_plan = True
        elif plan_mode == "never":
            use_plan = False
        elif plan_mode == "random":
            import random
            use_plan = random.choice([True, False])
        else:
            raise ValueError(f"Unknown plan_mode: {plan_mode}")
        
        if use_plan:
            instruction = instruction.strip() + "\nHere is an instruction to help you complete the task: \n" + plan_text

            

    system_prompt = COMPUTER_USE_PROMPT if not use_call_user else COMPUTER_USE_PROMPT_WITH_CALL_USER
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": system_prompt.format(
                        instruction=instruction, 
                        language="English"
                )}
            ]
        }
    ]
    

    return TaskInfo(
        messages = messages,
        instruction = instruction,
        task_config = task_data
    )



def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor


def process_text_sync(processor, messages):
        formatted = ""
        for m in messages:
            content = m["content"]

            # 如果 content 是 list（多模态消息）
            if isinstance(content, list):
                # 只取文本部分
                texts = [c["text"] for c in content if c.get("type") == "text"]
                content_str = "\n".join(texts)
            else:
                # 普通 string
                content_str = content

            formatted += f"{content_str}<|im_end|>\n"
        model_inputs = processor(text=[formatted], images=None, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        return input_ids[0]


import os
import json
import torch

# 假设这些变量和函数已经在别处定义好了
# def hf_processor(path, trust_remote_code, use_fast): ...
# def process_text_sync(processor, messages): ...
# COMPUTER_USE_PROMPT = "..."

def process_system_prompt(instruction_root='evaluation_examples/examples_linux_osworld_0912'):
    # 假设 hf_processor 和 COMPUTER_USE_PROMPT 已经定义
    processor = hf_processor('/workspace/huggingface/dart-gui-7b', trust_remote_code=True, use_fast=True)
    system_prompt = COMPUTER_USE_PROMPT  
    
    # --- 修改点 1: 定义输出文件路径 ---
    output_dir = 'evaluation_examples/system_prompt_token_ids'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'system_prompts.pt') # 定义最终的输出文件名
    print(f"Output will be saved to '{output_file_path}'")

    # --- 修改点 2: 初始化一个字典来收集所有数据 ---
    all_token_ids = {}

    # 遍历所有子目录
    for subdir_name in sorted(os.listdir(instruction_root)):
        subdir_path = os.path.join(instruction_root, subdir_name)
        if not os.path.isdir(subdir_path):
            continue
            
        # 遍历目录中的所有文件
        for file_name in sorted(os.listdir(subdir_path)):
            if not file_name.endswith('.json'):
                continue

            file_path = os.path.join(subdir_path, file_name)
            task_id = file_name.split('.')[0]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                instruction = data.get('instruction')
                if not instruction:
                    print(f"Warning: 'instruction' key not found or empty in {file_path}. Skipping.")
                    continue

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": system_prompt.format(
                                    instruction=instruction, 
                                    language="English"
                            )}
                        ]
                    }
                ]
                
                input_ids = process_text_sync(processor, messages)
                
                # --- 修改点 3: 将结果存入字典，而不是保存为单个文件 ---
                if input_ids is not None:
                    all_token_ids[task_id] = input_ids
                    print(f"Processed and collected token IDs for task: {task_id}")
                else:
                    print(f"Warning: process_text_sync returned None for task {task_id}. Skipping.")
                 
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {file_path}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred while processing {file_path}: {e}")

    # --- 修改点 4: 循环结束后，将整个字典保存到单个文件 ---
    if all_token_ids:
        torch.save(all_token_ids, output_file_path)
        print(f"\nSuccessfully saved {len(all_token_ids)} token ID sets to {output_file_path}")
    else:
        print("\nNo token IDs were generated. Output file was not created.")


def main():
    process_system_prompt()

if __name__ == "__main__":
    main()