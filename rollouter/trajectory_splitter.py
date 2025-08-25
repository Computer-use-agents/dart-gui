import os
import json
import copy
from collections import defaultdict

class TrajectorySplitter:
    def __init__(
            self,
            root_dir: str,
            window_size: int = 5,
            stride_size: int = 5,
            text_num: int = 30
        ) -> None:
        """
        初始化。

        Args:
            root_dir (str): 包含原始轨迹数据的根目录。
            window_size (int): 每个拆分片段包含的对话轮次 (user + assistant) 数量。
            stride_size (int): 滑动窗口的步长（单位：对话轮次）。
            text_num (int): 窗口前保留的单独的文本消息数量
        """
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride_size = stride_size
        self.text_num = max(text_num, 0)    # 理论上文本数量应大于图片数量，保险起见加入防越界逻辑
        print(f"Rollout splitter initialized with window_size={window_size}, stride_size={stride_size}")
        
    def split_and_save(
            self,
            dataset_id: str,
            output_dir: str,
            full_messages: list[dict],
            task_config: dict,
            reward: float
        ) -> int:
        """
        从内存中接收轨迹数据，进行拆分，并将片段保存到输出目录。

        Args:
            dataset_id (str): 轨迹的唯一标识符，用于命名输出文件。
            output_dir (str): 用于保存拆分后的JSON文件的目录。
            full_messages (list[dict]): 完整的消息列表（内容同 final_messages.json）。
            task_config (dict): 任务配置字典（内容同 task_config.json）。
            reward (float): 环境给出的奖励值（内容同 reward.txt）

        Returns:
            int: 成功生成的片段数量。
        """
        print(f"Processing trajectory from memory: {dataset_id}...")
        
        message_chunks = self._split_trajectory(full_messages)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        output_filenames = []
        for i, messages in enumerate(message_chunks):
            output_data = {
                "task_config": task_config,   # 直接用传入的 task_config
                "messages": messages,
                "source_dataset_id": dataset_id,
                "chunk_index": i,
                "reward": reward
            }
            output_filename = f"{dataset_id}_chunk_{i}.json"
            output_filepath = os.path.join(output_dir, output_filename)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            output_filenames.append(output_filename)
        
        num_chunks = len(message_chunks)
        if num_chunks > 0:
            print(f"Successfully split {dataset_id} into {num_chunks} chunks in '{output_dir}'.")
        else:
            print(f"Trajectory {dataset_id} did not produce any chunks based on the window/stride settings.")
            
        # 组装并保存 manifest
        meta = {
            "stride_size": self.stride_size,
            "window_size": self.window_size,
            "num_chunks": num_chunks,
            "output_filenames": output_filenames,
        }
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
        print(f"Saved manifest to {manifest_path}.")
            
        return meta
    
    def _split_trajectory(self, dataset: list[dict]) -> list[list[dict]]:
        """
        (内部方法) 对内存中的轨迹数据执行滑动窗口拆分逻辑。
        只负责拆分消息列表。
        """
        start = 1
        end = start + 2 * self.window_size
        n_msg = len(dataset)
        
        if n_msg < 3:
            return []

        message_chunks = []
        instruction = copy.deepcopy(dataset[1])
        is_first_turn = True

        while end <= n_msg+1:
            assert dataset[start]["role"] == "user", f"Expected 'user' role at index {start}, but got {dataset[start]['role']}"
            
            if len(dataset[start]["content"]) == 1:
                instruction["content"] = instruction["content"][:1]

            item = self._process_item(dataset, copy.deepcopy(instruction), start, end, is_first_turn)
            is_first_turn = False
            
            message_chunks.append(item)
            
            start += 2 * self.stride_size
            end = start + 2 * self.window_size
            
        return message_chunks

    def _process_item(self, dataset: list, instruction: dict, start: int, end: int, is_first_turn: bool) -> list:
        """
        (内部方法) 根据窗口的起止位置，构建一个消息片段。 (此方法无需改动)
        """
        system_prompt = dataset[0]
        
        message_body = []
        for i in range(max(start-self.text_num*2, 0), start): # 仅保留一定数量的文本消息，与rollout逻辑一致
            if dataset[i]["role"] == "assistant":
                message_body.append(copy.deepcopy(dataset[i]))
        
        current_instruction = copy.deepcopy(instruction)
        
        if is_first_turn:
            message_body.extend(copy.deepcopy(dataset[start+1 : end]))
        else:
            message_body.extend(copy.deepcopy(dataset[start : end]))
            
        item = [
            copy.deepcopy(system_prompt),
            current_instruction,
        ]
        item.extend(message_body)
        
        return item  

    # ---- 根据trainer代码迁移过来的逻辑，已进行优化，此处做备份 ----
    # def process_and_save(self, dataset_id: str, output_dir: str):
    #     """
    #     处理单个轨迹并将其拆分后的片段保存到输出目录。

    #     Args:
    #         dataset_id (str): 要处理的轨迹ID。
    #         output_dir (str): 用于保存拆分后的JSON文件的目录。
        
    #     Returns:
    #         int: 成功生成的片段数量。
    #     """
    #     print(f"Processing trajectory: {dataset_id}...")
    #     try:
    #         split_data = self._split_dataset_id(dataset_id)
    #     except FileNotFoundError:
    #         print(f"[Error] Could not find data for dataset_id: {dataset_id} in {self.root_dir}. Skipping.")
    #         return 0
        
    #     # 确保输出目录存在
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     for i, (messages, task_config, reward) in enumerate(split_data):
    #         # 将消息和任务配置合并到一个字典中
    #         output_data = {
    #             "task_config": task_config,
    #             "messages": messages,
    #             "source_dataset_id": dataset_id,
    #             "chunk_index": i
    #         }
            
    #         # 定义输出文件名
    #         output_filename = os.path.join(output_dir, f"{dataset_id}_chunk_{i}.json")
            
    #         # 保存为JSON文件
    #         with open(output_filename, 'w', encoding='utf-8') as f:
    #             json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    #     print(f"Successfully split {dataset_id} into {len(split_data)} chunks in '{output_dir}'.")
    #     return len(split_data)

    # def _split_dataset_id(self, dataset_id: str) -> list[tuple[list[dict], dict]]:
    #     """
    #     (内部方法) 执行滑动窗口拆分逻辑。
    #     """
    #     dataset_dir = os.path.join(self.root_dir, dataset_id)
    #     message_path = os.path.join(dataset_dir, "final_messages.json")
    #     with open(message_path, 'r', encoding='utf-8') as f:
    #         dataset = json.load(f)
        
    #     config_path = os.path.join(dataset_dir, "task_config.json")
    #     with open(config_path, 'r', encoding='utf-8') as f:
    #         task_config = json.load(f)

    #     # 窗口逻辑基于消息索引。一个 "turn" (stride/window) 包含 user 和 assistant 两个消息。
    #     # 因此，索引步长为 2 * stride_size。
    #     start = 1  # 从第一个 'user' 消息开始
    #     end = start + 2 * self.window_size
    #     n_msg = len(dataset)
        
    #     batch_data = []
    #     # instruction 可能是指第一个 'user' 消息，这里我们对其进行深拷贝以保证安全
    #     instruction = copy.deepcopy(dataset[1])
    #     is_first_turn = True

    #     while end <= n_msg:
    #         # 确保窗口的起始消息是 'user'
    #         assert dataset[start]["role"] == "user", f"Expected 'user' role at index {start}, but got {dataset[start]['role']}"
            
    #         # 如果是第一个窗口，且user消息只有一个文本内容（没有图片），
    #         # 那么instruction也应该只保留文本。
    #         if len(dataset[start]["content"]) == 1:
    #             instruction["content"] = instruction["content"][:1]

    #         item = self._process_item(dataset, copy.deepcopy(instruction), start, end, is_first_turn)
    #         is_first_turn = False
            
    #         batch_data.append((item, copy.deepcopy(task_config)))
            
    #         # 移动窗口
    #         start += 2 * self.stride_size
    #         end = start + 2 * self.window_size
            
    #     return batch_data

    # def _process_item(self, dataset: list, instruction: dict, start: int, end: int, is_first_turn: bool) -> list:
    #     """
    #     (内部方法) 根据窗口的起止位置，构建一个消息片段。
    #     """
    #     system_prompt = dataset[0]
        
    #     # 包含窗口之前所有的 assistant 回复
    #     message_body = []
    #     for i in range(max(start-self.text_num*2, 0), start):
    #         if dataset[i]["role"] == "assistant":
    #             message_body.append(copy.deepcopy(dataset[i]))
        
    #     current_instruction = copy.deepcopy(instruction)
        
    #     # 根据是否为第一个片段，决定是否包含第一个'user'消息(instruction)
    #     if is_first_turn:
    #         message_body.extend(copy.deepcopy(dataset[start+1 : end]))
    #     else:
    #         message_body.extend(copy.deepcopy(dataset[start : end]))
            
    #     # 组装最终的消息列表
    #     item = [
    #         copy.deepcopy(system_prompt),
    #         current_instruction,
    #     ]
    #     item.extend(message_body)
        
    #     return item

# --- 使用示例 ---
def create_dummy_data(base_dir, dataset_id, num_turns=10):
    """一个辅助函数，用于创建测试所需的虚拟数据。"""
    traj_path = os.path.join(base_dir, dataset_id)
    os.makedirs(traj_path, exist_ok=True)
    
    # 创建 task_config.json
    with open(os.path.join(traj_path, "task_config.json"), 'w') as f:
        json.dump({"id": f"task_for_{dataset_id}"}, f)
        
    # 创建 final_messages.json
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(num_turns):
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"User turn {i+1}"}, {"type": "image", "image": f"img_{i+1}"}]
        })
        messages.append({"role": "assistant", "content": f"Assistant response {i+1}"})
    with open(os.path.join(traj_path, "final_messages.json"), 'w') as f:
        json.dump(messages, f, indent=2)
        
def test_on_dummy_data():
    
    # 建立临时的测试目录结构
    import tempfile
    import shutil
    
     # 使用 with 语句确保临时目录在结束后被清理
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_data_dir = os.path.join(temp_dir, "raw_trajectories")
        processed_data_dir = os.path.join(temp_dir, "processed_chunks")
        
        print(f"Created temporary raw data directory: {raw_data_dir}")
        print(f"Created temporary processed data directory: {processed_data_dir}")

        # 1. 创建虚拟的原始数据
        DUMMY_DATASET_ID = "trajectory_abc_123"
        create_dummy_data(raw_data_dir, DUMMY_DATASET_ID, num_turns=12)

        # 2. 初始化 TrajectorySplitter
        # window_size=5, stride_size=5, num_turns=12
        # 第一次: start=1, end=11. (n_msg=25) -> 生成1个chunk
        # 第二次: start=11, end=21. -> 生成1个chunk
        # 第三次: start=21, end=31. (end > n_msg) -> 停止
        # 预期会生成 2 个文件
        splitter = TrajectorySplitter(
            root_dir=raw_data_dir,
            window_size=5,
            stride_size=5
        )

        # 3. 执行拆分和保存
        num_chunks = splitter.process_and_save(
            dataset_id=DUMMY_DATASET_ID,
            output_dir=processed_data_dir
        )
        
        # 4. 验证结果
        print("\n--- Verification ---")
        print(f"Number of chunks created: {num_chunks}")
        
        output_files = os.listdir(processed_data_dir)
        print(f"Files found in output directory: {output_files}")
        
        assert num_chunks == 2, f"Expected 2 chunks, but got {num_chunks}"
        assert len(output_files) == 2, f"Expected 2 files in output directory, but found {len(output_files)}"
        
        # 随机抽查一个文件的内容
        with open(os.path.join(processed_data_dir, output_files[0]), 'r') as f:
            first_chunk_data = json.load(f)
        
        assert "task_config" in first_chunk_data
        assert "messages" in first_chunk_data
        assert "source_dataset_id" in first_chunk_data
        assert first_chunk_data["source_dataset_id"] == DUMMY_DATASET_ID
        print("\nContent of the first chunk looks correct.")
        print(json.dumps(first_chunk_data, indent=2, ensure_ascii=False))

        print("\nRollout splitter test completed successfully!")
    

if __name__ == '__main__':
    
    raw_data_dir = "results/pass@1_all_32env_6model_tmp07_max-texts-35"
    splitter = TrajectorySplitter(
                root_dir=raw_data_dir,
                window_size=5,
                stride_size=5
                )
    
    num_chunks = splitter.process_and_save(
            dataset_id="0a0faba3-5580-44df-965d-f562a99b291c_1754049580762",
            output_dir="results_splitted"
        )
    
    print(f"Number of chunks created: {num_chunks}")