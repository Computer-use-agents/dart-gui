import asyncio, base64, io, ray
from typing import Dict, Any, List, Tuple, Optional
import os, pathlib, json
from PIL import Image
import time, datetime
import logging
from functools import partial

from env_k8s import RemoteDesktopEnv
from ui_tars_utils import *

from log_config import setup_logging
from copy import deepcopy


class ResourceExhaustedException(Exception):
    """资源耗尽异常"""
    pass

# 设置统一的日志系统
setup_logging()
logger = logging.getLogger(__name__)

"""
refer to run_agent_loop.TrajectoryRunner
"""

def pil_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

@ray.remote
class TrajectoryRunnerActor:
    def __init__(self, task_info: Dict[str, Any], runner_cfg):
        
        # --- load config ---
        self.runner_cfg = runner_cfg
        self.max_images = self.runner_cfg.max_images
        self.max_tests = self.runner_cfg.max_texts
        self.max_steps = self.runner_cfg.max_steps
        # self.rollout_n = self.runner_cfg.rollout_n

        # --- process task info for running ---
        self.base_messages: List = deepcopy(task_info["messages"])
        self.base_messages_for_save: List = deepcopy(task_info["messages"])
        
        # 在最终保存的message中去掉plan信息
        self.base_messages_for_save[1]["content"][0]["text"] = self.base_messages_for_save[1]["content"][0]["text"].split("Here is an instruction to help you complete the task: \n", 1)[0]
        
        self.task_cfg: Dict[str, Any] = task_info["task_config"]
        self.task_id = self.task_cfg["raw"]["task_id"] 
        self.trace_id = task_info.get("trace_id", "unknown_trace")
        
        self.task_root = f"{self.task_id}_{self.trace_id}"

        self.prompt_dialogue: List[Dict] = []   # for model client
        self.save_dialogue:   List[Dict] = []   # for partial storage
        self.save_dialogue_full: List[Dict] = []    # for full storage
        self.image_refs: List[Dict] = []  # record image position

        # --- initialize env ---
        self.env = None    # will be defined in _init_env()     
        self.env_ready = asyncio.Event()
        self.env_init_success = False  # 标识环境是否初始化成功
    def get_service_id(self) -> Optional[str]:
        """获取环境的service_id"""
        if not self.env_init_success:
            logger.warning(f"[{self.trace_id}] 环境初始化失败，无法获取service_id - task_id: {self.task_id}")
            return None
        elif self.env is None:
            logger.warning(f"[{self.trace_id}] 环境未初始化，无法获取service_id - task_id: {self.task_id}")
            return None
        elif not hasattr(self.env, 'service_id'):
            logger.warning(f"[{self.trace_id}] 环境对象缺少service_id属性 - task_id: {self.task_id}")
            return None
        elif self.env.service_id is None:
            logger.warning(f"[{self.trace_id}] 环境service_id为空 - task_id: {self.task_id}")
            return None
        else:
            logger.info(f"[{self.trace_id}] 成功获取service_id: {self.env.service_id} - task_id: {self.task_id}")
            return self.env.service_id
    async def wait_for_env_ready(self, timeout: float = 120.0) -> bool:
        """等待环境初始化完成（供外部调用）"""
        try:
            await asyncio.wait_for(self.env_ready.wait(), timeout=timeout)
            return self.env_init_success
        except asyncio.TimeoutError:
            return False
    def is_env_ready(self) -> bool:
        """检查环境是否初始化成功"""
        return self.env_init_success

    async def run_episode(self, model_pool, storage, mysql_writer):
        """
        single task rollout loop
        """
        # --- save task config ---
        storage_root = await storage.save_task_config.remote(self.task_root, self.task_cfg)

        logger.info(f"[{self.trace_id}] TrajectoryRunnerActor初始化 - task_id: {self.task_id}, task_root: {self.task_root}")
        
        try:
            # --- initialize env and first frame ----
            try:
                self._init_env(self.runner_cfg.env)
                self.env_init_success = True
                logger.info(f"[{self.trace_id}] 环境初始化成功 - task_id: {self.task_id}")
            except Exception as e:
                self.env_init_success = False
                logger.error(f"[{self.trace_id}] 环境初始化失败 - task_id: {self.task_id}, 错误: {e}")
            finally:
                # 无论成功或失败都设置事件，让等待方知道初始化过程已完成
                self.env_ready.set()
            
            # 如果初始化失败，抛出异常
            if not self.env_init_success:
                raise RuntimeError(f"[{self.trace_id}] 环境初始化失败 - task_id: {self.task_id}")
            
            all_img = []
            # --- get initial observation as first frame ----
            obs = self.env._get_obs()
            obs_img = obs["screenshot"]
            all_img.append(obs["image"])
            image_size = Image.open(BytesIO(obs["screenshot"])).size
            frame0 = await storage.save_frame.remote(self.task_root, 0, obs_img)
            self._set_first_frame(obs_img, frame0)
            

            step, done = 0, False
            logger.info(f"[{self.trace_id}] 环境初始化完成，开始主循环 - task_id: {self.task_id}")

            # def format_chat(messages):
            #     formatted = ""
            #     for m in messages:
            #         content = m["content"]

            #         # 如果 content 是 list（多模态消息）
            #         if isinstance(content, list):
            #             # 只取文本部分
            #             texts = [c["text"] for c in content if c.get("type") == "text"]
            #             content_str = "\n".join(texts)
            #         else:
            #             # 普通 string
            #             content_str = content

            #         formatted += f"{content_str}<|im_end|>\n"

            #     return formatted

            # base_messages = format_chat(self.base_messages_for_save)

            # prompt_token_ids = await self._call_model_tokenize(
            #                 model_pool, base_messages,
            #                 self.runner_cfg.model_pool)

            prompt_token_ids = await model_pool.process_text.remote(self.base_messages_for_save)

            # --- main loop ----
            # MAX_T, RETRIES, BACKOFF = 10, 3, 2
            while step < self.max_steps and not done:
                
                if step + 1 == self.max_steps:
                    action = "FAIL"
                    logger.warning(f"[{self.trace_id}] 达到最大步数限制，设置动作为FAIL - task_id: {self.task_id}, step: {step}")

                else:
                    st_step = time.time()
                    # build prompt
                    messages = self._build_messages()

                    # ---- call vLLM serve ----

                    # if local model pool not avaiable, use pre launched model server
                    st = time.time()
                    if model_pool:
                        response, model_path, vllm_logp, token_ids = await self._call_model(
                            model_pool, messages,
                            self.runner_cfg.model_pool)
                    # else:
                    #     response = await self._call_prelaunched_model(messages, self.runner_cfg.prelaunched_model)
                        
                    model_duration = time.time() - st
                    logger.info(f"[{self.trace_id}] 模型响应 - task_id: {self.task_id}, step: {step}, "
                                f"耗时: {model_duration:.2f}s, response_length: {len(response) if response else 0}")
                    logger.debug(f"[{self.trace_id}] 模型完整响应: {response}")

                    if response is None:
                        action = "FAIL"
                        logger.warning(f"[{self.trace_id}] 模型响应为空，设置动作为FAIL - task_id: {self.task_id}, step: {step}")

                    else:
                        self._add_text(response)
                        try:
                            action = self._parse(response, image_size)
                        except Exception as e:
                            logger.warning(f"Action解析失败！task_id: {self.task_id}, 错误: {e}")
                            action = "FAIL"


                # ---- interact with env ----
                if action in ["DONE", "FAIL"]:
                    logger.info(f"[{self.trace_id}] 任务完成 - task_id: {self.task_id}, step: {step}, action: {action}")

                # Execute the action
                st = time.time()
                obs, reward, done, info = await self._run_step_async(action)
                obs_img = obs["screenshot"]

                env_duration = time.time() - st
                logger.info(f"[{self.trace_id}] 环境步骤执行完成 - task_id: {self.task_id}, step: {step}, "
                            f"耗时: {env_duration:.2f}s, done: {done}")

                # ---- save screenshot ----
                if action not in ["DONE", "FAIL"]:
                    frame_path = await storage.save_frame.remote(
                        self.task_root, step + 1, obs_img)
                    
                    self._add_image(obs_img, frame_path)

                    all_img.append(obs["image"])
                
                # ---- save current trajectory
                await storage.save_partial_traj.remote(self.task_root, step + 1, self._build_trajectory())

                # ---- save vllm logp ----
                await storage.save_partial_pt.remote(self.task_root, step + 1, vllm_logp, token_ids, prompt_token_ids)
                
                step_duration = time.time() - st_step
                self._log_latency(step, model_duration, env_duration, step_duration)
                
                step += 1

            # calculate and save reward
            reward = self.env.evaluate()
            await storage.save_reward.remote(self.task_root, reward)
            logger.info(f"[{self.trace_id}] 任务评估完成 - task_id: {self.task_id}, reward: {reward}")
            
            # save trajectory json
            full_messages = self.base_messages_for_save + self.save_dialogue_full
            if full_messages:
                last_msg = full_messages[-1]
                if (
                    last_msg.get("role") == "user" and
                    len(last_msg.get("content", [])) == 1 and
                    last_msg["content"][0].get("type") == "image_url"
                ):
                    full_messages.pop()
                    all_img.pop()
            await storage.save_episode.remote(self.task_root, full_messages)

            # save img
            if False:
                image_grid_thw, num_patches_list, pixel_values = await model_pool.process_images.remote(all_img)
                await storage.save_img_pt.remote(self.task_root, all_img, image_grid_thw, num_patches_list, pixel_values)
            
            # split and save trajectory
            storage_root, split_dir, split_meta = await storage.split_episode.remote(self.task_root,
                                                                        full_messages,
                                                                        self.task_cfg,
                                                                        reward)
            n_chunks = split_meta["num_chunks"]
            logger.info(f"[{self.trace_id}] 拆分轨迹完成，共拆分为{n_chunks}条轨迹片段，已存入{split_dir}")
            
            # 插入mysql数据库
            if self.runner_cfg.write_to_mysql:
                meta = {
                'run_id': storage_root,
                'trajectory_id': self.task_root,
                'task_id': self.task_id,
                'trace_id': self.trace_id,
                'split_dir': split_dir,
                'reward': reward,
                'num_chunks': n_chunks,
                'model_version': model_path
                }
                chunks = [{'chunk_index': i, 'json_path': os.path.join(meta["split_dir"], filename)}
                        for i, filename in enumerate(split_meta["output_filenames"])]
                await mysql_writer.insert_run_and_chunks.remote(meta, chunks)
                logger.info(f"[{self.trace_id}] 已存入mySQL数据库")
                
                # # 更新训练数据表
                # insert_num = await mysql_writer.process_and_insert_trainable_group.remote(storage_root, self.rollout_n)
                # logger.info(f"Inserted {insert_num} rows into trainable_group")
            
            logger.info(f"[{self.trace_id}] 任务轨迹执行完成 - task_id: {self.task_id}, 总步数: {step}")
        except Exception as e:
            logger.error(f"[{self.trace_id}] 任务轨迹执行失败 - task_id: {self.task_id}, 错误: {e}")
        finally:
            # 无论成功或失败都确保环境被释放
            if self.env is not None:
                try:
                    self.env.close()
                    logger.info(f"[{self.trace_id}] 环境已关闭 - task_id: {self.task_id}")
                except Exception as e:
                    logger.error(f"[{self.trace_id}] 关闭环境时发生错误 - task_id: {self.task_id}, 错误: {e}")
                finally:
                    self.env = None
    
        return True
    
    def _init_env(self, env_cfg):  
        # Add retry logic for RemoteDesktopEnv initialization
        max_retries = env_cfg.max_retries
        for attempt in range(max_retries):
            try:
                self.env = RemoteDesktopEnv(
                    server_url=env_cfg.server_url,
                    user_token=env_cfg.user_token,
                    action_space=env_cfg.action_space,
                    screen_size=env_cfg.screen_size,
                    headless=env_cfg.headless,
                    os_type=env_cfg.os_type,
                    require_a11y_tree=env_cfg.require_a11y_tree,
                    task_config=self.task_cfg
                )
                logger.info(f"[{self.trace_id}] 环境初始化成功 - task_id: {self.task_id}, 尝试次数: {attempt + 1}")
                break
            except Exception as e:
                logger.warning(f"[{self.trace_id}] 环境初始化失败 - task_id: {self.task_id}, 尝试次数: {attempt + 1}, 错误: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"[{self.trace_id}] 所有重试尝试失败 - task_id: {self.task_id}")
                    raise
                logger.info(f"[{self.trace_id}] 准备重试 - task_id: {self.task_id}, 下次尝试: {attempt + 2}/{max_retries}")


    async def _call_model(self, model_pool, messages: str, model_cfg):
        """
        调模型；失败重试 RETRIES 次，指数退避 (backoff**attempt) 秒。
        返回 response (str) 或 None.
        """
        timeout = model_cfg.timeout
        retries = model_cfg.retries
        backoff = model_cfg.backoff
        DEFAULT_MODEL_PATH = model_cfg.ckpt_path
        
        for attempt in range(1, retries + 1):
            try:
                response, model_path, vllm_logp, token_ids = await asyncio.wait_for(
                    model_pool.generate.remote(messages,
                                               frequency_penalty=model_cfg.frequency_penalty,
                                               temperature=model_cfg.temperature,
                                               top_p=model_cfg.top_p,
                                               max_tokens=model_cfg.max_tokens,
                                               seed=model_cfg.seed,
                                               logprobs=model_cfg.logprobs,
                                               return_tokens_as_token_ids=model_cfg.return_tokens_as_token_ids), 
                    timeout=timeout
                )
                return response, model_path, vllm_logp, token_ids
            except (asyncio.TimeoutError, ray.exceptions.RayError) as e:
                if attempt == retries:
                    model_path = await asyncio.wait_for(
                        model_pool.get_last_model_version.remote(),
                        timeout=timeout
                    )
                    if not model_path:
                        logger.error(f" Can't get last model version, use default model path in model_cfg")
                        model_path = DEFAULT_MODEL_PATH
                    logger.error(f"[{self.trace_id}] 模型调用失败 - task_id: {self.task_id}, 重试次数: {attempt}, return last model version 错误: {e}")
                    return None, model_path, None, None
                await asyncio.sleep(backoff ** attempt)   # 2s, 4s, 8s …

    async def _call_model_tokenize(self, model_pool, messages: str, model_cfg):
        """
        调用 tokenize 接口，对格式化后的 base_messages 做 tokenization。
        返回 token_ids (List[int])。
        """
        timeout = model_cfg.timeout
        retries = model_cfg.retries
        backoff = model_cfg.backoff
        DEFAULT_MODEL_PATH = model_cfg.ckpt_path

        for attempt in range(1, retries + 1):
            try:
                token_ids = await asyncio.wait_for(
                    model_pool.tokenize.remote(messages),
                    timeout=timeout
                )
                return token_ids
            except (asyncio.TimeoutError, ray.exceptions.RayError) as e:
                if attempt == retries:
                    logger.error(f"[{self.trace_id}] tokenize 调用失败 - task_id: {self.task_id}, 重试次数: {attempt}, 错误: {e}")
                    return None
                await asyncio.sleep(backoff ** attempt) 
    
    async def _call_prelaunched_model(self, messages, model_cfg):
        
        vlm = OpenAI(
            api_key="EMPTY", 
            base_url=model_cfg.base_url)
        
        response = vlm.chat.completions.create(
                    model=model_cfg.model,
                    messages=messages,
                    frequency_penalty=model_cfg.frequency_penalty,
                    max_tokens=model_cfg.max_tokens,
                    temperature=model_cfg.temperature,
                    top_p=model_cfg.top_p,
                    seed=model_cfg.seed
                )
        # print("Raw resp: ", response)
        return response.choices[0].message.content.strip()
         
    def _add_image(self, img_bytes: bytes, frame_path: str):
        
        self.prompt_dialogue.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image;base64," + pil_to_base64(img_bytes)}
            }]
        })

        self.save_dialogue.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": frame_path
            }]
        })
        
        self.save_dialogue_full.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": frame_path
            }]
        })
        
        self.image_refs.append(
            {"source": "dialogue", "msg_idx": len(self.prompt_dialogue) - 1,
            "content_idx": None}
        )
        
        self._trim()
    
    
    def _add_text(self, assistant_txt: str):
        
        msg = {
            "role": "assistant",
            "content": add_box_token(assistant_txt)
        }
        self.prompt_dialogue.append(msg)
        self.save_dialogue.append(msg)
        self.save_dialogue_full.append(msg)
        # logger.info("Dialogue:", self.save_dialogue)
        self._trim()

        
    def _trim(self):
        """ensure image num ≤ self.max_images and assistant text num ≤ self.max_tests."""
        img_cnt  = len(self.image_refs)
        txt_cnt  = sum(m["role"] == "assistant" for m in self.prompt_dialogue)

        while img_cnt > self.max_images or txt_cnt > self.max_tests:

            # --- 图片超限：最早一张 ---
            if img_cnt > self.max_images:
                ref = self.image_refs.pop(0)
                if ref["source"] == "base":
                    self.base_messages[ref["msg_idx"]]["content"].pop(ref["content_idx"])
                else:               # dialogue 图
                    self._remove_dialogue_msg(ref["msg_idx"])
                img_cnt -= 1
                continue

            # --- 文本超限：最早 assistant 文本 ---
            if txt_cnt > self.max_tests:
                for i, m in enumerate(self.prompt_dialogue):
                    if m["role"] == "assistant":
                        self._remove_dialogue_msg(i)
                        txt_cnt -= 1
                        break
                

    def _build_messages(self) -> List[Dict]:
        return self.base_messages + self.prompt_dialogue


    def _build_trajectory(self) -> List[Dict]:
        if len(self.base_messages[1]['content']) == 1:
            return self.base_messages + self.save_dialogue
        else:
            return self.base_messages_for_save + self.save_dialogue
    
    
    def _parse(self, response, image_size):
                
        # Parse the action
        parsed_responses = parse_action_to_structure_output(
            response,
            factor=1000,  # TODO: Make this configurable
            origin_resized_height=image_size[0],
            origin_resized_width=image_size[1],
            model_type="qwen25vl",
            max_pixels=16384*28*28,
            min_pixels=100*28*28
        )
        logger.debug(f"[{self.trace_id}] 解析响应结果 - task_id: {self.task_id}, parsed_responses: {parsed_responses}")

        # Convert to pyautogui code
        action_code = parsing_response_to_pyautogui_code(
            parsed_responses,
            image_height=image_size[0],
            image_width=image_size[1],
            input_swap=False  # TODO: Make this configurable
        )
        
        logger.info(f"[{self.trace_id}] 解析动作代码 - task_id: {self.task_id}, action: {action_code}")
        
        return action_code
    
    
    async def _run_step_async(self, action):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.env.step, action))
    
    def _set_first_frame(self, obs_img, frame0):
        self.base_messages[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image;base64," + pil_to_base64(obs_img)}
            }
        )
        self.base_messages_for_save[1]["content"].append(
            {
                "type": "image_url",
                "image_url": frame0
            }
        )
        
        self.image_refs.append(
            {"source": "base", "msg_idx": 1,
            "content_idx": len(self.base_messages[1]["content"]) - 1}
        )
        
    def _log_latency(self, step, model_duration, env_duration, step_duration, out_path="timings.jsonl"):
        record = {
            "ts": datetime.datetime.now().isoformat(),
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "step": step,
            "model_duration": round(model_duration, 4),
            "env_duration": round(env_duration, 4),
            "step_duration": round(step_duration, 4),
        }
        pathlib.Path(out_path).parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            
    def _remove_dialogue_msg(self, idx: int):
        """
        删除 prompt/save 中第 idx 条 dialogue 消息，
        并同步修正 image_refs 里 dialogue 源的 msg_idx。
        """
        self.prompt_dialogue.pop(idx)
        self.save_dialogue.pop(idx)

        # 更新 image_refs
        self.image_refs = [
            r if not (r["source"] == "dialogue" and r["msg_idx"] == idx)
            else None                           # 同一条被删掉的图引用直接丢弃
            for r in self.image_refs
        ]
        self.image_refs = [
            (
                {**r, "msg_idx": r["msg_idx"] - 1}
                if r and r["source"] == "dialogue" and r["msg_idx"] > idx # idx后的图片索引均-1
                else r
            )
            for r in self.image_refs
            if r                                 # 剔除 None
        ]
