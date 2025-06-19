import datetime
import json
import logging
import os
import time

logger = logging.getLogger("desktopenv.experiment")

def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    agent.reset(logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    log_data = None
    action_history = ""
    while not done and step_idx < max_steps:
        
        response, actions = agent.predict(
            instruction,
            obs
        )
        action_history += f"Step {step_idx + 1}:\n {response}\n"
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            log_data = {
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
            }
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps(log_data))
                f.write("\n")
            
            with open(os.path.join(example_result_dir, "action_history.txt"), "w") as f:
                f.write(action_history)
                f.write("\n")

            if done:
                logger.info("The episode is done.")

                break
        step_idx += 1
    if log_data is not None:
        result = env.evaluate(
            task=instruction, 
            screenshot_path=os.path.join(example_result_dir, log_data["screenshot_file"]),
            action_history=action_history
        )
        
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
