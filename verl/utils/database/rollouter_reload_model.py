
import requests
import warnings

def reload_model(service_url: str, new_ckpt_path: str, batch_size: int = 1, timeout: int = 30):

    url = f"{service_url.rstrip('/')}/reload"
    payload = {
        "new_ckpt_path": new_ckpt_path,
        "batch_size": int(batch_size),
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        warnings.warn(f"[reload_model] 请求失败（网络/超时等）：{e}")
        return {"ok": False, "status": None, "data": None, "error": str(e)}

    if resp.status_code >= 400:
        text = resp.text
        warnings.warn(f"[reload_model] 接口返回错误 {resp.status_code}: {text}")
        return {"ok": False, "status": resp.status_code, "data": None, "error": text}

    try:
        data = resp.json()
        return {"ok": True, "status": resp.status_code, "data": data, "error": None}
    except ValueError:
        warnings.warn("[reload_model] 成功返回但非 JSON，已用文本替代。")
        return {"ok": True, "status": resp.status_code, "data": resp.text, "error": None}
    
    
if __name__=='__main__':
    reload_result = reload_model(
    service_url='http://172.19.171.69:15959',
    new_ckpt_path='/workspace/computer-use/verl/checkpoints/verl_osworld_grpo/sync_pass8_singlehard_lr1e-6_bz1_minibs30_paddingmask_stepwise_kl_maxstep30/global_step_9/actor/huggingface',
    batch_size=1)
    print(reload_result)