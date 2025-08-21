
from __future__ import annotations

import os
import re
import json
import html
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Iterable, Optional

# -----------------------------
# Minimal SampleVisualizer
# -----------------------------
class SampleVisualizer:
    """
    Minimal, dependency-free visualizer for debugging collated samples.
    - Keeps prompts/responses structure
    - Replaces image sentinels with original file paths (best-effort)
    - Shows key metadata (dataset_id, uid, step, token counts)
    - Highlights response_mask at per-token granularity
    """

    def __init__(
        self,
        out_dir: str,
        dataset_root: str,
        viz_every: int = 200,
        max_samples: int = 4,
        max_chars: int = 12000,
        autorefresh_sec: Optional[int] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.dataset_root = Path(dataset_root)
        self.viz_every = int(viz_every)
        self.max_samples = int(max_samples)
        self.max_chars = int(max_chars)
        self.autorefresh_sec = autorefresh_sec

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Cache: dataset_id -> list[str] image paths (order they appear in final_messages.json)
        self._image_map_cache: Dict[str, List[str]] = {}
        self._worker_thread: Optional[threading.Thread] = None
        self._q: "queue.Queue[Tuple[Any, Any, int, Dict[str, Any], Optional[str]]]" = queue.Queue(maxsize=4)
        self._lock = threading.Lock()

    # -----------------------------
    # Public API
    # -----------------------------
    def render(
        self,
        batch: Dict[str, Any],
        tokenizer: Any,
        step: int,
        meta: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Render one HTML report synchronously and return the saved path."""
        meta = meta or {}
        # Select up to max_samples from batch
        n_samples = self._infer_batch_size(batch)
        indices = list(range(min(n_samples, self.max_samples)))

        cards_html = []
        for i in indices:
            cards_html.append(self._render_sample_card(batch, tokenizer, i))

        html_doc = self._wrap_html("\n".join(cards_html), step=step, meta=meta)
        if save_path is None:
            save_path = str(self.out_dir / f"viz_step_{int(step):07d}.html")
        Path(save_path).write_text(html_doc, encoding="utf-8")

        # Also maintain latest.html as a stable entry
        (self.out_dir / "latest.html").write_text(html_doc, encoding="utf-8")

        return save_path

    def render_async(
        self,
        batch: Dict[str, Any],
        tokenizer: Any,
        step: int,
        meta: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Enqueue a render task; the newest tasks preempt older ones when queue is full."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
        # non-blocking put: drop oldest on overflow
        try:
            self._q.put_nowait((batch, tokenizer, step, meta or {}, save_path))
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except Exception:
                pass
            self._q.put_nowait((batch, tokenizer, step, meta or {}, save_path))

    # -----------------------------
    # Worker
    # -----------------------------
    def _worker_loop(self) -> None:
        while True:
            batch, tokenizer, step, meta, save_path = self._q.get()
            try:
                self.render(batch, tokenizer, step, meta, save_path)
            except Exception as e:
                # best-effort: swallow errors to avoid crashing training
                err_path = self.out_dir / "visualizer_error.log"
                with err_path.open("a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().isoformat()}] step={step} error: {repr(e)}\n")
            finally:
                self._q.task_done()

    # -----------------------------
    # HTML helpers
    # -----------------------------
    def _wrap_html(self, inner: str, step: int, meta: Dict[str, Any]) -> str:
        title = f"Samples Visualization — step {step}"
        meta_kv = " • ".join([f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in meta.items()])
        refresh_tag = (
            f'<meta http-equiv="refresh" content="{int(self.autorefresh_sec)}">'
            if self.autorefresh_sec and self.autorefresh_sec > 0
            else ""
        )
        styles = self._css()
        legend = self._legend_html()
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
{refresh_tag}
<title>{html.escape(title)}</title>
<style>
{styles}
</style>
</head>
<body>
<div class="page">
  <header class="page-header">
    <h1>{html.escape(title)}</h1>
    <div class="meta">Generated at {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} • {html.escape(meta_kv)}</div>
    {legend}
  </header>
  <main>
    {inner}
  </main>
  <footer class="page-footer">
    <div>Saved by SampleVisualizer • dataset_root={html.escape(str(self.dataset_root))}</div>
  </footer>
</div>
</body>
</html>"""

    def _css(self) -> str:
        return """
:root { --bg:#0b0d10; --card:#12151a; --muted:#8a9099; --text:#e6e6e6; --accent:#4da3ff; --ok:#59d38c; --warn:#ffcc66; }
*{ box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,Monaco,monospace; margin:0; }
.page { width: 1100px; margin: 18px auto; }
.page-header { margin-bottom: 18px; }
.page-header h1 { margin: 0 0 4px 0; font-size: 20px; }
.page-header .meta { color: var(--muted); font-size: 12px; }
.page-footer { color: var(--muted); font-size: 12px; margin: 12px 0 40px 0; }

.legend { margin: 10px 0 18px 0; font-size: 12px; color: var(--muted); }
.legend .chip { display:inline-block; padding: 2px 8px; border-radius: 12px; margin-right: 8px; }
.legend .mask1 { background: rgba(89,211,140,0.2); color: var(--text); border: 1px solid rgba(89,211,140,0.35); }
.legend .mask0 { background: rgba(255,204,102,0.18); color: var(--text); border: 1px solid rgba(255,204,102,0.4); }
.legend .img   { background: rgba(77,163,255,0.2); color: var(--text); border: 1px solid rgba(77,163,255,0.35); }

.card { background: var(--card); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; margin-bottom: 16px; overflow: hidden; }
.card .card-hd { padding: 10px 12px; border-bottom: 1px dashed rgba(255,255,255,0.08); display:flex; align-items:center; gap:10px; }
.card .tag { font-size: 11px; color: var(--muted); background: rgba(255,255,255,0.04); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.08); }
.card .stats { margin-left:auto; font-size: 11px; color: var(--muted); }
.card .title { font-size: 13px; font-weight: 600; }
.card .sec { display:flex; }
.card .col { width: 50%; padding: 10px 12px; }
.card h3 { margin: 0 0 6px 0; font-size: 12px; color: var(--muted); font-weight: 600; letter-spacing: .02em; }
.card pre, .card .resp { font-size: 12px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; max-height: 480px; overflow: auto; padding: 10px; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; background: rgba(255,255,255,0.02); }
.card .ledger { padding: 8px 12px 12px 12px; border-top: 1px dashed rgba(255,255,255,0.08); font-size: 12px; color: var(--muted); }
.card .ledger ul { margin: 6px 0 0 18px; }

/* token coloring */
.token { padding: 0 0px; }
.token.mask1 { font-weight: 700; }         /* participated (unmasked for loss) */
.token.mask0 { opacity: 0.35; }            /* excluded */
.token.img   { color: var(--accent); font-weight: 600; }

.small { font-size: 11px; color: var(--muted); }
        """

    def _legend_html(self) -> str:
        return """
<div class="legend">
  <span class="chip mask1">mask==1 (参与学习)</span>
  <span class="chip mask0">mask==0 (未参与/填充/忽略)</span>
  <span class="chip img">[IMG: path] 图片占位</span>
</div>
        """

    # -----------------------------
    # Sample card rendering
    # -----------------------------
    def _render_sample_card(self, batch: Dict[str, Any], tokenizer: Any, i: int) -> str:
        # --- gather fields ---
        dataset_id = self._get_str(batch.get("dataset_ids"), i)
        uid = (
            self._get_str(batch.get("uids"), i)
            or self._get_str(batch.get("uid"), i)
            or self._get_str(batch.get("ids"), i)
        )

        prompts_ids = self._get_list(batch.get("prompts"), i)
        responses_ids = self._get_list(batch.get("responses"), i)
        response_mask = self._get_list(batch.get("response_mask"), i)
        attention_mask = self._get_list(batch.get("attention_mask"), i)  # optional

        # Trim pads for prompts/responses
        pad_id = getattr(tokenizer, "pad_token_id", None)
        prompts_ids = self._trim_trailing_pad(prompts_ids, pad_id)
        responses_ids, response_mask = self._trim_pair_by_pad(responses_ids, response_mask, pad_id)

        # Build image list (best-effort) for this dataset
        image_paths_all = self._ensure_image_list(dataset_id)

        # --- render prompts text ---
        prompts_text = self._decode_sequence(prompts_ids, tokenizer)
        prompts_text = self._inject_images_best_effort(prompts_text, image_paths_all)[0]
        prompts_text = self._truncate_chars(prompts_text, self.max_chars)

        # --- render responses with token-level mask coloring ---
        resp_html, used_paths = self._render_responses_tokens(
            responses_ids, response_mask, tokenizer, image_paths_all
        )

        # Stats
        total = len(responses_ids)
        num_ones = sum(1 for v in response_mask if int(v) == 1)
        ratio = f"{num_ones}/{total} = {100.0 * num_ones / max(total,1):.1f}%"

        # Layout
        title = html.escape(f"{dataset_id or '?'} — {uid or 'sample'}")
        stats = html.escape(f"resp_mask: {ratio} • #prompt_tok={len(prompts_ids)} • #resp_tok={len(responses_ids)}")

        ledger_items = "".join([f"<li>{html.escape(p)}</li>" for p in used_paths]) or "<li>（无）</li>"

        return f"""
<div class="card">
  <div class="card-hd">
    <span class="title">{title}</span>
    <span class="tag">dataset_id</span><span class="tag">{html.escape(dataset_id or "?")}</span>
    <span class="tag">uid</span><span class="tag">{html.escape(uid or "?")}</span>
    <div class="stats">{stats}</div>
  </div>
  <div class="sec">
    <div class="col">
      <h3>Prompts</h3>
      <pre>{html.escape(prompts_text)}</pre>
    </div>
    <div class="col">
      <h3>Responses <span class="small">(逐 token 高亮)</span></h3>
      <div class="resp">{resp_html}</div>
    </div>
  </div>
  <div class="ledger">
    <div><strong>Image Ledger</strong>（本样本文本中出现顺序）</div>
    <ul>{ledger_items}</ul>
  </div>
</div>
        """

    def _render_responses_tokens(
        self,
        token_ids: List[int],
        mask: List[int],
        tokenizer: Any,
        image_paths_all: List[str],
    ) -> Tuple[str, List[str]]:
        """Return HTML (token-colored) and used image paths."""
        out = []
        used_paths: List[str] = []
        img_idx = 0

        n = min(len(token_ids), len(mask))
        for j in range(n):
            tid = int(token_ids[j])
            m = int(mask[j])  # 0/1
            piece = self._decode_single_token(tid, tokenizer)
            cls = "mask1" if m == 1 else "mask0"

            # Best-effort image replacement on single-token sentinels
            # Match tokens like "<|vision_start|>", "<image>", "<image_0>"
            if piece in ("<|vision_start|>", "<|image|>") or re.fullmatch(r"<image_\d+>", piece or ""):
                path = image_paths_all[img_idx] if img_idx < len(image_paths_all) else "UNKNOWN_IMG_PATH"
                img_idx += 1
                out.append(f'<span class="token img">[IMG: {html.escape(path)}]</span>')
                used_paths.append(path)
                continue
            if piece in ("<|vision_end|>",):
                # Skip rendering the end marker to reduce clutter
                continue

            out.append(f'<span class="token {cls}">{html.escape(piece or "")}</span>')

        return "".join(out), used_paths

    # -----------------------------
    # Token helpers
    # -----------------------------
    def _decode_single_token(self, tok_id: int, tokenizer: Any) -> str:
        try:
            return tokenizer.decode([int(tok_id)], skip_special_tokens=False)
        except Exception:
            # Fallback: raw id as string
            return f"<{int(tok_id)}>"

    def _decode_sequence(self, token_ids: List[int], tokenizer: Any) -> str:
        try:
            return tokenizer.decode([int(t) for t in token_ids], skip_special_tokens=False)
        except Exception:
            return " ".join(str(int(t)) for t in token_ids)

    def _trim_trailing_pad(self, ids: List[int], pad_id: Optional[int]) -> List[int]:
        if pad_id is None or not ids:
            return ids
        end = len(ids) - 1
        while end >= 0 and int(ids[end]) == int(pad_id):
            end -= 1
        return ids[: end + 1]

    def _trim_pair_by_pad(self, ids: List[int], mask: List[int], pad_id: Optional[int]) -> Tuple[List[int], List[int]]:
        ids2 = self._trim_trailing_pad(ids, pad_id)
        m2 = mask[: len(ids2)]
        return ids2, m2

    def _truncate_chars(self, s: str, max_chars: int) -> str:
        if len(s) <= max_chars:
            return s
        return s[: max_chars] + "\n… [truncated]"

    # -----------------------------
    # Image mapping
    # -----------------------------
    def _ensure_image_list(self, dataset_id: Optional[str]) -> List[str]:
        if not dataset_id:
            return []
        if dataset_id in self._image_map_cache:
            return self._image_map_cache[dataset_id]

        # Search final_messages.json under dataset_root/dataset_id/
        base = self.dataset_root / dataset_id
        candidates = [
            base / "final_messages.json",
            base / "final_message.json",
            base / "messages.json",
        ]
        paths: List[str] = []
        for p in candidates:
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    paths = self._extract_images_from_messages_json(data)
                    break
                except Exception:
                    continue

        self._image_map_cache[dataset_id] = paths
        return paths

    def _extract_images_from_messages_json(self, data: Any) -> List[str]:
        """Flatten images in order of appearance. Supports two common layouts: list[message] or dict with key 'messages'."""
        msgs = data
        if isinstance(data, dict):
            if "messages" in data and isinstance(data["messages"], list):
                msgs = data["messages"]
            else:
                # attempt to find list under known keys
                for k, v in data.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict) and "role" in v[0]:
                        msgs = v
                        break

        images: List[str] = []
        if not isinstance(msgs, list):
            return images

        for m in msgs:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "image":
                        # Support either 'image' or 'image_url' fields
                        path = c.get("image") or c.get("image_url") or c.get("path")
                        if path:
                            images.append(str(path))
        return images

    def _inject_images_best_effort(self, text: str, image_paths: List[str]) -> Tuple[str, List[str]]:
        """
        Replace image placeholders in decoded text with original file paths (best-effort).
        We match blocks or single tokens:
          - <|vision_start|> ... <|vision_end|>
          - <image>, <image_0>, <image_1>, ...
        Replacement order follows image appearance order from final_messages.json.
        """
        used: List[str] = []
        idx = 0

        def repl_block(match: "re.Match[str]") -> str:
            nonlocal idx
            path = image_paths[idx] if idx < len(image_paths) else "UNKNOWN_IMG_PATH"
            idx += 1
            used.append(path)
            return f"[IMG: {path}]"

        # non-greedy block replacement for vision sentinel pairs
        pattern_block = re.compile(r"<\|vision_start\|>.*?<\|vision_end\|>", flags=re.DOTALL)
        text = re.sub(pattern_block, repl_block, text)

        # single-token image placeholders
        pattern_single = re.compile(r"<image(?:_\d+)?>")
        text = re.sub(pattern_single, repl_block, text)

        return text, used

    # -----------------------------
    # Batch / tensor helpers
    # -----------------------------
    def _infer_batch_size(self, batch: Dict[str, Any]) -> int:
        for k in ("responses", "prompts", "input_ids", "dataset_ids"):
            if k in batch and hasattr(batch[k], "__len__"):
                try:
                    return len(batch[k])
                except Exception:
                    continue
        # fallback
        return 1

    def _get_list(self, x: Any, i: int) -> List[int]:
        if x is None:
            return []
        try:
            # torch.Tensor or numpy
            if hasattr(x, "ndim") and getattr(x, "ndim") >= 2:
                row = x[i]
                return row.tolist() if hasattr(row, "tolist") else list(row)
            if hasattr(x, "__getitem__"):
                row = x[i]
                return row.tolist() if hasattr(row, "tolist") else list(row)
        except Exception:
            pass
        # already a list-of-lists
        try:
            return list(x[i])
        except Exception:
            return []

    def _get_str(self, x: Any, i: int) -> Optional[str]:
        if x is None:
            return None
        try:
            v = x[i]
            if hasattr(v, "item"):
                v = v.item()
            return str(v)
        except Exception:
            try:
                return str(x)
            except Exception:
                return None



def _has_len(x):
    try:
        len(x)
        return True
    except Exception:
        return False

def _slice_first_n(x, n: Optional[int]):
    if n is None:
        return x
    try:
        return x[:n]
    except Exception:
        # e.g., torch.Tensor-like but no slicing overridden
        try:
            return x[0:n]
        except Exception:
            return x

def _to_jsonable(x: Any):
    """Best-effort: 将张量/ndarray/标量/容器转成 JSON 可序列化的数据。"""
    # 延迟导入，避免在无 torch/np 环境下报错
    try:
        import torch
    except Exception:
        torch = None
    try:
        import numpy as np
    except Exception:
        np = None

    # torch.Tensor
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        # 注意：大 batch 会很大；这里只用于 debug 单批
        return x.tolist()

    # numpy
    if np is not None and isinstance(x, np.ndarray):
        return x.tolist()

    # 标量
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x

    # 映射
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # 可迭代
    if isinstance(x, Iterable) and _has_len(x):
        try:
            return [_to_jsonable(v) for v in x]
        except Exception:
            pass

    # 兜底：转字符串
    return str(x)

def save_batch_for_viz(
    batch: Dict[str, Any],
    json_path: str,
    fields: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
) -> str:
    """
    将 batch（dict）保存为 JSON 文件，便于可视化调试。
    - fields: 仅保存指定键；默认保存和可视化直接相关的键
    - max_samples: 只保存前 N 个样本（减少体积）
    """
    if fields is None:
        # 你可以按需增删；下列是 SampleVisualizer 直接可用的键
        fields = [
            "prompts", "responses", "response_mask", "attention_mask",
            "dataset_ids", "uids", "uid", "ids"
        ]

    out: Dict[str, Any] = {}
    for k in fields:
        if k not in batch:
            continue
        v = batch[k]
        # 尝试按样本维度截断
        if max_samples is not None:
            v = _slice_first_n(v, max_samples)
        out[k] = _to_jsonable(v)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return json_path

if __name__ == '__main__':
    if __name__ == "__main__":
        import argparse, json, os, sys

        def _load_tokenizer_from_model(model_path: str, trust_remote_code: bool = True, use_fast: bool = True):
            """
            兼容多种形式：
            - AutoProcessor.from_pretrained(...).tokenizer
            - AutoTokenizer.from_pretrained(...)
            - 一些模型 Processor 本身就实现了 decode，则直接用它
            """
            tok = None
            proc = None
            try:
                from transformers import AutoProcessor, AutoTokenizer  # type: ignore
            except Exception as e:
                print("[error] transformers 未安装或导入失败：", repr(e))
                raise

            # 首选 Processor
            try:
                proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=use_fast)
            except Exception as e:
                print("[warn] AutoProcessor 加载失败，尝试 AutoTokenizer：", repr(e))
                proc = None

            if proc is not None:
                tok = getattr(proc, "tokenizer", None) or proc

            if tok is None:
                # 退回到 AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=use_fast)

            # 简单校验
            if not hasattr(tok, "decode"):
                raise RuntimeError("加载到的对象不含 decode 方法，请检查模型/processor 路径是否正确。")
            return tok

        parser = argparse.ArgumentParser("SampleVisualizer debug runner")
        parser.add_argument("--batch_json", type=str, default="", help="保存的 batch JSON 路径")
        parser.add_argument("--model_or_processor_path", type=str, default="", help="本地模型/processor 路径（用于加载 tokenizer/processor）")
        parser.add_argument("--dataset_root", type=str, default="", help="数据根目录（包含 {dataset_id}/final_messages.json）")
        parser.add_argument("--out_dir", type=str, default="viz", help="HTML 输出目录")
        parser.add_argument("--max_samples", type=int, default=4, help="最多可视化的样本数")
        parser.add_argument("--step", type=int, default=0, help="用于标题显示的 step 编号")
        parser.add_argument("--autorefresh", type=int, default=0, help=">0 则页面自动刷新（秒）")
        parser.add_argument("--no_trust_remote_code", action="store_true", help="禁用 trust_remote_code")
        parser.add_argument("--no_use_fast", action="store_true", help="禁用 use_fast tokenizer")
        args = parser.parse_args()

        # 1) 读取 batch JSON
        with open(args.batch_json, "r", encoding="utf-8") as f:
            batch = json.load(f)

        # 2) 加载 tokenizer / processor
        tokenizer = _load_tokenizer_from_model(
            model_path=args.model_or_processor_path,
            trust_remote_code=not args.no_trust_remote_code,
            use_fast=not args.no_use_fast,
        )

        # 3) 初始化可视化器并渲染
        viz = SampleVisualizer(
            out_dir=args.out_dir,
            dataset_root=args.dataset_root,
            max_samples=args.max_samples,
            autorefresh_sec=args.autorefresh if args.autorefresh > 0 else None,
        )
        out_path = viz.render(batch=batch, tokenizer=tokenizer, step=args.step)
        print("[viz] wrote:", out_path)
        print("[viz] latest:", os.path.join(args.out_dir, "latest.html"))