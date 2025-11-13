from pathlib import Path
import json
import math
from typing import List, Tuple, Dict

# ─── CONFIGURATION ─────────────────────────
INPUT_JSON      = "evaluation_examples/test_all.json"  # source JSON file
OUTPUT_DIR      = "evaluation_examples/chunk"         # where chunk files go
TASKS_PER_CHUNK = 16                          # tasks per chunk file
# ────────────────────────────────────────────

src_path = Path(INPUT_JSON)
out_dir  = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

# Load the data *once* so repeated calls are cheap
_data: Dict[str, List[str]] = json.loads(src_path.read_text("utf-8"))
_flat: List[Tuple[str, str]] = [
    (domain, task) for domain, tasks in _data.items() for task in tasks
]
_total_chunks: int = math.ceil(len(_flat) / TASKS_PER_CHUNK)


def _unflatten(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Convert a list of (domain, task) back to domain→tasks mapping."""
    out: Dict[str, List[str]] = {}
    for domain, task in pairs:
        out.setdefault(domain, []).append(task)
    return out


def write_chunk(index: int) -> bool:
    
    if index < 0:
        raise ValueError("index must be non‑negative")

    start = index * TASKS_PER_CHUNK
    end   = start + TASKS_PER_CHUNK
    slice_ = _flat[start:end]

    if not slice_:
        # Nothing left to write
        return False

    chunk_dict = _unflatten(slice_)
    out_path = out_dir / f"{src_path.stem}_part_{index}.json"
    out_path.write_text(
        json.dumps(chunk_dict, ensure_ascii=False, indent=2), "utf-8"
    )
    print(
        f"Wrote {out_path} \u00b7 chunk {index + 1}/{_total_chunks} "
        f"(tasks {len(slice_)}/{len(_flat)})"
    )
    return True


# Optional: quick self‑test when run directly -------------------------------
if __name__ == "__main__":
    i = 0
    while write_chunk(i) and i < 2:
        i += 1
    print("All chunks written.")
