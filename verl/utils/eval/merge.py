import os
import torch
from torch.distributed.tensor import DTensor


def merge_shards_force_concat(ckpt_dir, output_path, world_size):
    merged_state = {}

    for rank in range(world_size):
        path = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        print(f"Loading shard from: {path}")
        shard = torch.load(path, map_location='cpu', weights_only=False)

        # Convert DTensor -> local tensor
        new_shard = {}
        for k, v in shard.items():
            new_shard[k] = v.to_local() if isinstance(v, DTensor) else v

        for k, v in new_shard.items():
            merged_state.setdefault(k, []).append(v)

    merged_state_dict = {}
    concat_failures = 0
    take_first_only = 0

    for k, parts in merged_state.items():
        if len(parts) == 1:
            merged_state_dict[k] = parts[0]
        else:
            try:
                merged_state_dict[k] = torch.cat(parts, dim=0)
            except RuntimeError as e:
                print(f"Concat failed for {k}: {e}")
                try:
                    parts_float = [p.float() for p in parts]
                    merged_state_dict[k] = torch.cat(parts_float, dim=0)
                except Exception:
                    merged_state_dict[k] = parts[0]
                    take_first_only += 1
                    print(f"[WARN] Used only first part of {k}")

    print(f"[INFO] Merged parameters: {len(merged_state_dict)} (from {len(merged_state)})")
    print(f"[INFO] Params with partial concat: {take_first_only}")

    print(f"[INFO] Saving to {output_path}")
    torch.save(merged_state_dict, output_path)
