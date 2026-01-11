import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from stage_utils import PipelineConfig, ensure_static_kv, load_array, save_array

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_idx", type=int, required=True, help="Layer index (integer)")
    parser.add_argument("--onnx_dir", type=str, default="../onnx_models", help="Folder with block ONNX files")
    parser.add_argument("--hidden_in", type=str, default="hidden_block.npy")
    parser.add_argument("--hidden_out", type=str, default="hidden_block.npy")
    parser.add_argument("--attention_mask", type=str, default="attention_mask.npy")
    parser.add_argument("--position_ids", type=str, default="position_ids.npy")
    parser.add_argument("--past_key", type=str, default="past_key_block.npy")
    parser.add_argument("--past_value", type=str, default="past_value_block.npy")
    parser.add_argument("--meta", type=str, default="meta.json", help="Path to the step metadata JSON.")
    return parser


def _load_inputs(args):
    hidden = load_array(args.hidden_in).astype(np.float32)
    attn = load_array(args.attention_mask).astype(np.float32)
    pos = load_array(args.position_ids).astype(np.int64)
    past_key = load_array(args.past_key)
    past_value = load_array(args.past_value)
    return hidden, attn, pos, past_key, past_value

def run_layer(args: argparse.Namespace):
    block_idx = args.layer_idx
    onnx_name = f"layer_{block_idx}.onnx"

    onnx_path = Path(args.onnx_dir) / onnx_name
    cfg = PipelineConfig(args.onnx_dir)

    with open(args.meta, "r", encoding="utf-8") as fr:
        meta = json.load(fr)
    q_len = int(meta["q_len"])
    past_len = int(meta["past_len"])
    max_cache_len = int(meta["max_cache_len"])
    max_input_len = int(meta["max_input_len"])
    if q_len > max_input_len:
        raise ValueError(f"Meta q_len {q_len} exceeds max_input_len {max_input_len}")
    if past_len + q_len > max_cache_len:
        raise ValueError(
            f"Meta past_len ({past_len}) + q_len ({q_len}) exceeds max_cache_len ({max_cache_len})."
            " Re-run token embed with consistent arguments."
        )

    ensure_static_kv(Path(args.past_key), 1, cfg.num_key_value_heads, cfg.head_dim, max_cache_len)
    ensure_static_kv(Path(args.past_value), 1, cfg.num_key_value_heads, cfg.head_dim, max_cache_len)

    hidden, attn, pos, past_key, past_value = _load_inputs(args)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {
        "hidden_states": hidden,
        "attention_mask": attn,
        # "position_ids": pos.astype(np.float32, copy=False),
        "position_ids": pos,
        "past_key": past_key[0].astype(np.float32, copy=False),
        "past_value": past_value[0].astype(np.float32, copy=False),
    }
    hidden_out, present_key, present_value = sess.run(None, feeds)
    save_array(args.hidden_out, hidden_out.astype(np.float32))

    if q_len > 0:
        present_key = present_key.astype(np.float16, copy=False)
        present_value = present_value.astype(np.float16, copy=False)
        start = past_len
        end = start + q_len
        past_key[:, :, :, start:end, :] = present_key[:, :, :q_len, :]
        past_value[:, :, :, start:end, :] = present_value[:, :, :q_len, :]

    save_array(args.past_key, past_key)
    save_array(args.past_value, past_value)

    kv_len = past_len + q_len
    print(
        f"Block {block_idx} processed hidden shape {hidden.shape} -> {hidden_out.shape}. "
        f"KV len now {kv_len}/{max_cache_len}."
    )



def main():
    parser = build_parser()
    args = parser.parse_args()
    run_layer(args)


if __name__ == "__main__":
    main()