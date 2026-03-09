import argparse
import json
from pathlib import Path

import numpy as np
from acl_runner import ACLModel

from stage_utils import PipelineConfig, ensure_static_kv, load_array, save_array

BLOCK_LAYOUT = [
    (0, 7, "layers_0_6.om"),
    (7, 14, "layers_7_13.om"),
    (14, 21, "layers_14_20.om"),
    (21, 28, "layers_21_27.om"),
]


def build_parser(block_idx: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run block {block_idx} OM chunk")
    parser.add_argument("--om_dir", type=str, default="../om_models")
    parser.add_argument("--hidden_in", type=str, default=f"hidden_block{block_idx}.npy")
    parser.add_argument("--hidden_out", type=str, default=f"hidden_block{block_idx + 1}.npy")
    parser.add_argument("--attention_mask", type=str, default="attention_mask.npy")
    parser.add_argument("--position_ids", type=str, default="position_ids.npy")
    parser.add_argument("--past_key", type=str, default=f"past_key_block{block_idx}.npy")
    parser.add_argument("--past_value", type=str, default=f"past_value_block{block_idx}.npy")
    parser.add_argument("--meta", type=str, default="meta.json")
    return parser


def _load_inputs(args):
    hidden = load_array(args.hidden_in).astype(np.float32)
    attn = load_array(args.attention_mask).astype(np.float32)
    pos = load_array(args.position_ids).astype(np.int64)
    past_key = load_array(args.past_key)
    past_value = load_array(args.past_value)
    return hidden, attn, pos, past_key, past_value


def run_block(block_idx: int, args: argparse.Namespace):
    start, end, om_name = BLOCK_LAYOUT[block_idx]
    layers = end - start

    om_path = Path(args.om_dir) / om_name
    cfg = PipelineConfig(args.om_dir)

    with open(args.meta, "r", encoding="utf-8") as fr:
        meta = json.load(fr)

    q_len = int(meta["q_len"])
    past_len = int(meta["past_len"])
    max_cache_len = int(meta["max_cache_len"])
    max_input_len = int(meta["max_input_len"])

    if q_len > max_input_len:
        raise ValueError(f"q_len {q_len} > max_input_len {max_input_len}")
    if past_len + q_len > max_cache_len:
        raise ValueError("KV cache overflow")

    # 确保 KV cache 是静态 shape
    ensure_static_kv(
        Path(args.past_key),
        layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        max_cache_len,
    )
    ensure_static_kv(
        Path(args.past_value),
        layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        max_cache_len,
    )

    hidden, attn, pos, past_key, past_value = _load_inputs(args)

    # ====== ACL Runner ======
    runner = ACLModel(str(om_path))
    runner.init()

    outputs = runner.execute([
        hidden,
        attn,
        pos,
        past_key.astype(np.float32, copy=False),
        past_value.astype(np.float32, copy=False),
    ])

    runner.finalize()

    # ====== 解包输出 ======
    hidden_out_raw, present_key_raw, present_value_raw = outputs

    hidden_out = hidden_out_raw.view(np.float32).reshape(hidden.shape)
    save_array(args.hidden_out, hidden_out.astype(np.float32))

    if q_len > 0:
        # =========================
        # 目标形状与元素数
        # =========================
        target_shape = (
            layers,
            hidden.shape[0],
            cfg.num_key_value_heads,
            16,
            cfg.head_dim,
        )
        num = np.prod(target_shape)

        # =========================
        # present_key: flatten -> cut -> reshape
        # =========================
        present_key_flat = present_key_raw.view(np.float32).reshape(-1)
        present_key = present_key_flat[:num].reshape(target_shape)

        # =========================
        # present_value: flatten -> cut -> reshape
        # =========================
        present_value_flat = present_value_raw.view(np.float32).reshape(-1)
        present_value = present_value_flat[:num].reshape(target_shape)

        # =========================
        # 类型转换
        # =========================
        present_key = present_key.astype(np.float16, copy=False)
        present_value = present_value.astype(np.float16, copy=False)

        # =========================
        # 写入 past cache
        # =========================
        start = past_len
        end = start + q_len

        past_key[:, :, :, start:end, :] = present_key[:, :, :, :q_len, :]
        past_value[:, :, :, start:end, :] = present_value[:, :, :, :q_len, :]


    save_array(args.past_key, past_key)
    save_array(args.past_value, past_value)

    kv_len = past_len + q_len
    print(
        f"✅ Block {block_idx} done: hidden {hidden.shape} → {hidden_out.shape}, "
        f"KV {kv_len}/{max_cache_len}"
    )