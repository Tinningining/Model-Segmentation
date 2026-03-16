import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from qwen3_custom_modules import (
    load_base_qwen3,
    Qwen3EmbeddingModule,
    Qwen3BlockStackModule,
    Qwen3OutputModule,
)


class Qwen3BlockStackPrefillWrapper(nn.Module):
    """Prefill 阶段的 Block 包装器：无 past_key/past_value 输入。"""

    def __init__(self, block_stack: Qwen3BlockStackModule):
        super().__init__()
        self.block_stack = block_stack

    def forward(self, hidden_states, attention_mask, position_ids):
        return self.block_stack(hidden_states, attention_mask, position_ids)


def export_prefill_onnx(base, onnx_path: Path, prefill_len: int):
    """导出 Prefill 阶段 ONNX 模型（seq_len=prefill_len，无 past KV）。"""
    cfg = base.config

    # 1) Embedding
    emb = Qwen3EmbeddingModule(base).eval()
    ids = torch.zeros(1, prefill_len, dtype=torch.float32)
    torch.onnx.export(
        emb,
        (ids,),
        str(onnx_path / "embed.onnx"),
        input_names=["input_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "B", 1: "T"},
            "hidden_states": {0: "B", 1: "T"},
        },
        opset_version=13,
        export_params=True,
    )

    # 2) Blocks（无 past_key/past_value，prefill 首次处理）
    blocks = [
        ("layers_0_6", 0, 7),
        ("layers_7_13", 7, 14),
        ("layers_14_20", 14, 21),
        ("layers_21_27", 21, 28),
    ]
    for name, s, e in blocks:
        blk = Qwen3BlockStackModule(base, s, e).eval()
        wrapper = Qwen3BlockStackPrefillWrapper(blk).eval()
        hs = torch.zeros(1, prefill_len, cfg.hidden_size)
        # Prefill: causal mask，无 past，最后一维 = prefill_len
        attn = torch.zeros(1, 1, prefill_len, prefill_len)
        pos = torch.arange(prefill_len, dtype=torch.long).unsqueeze(0)
        torch.onnx.export(
            wrapper,
            (hs, attn, pos),
            str(onnx_path / f"{name}.onnx"),
            input_names=["hidden_states", "attention_mask", "position_ids"],
            output_names=["hidden_states_out", "present_key", "present_value"],
            dynamic_axes={
                "hidden_states": {0: "B", 1: "T"},
                "attention_mask": {0: "B", 2: "Q", 3: "KV"},
                "position_ids": {0: "B", 1: "T"},
                "hidden_states_out": {0: "B", 1: "T"},
                "present_key": {0: "L", 3: "KV_OUT"},
                "present_value": {0: "L", 3: "KV_OUT"},
            },
            opset_version=13,
            export_params=True,
        )

    # 3) Output
    out = Qwen3OutputModule(base).eval()
    hs = torch.zeros(1, prefill_len, cfg.hidden_size)
    torch.onnx.export(
        out,
        (hs,),
        str(onnx_path / "output.onnx"),
        input_names=["hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "hidden_states": {0: "B", 1: "T"},
            "logits": {0: "B", 1: "T"},
        },
        opset_version=13,
        export_params=True,
    )

    print(f"Exported Prefill ONNX to {onnx_path.resolve()}")


def export_decode_onnx(base, onnx_path: Path, max_cache_len: int):
    """导出 Decode 阶段 ONNX 模型（seq_len=1，带 past KV）。"""
    cfg = base.config
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    decode_len = 1

    # 1) Embedding
    emb = Qwen3EmbeddingModule(base).eval()
    ids = torch.zeros(1, decode_len, dtype=torch.float32)
    torch.onnx.export(
        emb,
        (ids,),
        str(onnx_path / "embed.onnx"),
        input_names=["input_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "B", 1: "T"},
            "hidden_states": {0: "B", 1: "T"},
        },
        opset_version=13,
        export_params=True,
    )

    # 2) Blocks（带 past_key/past_value）
    blocks = [
        ("layers_0_6", 0, 7),
        ("layers_7_13", 7, 14),
        ("layers_14_20", 14, 21),
        ("layers_21_27", 21, 28),
    ]
    for name, s, e in blocks:
        blk = Qwen3BlockStackModule(base, s, e).eval()
        hs = torch.zeros(1, decode_len, cfg.hidden_size)
        # Decode: 1 token attends to past + self, 最后一维 = max_cache_len + 1
        attn = torch.zeros(1, 1, decode_len, max_cache_len + decode_len)
        pos = torch.zeros(1, decode_len, dtype=torch.long)
        past_shape = (e - s, 1, kv_heads, max_cache_len, head_dim)
        past_key = torch.zeros(past_shape)
        past_value = torch.zeros(past_shape)
        torch.onnx.export(
            blk,
            (hs, attn, pos, past_key, past_value),
            str(onnx_path / f"{name}.onnx"),
            input_names=[
                "hidden_states",
                "attention_mask",
                "position_ids",
                "past_key",
                "past_value",
            ],
            output_names=["hidden_states_out", "present_key", "present_value"],
            dynamic_axes={
                "hidden_states": {0: "B", 1: "T"},
                "attention_mask": {0: "B", 2: "Q", 3: "KV"},
                "position_ids": {0: "B", 1: "T"},
                "past_key": {0: "L", 3: "KV_IN"},
                "past_value": {0: "L", 3: "KV_IN"},
                "hidden_states_out": {0: "B", 1: "T"},
                "present_key": {0: "L", 3: "KV_OUT"},
                "present_value": {0: "L", 3: "KV_OUT"},
            },
            opset_version=13,
            export_params=True,
        )

    # 3) Output
    out = Qwen3OutputModule(base).eval()
    hs = torch.zeros(1, decode_len, cfg.hidden_size)
    torch.onnx.export(
        out,
        (hs,),
        str(onnx_path / "output.onnx"),
        input_names=["hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "hidden_states": {0: "B", 1: "T"},
            "logits": {0: "B", 1: "T"},
        },
        opset_version=13,
        export_params=True,
    )

    print(f"Exported Decode ONNX to {onnx_path.resolve()}")


def export_onnx(model_path: str, onnx_dir: str, prefill_len: int = 512, max_cache_len: int = 1024):
    """导出 Prefill 和 Decode 两组 ONNX 模型。"""
    base = load_base_qwen3(model_path)
    cfg_dict = base.config.to_dict()

    # Prefill 组
    prefill_path = Path(onnx_dir) / "prefill"
    prefill_path.mkdir(parents=True, exist_ok=True)
    export_prefill_onnx(base, prefill_path, prefill_len)
    with open(prefill_path / "config.json", "w", encoding="utf-8") as fw:
        json.dump(cfg_dict, fw, indent=2)

    # Decode 组
    decode_path = Path(onnx_dir) / "decode"
    decode_path.mkdir(parents=True, exist_ok=True)
    export_decode_onnx(base, decode_path, max_cache_len)
    with open(decode_path / "config.json", "w", encoding="utf-8") as fw:
        json.dump(cfg_dict, fw, indent=2)

    print(f"\nAll ONNX exported to {Path(onnx_dir).resolve()}")
    print(f"  Prefill: {prefill_path.resolve()} (seq_len={prefill_len})")
    print(f"  Decode:  {decode_path.resolve()} (seq_len=1, max_cache={max_cache_len})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home/szm/atlas/qwen3_1.7b")
    ap.add_argument("--onnx_dir", default="./onnx_models")
    ap.add_argument("--prefill_len", type=int, default=512)
    ap.add_argument("--max_cache_len", type=int, default=1024)
    args = ap.parse_args()
    export_onnx(args.model_path, args.onnx_dir, args.prefill_len, args.max_cache_len)
