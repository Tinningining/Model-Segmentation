import argparse
import json
from pathlib import Path

import torch

from qwen3_custom_modules import (
    load_base_qwen3,
    Qwen3EmbeddingModule,
    Qwen3BlockStackModule,
    Qwen3OutputModule,
)


def export_onnx(model_path: str, onnx_dir: str, seq_len: int = 8):
    """导出嵌入、7 层一份的 4 个 Block 以及输出头 ONNX。

    Block 的输入/输出包含 hidden_states、attention_mask、position_ids 以及
    past/present KV，用于后续流水线中的静态 KV cache 推理。
    """

    onnx_path = Path(onnx_dir)
    onnx_path.mkdir(parents=True, exist_ok=True)

    base = load_base_qwen3(model_path)
    cfg_dict = base.config.to_dict()

    # 1) Embedding
    emb = Qwen3EmbeddingModule(base).eval()
    # ids = torch.zeros(1, seq_len, dtype=torch.long)
    ids = torch.zeros(1, seq_len, dtype=torch.float32)
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

    # 2) Blocks (7 layers each)
    blocks = [
        ("layers_0_6", 0, 7),
        ("layers_7_13", 7, 14),
        ("layers_14_20", 14, 21),
        ("layers_21_27", 21, 28),
    ]
    kv_heads = base.config.num_key_value_heads
    head_dim = base.config.hidden_size // base.config.num_attention_heads
    for name, s, e in blocks:
        blk = Qwen3BlockStackModule(base, s, e).eval()
        hs = torch.zeros(1, seq_len, base.config.hidden_size)
        attn = torch.zeros(1, 1, seq_len, seq_len * 2)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        # pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
        past_shape = (e - s, 1, kv_heads, seq_len, head_dim)
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
    hs = torch.zeros(1, seq_len, base.config.hidden_size)
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

    with open(onnx_path / "config.json", "w", encoding="utf-8") as fw:
        json.dump(cfg_dict, fw, indent=2)

    print(f"Exported ONNX (and config) to {onnx_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home/szm/atlas/qwen3_1.7b")
    ap.add_argument("--onnx_dir", default="./onnx_models")
    ap.add_argument("--seq_len", type=int, default=16)
    args = ap.parse_args()
    export_onnx(args.model_path, args.onnx_dir, args.seq_len)
