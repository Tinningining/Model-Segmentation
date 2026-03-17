import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from stage_utils import (
    PipelineConfig,
    build_system_attention_mask,
    build_prefill_attention_mask,
    build_decode_attention_mask,
    build_static_position_ids,
    load_array,
    save_array,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize text and run embedding ONNX stage.")
    parser.add_argument("--prompt", type=str, default="prompt.txt")
    parser.add_argument("--tokens_file", type=str, default="tokens.npy")
    parser.add_argument("--onnx_dir", type=str, default="../onnx_models")
    parser.add_argument("--tokenizer_dir", type=str, default="../models/tokenizer")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--past_len", type=int, default=0)
    parser.add_argument("--write_tokens", action="store_true")
    parser.add_argument("--max_cache_len", type=int, default=None)
    parser.add_argument("--max_input_len", type=int, default=1)
    parser.add_argument("--mode", type=str, choices=["system", "prefill", "decode"], default="decode",
                        help="system: 固定 system prompt 无 past KV; prefill: 用户输入带 past KV; decode: 逐 token 生成")
    return parser.parse_args()


def ensure_tokens(args, tokenizer):
    tokens_path = Path(args.tokens_file)
    if tokens_path.exists() and not args.write_tokens:
        return load_array(tokens_path)
    prompt_text = Path(args.prompt).read_text(encoding="utf-8")
    encoded = tokenizer(prompt_text, return_tensors="np", add_special_tokens=True)
    tokens = encoded["input_ids"].astype(np.int64)
    save_array(tokens_path, tokens)
    return tokens


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(args.onnx_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    pad_id = int(tokenizer.pad_token_id)

    max_cache_len = args.max_cache_len or cfg.max_position_embeddings
    max_input_len = max(1, args.max_input_len)
    tokens = ensure_tokens(args, tokenizer)
    q_len = tokens.shape[1]
    past_len = args.past_len
    mode = args.mode

    if q_len > max_input_len:
        raise ValueError(f"Token length {q_len} exceeds max_input_len={max_input_len}.")

    # Build attention mask based on mode
    if mode == "system":
        # System: 纯 causal mask，无 past KV
        attn = build_system_attention_mask(q_len, max_input_len)
    elif mode == "prefill":
        # Prefill: 带 past KV 的 causal mask（past KV 来自 system 阶段）
        attn = build_prefill_attention_mask(q_len, max_input_len, past_len, max_cache_len)
    else:
        # Decode
        if past_len + q_len > max_cache_len:
            raise ValueError(f"past_len({past_len}) + q_len({q_len}) > max_cache_len({max_cache_len})")
        attn = build_decode_attention_mask(past_len, q_len, max_cache_len, max_input_len)

    pos = build_static_position_ids(past_len, q_len, max_input_len)
    save_array(output_dir / "attention_mask.npy", attn)
    save_array(output_dir / "position_ids.npy", pos)
    save_array(output_dir / "tokens.npy", tokens)

    # Run embedding
    embed_path = str(Path(args.onnx_dir) / "embed.onnx")
    embed_session = ort.InferenceSession(embed_path, providers=["CPUExecutionProvider"])
    embed_ids = np.full((1, max_input_len), pad_id, dtype=np.float32)
    embed_ids[:, :q_len] = tokens
    hidden = embed_session.run(None, {"input_ids": embed_ids})[0].astype(np.float32)
    save_array(output_dir / "hidden_block0.npy", hidden)

    meta = {
        "mode": mode,
        "past_len": past_len,
        "q_len": q_len,
        "next_past_len": past_len + q_len,
        "max_cache_len": max_cache_len,
        "max_input_len": max_input_len,
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as fw:
        json.dump(meta, fw, indent=2)
    print(f"[{mode}] Tokenized {q_len} tokens (past_len={past_len}). Hidden saved to hidden_block0.npy")


if __name__ == "__main__":
    main()
