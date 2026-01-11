import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from stage_utils import PipelineConfig, build_static_attention_mask, build_static_position_ids, load_array, save_array


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize text and run embedding ONNX stage.")
    parser.add_argument("--prompt", type=str, default="prompt.txt", help="Input prompt text file.")
    parser.add_argument(
        "--tokens_file",
        type=str,
        default="tokens.npy",
        help="Optional existing token file to reuse (int64).",
    )
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default="../onnx_models",
        help="Directory containing embed.onnx and config.json.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="../models/tokenizer",
        help="Tokenizer folder (AutoTokenizer).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Where to write hidden states and masks (default: current folder).",
    )
    parser.add_argument(
        "--past_len",
        type=int,
        default=0,
        help="Existing KV cache length. Use 0 for fresh prefill, or accumulated length for decode.",
    )
    parser.add_argument(
        "--write_tokens",
        action="store_true",
        help="Force re-tokenization of prompt even if tokens_file exists.",
    )
    parser.add_argument(
        "--max_cache_len",
        type=int,
        default=None,
        help="Static KV cache length (defaults to model max_position_embeddings).",
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=1,
        help="Static chunk size per ONNX call. Prompt chunk must not exceed this length.",
    )
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
    if q_len > max_input_len:
        raise ValueError(
            f"Token chunk length {q_len} exceeds max_input_len={max_input_len}. Increase the limit or split the prompt."
        )
    if past_len + q_len > max_cache_len:
        raise ValueError(
            f"past_len ({past_len}) + q_len ({q_len}) exceeds max_cache_len={max_cache_len}. "
            "Increase the cache length or reset KV cache."
        )

    attn = build_static_attention_mask(past_len, q_len, max_cache_len, max_input_len)
    pos = build_static_position_ids(past_len, q_len, max_input_len)
    save_array(output_dir / "attention_mask.npy", attn)
    save_array(output_dir / "position_ids.npy", pos)
    save_array(output_dir / "tokens.npy", tokens)

    embed_session = ort.InferenceSession(str(Path(args.onnx_dir) / "embed.onnx"), providers=["CPUExecutionProvider"])
    # embed_ids = np.full((1, max_input_len), pad_id, dtype=np.float32)
    embed_ids = np.full((1, max_input_len), pad_id, dtype=np.long)
    embed_ids[:, :q_len] = tokens
    hidden = embed_session.run(None, {"input_ids": embed_ids})[0].astype(np.float32)
    save_array(output_dir / "hidden_block0.npy", hidden)

    meta = {
        "past_len": past_len,
        "q_len": q_len,
        "next_past_len": past_len + q_len,
        "max_cache_len": max_cache_len,
        "max_input_len": max_input_len,
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as fw:
        json.dump(meta, fw, indent=2)
    print(
        f"Tokenized {q_len} tokens (past_len={past_len}, max_cache_len={max_cache_len}, max_input_len={max_input_len}). "
        "Hidden states saved to hidden_block0.npy"
    )


if __name__ == "__main__":
    main()
