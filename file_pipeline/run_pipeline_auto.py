import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).parent


def run_cmd(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def call_stage_token_embed(
    prompt_path, tokens_file, out_dir, onnx_dir, tokenizer_dir,
    past_len, write_tokens, max_cache_len, max_input_len, mode,
):
    cmd = [
        sys.executable, str(SCRIPT_DIR / "stage_token_embed.py"),
        "--prompt", str(prompt_path),
        "--onnx_dir", str(onnx_dir),
        "--tokenizer_dir", str(tokenizer_dir),
        "--output_dir", str(out_dir),
        "--tokens_file", str(tokens_file),
        "--past_len", str(past_len),
        "--max_cache_len", str(max_cache_len),
        "--max_input_len", str(max_input_len),
        "--mode", mode,
    ]
    if write_tokens:
        cmd.append("--write_tokens")
    run_cmd(cmd)


def call_stage_block(block_idx, step_dir, onnx_dir, kv_dir):
    cmd = [
        sys.executable, str(SCRIPT_DIR / f"stage_block{block_idx}.py"),
        "--onnx_dir", str(onnx_dir),
        "--hidden_in", str(step_dir / f"hidden_block{block_idx}.npy"),
        "--hidden_out", str(step_dir / f"hidden_block{block_idx + 1}.npy"),
        "--attention_mask", str(step_dir / "attention_mask.npy"),
        "--position_ids", str(step_dir / "position_ids.npy"),
        "--past_key", str(kv_dir / f"past_key_block{block_idx}.npy"),
        "--past_value", str(kv_dir / f"past_value_block{block_idx}.npy"),
        "--meta", str(step_dir / "meta.json"),
    ]
    run_cmd(cmd)


def call_stage_output(step_dir, onnx_dir, tokenizer_dir, logits_path, tokens_file,
                      temperature, top_k, top_p, greedy):
    cmd = [
        sys.executable, str(SCRIPT_DIR / "stage_output.py"),
        "--onnx_dir", str(onnx_dir),
        "--tokenizer_dir", str(tokenizer_dir),
        "--hidden_in", str(step_dir / "hidden_block4.npy"),
        "--logits", str(logits_path),
        "--next_token_file", str(step_dir / "next_token_id.txt"),
        "--next_token_text", str(step_dir / "next_token.txt"),
        "--tokens_file", str(tokens_file),
        "--temperature", str(temperature),
        "--top_k", str(top_k),
        "--top_p", str(top_p),
        "--meta", str(step_dir / "meta.json"),
    ]
    if greedy:
        cmd.append("--greedy")
    run_cmd(cmd)
    next_id = int((step_dir / "next_token_id.txt").read_text(encoding="utf-8").strip())
    token_text = (step_dir / "next_token.txt").read_text(encoding="utf-8")
    return next_id, token_text


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3 file-based pipeline (prefill+decode)")
    p.add_argument("--prompt", type=str, default="prompt.txt")
    p.add_argument("--prefill_onnx_dir", type=str, default="../onnx_models/prefill",
                   help="ONNX dir for prefill models (no past KV)")
    p.add_argument("--decode_onnx_dir", type=str, default="../onnx_models/decode",
                   help="ONNX dir for decode models (with past KV)")
    p.add_argument("--tokenizer_dir", type=str, default="../models/tokenizer")
    p.add_argument("--run_root", type=str, default="./runs_auto")
    p.add_argument("--kv_dir", type=str, default="./kv_cache")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--prefill_len", type=int, default=512, help="Max input len for prefill stage")
    p.add_argument("--max_cache_len", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    prompt_path = Path(args.prompt)
    prefill_onnx = Path(args.prefill_onnx_dir)
    decode_onnx = Path(args.decode_onnx_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    run_root = Path(args.run_root)
    kv_dir = Path(args.kv_dir)

    if args.clean:
        for t in (run_root, kv_dir):
            if t.exists():
                shutil.rmtree(t)
    run_root.mkdir(parents=True, exist_ok=True)
    kv_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    generated_ids = []
    generated_text = []
    past_len = 0
    next_token_id = None

    for idx in range(args.steps + 1):
        step_dir = run_root / f"step_{idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        tokens_file = step_dir / "tokens.npy"

        if idx == 0:
            # Prefill step
            call_stage_token_embed(
                prompt_path, tokens_file, step_dir, prefill_onnx, tokenizer_dir,
                past_len, write_tokens=True,
                max_cache_len=args.max_cache_len, max_input_len=args.prefill_len,
                mode="prefill",
            )
            prompt_tokens = np.load(tokens_file)
            step_onnx = prefill_onnx
        else:
            # Decode step
            np.save(tokens_file, np.array([[next_token_id]], dtype=np.int64))
            call_stage_token_embed(
                prompt_path, tokens_file, step_dir, decode_onnx, tokenizer_dir,
                past_len, write_tokens=False,
                max_cache_len=args.max_cache_len, max_input_len=1,
                mode="decode",
            )
            step_onnx = decode_onnx

        for block_idx in range(4):
            call_stage_block(block_idx, step_dir, step_onnx, kv_dir)

        logits_path = step_dir / "logits.npy"
        next_token_id, token_text = call_stage_output(
            step_dir, step_onnx, tokenizer_dir, logits_path, tokens_file,
            args.temperature, args.top_k, args.top_p, args.greedy,
        )

        generated_ids.append(next_token_id)
        generated_text.append(token_text)

        meta = json.load(open(step_dir / "meta.json", "r", encoding="utf-8"))
        past_len = meta["next_past_len"]

        if idx == args.steps:
            break

    full_ids = np.concatenate([prompt_tokens, np.array([generated_ids], dtype=np.int64)], axis=1)
    full_text = tokenizer.decode(full_ids[0], skip_special_tokens=False)
    generated_only = "".join(generated_text)

    summary = {
        "prompt_tokens": prompt_tokens.tolist(),
        "generated_ids": generated_ids,
        "generated_text": generated_only,
        "full_text": full_text,
    }
    report_path = run_root / "summary.json"
    with open(report_path, "w", encoding="utf-8") as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)

    print(f"Generation complete. Summary: {report_path}")
    print(f"Generated: {generated_only}")


if __name__ == "__main__":
    main()
