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
    p = argparse.ArgumentParser(description="Qwen3 pipeline (system+prefill+decode)")
    p.add_argument("--system_prompt", type=str, default="system_prompt.txt")
    p.add_argument("--user_prompt", type=str, default="user_prompt.txt")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--system_onnx_dir", type=str, default="../onnx_models/system")
    p.add_argument("--prefill_onnx_dir", type=str, default="../onnx_models/prefill")
    p.add_argument("--decode_onnx_dir", type=str, default="../onnx_models/decode")
    p.add_argument("--tokenizer_dir", type=str, default="../models/tokenizer")
    p.add_argument("--run_root", type=str, default="./runs_auto")
    p.add_argument("--kv_dir", type=str, default="./kv_cache")
    p.add_argument("--system_kv_dir", type=str, default="./system_kv_cache",
                   help="System KV cache 独立存储目录，首次生成后可复用")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--system_len", type=int, default=256)
    p.add_argument("--prefill_len", type=int, default=512)
    p.add_argument("--max_cache_len", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--no_system", action="store_true")
    p.add_argument("--force_rebuild_system_kv", action="store_true",
                   help="强制重新运行 system 阶段生成 KV cache（即使已存在）")
    return p.parse_args()


def main():
    args = parse_args()
    system_prompt_path = Path(args.system_prompt)
    user_prompt_path = Path(args.user_prompt)
    prompt_path = Path(args.prompt) if args.prompt else None
    system_onnx = Path(args.system_onnx_dir)
    prefill_onnx = Path(args.prefill_onnx_dir)
    decode_onnx = Path(args.decode_onnx_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    run_root = Path(args.run_root)
    kv_dir = Path(args.kv_dir)
    system_kv_dir = Path(args.system_kv_dir)

    if args.clean:
        for t in (run_root, kv_dir):
            if t.exists():
                shutil.rmtree(t)
        # 注意：system_kv_dir 不随 --clean 删除，因为它是可复用的
    run_root.mkdir(parents=True, exist_ok=True)
    kv_dir.mkdir(parents=True, exist_ok=True)
    system_kv_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    generated_ids = []
    generated_text = []
    past_len = 0
    next_token_id = None

    # ========== System Stage ==========
    if not args.no_system:
        # 检查 system KV cache 是否已存在
        system_kv_meta_path = system_kv_dir / "system_kv_meta.json"
        system_kv_exists = system_kv_meta_path.exists() and not args.force_rebuild_system_kv

        if system_kv_exists:
            # 复用已有的 system KV cache
            print("\n===== System Stage: Reusing cached system KV =====")
            with open(system_kv_meta_path, "r", encoding="utf-8") as fr:
                sys_kv_meta = json.load(fr)
            system_q_len = sys_kv_meta["system_q_len"]

            # 将 system KV cache 复制到工作 kv_dir
            for block_idx in range(4):
                for kv_type in ("past_key", "past_value"):
                    src = system_kv_dir / f"{kv_type}_block{block_idx}.npy"
                    dst = kv_dir / f"{kv_type}_block{block_idx}.npy"
                    if src.exists():
                        shutil.copy2(str(src), str(dst))
                    else:
                        raise FileNotFoundError(f"System KV cache missing: {src}")

            past_len = system_q_len
            print(f"Loaded system KV cache. system_q_len={system_q_len}, past_len={past_len}")
        else:
            # 首次运行：用 system 模型生成 KV cache
            print("\n===== System Stage: Computing system KV cache =====")
            sys_step_dir = run_root / "step_system"
            sys_step_dir.mkdir(parents=True, exist_ok=True)
            sys_tokens_file = sys_step_dir / "tokens.npy"

            call_stage_token_embed(
                system_prompt_path, sys_tokens_file, sys_step_dir,
                system_onnx, tokenizer_dir,
                past_len=0, write_tokens=True,
                max_cache_len=args.max_cache_len,
                max_input_len=args.system_len,
                mode="system",
            )
            system_tokens = np.load(sys_tokens_file)
            system_q_len = system_tokens.shape[1]

            for block_idx in range(4):
                call_stage_block(block_idx, sys_step_dir, system_onnx, kv_dir)

            # 将生成的 system KV cache 保存到独立目录
            for block_idx in range(4):
                for kv_type in ("past_key", "past_value"):
                    src = kv_dir / f"{kv_type}_block{block_idx}.npy"
                    dst = system_kv_dir / f"{kv_type}_block{block_idx}.npy"
                    if src.exists():
                        shutil.copy2(str(src), str(dst))

            # 保存元信息
            sys_kv_meta = {
                "system_q_len": system_q_len,
                "max_cache_len": args.max_cache_len,
                "system_len": args.system_len,
                "system_prompt": str(system_prompt_path),
            }
            with open(system_kv_meta_path, "w", encoding="utf-8") as fw:
                json.dump(sys_kv_meta, fw, indent=2)

            past_len = system_q_len
            print(f"System stage done. q_len={system_q_len}, past_len={past_len}")
            print(f"System KV cache saved to {system_kv_dir}")

    # ========== Determine prefill prompt ==========
    if args.no_system:
        prefill_prompt = prompt_path if prompt_path else Path("prompt.txt")
    else:
        prefill_prompt = user_prompt_path

    # ========== Prefill + Decode ==========
    for idx in range(args.steps + 1):
        step_dir = run_root / f"step_{idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        tokens_file = step_dir / "tokens.npy"

        if idx == 0:
            # Prefill step
            if args.no_system:
                # Old mode: prefill without past KV
                call_stage_token_embed(
                    prefill_prompt, tokens_file, step_dir,
                    prefill_onnx, tokenizer_dir,
                    past_len=0, write_tokens=True,
                    max_cache_len=args.max_cache_len,
                    max_input_len=args.prefill_len,
                    mode="prefill",
                )
            else:
                # New mode: prefill with past KV from system
                call_stage_token_embed(
                    prefill_prompt, tokens_file, step_dir,
                    prefill_onnx, tokenizer_dir,
                    past_len=past_len, write_tokens=True,
                    max_cache_len=args.max_cache_len,
                    max_input_len=args.prefill_len,
                    mode="prefill",
                )
            prompt_tokens = np.load(tokens_file)
            step_onnx = prefill_onnx
        else:
            # Decode step
            np.save(tokens_file, np.array([[next_token_id]], dtype=np.int64))
            call_stage_token_embed(
                prefill_prompt, tokens_file, step_dir,
                decode_onnx, tokenizer_dir,
                past_len, write_tokens=False,
                max_cache_len=args.max_cache_len,
                max_input_len=1,
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

    # ========== Summary ==========
    if not args.no_system:
        sys_tokens_path = run_root / "step_system" / "tokens.npy"
        if sys_tokens_path.exists():
            sys_tok = np.load(sys_tokens_path)
            full_ids = np.concatenate([
                sys_tok, prompt_tokens,
                np.array([generated_ids], dtype=np.int64)
            ], axis=1)
        else:
            full_ids = np.concatenate([
                prompt_tokens,
                np.array([generated_ids], dtype=np.int64)
            ], axis=1)
    else:
        full_ids = np.concatenate([
            prompt_tokens,
            np.array([generated_ids], dtype=np.int64)
        ], axis=1)

    full_text = tokenizer.decode(full_ids[0], skip_special_tokens=False)
    generated_only = "".join(generated_text)

    summary = {
        "system_prompt": str(system_prompt_path) if not args.no_system else None,
        "user_prompt": str(prefill_prompt),
        "prompt_tokens": prompt_tokens.tolist(),
        "generated_ids": generated_ids,
        "generated_text": generated_only,
        "full_text": full_text,
    }
    report_path = run_root / "summary.json"
    with open(report_path, "w", encoding="utf-8") as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)

    print(f"\nGeneration complete. Summary: {report_path}")
    print(f"Generated: {generated_only}")


if __name__ == "__main__":
    main()
