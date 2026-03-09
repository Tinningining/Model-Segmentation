import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).parent


def run_cmd(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ============================================================
# Stage calls
# ============================================================

def call_stage_token_embed(
    tokens_file: Path,
    out_dir: Path,
    om_dir: Path,
    past_len: int,
    max_cache_len: int,
    max_input_len: int,
):
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "stage_token_embed.py"),
        "--tokens_file",
        str(tokens_file),
        "--om_dir",
        str(om_dir),
        "--output_dir",
        str(out_dir),
        "--past_len",
        str(past_len),
        "--max_cache_len",
        str(max_cache_len),
        "--max_input_len",
        str(max_input_len),
    ]
    run_cmd(cmd)


def call_stage_block(block_idx: int, step_dir: Path, om_dir: Path, kv_dir: Path):
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / f"stage_block{block_idx}.py"),
        "--om_dir",
        str(om_dir),
        "--hidden_in",
        str(step_dir / f"hidden_block{block_idx}.npy"),
        "--hidden_out",
        str(step_dir / f"hidden_block{block_idx + 1}.npy"),
        "--attention_mask",
        str(step_dir / "attention_mask.npy"),
        "--position_ids",
        str(step_dir / "position_ids.npy"),
        "--past_key",
        str(kv_dir / f"past_key_block{block_idx}.npy"),
        "--past_value",
        str(kv_dir / f"past_value_block{block_idx}.npy"),
        "--meta",
        str(step_dir / "meta.json"),
    ]
    run_cmd(cmd)


def call_stage_output(
    step_dir: Path,
    om_dir: Path,
    logits_path: Path,
    tokens_file: Path,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
) -> int:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "stage_output.py"),
        "--om_dir",
        str(om_dir),
        "--hidden_in",
        str(step_dir / "hidden_block4.npy"),
        "--logits",
        str(logits_path),
        "--next_token_file",
        str(step_dir / "next_token_id.txt"),
        "--tokens_file",
        str(tokens_file),
        "--temperature",
        str(temperature),
        "--top_k",
        str(top_k),
        "--top_p",
        str(top_p),
        "--meta",
        str(step_dir / "meta.json"),
    ]
    if greedy:
        cmd.append("--greedy")

    run_cmd(cmd)

    next_id = int(
        (step_dir / "next_token_id.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    return next_id


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Automate full OM pipeline (token-only, no tokenizer)."
    )
    parser.add_argument(
        "--init_tokens",
        type=str,
        required=True,
        help="Initial token id txt file",
    )
    parser.add_argument("--om_dir", type=str, default="../om_models")
    parser.add_argument("--run_root", type=str, default="./runs_auto")
    parser.add_argument("--kv_dir", type=str, default="./kv_cache")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=1)
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    om_dir = Path(args.om_dir)
    run_root = Path(args.run_root)
    kv_dir = Path(args.kv_dir)

    if args.clean:
        for p in (run_root, kv_dir):
            if p.exists():
                shutil.rmtree(p)

    run_root.mkdir(parents=True, exist_ok=True)
    kv_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Load initial tokens
    # ========================================================
    init_tokens = Path(args.init_tokens)
    text = init_tokens.read_text(encoding="utf-8").strip()
    prompt_ids = [int(x) for x in text.replace("\n", " ").split()]
    prompt_tokens = np.array([prompt_ids], dtype=np.int64)

    generated_ids: list[int] = []
    past_len = 0
    next_token_id = None

    for idx in range(args.steps + 1):
        step_dir = run_root / f"step_{idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        tokens_file = step_dir / "tokens.txt"

        if idx == 0:
            tokens_file.write_text(
                " ".join(str(x) for x in prompt_ids),
                encoding="utf-8",
            )
        else:
            tokens_file.write_text(
                str(next_token_id),
                encoding="utf-8",
            )

        # ========== Embed ==========
        call_stage_token_embed(
            tokens_file=tokens_file,
            out_dir=step_dir,
            om_dir=om_dir,
            past_len=past_len,
            max_cache_len=args.max_cache_len,
            max_input_len=args.max_input_len,
        )

        # ========== Blocks ==========
        for block_idx in range(4):
            call_stage_block(block_idx, step_dir, om_dir, kv_dir)

        # ========== Output ==========
        logits_path = step_dir / "logits.npy"
        next_token_id = call_stage_output(
            step_dir,
            om_dir,
            logits_path,
            tokens_file,
            args.temperature,
            args.top_k,
            args.top_p,
            args.greedy,
        )

        generated_ids.append(next_token_id)

        meta = json.load(
            open(step_dir / "meta.json", "r", encoding="utf-8")
        )
        past_len = meta["next_past_len"]

        if idx == args.steps:
            break

    # ========================================================
    # Save summary (token-only)
    # ========================================================
    full_ids = np.concatenate(
        [prompt_tokens, np.array([generated_ids], dtype=np.int64)],
        axis=1,
    )

    summary = {
        "prompt_tokens": prompt_tokens.tolist(),
        "generated_ids": generated_ids,
        "full_ids": full_ids.tolist(),
    }

    report_path = run_root / "summary.json"
    with open(report_path, "w", encoding="utf-8") as fw:
        json.dump(summary, fw, indent=2)

    print(f"✅ Generation complete (token-only)")
    print(f"Summary written to {report_path}")
    print(f"Generated token ids: {generated_ids}")


if __name__ == "__main__":
    main()
