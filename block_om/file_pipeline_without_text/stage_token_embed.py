import argparse
import json
from pathlib import Path

import numpy as np
from acl_runner import ACLModel

from stage_utils import (
    PipelineConfig,
    build_static_attention_mask,
    build_static_position_ids,
    load_array,
    save_array,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run embedding OM stage with token-id txt input (no tokenizer)."
    )
    parser.add_argument(
        "--tokens_file",
        type=str,
        default="tokens.txt",
        help="TXT file containing token ids (space or newline separated).",
    )
    parser.add_argument(
        "--om_dir",
        type=str,
        default="../om_models",
        help="Directory containing embed.om and config.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory",
    )
    parser.add_argument(
        "--past_len",
        type=int,
        default=0,
        help="Existing KV cache length",
    )
    parser.add_argument(
        "--max_cache_len",
        type=int,
        default=None,
        help="Static KV cache length",
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=1,
        help="Static chunk size per OM inference",
    )
    return parser.parse_args()


def load_tokens_from_txt(path: Path) -> np.ndarray:
    """
    Load token ids from txt file.
    Supports space or newline separated integers.
    Returns shape (1, q_len), dtype int64.
    """
    if not path.exists():
        raise FileNotFoundError(f"Token id file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Token id txt file is empty")

    ids = [int(x) for x in text.replace("\n", " ").split()]
    return np.array([ids], dtype=np.int64)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. Load config & tokens
    # =========================
    cfg = PipelineConfig(args.om_dir)

    tokens = load_tokens_from_txt(Path(args.tokens_file))
    q_len = tokens.shape[1]
    past_len = args.past_len

    max_cache_len = args.max_cache_len or cfg.max_position_embeddings
    max_input_len = max(1, args.max_input_len)

    if q_len > max_input_len:
        raise ValueError(f"q_len {q_len} exceeds max_input_len {max_input_len}")
    if past_len + q_len > max_cache_len:
        raise ValueError(
            f"past_len {past_len} + q_len {q_len} exceeds max_cache_len {max_cache_len}"
        )

    # =========================
    # 2. Build mask / position
    # =========================
    attn = build_static_attention_mask(
        past_len, q_len, max_cache_len, max_input_len
    )
    pos = build_static_position_ids(
        past_len, q_len, max_input_len
    )

    save_array(output_dir / "attention_mask.npy", attn)
    save_array(output_dir / "position_ids.npy", pos)
    save_array(output_dir / "tokens.npy", tokens)

    # =========================
    # 3. Run embed.om
    # =========================
    PAD_ID = 0  # fixed pad id

    embed_ids = np.full(
        (1, max_input_len),
        PAD_ID,
        dtype=np.float32,
    )
    embed_ids[:, :q_len] = tokens.astype(np.float32)

    embed_om_path = Path(args.om_dir) / "embed.om"
    if not embed_om_path.exists():
        raise FileNotFoundError(f"embed.om not found: {embed_om_path}")

    runner = ACLModel(str(embed_om_path))
    runner.init()

    (hidden_raw,) = runner.execute([embed_ids])

    runner.finalize()

    hidden = hidden_raw.view(np.float32).reshape(
        1, max_input_len, cfg.hidden_size
    )

    save_array(output_dir / "hidden_block0.npy", hidden.astype(np.float32))

    # =========================
    # 4. Save meta.json
    # =========================
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
        f"✅ Embed OM done | "
        f"q_len={q_len}, past_len={past_len}, "
        f"hidden saved → {output_dir / 'hidden_block0.npy'}"
    )


if __name__ == "__main__":
    main()
