import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from stage_utils import load_array, save_array


def parse_args():
    parser = argparse.ArgumentParser(description="Run output head, sample next token, decode text.")
    parser.add_argument("--onnx_dir", type=str, default="../onnx_models")
    parser.add_argument("--tokenizer_dir", type=str, default="../models/tokenizer")
    parser.add_argument("--hidden_in", type=str, default="hidden_block4.npy")
    parser.add_argument("--logits", type=str, default="logits.npy")
    parser.add_argument("--next_token_file", type=str, default="next_token_id.txt")
    parser.add_argument("--next_token_text", type=str, default="next_token.txt")
    parser.add_argument("--tokens_file", type=str, default="tokens.npy")
    parser.add_argument("--update_tokens", action="store_true", help="Append sampled token id to tokens.npy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", help="Force greedy argmax (overrides sampling knobs).")
    parser.add_argument("--meta", type=str, default="meta.json", help="Path to step meta (for q_len/max lengths).")
    return parser.parse_args()


def sample_logits(logits: np.ndarray, temperature: float, top_k: int, top_p: float, greedy: bool) -> int:
    logits = logits.astype(np.float64)
    if greedy or (top_k == 0 and top_p >= 1.0 and abs(temperature - 1.0) < 1e-6):
        return int(np.argmax(logits))

    logits = logits / max(temperature, 1e-5)
    probs = logits - np.max(logits)
    probs = np.exp(probs)
    probs = probs / np.sum(probs)

    working = probs
    if top_k > 0:
        idx = np.argpartition(-working, top_k - 1)[:top_k]
        mask = np.zeros_like(working)
        mask[idx] = working[idx]
        working = mask
    if top_p < 1.0:
        sorted_idx = np.argsort(-working)
        sorted_probs = working[sorted_idx]
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.argmax(cumulative >= top_p)
        keep = sorted_idx[: cutoff + 1]
        mask = np.zeros_like(working)
        mask[keep] = working[keep]
        working = mask

    working = working / np.sum(working)
    return int(np.random.choice(len(working), p=working))


def maybe_update_tokens(tokens_path: Path, next_id: int):
    if not tokens_path.exists():
        return
    arr = np.load(tokens_path)
    if arr.ndim != 2:
        raise ValueError("tokens.npy must have shape (1, T)")
    arr = np.concatenate([arr, np.array([[next_id]], dtype=arr.dtype)], axis=1)
    np.save(tokens_path, arr)


def main():
    args = parse_args()
    with open(args.meta, "r", encoding="utf-8") as fr:
        meta = json.load(fr)
    q_len = int(meta["q_len"])
    if q_len <= 0:
        raise ValueError("meta.json reports q_len <= 0; nothing to decode.")
    hidden = load_array(args.hidden_in).astype(np.float32)
    sess = ort.InferenceSession(str(Path(args.onnx_dir) / "output.onnx"), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"hidden_states": hidden})[0]
    valid_logits = logits[:, :q_len, :].astype(np.float32, copy=False)
    save_array(args.logits, valid_logits.astype(np.float32))

    last_logits = valid_logits[0, -1]
    next_id = sample_logits(last_logits, args.temperature, args.top_k, args.top_p, args.greedy)
    Path(args.next_token_file).write_text(str(next_id), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    token_text = tokenizer.decode([next_id], skip_special_tokens=False)
    Path(args.next_token_text).write_text(token_text, encoding="utf-8")

    if args.update_tokens:
        maybe_update_tokens(Path(args.tokens_file), next_id)

    print(
        f"Logits saved to {args.logits} shape {valid_logits.shape}. "
        f"Next token id {next_id} ({token_text!r}) saved to {args.next_token_file}."
    )


if __name__ == "__main__":
    main()
