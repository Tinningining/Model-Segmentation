import argparse
import json
from pathlib import Path

import numpy as np


def load_npy_safe(path):
    if not path.exists():
        return None
    return np.load(path)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def cosine(a, b):
    a_f = a.reshape(-1).astype(np.float64)
    b_f = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom == 0:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


def align_shape(a, b):
    """
    A = from static ONNX, bigger shape
    B = correct PyTorch shape
    goal: trim A to B's shape
    """

    if a is None or b is None:
        return a, b  # let compare_arrays handle

    if a.shape == b.shape:
        return a, b

    # 仅裁剪 time 维（第 1 维），假设形状 (batch, seq, hidden)
    a_batch, a_seq, a_dim = a.shape
    b_batch, b_seq, b_dim = b.shape

    if a_batch != b_batch or a_dim != b_dim:
        # 不支持的形状差异
        return a, b

    if a_seq < b_seq:
        # 静态 ONNX 的 seq 居然比正确的还短，不处理
        return a, b

    # 裁切 A 到 B 的长度
    a_trimmed = a[:, :b_seq, :]
    return a_trimmed, b


def compare_arrays(a, b):
    if a is None or b is None:
        return {
            "exists": (a is not None, b is not None),
            "shape_match": False,
            "mse": None,
            "cosine": None,
            "max_abs_err": None,
        }

    # -------- 新增：自动裁切 A → B shape --------
    a, b = align_shape(a, b)

    if a.shape != b.shape:
        return {
            "exists": (True, True),
            "shape_match": False,
            "mse": None,
            "cosine": None,
            "max_abs_err": None,
        }

    diff = a - b
    return {
        "exists": (True, True),
        "shape_match": True,
        "mse": mse(a, b),
        "cosine": cosine(a, b),
        "max_abs_err": float(np.max(np.abs(diff))),
    }


def compare_step(stepA, stepB, blocks):
    results = {}

    # Compare embedding
    fA = stepA / "hidden_block0.npy"
    fB = stepB / "hidden_block0.npy"
    arrA = load_npy_safe(fA)
    arrB = load_npy_safe(fB)
    results["hidden_block0"] = compare_arrays(arrA, arrB)

    # Compare intermediate blocks
    for blk in blocks:
        name = f"hidden_block{blk}.npy"
        arrA = load_npy_safe(stepA / name)
        arrB = load_npy_safe(stepB / name)
        results[f"hidden_block{blk}"] = compare_arrays(arrA, arrB)

    # Compare logits (无需裁切)
    arrA = load_npy_safe(stepA / "logits.npy")
    arrB = load_npy_safe(stepB / "logits.npy")
    results["logits"] = compare_arrays(arrA, arrB)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare two run folders (A = static ONNX, B = real PyTorch)")
    parser.add_argument("--runA", type=str, required=True)
    parser.add_argument("--runB", type=str, required=True)
    parser.add_argument("--blocks", nargs="*", default=["1","2","3","4"],
                        help="block index list (default 1,2,3,4)")
    parser.add_argument("--output", type=str, default="compare_result.json")
    args = parser.parse_args()

    runA = Path(args.runA)
    runB = Path(args.runB)

    all_steps = sorted([d.name for d in runA.iterdir() if d.is_dir() and d.name.startswith("step_")])
    results = {}

    for step in all_steps:
        sA = runA / step
        sB = runB / step
        if not sB.exists():
            print(f"Warning: {sB} missing")
            continue

        print(f"Comparing {step} ...")
        step_res = compare_step(sA, sB, args.blocks)
        results[step] = step_res

    with open(args.output, "w", encoding="utf-8") as fw:
        json.dump(results, fw, indent=2)

    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
