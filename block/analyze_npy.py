# #!/usr/bin/env python3
# """
# 分析一个 run 目录下每个 step 的 npy 文件大小、shape、dtype 等信息
# """

# import argparse
# from pathlib import Path
# import numpy as np
# import json
# import humanize  # 用于友好显示文件大小，可选，不安装可删除

# def analyze_npy_file(path: Path):
#     info = {}
#     if not path.exists():
#         info["exists"] = False
#         return info
#     info["exists"] = True
#     try:
#         arr = np.load(path)
#         info["shape"] = arr.shape
#         info["dtype"] = str(arr.dtype)
#         info["size_bytes"] = path.stat().st_size
#         info["size_human"] = humanize.naturalsize(path.stat().st_size) if "humanize" in globals() else path.stat().st_size
#     except Exception as e:
#         info["error"] = str(e)
#     return info

# def main():
#     parser = argparse.ArgumentParser(description="Analyze npy files in run directory")
#     parser.add_argument("--run_dir", type=str, required=True, help="Path to run folder containing step_* directories")
#     parser.add_argument("--output", type=str, default="npy_analysis.json", help="JSON output file")
#     args = parser.parse_args()

#     run_dir = Path(args.run_dir)
#     steps = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])

#     result = {}
#     for step_dir in steps:
#         step_info = {}
#         npy_files = sorted(step_dir.glob("*.npy"))
#         for npy_path in npy_files:
#             step_info[npy_path.name] = analyze_npy_file(npy_path)
#         result[step_dir.name] = step_info

#     # 保存 JSON
#     with open(args.output, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)
#     print(f"Analysis complete. Saved to {args.output}")

#     # 打印简要表格
#     for step_name, files in result.items():
#         print(f"\n{step_name}:")
#         for fname, info in files.items():
#             if info.get("exists", False):
#                 print(f"  {fname:20s} shape={info.get('shape')} dtype={info.get('dtype')} size={info.get('size_human')}")
#             else:
#                 print(f"  {fname:20s} MISSING")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
通用：分析某个文件夹下的所有 .npy 文件（支持递归）
"""

import argparse
from pathlib import Path
import numpy as np
import json

try:
    import humanize
    _HAS_HUMANIZE = True
except ImportError:
    _HAS_HUMANIZE = False


def analyze_npy(path: Path):
    info = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return info

    try:
        # mmap_mode=None -> 真实加载；'r' 可用于超大文件
        arr = np.load(path, allow_pickle=False)

        info.update({
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "ndim": arr.ndim,
            "num_elements": arr.size,
            "itemsize": arr.itemsize,
            "theoretical_bytes": arr.size * arr.itemsize,
            "file_bytes": path.stat().st_size,
            "file_size": (
                humanize.naturalsize(path.stat().st_size)
                if _HAS_HUMANIZE else path.stat().st_size
            ),
            "is_memmap": isinstance(arr, np.memmap),
            "c_contiguous": bool(arr.flags["C_CONTIGUOUS"]),
            "f_contiguous": bool(arr.flags["F_CONTIGUOUS"]),
        })
    except Exception as e:
        info["error"] = str(e)

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Analyze all .npy files under a directory"
    )
    parser.add_argument(
        "--dir", required=True, help="Target directory"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively search subdirectories"
    )
    parser.add_argument(
        "--output", default="npy_analysis.json", help="Output JSON file"
    )
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    if args.recursive:
        npy_files = sorted(base_dir.rglob("*.npy"))
    else:
        npy_files = sorted(base_dir.glob("*.npy"))

    results = []
    for path in npy_files:
        results.append(analyze_npy(path))

    # 保存 JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Analysis complete. {len(results)} files analyzed.")
    print(f"📄 Saved to {args.output}")

    # 简要打印
    for info in results:
        if "error" in info:
            print(f"[ERROR] {info['path']}: {info['error']}")
            continue

        print(
            f"{info['path']}\n"
            f"  shape={info['shape']} dtype={info['dtype']} "
            f"file={info['file_size']} mem={info['theoretical_bytes']} bytes"
        )


if __name__ == "__main__":
    main()
