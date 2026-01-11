# import argparse
# import json
# import shutil
# from pathlib import Path

# import numpy as np
# import torch
# from transformers import AutoTokenizer

# from qwen3_custom_modules import load_base_qwen3  # 本地加载模型

# # -----------------------------
# # 命令行参数
# # -----------------------------
# def parse_args():
#     parser = argparse.ArgumentParser(description="Run full Qwen3 model with intermediate outputs via hook and KV cache")
#     parser.add_argument("--prompt", type=str, default="prompt.txt", help="Prompt text file")
#     parser.add_argument("--run_root", type=str, default="./runs_full_model", help="Output directory")
#     parser.add_argument("--tokenizer_dir", type=str, default="qwen3_1.7b")
#     parser.add_argument("--model_path", type=str, default="qwen3_1.7b")
#     parser.add_argument("--steps", type=int, default=10, help="Number of tokens to generate")
#     parser.add_argument("--greedy", action="store_true", help="Greedy decoding")
#     parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
#     parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 disables)")
#     parser.add_argument("--top_p", type=float, default=0.0, help="Top-p (nucleus) sampling (0 disables)")
#     parser.add_argument("--clean", action="store_true", help="Remove run_root before starting")
#     return parser.parse_args()


# # -----------------------------
# # 主流程
# # -----------------------------
# def main():
#     args = parse_args()
#     run_root = Path(args.run_root)
#     if args.clean and run_root.exists():
#         shutil.rmtree(run_root)
#     run_root.mkdir(parents=True, exist_ok=True)

#     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

#     # 加载完整模型
#     base = load_base_qwen3(args.model_path).eval()

#     # target layers，用 hook 捕获
#     target_layers = [6, 13, 20, 27]  # 对应 block 层索引
#     intermediate_outputs = {}

#     def make_hook(name):
#         def hook(module, input, output):
#             intermediate_outputs[name] = output.detach().cpu().numpy()
#         return hook

#     for idx in target_layers:
#         layer = base.model.layers[idx]
#         layer.register_forward_hook(make_hook(f"block_{idx}"))

#     # 捕获 embedding
#     embedding_output = {}

#     def embed_hook(module, input, output):
#         embedding_output["embed"] = output.detach().cpu().numpy()

#     base.get_input_embeddings().register_forward_hook(embed_hook)

#     # 读取 prompt 文本
#     with open(args.prompt, "r", encoding="utf-8") as f:
#         text = f.read()
#     input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)], dtype=torch.long)

#     generated_ids = []
#     generated_text = []

#     # KV cache 初始化
#     past_key_values = None

#     for step_idx in range(args.steps + 1):
#         step_dir = run_root / f"step_{step_idx:04d}"
#         step_dir.mkdir(parents=True, exist_ok=True)

#         if step_idx == 0:
#             step_input_ids = input_ids
#         else:
#             step_input_ids = torch.tensor([[next_token_id]], dtype=torch.long)

#         with torch.no_grad():
#             # 支持 KV cache 的自回归 forward
#             outputs = base(step_input_ids, past_key_values=past_key_values, use_cache=True)
#             logits = outputs.logits
#             past_key_values = outputs.past_key_values  # 更新 KV cache

#         # 保存 embedding
#         np.save(step_dir / f"hidden_block0.npy", embedding_output["embed"])
#         # 保存每个 block 输出
#         for idx in target_layers:
#             np.save(step_dir / f"hidden_block{target_layers.index(idx)+1}.npy",
#                     intermediate_outputs[f"block_{idx}"])
#         # 保存 logits
#         logits_np = logits.detach().cpu().numpy()
#         np.save(step_dir / f"logits.npy", logits_np)

#         # 选择下一个 token
#         logits_tensor = logits[0, -1, :]  # (vocab_size,)

#         if args.greedy:
#             next_token_id = int(torch.argmax(logits_tensor).item())
#         else:
#             # temperature scaling
#             scaled_logits = logits_tensor / max(args.temperature, 1e-5)

#             # top-k
#             if args.top_k > 0:
#                 topk_vals, topk_indices = torch.topk(scaled_logits, args.top_k)
#                 mask = torch.full_like(scaled_logits, float("-inf"))
#                 mask[topk_indices] = scaled_logits[topk_indices]
#                 scaled_logits = mask

#             # top-p
#             if args.top_p > 0.0:
#                 sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
#                 cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
#                 cutoff = cumulative_probs > args.top_p
#                 cutoff[1:] = cutoff[:-1] & cutoff[1:]  # 保留至少一个 token
#                 sorted_logits[cutoff] = float("-inf")
#                 # revert to original order
#                 temp = torch.full_like(scaled_logits, float("-inf"))
#                 temp[sorted_indices] = sorted_logits
#                 scaled_logits = temp

#             probs = torch.softmax(scaled_logits, dim=-1)
#             next_token_id = int(torch.multinomial(probs, num_samples=1).item())

#         next_token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
#         generated_ids.append(next_token_id)
#         generated_text.append(next_token_text)

#         # 保存 meta.json
#         meta = {"next_past_len": step_input_ids.shape[1]}
#         with open(step_dir / "meta.json", "w", encoding="utf-8") as f:
#             json.dump(meta, f, indent=2)

#     # 汇总结果
#     full_ids = torch.cat([input_ids, torch.tensor([generated_ids], dtype=torch.long)], dim=1)
#     full_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
#     generated_only = "".join(generated_text)

#     summary = {
#         "prompt_tokens": input_ids.tolist(),
#         "generated_ids": generated_ids,
#         "generated_text": generated_only,
#         "full_text": full_text,
#     }
#     report_path = run_root / "summary_full_model.json"
#     with open(report_path, "w", encoding="utf-8") as fw:
#         json.dump(summary, fw, ensure_ascii=False, indent=2)

#     print(f"Generation complete. Summary written to {report_path}")
#     print(f"Generated text: {generated_only}")


# if __name__ == "__main__":
#     main()



import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from qwen3_custom_modules import load_base_qwen3  # 本地加载模型


# ----------------------------------------------------
# 这些是固定捕获的层（embedding + 4 个 block）
# ----------------------------------------------------
FIXED_LAYERS = [6, 13, 20, 27]   # 永远捕获的四层


def parse_args():
    parser = argparse.ArgumentParser(description="Full Qwen3 model with intermediates + KV cache")
    parser.add_argument("--prompt", type=str, default="prompt.txt")
    parser.add_argument("--run_root", type=str, default="./runs_full_model")
    parser.add_argument("--tokenizer_dir", type=str, default="qwen3_1.7b")
    parser.add_argument("--model_path", type=str, default="qwen3_1.7b")
    parser.add_argument("--steps", type=int, default=10)

    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)

    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_root = Path(args.run_root)

    if args.clean and run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # ---- 加载模型 ----
    base = load_base_qwen3(args.model_path).eval().to("cpu")

    # ----------------------------------------------------
    # 注册 embedding hook
    # ----------------------------------------------------
    embedding_output = {}

    def embed_hook(module, inp, out):
        embedding_output["embed"] = out.detach().cpu().numpy()

    base.get_input_embeddings().register_forward_hook(embed_hook)

    # ----------------------------------------------------
    # 注册中间层 hook
    # ----------------------------------------------------
    intermediate_outputs = {}

    def make_hook(idx):
        def hook(module, inp, out):
            intermediate_outputs[idx] = out.detach().cpu().numpy()
        return hook

    for layer_idx in FIXED_LAYERS:
        base.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))

    # ----------------------------------------------------
    # 读取 prompt
    # ----------------------------------------------------
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    input_ids = torch.tensor(
        [tokenizer.encode(prompt_text, add_special_tokens=True)],
        dtype=torch.long
    )

    generated_ids = []
    generated_text = []
    past_key_values = None

    # ----------------------------------------------------
    # 主循环：逐 token 生成并保存
    # ----------------------------------------------------
    for step_idx in range(args.steps + 1):
        step_dir = run_root / f"step_{step_idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        if step_idx == 0:
            step_input = input_ids
        else:
            step_input = torch.tensor([[next_token_id]], dtype=torch.long)

        # 本 step 先清空 capture 缓存
        embedding_output.clear()
        intermediate_outputs.clear()

        # 正常 forward（与 generate 等价）
        with torch.no_grad():
            outputs = base(
                step_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        # ----------------------------------------
        # 保存 embedding → hidden_block0.npy
        # ----------------------------------------
        if "embed" in embedding_output:
            np.save(step_dir / "hidden_block0.npy", embedding_output["embed"])

        # ----------------------------------------
        # 保存 4 个 block 输出 → hidden_block1~4
        # ----------------------------------------
        sorted_layers = FIXED_LAYERS  # 保证顺序一致：6,13,20,27
        for i, layer_idx in enumerate(sorted_layers):
            if layer_idx in intermediate_outputs:
                np.save(step_dir / f"hidden_block{i+1}.npy",
                        intermediate_outputs[layer_idx])

        # ----------------------------------------
        # 保存 logits
        # ----------------------------------------
        np.save(step_dir / "logits.npy",
                logits.detach().cpu().numpy())

        # ----------------------------------------
        # 选择下一个 token
        # ----------------------------------------
        logits_vec = logits[0, -1, :]

        if args.greedy:
            next_token_id = int(torch.argmax(logits_vec).item())
        else:
            # temperature
            scaled = logits_vec / max(args.temperature, 1e-5)

            # top-k
            if args.top_k > 0:
                v, idxs = torch.topk(scaled, args.top_k)
                mask = torch.full_like(scaled, float("-inf"))
                mask[idxs] = scaled[idxs]
                scaled = mask

            # top-p
            if args.top_p > 0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cumprob = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cumprob > args.top_p
                cutoff[1:] = cutoff[:-1] & cutoff[1:]
                sorted_logits[cutoff] = float("-inf")
                tmp = torch.full_like(scaled, float("-inf"))
                tmp[sorted_indices] = sorted_logits
                scaled = tmp

            probs = torch.softmax(scaled, dim=-1)
            next_token_id = int(torch.multinomial(probs, 1).item())

        generated_ids.append(next_token_id)
        generated_text.append(tokenizer.decode([next_token_id], skip_special_tokens=True))

        # 保存 meta
        with open(step_dir / "meta.json", "w") as f:
            json.dump({"input_len": step_input.shape[1]}, f, indent=2)

    # ----------------------------------------------------
    # 汇总 summary.json
    # ----------------------------------------------------
    full_ids = torch.cat([input_ids, torch.tensor([generated_ids])], dim=1)
    summary = {
        "prompt": prompt_text,
        "prompt_ids": input_ids.tolist(),
        "generated_ids": generated_ids,
        "generated_text": "".join(generated_text),
        "full_text": tokenizer.decode(full_ids[0], skip_special_tokens=True),
    }

    with open(run_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Generation finished.")
    print("Generated text:", "".join(generated_text))


if __name__ == "__main__":
    main()