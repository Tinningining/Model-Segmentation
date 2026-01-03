#!/usr/bin/env python3
"""
使用 transformers 标准库直接调用完整 Qwen3 模型进行推理。
最简单的方式，不拆分、不导出 ONNX，纯原版调用。
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def main():
    parser = argparse.ArgumentParser(description="Qwen3 标准推理")
    parser.add_argument("--model_dir", type=str, required=True, help="模型路径")
    parser.add_argument("--prompt", type=str, default="你好", help="输入提示词")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K 采样")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-P 采样")
    parser.add_argument("--greedy", action="store_true", help="贪婪解码")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备 (cpu/cuda)")
    args = parser.parse_args()

    print(f"[INFO] 加载模型: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(args.device).eval()

    print(f"[INFO] 输入: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    print(f"[INFO] Token IDs: {input_ids.tolist()}")

    # 生成参数
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if args.greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_k"] = args.top_k
        gen_kwargs["top_p"] = args.top_p

    print(f"[INFO] 开始生成...")
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    # 解码输出
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("=" * 50)
    print("[输出]")
    print(generated_text)
    print("=" * 50)

    # 只显示新生成的部分
    new_tokens = output_ids[0][input_ids.shape[1]:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"[新生成部分] {new_text}")


if __name__ == "__main__":
    main()

