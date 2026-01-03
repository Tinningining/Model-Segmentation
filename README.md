# Qwen3 阶段 1-2 归档说明

本目录保存了“模型切割 → 导出 ONNX → ONNX 分阶段/整体推理（静态形状）”完整流程的脚本与说明。结构如下：

```
atlas/
├── convert_to_onnx.py
├── qwen3_custom_modules.py
├── config.json (示例模型配置)
└── file_pipeline/
    ├── stage_token_embed.py
    ├── stage_block_common.py + stage_block{0..3}.py
    ├── stage_output.py
    ├── stage_utils.py
    └── run_pipeline_auto.py
```

## 各文件用途

- `convert_to_onnx.py`：调用 `qwen3_custom_modules.py` 中的模块，将原始 Qwen3 模型拆分为 embed、四个 block（每个 7 层）、output 头等 ONNX 文件，支持指定 `--seq_len` 输出 prefill 或 decode 模型。
- `qwen3_custom_modules.py`：封装 HuggingFace Qwen3 模型的嵌入、BlockStack、输出头，供导出脚本引用。
- `config.json`：示例模型配置（取自 ONNX 目录的 `config.json`），提供 hidden_size、头数等参数，ONNX 推理脚本会读取。
- `file_pipeline/stage_token_embed.py`：分阶段脚本第 1 步，负责 tokenize + 嵌入，并根据 `--max_cache_len/--max_input_len` 构造**固定形状**的 `hidden_block0.npy`、`attention_mask.npy`、`position_ids.npy`、`meta.json`。
- `file_pipeline/stage_block_common.py` + `stage_block{0..3}.py`：每个子脚本加载对应 block ONNX，读取 `meta.json` 中的有效长度，仅在固定 KV buffer 中覆盖 `[past_len : past_len + q_len]` 区间。
- `file_pipeline/stage_output.py`：调用 `output.onnx` 计算 logits，裁剪成 `q_len` 的真实部分后再做采样。
- `file_pipeline/stage_utils.py`：提供静态 KV/mask/position 的构造与写入工具（无动态 concat）。
- `file_pipeline/run_pipeline_auto.py`：自动串联上述阶段，贯穿同一组 `--max_cache_len/--max_input_len` 参数，可指定 prefill/decode 双目录。

## 使用指南

1. **准备 ONNX（prefill + decode）**：至少导出两份 ONNX，示例：
   ```bash
   python convert_to_onnx.py \
     --model_path /path/to/qwen3_1.7b \
     --onnx_dir /path/to/onnx_prefill \
     --seq_len 16    # prefill: 可容纳较长 prompt
   python convert_to_onnx.py \
     --model_path /path/to/qwen3_1.7b \
     --onnx_dir /path/to/onnx_decode \
     --seq_len 1     # decode: 单 token
   ```
   生成的 `config.json` 会被分阶段脚本读取，确定 hidden_size、头数等参数。若只需单一场景，也可以只导出一种 `seq_len`。

2. **分阶段推理（手动）**：在某个目录中准备 `prompt.txt`，然后依次执行（以 prefill 为例）：
   ```bash
   python file_pipeline/stage_token_embed.py \
     --prompt prompt.txt \
     --onnx_dir /path/to/onnx_prefill \
     --tokenizer_dir /path/to/tokenizer_dir \
     --output_dir step_0000 \
     --past_len 0 \
     --max_cache_len 1024 \
     --max_input_len 16 \
     --write_tokens

   python file_pipeline/stage_block0.py --onnx_dir /path/to/onnx_prefill --hidden_in step_0000/hidden_block0.npy --hidden_out step_0000/hidden_block1.npy --attention_mask step_0000/attention_mask.npy --position_ids step_0000/position_ids.npy --past_key kv_cache/past_key_block0.npy --past_value kv_cache/past_value_block0.npy --meta step_0000/meta.json
   # 依次运行 stage_block1/2/3，参数同上

   python file_pipeline/stage_output.py \
     --onnx_dir /path/to/onnx_prefill \
     --tokenizer_dir /path/to/tokenizer_dir \
     --hidden_in step_0000/hidden_block4.npy \
     --logits step_0000/logits.npy \
     --next_token_file step_0000/next_token_id.txt \
     --next_token_text step_0000/next_token.txt \
     --tokens_file step_0000/tokens.npy \
     --max_input_len 16 \
     --meta step_0000/meta.json \
     --greedy
   ```
   每个 step 结束后 `meta.json` 会写入 `past_len/q_len/max_cache_len/max_input_len`，下一步 decode 时传入 `--past_len meta["next_past_len"]`，并沿用同一套静态长度。提示：当前版本未自动拆分 prompt，如果 prompt token 数超过 `max_input_len` 需要手动调大 `--max_input_len`。

3. **自动运行（推荐）**：用 `run_pipeline_auto.py` 串联 prefill + decode：
   ```bash
   python file_pipeline/run_pipeline_auto.py \
     --prompt prompt.txt \
     --prefill_onnx_dir /path/to/onnx_prefill \
     --decode_onnx_dir /path/to/onnx_decode \
     --tokenizer_dir /path/to/tokenizer_dir \
     --run_root runs_auto \
     --kv_dir kv_cache \
     --steps 50 \
     --max_cache_len 1024 \
     --max_input_len 16 \
     --temperature 1.0 --top_k 0 --top_p 1.0 \
     --greedy \
     --clean
   ```
   - 第 0 步使用 `prefill_onnx_dir`，后续使用 `decode_onnx_dir`；
   - `--max_input_len` 必须 ≥ prompt 的 token 数，否则脚本会报错提示；
   - `runs_auto/step_xxxx` 下会保存所有 `.npy`、`meta.json`、`next_token*.txt`，整个生成结果写入 `summary.json`。

4. **备份/迁移**：将 `archive_stage12/` 整个目录压缩或同步到其他机器即可，使用时只需按 README 中命令配置对应路径。

## 注意事项

- 需要 HuggingFace `transformers`、`onnxruntime`、`numpy` 等依赖
- 确保 tokenizer 目录完整（`tokenizer.json`、`tokenizer_config.json`、merges/vocab 等）。
- KV cache `.npy` 需保持固定 shape（float16，形如 `(layers,1,kv_heads,max_cache_len,head_dim)`），脚本会自动初始化/校验，勿手动改动维度。
- `meta.json` 是各 stage 的约束来源，若需重新开始 decode，记得更新 `--past_len` 或清空 KV 目录。

