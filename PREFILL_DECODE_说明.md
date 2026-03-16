# Prefill / Decode 双模型方案说明

## 背景

原始方案只有一组模型，所有推理步骤使用相同的固定 seq_len（如 16 或 512），导致：
- 输入长度受限于编译时的 seq_len
- Decode 阶段每步只生成 1 个 token，却仍然使用 seq_len=512 的模型，浪费大量算力

新方案将模型拆分为 Prefill 和 Decode 两组，各自针对不同阶段优化。

## 推理流程

```
用户输入 prompt (最多 512 tokens)
        │
        ▼
┌─────────────────────────────┐
│  Prefill 阶段（首次处理）     │
│  使用 prefill 组模型          │
│  一次性处理全部 prompt tokens  │
│  seq_len = 512               │
│  无 past KV 输入              │
│  输出: hidden + present KV    │
└──────────┬──────────────────┘
           │ present KV 写入 KV Cache
           ▼
┌─────────────────────────────┐
│  Decode 阶段（逐 token 生成） │
│  使用 decode 组模型           │
│  每次处理 1 个新 token        │
│  seq_len = 1                 │
│  带 past KV 输入（从 cache）  │
│  输出: hidden + 新 KV         │
│  循环直到 EOS 或达到上限       │
└─────────────────────────────┘
```

## 文件结构

```
onnx_models/
├── prefill/          # Prefill 阶段 ONNX
│   ├── embed.onnx
│   ├── layers_0_6.onnx
│   ├── layers_7_13.onnx
│   ├── layers_14_20.onnx
│   ├── layers_21_27.onnx
│   ├── output.onnx
│   └── config.json
└── decode/           # Decode 阶段 ONNX
    ├── embed.onnx
    ├── layers_0_6.onnx
    ├── layers_7_13.onnx
    ├── layers_14_20.onnx
    ├── layers_21_27.onnx
    ├── output.onnx
    └── config.json

model_om/
├── prefill/          # Prefill 阶段 OM
│   ├── embed.om
│   ├── layers_0_6.om
│   ├── ...
│   └── output.om
└── decode/           # Decode 阶段 OM
    ├── embed.om
    ├── layers_0_6.om
    ├── ...
    └── output.om
```

## 两组模型的输入输出对比

### Embedding

| 阶段 | 输入 | 输出 |
|------|------|------|
| Prefill | input_ids: `[1, 512]` | hidden_states: `[1, 512, 2048]` |
| Decode  | input_ids: `[1, 1]`   | hidden_states: `[1, 1, 2048]`   |

### Transformer Block（以 layers_0_6 为例，7 层）

**Prefill（无 past KV）：**

| 输入名 | 形状 | 说明 |
|--------|------|------|
| hidden_states | `[1, 512, 2048]` | 来自上一层输出 |
| attention_mask | `[1, 1, 512, 512]` | causal mask，无 past |
| position_ids | `[1, 512]` | 位置 0~511 |

| 输出名 | 形状 | 说明 |
|--------|------|------|
| hidden_states_out | `[1, 512, 2048]` | 传给下一层 |
| present_key | `[7, 1, 8, 512, 128]` | 写入 KV Cache |
| present_value | `[7, 1, 8, 512, 128]` | 写入 KV Cache |

**Decode（带 past KV）：**

| 输入名 | 形状 | 说明 |
|--------|------|------|
| hidden_states | `[1, 1, 2048]` | 当前 1 个 token |
| attention_mask | `[1, 1, 1, 1025]` | 1 token attend to 1024 past + 1 self |
| position_ids | `[1, 1]` | 当前位置 |
| past_key | `[7, 1, 8, 1024, 128]` | 从 KV Cache 读取 |
| past_value | `[7, 1, 8, 1024, 128]` | 从 KV Cache 读取 |

| 输出名 | 形状 | 说明 |
|--------|------|------|
| hidden_states_out | `[1, 1, 2048]` | 传给下一层 |
| present_key | `[7, 1, 8, 1, 128]` | 新 token 的 KV，追加到 cache |
| present_value | `[7, 1, 8, 1, 128]` | 新 token 的 KV，追加到 cache |

### Output（RMSNorm + LM Head）

| 阶段 | 输入 | 输出 |
|------|------|------|
| Prefill | hidden_states: `[1, 512, 2048]` | logits: `[1, 512, 151936]` |
| Decode  | hidden_states: `[1, 1, 2048]`   | logits: `[1, 1, 151936]`   |

## 代码修改说明

### convert_to_onnx.py

1. 新增 `Qwen3BlockStackPrefillWrapper` 类：
   - 包装 `Qwen3BlockStackModule`，forward 只接收 3 个参数（hidden_states, attention_mask, position_ids）
   - 不传 past_key/past_value，内部默认为 None
   - 这样导出的 ONNX 图中不包含 past KV 相关的输入节点

2. `export_prefill_onnx()` 函数：
   - 用 `Qwen3BlockStackPrefillWrapper` 导出 Block
   - attention_mask 形状 `[1, 1, prefill_len, prefill_len]`（纯 causal，无 past）
   - 不导出 past_key/past_value 输入

3. `export_decode_onnx()` 函数：
   - 直接用 `Qwen3BlockStackModule` 导出（保留 past_key/past_value 输入）
   - seq_len=1，attention_mask 形状 `[1, 1, 1, max_cache_len+1]`
   - past_key/past_value 形状 `[7, 1, 8, max_cache_len, 128]`

4. 两组模型共享完全相同的权重，数学计算等价，输出一致。

### convert_to_om.sh

1. 分为 Prefill 和 Decode 两个转换阶段
2. Prefill Block 的 `--input_shape` 只有 3 个输入（无 past KV）
3. Decode Block 的 `--input_shape` 有 5 个输入（含 past KV）
4. attention_mask 最后一维：
   - Prefill: `512`（= prefill_len，无 past）
   - Decode: `1025`（= max_cache_len + 1，past + self）

## 使用方法

```bash
# 1. 导出两组 ONNX
cd qwen
python convert_to_onnx.py --model_path /path/to/qwen3_1.7b --prefill_len 512 --max_cache_len 1024

# 2. 转换两组 OM
bash convert_to_om.sh
```

## 优势

- Prefill 阶段一次处理 512 tokens，充分利用并行计算
- Decode 阶段每步只处理 1 token，计算量最小化
- KV Cache 只在 Decode 阶段作为输入，Prefill 阶段直接输出 present KV
- 总上下文长度：512（prompt）+ 最多 512（生成）= 1024（max_cache_len）
