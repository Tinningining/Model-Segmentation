# Llama OM 模型输入输出规格说明

本文档详细说明 Llama 模型切分后每个 OM 文件的输入输出规格。

## 模型切分概览

TinyLlama-1.1B 共 22 层（layers 0-21），切分为 4 个模型：

```
┌─────────────────────────────────────────────────────────────────┐
│  M0: Embedding + Layers 0-4    (5 layers)                       │
│  M1: Layers 5-10               (6 layers)                       │
│  M2: Layers 11-16              (6 layers)                       │
│  M3: Layers 17-21 + Norm + LMHead (5 layers)                   │
└─────────────────────────────────────────────────────────────────┘
```

## 通用说明

### 数据类型约定

- `int64`: 整数类型（token IDs, position IDs）
- `float32`: 浮点类型（hidden states, attention mask, KV cache）
- `float16`: 半精度浮点（某些 KV cache 可能使用）

### 形状约定

- `B`: batch size（通常为 1）
- `L`: 序列长度（max_input_len，通常为 16）
- `S`: KV cache 长度（max_cache_len，通常为 1024）
- `H`: hidden size（2048）
- `V`: vocab size（32000）
- `N`: num_kv_heads（32）
- `D`: head_dim（64）

---

## M0: llama_m0_embed_layers_0_4.om

**功能**: Embedding + Layers 0-4

### 输入 (Inputs)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `input_ids` | `[B, L]` | int64 | 输入 token IDs |
| `attention_mask` | `[B, L, S]` | float32 | 注意力掩码 |
| `position_ids` | `[B, L]` | int64 | 位置 IDs |
| `past_key_values_0` | `[1, B, N, S, D]` | float32 | Layer 0 的 past key |
| `past_key_values_0` (value) | `[1, B, N, S, D]` | float32 | Layer 0 的 past value |
| `past_key_values_1` | `[1, B, N, S, D]` | float32 | Layer 1 的 past key |
| `past_key_values_1` (value) | `[1, B, N, S, D]` | float32 | Layer 1 的 past value |
| `past_key_values_2` | `[1, B, N, S, D]` | float32 | Layer 2 的 past key |
| `past_key_values_2` (value) | `[1, B, N, S, D]` | float32 | Layer 2 的 past value |
| `past_key_values_3` | `[1, B, N, S, D]` | float32 | Layer 3 的 past key |
| `past_key_values_3` (value) | `[1, B, N, S, D]` | float32 | Layer 3 的 past value |
| `past_key_values_4` | `[1, B, N, S, D]` | float32 | Layer 4 的 past key |
| `past_key_values_4` (value) | `[1, B, N, S, D]` | float32 | Layer 4 的 past value |

**总计**: 3 个基础输入 + 10 个 KV cache 输入（5层 × 2）= **13 个输入**

### 输出 (Outputs)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `/model/layers.4/Add_1_output_0` | `[B, L, H]` | float32 | Layer 4 的输出 hidden states |
| `present_key_values_0` (key) | `[1, B, N, L, D]` | float32 | Layer 0 的 present key |
| `present_key_values_0` (value) | `[1, B, N, L, D]` | float32 | Layer 0 的 present value |
| `present_key_values_1` (key) | `[1, B, N, L, D]` | float32 | Layer 1 的 present key |
| `present_key_values_1` (value) | `[1, B, N, L, D]` | float32 | Layer 1 的 present value |
| `present_key_values_2` (key) | `[1, B, N, L, D]` | float32 | Layer 2 的 present key |
| `present_key_values_2` (value) | `[1, B, N, L, D]` | float32 | Layer 2 的 present value |
| `present_key_values_3` (key) | `[1, B, N, L, D]` | float32 | Layer 3 的 present key |
| `present_key_values_3` (value) | `[1, B, N, L, D]` | float32 | Layer 3 的 present value |
| `present_key_values_4` (key) | `[1, B, N, L, D]` | float32 | Layer 4 的 present key |
| `present_key_values_4` (value) | `[1, B, N, L, D]` | float32 | Layer 4 的 present value |

**总计**: 1 个 hidden states + 10 个 KV cache 输出（5层 × 2）= **11 个输出**

---

## M1: llama_m1_layers_5_10.om

**功能**: Layers 5-10

### 输入 (Inputs)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `/model/layers.4/Add_1_output_0` | `[B, L, H]` | float32 | 来自 M0 的 hidden states |
| `input_ids` | `[B, L]` | int64 | 输入 token IDs（用于 RoPE） |
| `attention_mask` | `[B, L, S]` | float32 | 注意力掩码 |
| `position_ids` | `[B, L]` | int64 | 位置 IDs |
| `past_key_values_0` | `[1, B, N, S, D]` | float32 | Layer 0 的 KV（用于 RoPE 基准） |
| `past_key_values_5` | `[1, B, N, S, D]` | float32 | Layer 5 的 past key |
| `past_key_values_5` (value) | `[1, B, N, S, D]` | float32 | Layer 5 的 past value |
| `past_key_values_6` | `[1, B, N, S, D]` | float32 | Layer 6 的 past key |
| `past_key_values_6` (value) | `[1, B, N, S, D]` | float32 | Layer 6 的 past value |
| ... | ... | ... | Layers 7-10 的 KV cache |
| `past_key_values_10` | `[1, B, N, S, D]` | float32 | Layer 10 的 past key |
| `past_key_values_10` (value) | `[1, B, N, S, D]` | float32 | Layer 10 的 past value |

**总计**: 5 个基础输入 + 12 个 KV cache 输入（6层 × 2）= **17 个输入**

### 输出 (Outputs)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `/model/layers.10/Add_1_output_0` | `[B, L, H]` | float32 | Layer 10 的输出 hidden states |
| `present_key_values_5` (key) | `[1, B, N, L, D]` | float32 | Layer 5 的 present key |
| `present_key_values_5` (value) | `[1, B, N, L, D]` | float32 | Layer 5 的 present value |
| ... | ... | ... | Layers 6-10 的 present KV |
| `present_key_values_10` (key) | `[1, B, N, L, D]` | float32 | Layer 10 的 present key |
| `present_key_values_10` (value) | `[1, B, N, L, D]` | float32 | Layer 10 的 present value |

**总计**: 1 个 hidden states + 12 个 KV cache 输出（6层 × 2）= **13 个输出**

---

## M2: llama_m2_layers_11_16.om

**功能**: Layers 11-16

### 输入 (Inputs)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `/model/layers.10/Add_1_output_0` | `[B, L, H]` | float32 | 来自 M1 的 hidden states |
| `input_ids` | `[B, L]` | int64 | 输入 token IDs（用于 RoPE） |
| `attention_mask` | `[B, L, S]` | float32 | 注意力掩码 |
| `position_ids` | `[B, L]` | int64 | 位置 IDs |
| `past_key_values_0` | `[1, B, N, S, D]` | float32 | Layer 0 的 KV（用于 RoPE 基准） |
| `past_key_values_11` | `[1, B, N, S, D]` | float32 | Layer 11 的 past key |
| `past_key_values_11` (value) | `[1, B, N, S, D]` | float32 | Layer 11 的 past value |
| ... | ... | ... | Layers 12-16 的 KV cache |
| `past_key_values_16` | `[1, B, N, S, D]` | float32 | Layer 16 的 past key |
| `past_key_values_16` (value) | `[1, B, N, S, D]` | float32 | Layer 16 的 past value |

**总计**: 5 个基础输入 + 12 个 KV cache 输入（6层 × 2）= **17 个输入**

### 输出 (Outputs)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `/model/layers.16/Add_1_output_0` | `[B, L, H]` | float32 | Layer 16 的输出 hidden states |
| `present_key_values_11` (key) | `[1, B, N, L, D]` | float32 | Layer 11 的 present key |
| `present_key_values_11` (value) | `[1, B, N, L, D]` | float32 | Layer 11 的 present value |
| ... | ... | ... | Layers 12-16 的 present KV |
| `present_key_values_16` (key) | `[1, B, N, L, D]` | float32 | Layer 16 的 present key |
| `present_key_values_16` (value) | `[1, B, N, L, D]` | float32 | Layer 16 的 present value |

**总计**: 1 个 hidden states + 12 个 KV cache 输出（6层 × 2）= **13 个输出**

---

## M3: llama_m3_layers_17_21_lmhead.om

**功能**: Layers 17-21 + RMSNorm + LM Head

### 输入 (Inputs)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `/model/layers.16/Add_1_output_0` | `[B, L, H]` | float32 | 来自 M2 的 hidden states |
| `input_ids` | `[B, L]` | int64 | 输入 token IDs（用于 RoPE） |
| `attention_mask` | `[B, L, S]` | float32 | 注意力掩码 |
| `position_ids` | `[B, L]` | int64 | 位置 IDs |
| `past_key_values_0` | `[1, B, N, S, D]` | float32 | Layer 0 的 KV（用于 RoPE 基准） |
| `past_key_values_17` | `[1, B, N, S, D]` | float32 | Layer 17 的 past key |
| `past_key_values_17` (value) | `[1, B, N, S, D]` | float32 | Layer 17 的 past value |
| ... | ... | ... | Layers 18-21 的 KV cache |
| `past_key_values_21` | `[1, B, N, S, D]` | float32 | Layer 21 的 past key |
| `past_key_values_21` (value) | `[1, B, N, S, D]` | float32 | Layer 21 的 past value |

**总计**: 5 个基础输入 + 10 个 KV cache 输入（5层 × 2）= **15 个输入**

### 输出 (Outputs)

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `logits` | `[B, L, V]` | float32 | 最终的 logits 输出 |
| `present_key_values_17` (key) | `[1, B, N, L, D]` | float32 | Layer 17 的 present key |
| `present_key_values_17` (value) | `[1, B, N, L, D]` | float32 | Layer 17 的 present value |
| ... | ... | ... | Layers 18-21 的 present KV |
| `present_key_values_21` (key) | `[1, B, N, L, D]` | float32 | Layer 21 的 present key |
| `present_key_values_21` (value) | `[1, B, N, L, D]` | float32 | Layer 21 的 present value |

**总计**: 1 个 logits + 10 个 KV cache 输出（5层 × 2）= **11 个输出**

---

## 关键注意事项

### 1. past_key_values_0 的特殊作用

**所有中间节点和尾节点都需要 `past_key_values_0`**，原因：

- RoPE（旋转位置编码）需要第 0 层的 KV cache 作为基准
- 用于计算正确的位置信息
- 即使节点不处理 Layer 0，也需要这个输入

### 2. KV Cache 的形状变化

- **输入 past_key_values**: `[1, B, N, S, D]` - 完整的历史缓存
- **输出 present_key_values**: `[1, B, N, L, D]` - 当前步的新 KV

在实际使用中，需要将 present 拼接到 past 中以形成下一步的输入。

### 3. 实际使用的形状（默认配置）

```python
B = 1           # batch_size
L = 16          # max_input_len
S = 1024        # max_cache_len
H = 2048        # hidden_size
V = 32000       # vocab_size
N = 32          # num_kv_heads
D = 64          # head_dim
```

### 4. 内存占用估算

每个 KV cache（单层，单个 key 或 value）：
```
1 × 1 × 32 × 1024 × 64 × 4 bytes (float32) = 8 MB
```

M0 的 KV cache 总量（5层 × 2）：
```
10 × 8 MB = 80 MB
```

### 5. 数据流示例

**第一步（prompt phase，q_len=5）：**

```
M0: input_ids[1,5] → hidden[1,5,2048] + present_kv_0..4[1,1,32,5,64]
    ↓
M1: hidden[1,5,2048] → hidden[1,5,2048] + present_kv_5..10[1,1,32,5,64]
    ↓
M2: hidden[1,5,2048] → hidden[1,5,2048] + present_kv_11..16[1,1,32,5,64]
    ↓
M3: hidden[1,5,2048] → logits[1,5,32000] + present_kv_17..21[1,1,32,5,64]
    ↓
采样: logits[1,4,32000] → next_token (取最后一个位置)
```

**后续步骤（generation phase，q_len=1）：**

```
M0: input_ids[1,1] + past_kv[1,1,32,1024,64] → hidden[1,1,2048] + present_kv[1,1,32,1,64]
    ↓
M1: hidden[1,1,2048] + past_kv[1,1,32,1024,64] → hidden[1,1,2048] + present_kv[1,1,32,1,64]
    ↓
M2: hidden[1,1,2048] + past_kv[1,1,32,1024,64] → hidden[1,1,2048] + present_kv[1,1,32,1,64]
    ↓
M3: hidden[1,1,2048] + past_kv[1,1,32,1024,64] → logits[1,1,32000] + present_kv[1,1,32,1,64]
    ↓
采样: logits[1,0,32000] → next_token
```

---

## 与代码的对应关系

在 `llama_distributed` 框架中：

### node_head.py (M0)
```python
inputs = [
    padded_ids,           # input_ids [1, 16]
    attention_mask,       # [1, 16, 1024]
    position_ids,         # [1, 16]
    past_key[0],          # Layer 0 key
    past_value[0],        # Layer 0 value
    past_key[1],          # Layer 1 key
    past_value[1],        # Layer 1 value
    ...                   # Layers 2-4
]

outputs = [
    hidden_states,        # [1, 16, 2048]
    present_key[0],       # Layer 0 key
    present_value[0],     # Layer 0 value
    ...                   # Layers 1-4
]
```

### node_middle.py (M1/M2)
```python
inputs = [
    hidden,               # [1, 16, 2048]
    input_ids,            # [1, 16]
    attention_mask,       # [1, 16, 1024]
    position_ids,         # [1, 16]
    past_key_values_0,    # Layer 0 KV (for RoPE)
    past_key[0],          # 当前节点第一层 key
    past_value[0],        # 当前节点第一层 value
    ...                   # 其他层
]
```

### node_tail.py (M3)
```python
outputs = [
    logits,               # [1, 16, 32000]
    present_key[0],       # Layer 17 key
    present_value[0],     # Layer 17 value
    ...                   # Layers 18-21
]
```

---

## 总结

| 模型 | 层范围 | 输入数 | 输出数 | 主要功能 |
|------|--------|--------|--------|----------|
| M0 | Embed + 0-4 | 13 | 11 | Token embedding + 前5层 |
| M1 | 5-10 | 17 | 13 | 中间6层 |
| M2 | 11-16 | 17 | 13 | 中间6层 |
| M3 | 17-21 + LM | 15 | 11 | 后5层 + 输出层 |

所有模型都需要 `past_key_values_0` 用于 RoPE 计算，这是 Llama 架构的特殊要求。
