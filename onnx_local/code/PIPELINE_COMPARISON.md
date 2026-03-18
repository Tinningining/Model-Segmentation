# File Pipeline vs Run Local 输入输出对比

## file_pipeline 执行流程

### Prefill 阶段（无 system KV，past_len=0）

1. **stage_token_embed.py**
   - 输入：prompt text
   - 输出：
     - `tokens.npy`: shape (1, q_len), dtype=int64
     - `hidden_block0.npy`: shape (1, max_input_len, hidden_size), dtype=float32
     - `attention_mask.npy`: shape (1, 1, max_input_len, max_input_len), dtype=float32
     - `position_ids.npy`: shape (1, max_input_len), dtype=int64
     - `meta.json`: {"mode": "prefill", "past_len": 0, "q_len": q_len, ...}

2. **stage_block_common.py** (每个 block)
   - 输入：
     - `hidden_states`: shape (1, max_input_len, hidden_size)
     - `attention_mask`: shape (1, 1, max_input_len, max_input_len)
     - `position_ids`: shape (1, max_input_len)
     - **无 past_key/past_value**（因为 past_len=0 且是首次 prefill）
   - 输出：
     - `hidden_states`: shape (1, max_input_len, hidden_size)
     - `present_key`: shape (num_layers, 1, num_kv_heads, q_len, head_dim)
     - `present_value`: shape (num_layers, 1, num_kv_heads, q_len, head_dim)

### Prefill 阶段（有 system KV，past_len>0）

1. **stage_token_embed.py**
   - 输入：user prompt text
   - 输出：
     - `tokens.npy`: shape (1, q_len)
     - `hidden_block0.npy`: shape (1, max_input_len, hidden_size)
     - `attention_mask.npy`: shape (1, 1, max_input_len, **max_cache_len + max_input_len**)
     - `position_ids.npy`: shape (1, max_input_len)
     - `meta.json`: {"mode": "prefill", "past_len": past_len, "q_len": q_len, ...}

2. **stage_block_common.py** (每个 block)
   - 输入：
     - `hidden_states`: shape (1, max_input_len, hidden_size)
     - `attention_mask`: shape (1, 1, max_input_len, **max_cache_len + max_input_len**)
     - `position_ids`: shape (1, max_input_len)
     - `past_key`: shape (num_layers, 1, num_kv_heads, **max_cache_len**, head_dim)
     - `past_value`: shape (num_layers, 1, num_kv_heads, **max_cache_len**, head_dim)
   - 输出：
     - `hidden_states`: shape (1, max_input_len, hidden_size)
     - `present_key`: shape (num_layers, 1, num_kv_heads, q_len, head_dim)
     - `present_value`: shape (num_layers, 1, num_kv_heads, q_len, head_dim)

### Decode 阶段

1. **stage_token_embed.py**
   - 输入：single token
   - 输出：
     - `tokens.npy`: shape (1, 1)
     - `hidden_block0.npy`: shape (1, 1, hidden_size)
     - `attention_mask.npy`: shape (1, 1, 1, max_cache_len + 1)
     - `position_ids.npy`: shape (1, 1)
     - `meta.json`: {"mode": "decode", "past_len": past_len, "q_len": 1, ...}

2. **stage_block_common.py** (每个 block)
   - 输入：
     - `hidden_states`: shape (1, 1, hidden_size)
     - `attention_mask`: shape (1, 1, 1, max_cache_len + 1)
     - `position_ids`: shape (1, 1)
     - `past_key`: shape (num_layers, 1, num_kv_heads, max_cache_len, head_dim)
     - `past_value`: shape (num_layers, 1, num_kv_heads, max_cache_len, head_dim)
   - 输出：
     - `hidden_states`: shape (1, 1, hidden_size)
     - `present_key`: shape (num_layers, 1, num_kv_heads, 1, head_dim)
     - `present_value`: shape (num_layers, 1, num_kv_heads, 1, head_dim)

## run_local 当前实现问题

### 问题 1: Prefill 阶段 past_len=0 时的处理

**file_pipeline 行为**：
- 当 past_len=0 时，prefill 模型**不接受** past_key/past_value 输入
- attention_mask shape = (1, 1, max_input_len, max_input_len)

**run_local 当前行为**：
- 总是传入 past_key/past_value（即使是全零）
- 这会导致模型报错，因为 prefill 模型在 past_len=0 时不期望这些输入

### 问题 2: 模型选择逻辑

**file_pipeline**：
- past_len=0: 使用 prefill 模型，**不传** past KV
- past_len>0: 使用 prefill 模型，**传入** past KV

**run_local 需要修复**：
- 需要检查 prefill ONNX 模型的输入签名
- 如果模型有两个版本（with/without past KV），需要分别处理
- 如果模型只有一个版本，需要确认其输入要求

## 修复方案

需要在 `onnx_model.py` 中检查模型的输入签名，然后在 `run_local.py` 中根据签名决定是否传入 past KV。
