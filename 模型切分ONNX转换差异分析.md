# Qwen 与 Llama 模型切分为 ONNX 的过程差异分析

## 概述

本文档详细分析了 Qwen 和 Llama 两个大语言模型在转换为 ONNX 格式过程中的关键差异。两者虽然都是基于 Transformer 架构的因果语言模型，但在模型切分策略、量化方法、导出流程等方面存在显著不同。

---

## 1. 模型切分策略

### 1.1 Qwen 的切分方式

**特点：模块化切分，分为 6 个独立部分**

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen 模型切分架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                                           │
│  │  embed.onnx  │  ← Embedding 层（独立）                    │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────┐                                       │
│  │ layers_0_6.onnx  │  ← Transformer Block 0-6 (7层)        │
│  └──────┬───────────┘                                       │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────┐                                       │
│  │ layers_7_13.onnx │  ← Transformer Block 7-13 (7层)       │
│  └──────┬───────────┘                                       │
│         │                                                   │
│         ▼                                                   │
│  ┌───────────────────┐                                      │
│  │ layers_14_20.onnx │  ← Transformer Block 14-20 (7层)     │
│  └──────┬────────────┘                                      │
│         │                                                   │
│         ▼                                                   │
│  ┌───────────────────┐                                      │
│  │ layers_21_27.onnx │  ← Transformer Block 21-27 (7层)     │
│  └──────┬────────────┘                                      │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ output.onnx  │  ← LM Head 输出层（独立）                  │
│  └──────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

总计：6 个 ONNX 文件
- 1 个 Embedding 模块
- 4 个 Block 模块（每个 7 层）
- 1 个 Output 模块
```

**代码实现：**

```python
# qwen/convert_to_onnx.py
def export_onnx(model_path: str, onnx_dir: str, seq_len: int = 8):
    # 1) 独立导出 Embedding
    emb = Qwen3EmbeddingModule(base).eval()
    torch.onnx.export(emb, ...)
    
    # 2) 分组导出 Blocks（每组 7 层）
    blocks = [
        ("layers_0_6", 0, 7),
        ("layers_7_13", 7, 14),
        ("layers_14_20", 14, 21),
        ("layers_21_27", 21, 28),
    ]
    for name, s, e in blocks:
        blk = Qwen3BlockStackModule(base, s, e).eval()
        torch.onnx.export(blk, ...)
    
    # 3) 独立导出 Output
    out = Qwen3OutputModule(base).eval()
    torch.onnx.export(out, ...)
```

### 1.2 Llama 的切分方式

**特点：整体导出后再切分**

```
┌─────────────────────────────────────────────────────────────┐
│                   Llama 模型切分架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第一步：导出完整模型                                         │
│  ┌───────────────────────────────────────────────────┐      │
│  │                                                   │      │
│  │         tiny-llama.onnx (完整模型)                 │      │
│  │                                                   │      │
│  │  • Embedding                                     │      │
│  │  • 22 层 Transformer Blocks                       │      │
│  │  • LM Head                                       │      │
│  │                                                   │      │
│  └───────────────────────────────────────────────────┘      │
│                          │                                  │
│                          ▼                                  │
│  第二步：使用 split_on_onnx.py 切分                          │
│  ┌──────────────────────────────────────────────────┐       │
│  │ llama_m0_embed_layers_0_4.onnx                   │       │
│  │ llama_m1_layers_5_10.onnx                        │       │
│  │ llama_m2_layers_11_16.onnx                       │       │
│  │ llama_m3_layers_17_21_lmhead.onnx                │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

总计：先导出 1 个完整 ONNX，再切分为 4 个子模型
```

**代码实现：**

```python
# llama/export_llama.py
def export_onnx(base_model, out_path, quant_cfg_path, act_path):
    # 加载完整模型
    model = LlamaForCausalLM.from_pretrained(base_model, ...)
    
    # 应用量化
    quantize(model, quantize_cfg)
    
    # 导出完整模型（包含所有层）
    torch.onnx.export(
        model,
        f=out_path,  # 输出为单个 ONNX 文件
        args=input_args,
        ...
    )

# llama/split_on_onnx.py
# 后续使用 ONNX 图操作工具切分模型
```

---

## 2. 自定义模块实现

### 2.1 Qwen 的自定义模块

**特点：完全重写 Transformer 组件**

Qwen 实现了一套完整的自定义模块，包括：

```python
# qwen/qwen3_custom_modules.py

# 1. 自定义 Attention 模块
class CustomQwen3Attention(nn.Module):
    """完全自定义的注意力实现"""
    def __init__(self, config: Qwen3Config, layer_idx: int):
        self.q_proj = nn.Linear(...)
        self.k_proj = nn.Linear(...)
        self.v_proj = nn.Linear(...)
        self.o_proj = nn.Linear(...)
        self.q_norm = Qwen3RMSNorm(...)  # QK Normalization
        self.k_norm = Qwen3RMSNorm(...)
    
    def forward(self, hidden_states, cos, sin, attention_mask, 
                past_key, past_value):
        # 手动实现 RoPE
        query_states, key_states = apply_rotary_pos_emb(...)
        # 手动实现 KV Cache 拼接
        key_states = torch.cat([past_key, key_states], dim=2)
        # 手动实现 GQA (Grouped Query Attention)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        ...

# 2. 自定义 MLP 模块
class CustomQwen3MLP(nn.Module):
    """SwiGLU 激活函数的 MLP"""
    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )

# 3. 自定义 Decoder Layer
class CustomQwen3DecoderLayer(nn.Module):
    """完整的 Transformer 层"""
    def __init__(self, config, layer_idx):
        self.self_attn = CustomQwen3Attention(...)
        self.mlp = CustomQwen3MLP(...)
        self.input_layernorm = Qwen3RMSNorm(...)
        self.post_attention_layernorm = Qwen3RMSNorm(...)

# 4. Block Stack 模块（多层打包）
class Qwen3BlockStackModule(nn.Module):
    """将连续的多个 Decoder Layer 打包"""
    def __init__(self, base_model, start_layer, end_layer):
        self.layers = nn.ModuleList()
        for idx in range(start_layer, end_layer):
            custom_layer = CustomQwen3DecoderLayer(...)
            # 从原模型加载权重
            custom_layer.load_state_dict(
                base_model.model.layers[idx].state_dict()
            )
            self.layers.append(custom_layer)
```

**优势：**
- ✅ 完全控制前向传播逻辑
- ✅ 可以精确控制 KV Cache 的输入输出格式
- ✅ 便于调试和优化
- ✅ 支持自定义操作（如 QK Normalization）

### 2.2 Llama 的模块修改

**特点：修改 transformers 库源码**

Llama 采用直接修改 `transformers` 库的方式：

```python
# llama/modeling_llama_4.35.py
# 这是修改后的 transformers/models/llama/modeling_llama.py

class LlamaForCausalLM(LlamaPreTrainedModel):
    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,  # 修改为分开的 KV 输入
        ...
    ):
        # 修改前向传播逻辑以支持 ONNX 导出
        # 将 past_key_values 从 tuple 改为独立的 tensor 列表
        ...
```

**使用方式：**

```bash
# 必须替换 transformers 库中的文件
TRANSFORMERS_PATH=$(python -c "import transformers; ...")
cp modeling_llama_4.35.py $TRANSFORMERS_PATH/models/llama/modeling_llama.py
```

**劣势：**
- ❌ 需要修改第三方库源码
- ❌ 版本兼容性问题
- ❌ 不便于维护和升级
- ❌ 可能影响其他使用 transformers 的项目

---

## 3. 量化策略

### 3.1 Qwen 的量化方式

**特点：不在导出阶段进行量化**

Qwen 的 ONNX 导出脚本中**没有量化逻辑**，保持原始精度（通常是 FP16 或 FP32）：

```python
# qwen/convert_to_onnx.py
def export_onnx(model_path: str, onnx_dir: str, seq_len: int = 8):
    base = load_base_qwen3(model_path)  # 加载原始模型
    
    # 直接导出，无量化
    emb = Qwen3EmbeddingModule(base).eval()
    torch.onnx.export(emb, ...)
    
    # Block 也是直接导出
    blk = Qwen3BlockStackModule(base, s, e).eval()
    torch.onnx.export(blk, ...)
```

**量化时机：**
- 可能在 ONNX → OM 转换时进行（使用 ATC 工具）
- 或在推理时使用硬件加速的量化

### 3.2 Llama 的量化方式

**特点：导出前进行量化**

Llama 在导出 ONNX 之前就应用了复杂的量化策略：

```python
# llama/export_llama.py
def export_onnx(base_model, out_path, quant_cfg_path, act_path):
    # 1. 加载模型
    model = LlamaForCausalLM.from_pretrained(...)
    
    # 2. 加载量化配置
    spec = importlib.util.spec_from_file_location("quant_cfg_module", quant_cfg_path)
    quant_cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant_cfg_module)
    quantize_cfg = quant_cfg_module.get(model_cfg, act_path)
    
    # 3. 应用量化（关键步骤）
    from quantize import quantize
    quantize(model, quantize_cfg)
    
    # 4. 导出量化后的模型
    torch.onnx.export(model, ...)
```

**支持的量化方法：**

```python
# llama/quantize.py

# 1. W8Linear - 仅权重 INT8 量化
class W8Linear(nn.Module):
    def __init__(self, origin_weight, bias, act_max, alpha=32):
        self.weight_q, self.max_val = quantize_mat(origin_weight)
    
    def forward(self, x):
        return nn.functional.linear(
            x, 
            dequantize_mat(self.weight_q, self.max_val),
            bias=self.bias
        )

# 2. W8X8Linear - 权重和激活都 INT8 量化
class W8X8Linear(nn.Module):
    def __init__(self, ori_w, bias, act_max, alpha=32):
        # Smooth Quant: 平衡权重和激活的量化难度
        self.scales = (act_max.pow(alpha) / 
                      ori_w.abs().max(dim=0)[0].pow(1 - alpha))
        ori_w = ori_w.mul(self.scales)
        self.weight_q, self.max_val = quantize_mat(ori_w)
    
    def forward(self, x):
        x = x.div(self.scales)  # Smooth
        x_q, x_max = quantize_mat(x)
        return qMatmul(x_q, x_max, self.weight_q, self.max_val, x.dtype)

# 3. W8SDLinear - 静态分解量化
class W8SDLinear(nn.Module):
    """将权重分解为量化部分和高精度部分"""
    def __init__(self, origin_weight, bias, act_max, alpha=32):
        # 找出异常值索引
        self.idx_unq, self.t = get_unq_idx_topk(act_max, alpha)
        # 分解权重
        self.weight_q, self.weight_unq = decomposition(...)

# 4. W8DXLinear - 动态分解量化
class W8DXLinear(nn.Module):
    """运行时动态分解激活值"""
    def forward(self, x):
        idx_unq, t = get_unq_idx_topk(x, self.alpha)
        x_q, x_unq = decomposition(x, idx_unq, t)
        ...
```

**量化配置文件示例：**

```python
# llama/config/w8x8.py
def get(model_cfg, act_scales_path):
    return {
        'act_scales_path': act_scales_path,
        'smooth': True,
        'alpha': 0.85,
        '0.q_proj': {'type': 'W8X8', 'alpha': 32},
        '0.k_proj': {'type': 'W8X8', 'alpha': 32},
        '0.v_proj': {'type': 'W8X8', 'alpha': 32},
        '0.o_proj': {'type': 'W8X8', 'alpha': 32},
        ...
    }
```

---

## 4. KV Cache 处理

### 4.1 Qwen 的 KV Cache

**特点：每层独立的 KV Cache**

```python
# 输入格式
past_key: [num_layers, batch, num_kv_heads, seq_len, head_dim]
past_value: [num_layers, batch, num_kv_heads, seq_len, head_dim]

# 例如 layers_0_6.onnx (7层)
past_key.shape = [7, 1, 8, 1024, 128]
past_value.shape = [7, 1, 8, 1024, 128]

# 输出格式（只返回新生成的 KV）
present_key: [num_layers, batch, num_kv_heads, new_seq_len, head_dim]
present_value: [num_layers, batch, num_kv_heads, new_seq_len, head_dim]
```

**代码实现：**

```python
# qwen/qwen3_custom_modules.py
class Qwen3BlockStackModule(nn.Module):
    def forward(self, hidden_states, attention_mask, position_ids,
                past_key, past_value):
        present_keys = []
        present_values = []
        
        for layer_idx, layer in enumerate(self.layers):
            # 提取当前层的 past KV
            layer_past_k = past_key[layer_idx] if past_key is not None else None
            layer_past_v = past_value[layer_idx] if past_value is not None else None
            
            # 前向传播
            hidden_states, pk, pv = layer(
                hidden_states, cos, sin, attention_mask,
                layer_past_k, layer_past_v
            )
            
            present_keys.append(pk)
            present_values.append(pv)
        
        # 堆叠所有层的 KV
        present_key = torch.stack(present_keys, dim=0)
        present_value = torch.stack(present_values, dim=0)
        
        return hidden_states, present_key, present_value
```

### 4.2 Llama 的 KV Cache

**特点：所有层的 KV Cache 作为独立输入**

```python
# 输入格式（每层单独作为输入）
input_names = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "past_key_values_0",   # 第 0 层
    "past_key_values_1",   # 第 1 层
    ...
    "past_key_values_21",  # 第 21 层
]

# 每层的 KV Cache 格式
past_key_values_i.shape = [2, batch, num_kv_heads, kv_len, head_dim]
# 其中 [0] 是 key, [1] 是 value

# 输出格式（每层单独输出）
output_names = [
    "logits",
    "present_key_values_0",
    "present_key_values_1",
    ...
    "present_key_values_21",
    "attn_scores"
]
```

**代码实现：**

```python
# llama/export_llama.py
def export_onnx(base_model, out_path, quant_cfg_path, act_path):
    # 构造输入名称
    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["logits"]
    
    for i in range(model_cfg.num_hidden_layers):
        input_names.append(f"past_key_values_{i}")
        output_names.append(f"present_key_values_{i}")
    
    # 构造 KV Cache 输入
    past_key_values = []
    for _ in range(n_layers):
        past_key_values.append(
            torch.rand((2, batch_size, n_heads, kv_len, head_dim), 
                      dtype=torch.float16).to(device)
        )
    
    # 导出
    torch.onnx.export(
        model,
        args=(input_ids, attention_mask, position_ids, 
              past_key_values, None, None, True, True),
        input_names=input_names,
        output_names=output_names,
        ...
    )
```

---

## 5. 动态轴配置

### 5.1 Qwen 的动态轴

```python
# qwen/convert_to_onnx.py

# Embedding 层
dynamic_axes={
    "input_ids": {0: "B", 1: "T"},
    "hidden_states": {0: "B", 1: "T"},
}

# Block 层
dynamic_axes={
    "hidden_states": {0: "B", 1: "T"},
    "attention_mask": {0: "B", 2: "Q", 3: "KV"},
    "position_ids": {0: "B", 1: "T"},
    "past_key": {0: "L", 3: "KV_IN"},      # 层数和序列长度可变
    "past_value": {0: "L", 3: "KV_IN"},
    "hidden_states_out": {0: "B", 1: "T"},
    "present_key": {0: "L", 3: "KV_OUT"},
    "present_value": {0: "L", 3: "KV_OUT"},
}

# Output 层
dynamic_axes={
    "hidden_states": {0: "B", 1: "T"},
    "logits": {0: "B", 1: "T"},
}
```

### 5.2 Llama 的动态轴

```python
# llama/export_llama.py

dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "seq_length"},
    "attention_mask": {0: "batch_size", 1: "all_len"},
    "position_ids": {0: "batch_size", 1: "seq_length"},
}

# 每一层的 KV Cache 都是独立的动态轴
for i in range(model_cfg.num_hidden_layers):
    dynamic_axes[f"past_key_values_{i}"] = {
        1: "batch_size", 
        3: "kv_len"
    }
    dynamic_axes[f"present_key_values_{i}"] = {
        1: "batch_size", 
        3: "kv_len"
    }
```

---

## 6. 推理部署差异

### 6.1 Qwen 的部署方式

**4 节点流水线并行：**

```
Node 0 (设备1)          Node 1 (设备2)          Node 2 (设备3)          Node 3 (设备4)
┌─────────────┐        ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ embed.om    │        │             │        │             │        │             │
│ layers_0_6  │───────▶│ layers_7_13 │───────▶│layers_14_20 │───────▶│layers_21_27 │
│             │        │             │        │             │        │ output.om   │
│ [头节点]     │        │ [中间节点]   │        │ [中间节点]   │        │ [尾节点]     │
└─────────────┘        └─────────────┘        └─────────────┘        └─────────────┘
      ▲                                                                      │
      │                                                                      │
      └──────────────────────── next_token ◀─────────────────────────────────┘
```

**特点：**
- ✅ 模块化设计，每个节点职责清晰
- ✅ 支持真正的分布式部署（多台设备）
- ✅ 网络传输数据量小（只传 hidden_states 和 token）
- ✅ KV Cache 保存在本地，无需网络传输

### 6.2 Llama 的部署方式

**先整体后切分：**

```
步骤 1: 导出完整模型
┌────────────────────────────────────┐
│     tiny-llama.onnx (完整)          │
│  • Embedding + 22 Layers + LM Head │
└────────────────────────────────────┘
                │
                ▼
步骤 2: 使用 split_on_onnx.py 切分
┌────────────────────────────────────┐
│ llama_m0_embed_layers_0_4.om       │
│ llama_m1_layers_5_10.om            │
│ llama_m2_layers_11_16.om           │
│ llama_m3_layers_17_21_lmhead.om    │
└────────────────────────────────────┘
                │
                ▼
步骤 3: 分布式推理（类似 Qwen）
```

**特点：**
- ❌ 需要额外的切分步骤
- ❌ 切分逻辑复杂（需要操作 ONNX 图）
- ✅ 灵活性高（可以任意切分）
- ✅ 支持不同的切分策略

---

## 7. 代码复杂度对比

### 7.1 Qwen

| 文件 | 行数 | 复杂度 | 说明 |
|------|------|--------|------|
| `convert_to_onnx.py` | ~100 | 低 | 简洁的导出脚本 |
| `qwen3_custom_modules.py` | ~300 | 中 | 自定义模块实现 |
| **总计** | **~400** | **中** | 代码清晰，易于维护 |

### 7.2 Llama

| 文件 | 行数 | 复杂度 | 说明 |
|------|------|--------|------|
| `export_llama.py` | ~100 | 中 | 导出脚本 |
| `quantize.py` | ~200 | 高 | 复杂的量化逻辑 |
| `modeling_llama_4.35.py` | ~1000+ | 高 | 修改后的 transformers 源码 |
| `split_on_onnx.py` | ~200 | 高 | ONNX 图切分逻辑 |
| `config/*.py` | ~50×5 | 中 | 多个量化配置 |
| **总计** | **~1800+** | **高** | 代码复杂，维护困难 |

---

## 8. 优缺点总结

### 8.1 Qwen 方案

**优点：**
- ✅ **模块化设计**：每个部分独立导出，职责清晰
- ✅ **代码简洁**：无需修改第三方库，易于维护
- ✅ **部署灵活**：可以灵活组合不同的模块
- ✅ **调试友好**：每个模块可以单独测试
- ✅ **版本兼容**：不依赖特定版本的 transformers

**缺点：**
- ❌ **需要自定义模块**：需要手动实现 Transformer 组件
- ❌ **初期开发成本高**：需要深入理解模型结构
- ❌ **无内置量化**：量化需要在其他阶段进行

**适用场景：**
- 需要精确控制模型结构
- 需要自定义操作或优化
- 长期维护的项目
- 多设备分布式部署

### 8.2 Llama 方案

**优点：**
- ✅ **量化丰富**：支持多种量化策略（W8, W8X8, W8SD, W8DX）
- ✅ **Smooth Quant**：先进的量化技术，精度损失小
- ✅ **灵活切分**：可以任意切分模型
- ✅ **快速原型**：基于现有库，开发速度快

**缺点：**
- ❌ **修改第三方库**：需要替换 transformers 源码
- ❌ **版本锁定**：依赖特定版本的 transformers (4.35)
- ❌ **维护困难**：代码复杂，不易调试
- ❌ **两步流程**：先导出完整模型，再切分
- ❌ **兼容性问题**：可能影响其他项目

**适用场景：**
- 需要高精度量化
- 单机或小规模部署
- 快速实验和原型开发
- 对模型精度要求高

---

## 9. 技术细节对比表

| 对比项 | Qwen | Llama |
|--------|------|-------|
| **导出方式** | 模块化分别导出 | 整体导出后切分 |
| **ONNX 文件数** | 6 个（embed + 4×blocks + output） | 1 个完整模型 → 4 个切分模型 |
| **自定义模块** | 完全自定义 Transformer 组件 | 修改 transformers 源码 |
| **量化时机** | 导出后（ATC 或推理时） | 导出前（PyTorch 层面） |
| **量化方法** | 硬件加速量化 | W8/W8X8/W8SD/W8DX |
| **KV Cache 格式** | 堆叠格式 [L,B,H,S,D] | 独立输入 per-layer |
| **动态轴** | 简洁（B, T, KV） | 详细（每层独立） |
| **代码行数** | ~400 行 | ~1800+ 行 |
| **维护难度** | 低 | 高 |
| **版本依赖** | 无特定版本要求 | transformers 4.35 |
| **调试难度** | 低（模块独立） | 高（整体复杂） |
| **部署灵活性** | 高（模块化） | 中（需切分） |
| **开发成本** | 高（需自定义） | 中（基于现有库） |

---

## 10. 关键技术差异深度解析

### 10.1 RoPE (Rotary Position Embedding) 实现

**Qwen：**
```python
# 在 Block 模块内部计算 RoPE
cos, sin = self.rotary_emb(hidden_states, position_ids)
query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin
)
```

**Llama：**
```python
# 在完整模型中计算，导出时包含在 ONNX 图中
# 使用 transformers 库的原生实现
```

### 10.2 Attention Mask 处理

**Qwen：**
```python
# 显式传入 attention_mask
# Shape: [batch, 1, seq_len, seq_len*2]
attn_weights = attn_weights + attention_mask
```

**Llama：**
```python
# 在模型内部构造 causal mask
# 导出时固化在 ONNX 图中
```

### 10.3 GQA (Grouped Query Attention) 实现

**Qwen：**
```python
# 手动实现 KV 重复
def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )
```

**Llama：**
```python
# 使用 transformers 库的实现
# 在导出时自动处理
```

---

## 11. 性能对比分析

### 11.1 导出速度

| 阶段 | Qwen | Llama |
|------|------|-------|
| 模型加载 | 快（按需加载） | 慢（完整加载） |
| 量化处理 | 无 | 慢（复杂量化） |
| ONNX 导出 | 快（分模块） | 慢（整体导出） |
| 模型切分 | 无需切分 | 需要额外切分 |
| **总时间** | **短** | **长** |

### 11.2 推理性能

| 指标 | Qwen | Llama |
|------|------|-------|
| 首 Token 延迟 | 低（模块化） | 中（整体） |
| 吞吐量 | 高（流水线） | 中 |
| 内存占用 | 低（分布式） | 高（单机） |
| 网络带宽 | 低（只传 hidden_states） | 中 |

### 11.3 模型精度

| 方面 | Qwen | Llama |
|------|------|-------|
| 量化精度 | 取决于 ATC 配置 | 高（Smooth Quant） |
| 数值稳定性 | 高（FP16/FP32） | 中（INT8 量化） |
| 精度损失 | 小 | 可控（多种量化策略） |

---

## 12. 实际应用建议

### 12.1 选择 Qwen 方案的情况

1. **多设备分布式部署**
   - 有多台昇腾设备（如 4 台香橙派）
   - 需要流水线并行推理
   - 网络带宽有限

2. **长期维护项目**
   - 需要频繁修改模型结构
   - 需要添加自定义操作
   - 团队有深度学习背景

3. **精确控制需求**
   - 需要自定义 KV Cache 管理
   - 需要特殊的注意力机制
   - 需要模块化测试和调试

### 12.2 选择 Llama 方案的情况

1. **快速原型开发**
   - 需要快速验证想法
   - 基于现有 transformers 库
   - 开发时间紧张

2. **高精度量化需求**
   - 需要 INT8 量化
   - 需要 Smooth Quant
   - 对模型精度要求高

3. **单机部署**
   - 只有一台设备
   - 不需要分布式
   - 内存充足

### 12.3 混合方案

可以结合两者优势：

```python
# 1. 使用 Llama 的量化方法
from llama.quantize import quantize, W8X8Linear

# 2. 使用 Qwen 的模块化导出
class QuantizedQwen3BlockStackModule(nn.Module):
    def __init__(self, base_model, start_layer, end_layer, quant_cfg):
        super().__init__()
        # 加载层
        for idx in range(start_layer, end_layer):
            layer = CustomQwen3DecoderLayer(...)
            # 应用量化
            quantize(layer, quant_cfg)
            self.layers.append(layer)
```

---

## 13. 总结

### 核心差异

1. **架构理念**
   - Qwen：模块化优先，分而治之
   - Llama：整体优先，后期切分

2. **实现方式**
   - Qwen：自定义模块，完全控制
   - Llama：修改源码，快速开发

3. **量化策略**
   - Qwen：后置量化，硬件加速
   - Llama：前置量化，软件实现

4. **部署目标**
   - Qwen：分布式流水线
   - Llama：灵活切分

### 最佳实践建议

1. **对于新项目**：优先考虑 Qwen 方案，模块化设计更利于长期维护

2. **对于快速验证**：可以使用 Llama 方案，快速得到结果

3. **对于生产环境**：
   - 多设备：Qwen 方案
   - 单设备：Llama 方案（带量化）
   - 混合部署：结合两者优势

4. **对于团队协作**：
   - Qwen 方案代码更清晰，便于多人协作
   - Llama 方案需要统一环境配置

### 未来发展方向

1. **统一框架**：开发统一的模型导出和切分框架
2. **自动量化**：自动选择最优量化策略
3. **动态切分**：根据硬件资源动态调整切分策略
4. **性能优化**：进一步优化网络传输和计算效率

---

## 附录

### A. 相关文件清单

**Qwen 项目：**
```
qwen/
├── convert_to_onnx.py              # ONNX 导出脚本
├── qwen3_custom_modules.py         # 自定义模块
├── config.json                     # 模型配置
├── distributed_inference/          # 分布式推理框架
│   ├── node_head.py               # 头节点
│   ├── node_middle.py             # 中间节点
│   ├── node_tail.py               # 尾节点
│   └── ...
└── model_om/                       # OM 模型文件
    ├── embed.om
    ├── layers_0_6.om
    ├── layers_7_13.om
    ├── layers_14_20.om
    ├── layers_21_27.om
    └── output.om
```

**Llama 项目：**
```
llama/
├── export_llama.py                 # ONNX 导出脚本
├── quantize.py                     # 量化实现
├── modeling_llama_4.35.py          # 修改后的 transformers 源码
├── split_on_onnx.py                # ONNX 切分脚本
├── config/                         # 量化配置
│   ├── w8.py
│   ├── w8x8.py
│   ├── w8dx.py
│   └── ...
└── inference_net/                  # 推理框架
    ├── main.py
    ├── engine.py
    └── ...
```

### B. 参考资源

- [ONNX 官方文档](https://onnx.ai/)
- [PyTorch ONNX 导出指南](https://pytorch.org/docs/stable/onnx.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [昇腾 CANN 开发文档](https://www.hiascend.com/document)
- [Smooth Quant 论文](https://arxiv.org/abs/2211.10438)

---

**文档版本：** v1.0  
**最后更新：** 2026-03-07  
**作者：** Kiro AI Assistant
