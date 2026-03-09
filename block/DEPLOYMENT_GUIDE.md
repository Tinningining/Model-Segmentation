# Qwen3 模型部署指南

本文档详细介绍了 Qwen3 模型的部署流程，包括模型适配、ONNX 转换、模型拆分、OM 转换以及分块推理运行。

---

## 目录

1. [适配模型信息](#1-适配模型信息)
2. [量化方案](#2-量化方案)
3. [PyTorch 转 ONNX](#3-pytorch-转-onnx)
4. [ONNX 模型拆分](#4-onnx-模型拆分)
5. [ONNX 转 OM](#5-onnx-转-om)
6. [分块推理运行](#6-分块推理运行)
7. [位置编码与精度问题](#7-位置编码与精度问题)

---

## 1. 适配模型信息

### 当前适配模型

| 模型名称 | 参数量 | 架构 |
|---------|--------|------|
| Qwen3-1.7B | 1.7B (17亿参数) | Qwen3ForCausalLM |

### 模型配置参数

从 `config.json` 中提取的关键参数：

```json
{
  "vocab_size": 151936,
  "max_position_embeddings": 40960,
  "hidden_size": 2048,
  "intermediate_size": 6144,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "attention_bias": false,
  "torch_dtype": "float32"
}
```

### 参数量计算

- **隐藏层维度**: 2048
- **中间层维度**: 6144
- **注意力头数**: 16
- **KV 头数**: 8 (GQA - Grouped Query Attention)
- **层数**: 28
- **词表大小**: 151936

---

## 2. 量化方案

### 当前量化策略

**模型权重**: 未使用量化，保持 **FP32** 精度

**KV Cache**: 使用 **FP16** 量化存储

```python
# KV Cache 存储格式 (来自 stage_utils.py)
def ensure_static_kv(path: Path, layers: int, kv_heads: int, head_dim: int, max_cache_len: int):
    target_shape = (layers, 1, kv_heads, max_cache_len, head_dim)
    empty = np.zeros(target_shape, dtype=np.float16)  # FP16 存储
    save_array(path, empty)
```

**推理计算**: FP32

```python
# 推理时转换为 FP32 (来自 stage_block_common.py)
feeds = {
    "hidden_states": hidden,
    "attention_mask": attn,
    "position_ids": pos,
    "past_key": past_key.astype(np.float32, copy=False),  # FP16 → FP32
    "past_value": past_value.astype(np.float32, copy=False),
}
```

---

## 3. PyTorch 转 ONNX

### 转换脚本

使用 `convert_to_onnx.py` 进行转换。

### 模型拆分结构

模型被拆分为以下组件：

1. **Embedding 模块** (`embed.onnx`)
2. **Decoder Block 模块** (4个，每个包含7层)
   - `layers_0_6.onnx` (第 0-6 层)
   - `layers_7_13.onnx` (第 7-13 层)
   - `layers_14_20.onnx` (第 14-20 层)
   - `layers_21_27.onnx` (第 21-27 层)
3. **Output 模块** (`output.onnx`)

### 转换命令

```bash
# 本地转换示例 (Windows)
python convert_to_onnx.py \
  --model_path D:\qwen_split\qwen3_1.7b \
  --onnx_dir onnx_models \
  --seq_len 16

# Linux 示例
python convert_to_onnx.py \
  --model_path /path/to/qwen3_1.7b \
  --onnx_dir onnx_models \
  --seq_len 16
```

**参数说明**:
- `--model_path`: HuggingFace 格式的 Qwen3 模型路径
- `--onnx_dir`: ONNX 模型输出目录
- `--seq_len`: 序列长度，决定了模型的输入形状

### 核心转换代码

```python
# 1) Embedding 导出
emb = Qwen3EmbeddingModule(base).eval()
ids = torch.zeros(1, seq_len, dtype=torch.float32)
torch.onnx.export(
    emb,
    (ids,),
    str(onnx_path / "embed.onnx"),
    input_names=["input_ids"],
    output_names=["hidden_states"],
    dynamic_axes={
        "input_ids": {0: "B", 1: "T"},
        "hidden_states": {0: "B", 1: "T"},
    },
    opset_version=13,
    export_params=True,
)

# 2) Block 导出 (每7层一个)
blocks = [
    ("layers_0_6", 0, 7),
    ("layers_7_13", 7, 14),
    ("layers_14_20", 14, 21),
    ("layers_21_27", 21, 28),
]
for name, s, e in blocks:
    blk = Qwen3BlockStackModule(base, s, e).eval()
    hs = torch.zeros(1, seq_len, base.config.hidden_size)
    attn = torch.zeros(1, 1, seq_len, seq_len * 2)
    pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    past_shape = (e - s, 1, kv_heads, seq_len, head_dim)
    past_key = torch.zeros(past_shape)
    past_value = torch.zeros(past_shape)
    torch.onnx.export(
        blk,
        (hs, attn, pos, past_key, past_value),
        str(onnx_path / f"{name}.onnx"),
        input_names=["hidden_states", "attention_mask", "position_ids", "past_key", "past_value"],
        output_names=["hidden_states_out", "present_key", "present_value"],
        dynamic_axes={...},
        opset_version=13,
        export_params=True,
    )

# 3) Output 导出
out = Qwen3OutputModule(base).eval()
torch.onnx.export(
    out,
    (hs,),
    str(onnx_path / "output.onnx"),
    input_names=["hidden_states"],
    output_names=["logits"],
    dynamic_axes={...},
    opset_version=13,
    export_params=True,
)
```

---

## 4. ONNX 模型拆分

### 拆分策略

采用 **按层分组** 的拆分策略，将 28 层 Decoder 分为 4 个 Block：

```python
BLOCK_LAYOUT = [
    (0, 7, "layers_0_6.onnx"),    # Block 0: 第 0-6 层
    (7, 14, "layers_7_13.onnx"),  # Block 1: 第 7-13 层
    (14, 21, "layers_14_20.onnx"), # Block 2: 第 14-20 层
    (21, 28, "layers_21_27.onnx"), # Block 3: 第 21-27 层
]
```

### 拆分注意事项

#### 1. KV Cache 形状一致性

每个 Block 的 KV Cache 必须保持固定形状：

```python
# KV Cache 形状: (layers, batch, kv_heads, max_cache_len, head_dim)
target_shape = (layers, 1, kv_heads, max_cache_len, head_dim)
# 例如 Block 0: (7, 1, 8, 1024, 128)
```

#### 2. Attention Mask 构造

静态 Attention Mask 需要正确处理 past_len 和 q_len：

```python
def build_static_attention_mask(past_len: int, q_len: int, max_cache_len: int, max_input_len: int):
    NEG_INF = -1e9
    total = max_cache_len + max_input_len
    mask = np.full((max_input_len, total), NEG_INF, dtype=np.float32)
    
    if q_len > 0:
        # 允许看到 past tokens
        if past_len > 0:
            mask[:q_len, :past_len] = 0.0
        # 因果 mask (只能看到当前及之前的 token)
        for row in range(q_len):
            cols_end = max_cache_len + row + 1
            mask[row, max_cache_len:cols_end] = 0.0
    
    return mask.reshape(1, 1, max_input_len, total)
```

#### 3. Position IDs 构造

```python
def build_static_position_ids(past_len: int, q_len: int, max_input_len: int):
    pos = np.zeros((1, max_input_len), dtype=np.int64)
    if q_len > 0:
        pos[0, :q_len] = np.arange(past_len, past_len + q_len, dtype=np.int64)
        # 填充部分使用最后一个有效位置
        if q_len < max_input_len:
            pos[0, q_len:] = pos[0, q_len - 1]
    return pos
```

#### 4. 动态轴配置

导出时需要正确配置动态轴：

```python
dynamic_axes={
    "hidden_states": {0: "B", 1: "T"},
    "attention_mask": {0: "B", 2: "Q", 3: "KV"},
    "position_ids": {0: "B", 1: "T"},
    "past_key": {0: "L", 3: "KV_IN"},
    "past_value": {0: "L", 3: "KV_IN"},
    "hidden_states_out": {0: "B", 1: "T"},
    "present_key": {0: "L", 3: "KV_OUT"},
    "present_value": {0: "L", 3: "KV_OUT"},
}
```

---

## 5. ONNX 转 OM

> **注意**: 当前代码库中未包含 ONNX 转 OM 的具体实现。以下是华为昇腾平台的通用转换流程。

### 转换工具

使用 ATC (Ascend Tensor Compiler) 工具进行转换：

```bash
# 基本转换命令
atc --model=embed.onnx \
    --framework=5 \
    --output=embed \
    --soc_version=Ascend310P3

# Block 转换 (需要指定动态 shape)
atc --model=layers_0_6.onnx \
    --framework=5 \
    --output=layers_0_6 \
    --soc_version=Ascend310P3 \
    --input_shape="hidden_states:1,-1,2048;attention_mask:1,1,-1,-1;position_ids:1,-1;past_key:7,1,8,-1,128;past_value:7,1,8,-1,128" \
    --dynamic_dims="16,16,32,16;1,1,2,1"
```

### 转换注意事项

1. **动态 Shape 处理**: 需要根据实际使用场景配置动态维度
2. **算子兼容性**: 检查 ONNX 算子是否被 ATC 支持
3. **精度模式**: 可选择 FP16/FP32 精度模式

---

## 6. 分块推理运行

### 推理流程

```
prompt.txt → Tokenize → Embedding → Block0 → Block1 → Block2 → Block3 → Output → Next Token
                ↓           ↓          ↓         ↓         ↓          ↓
            tokens.npy  hidden_0   hidden_1  hidden_2  hidden_3   hidden_4 → logits
                            ↓          ↓         ↓         ↓
                         KV Cache  KV Cache  KV Cache  KV Cache
```

### 自动化运行 (ONNX - 本地)

```bash
# Windows 本地运行示例
python file_pipeline\run_pipeline_auto.py \
  --prompt prompt.txt \
  --onnx_dir onnx_models \
  --tokenizer_dir D:\qwen_split\qwen3_1.7b \
  --run_root runs_auto \
  --kv_dir kv_cache \
  --steps 10 \
  --max_cache_len 1024 \
  --max_input_len 16 \
  --temperature 1.0 \
  --top_k 0 \
  --top_p 1.0 \
  --greedy \
  --clean

# Linux 运行示例
python file_pipeline/run_pipeline_auto.py \
  --prompt prompt.txt \
  --onnx_dir onnx_models \
  --tokenizer_dir /path/to/qwen3_1.7b \
  --run_root runs_auto \
  --kv_dir kv_cache \
  --steps 10 \
  --max_cache_len 1024 \
  --max_input_len 16 \
  --greedy \
  --clean
```

**参数说明**:
- `--prompt`: 输入提示词文件
- `--onnx_dir`: ONNX 模型目录
- `--tokenizer_dir`: Tokenizer 目录（与模型路径相同）
- `--run_root`: 运行输出目录
- `--kv_dir`: KV Cache 存储目录
- `--steps`: 生成的 token 数量
- `--max_cache_len`: KV Cache 最大长度
- `--max_input_len`: 单次输入最大长度
- `--greedy`: 使用贪婪解码
- `--clean`: 清空之前的运行结果

### 自动化运行 (OM - 昇腾开发板)

```bash
# 在昇腾开发板上运行 OM 模型
python file_pipeline_without_text/run_pipeline_auto.py \
  --init_tokens init_tokens.txt \
  --om_dir model_om \
  --run_root runs_auto \
  --kv_dir kv_cache \
  --steps 0 \
  --max_cache_len 1024 \
  --max_input_len 16 \
  --temperature 1.0 \
  --top_k 0 \
  --top_p 1.0 \
  --greedy \
  --clean
```

**参数说明**:
- `--init_tokens`: 初始 token 文件（替代 prompt.txt，因为开发板上没有 tokenizer）
- `--om_dir`: OM 模型目录
- `--steps`: 生成步数（0 表示仅 prefill）

### 核心推理代码

```python
# Block 推理核心逻辑 (stage_block_common.py)
def run_block(block_idx: int, args: argparse.Namespace):
    start, end, onnx_name = BLOCK_LAYOUT[block_idx]
    layers = end - start
    
    # 加载 ONNX 模型
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    # 准备输入
    feeds = {
        "hidden_states": hidden,
        "attention_mask": attn,
        "position_ids": pos,
        "past_key": past_key.astype(np.float32, copy=False),
        "past_value": past_value.astype(np.float32, copy=False),
    }
    
    # 推理
    hidden_out, present_key, present_value = sess.run(None, feeds)
    
    # 更新 KV Cache (仅更新有效区间)
    if q_len > 0:
        present_key = present_key.astype(np.float16, copy=False)
        present_value = present_value.astype(np.float16, copy=False)
        start = past_len
        end = start + q_len
        past_key[:, :, :, start:end, :] = present_key[:, :, :, :q_len, :]
        past_value[:, :, :, start:end, :] = present_value[:, :, :, :q_len, :]
```

---

## 7. 位置编码与精度问题

### 位置编码类型

Qwen3 使用 **RoPE (Rotary Position Embedding)** 旋转位置编码。

```python
# 配置参数
"rope_theta": 1000000,
"max_position_embeddings": 40960
```

### RoPE 实现

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将向量分成两半并旋转"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码到 Q 和 K"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### 精度问题

#### 问题描述

在 ONNX 导出和推理过程中，`position_ids` 的数据类型处理存在精度问题：

1. **ONNX 导出时的类型问题**: 某些情况下 `position_ids` 被错误地转换为 `float32`，导致转为 OM 模型后运行的结果出错
2. **RoPE 计算精度**: `cos`/`sin` 计算需要与 `hidden_states` 保持相同精度

#### 解决方案

**1. Position IDs 类型修正**

```python
# 错误方式 (导致精度问题)
# pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)

# 正确方式
pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
```

**2. Embedding 模块类型转换**

```python
class Qwen3EmbeddingModule(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.embed_tokens = base_model.model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 强制转换为 long，因为 embed_tokens 需要 long 类型
        input_ids = input_ids.to(torch.long)
        return self.embed_tokens(input_ids)
```

**3. RoPE 计算精度对齐**

```python
# 在 BlockStack 中确保精度一致
cos, sin = self.rotary_emb(hidden_states, position_ids)
cos = cos.to(hidden_states.dtype)  # 对齐到 hidden_states 的精度
sin = sin.to(hidden_states.dtype)
```

**4. Softmax 精度处理**

```python
# 在 Attention 计算中使用 FP32 进行 softmax
attn_weights = nn.functional.softmax(
    attn_weights, 
    dim=-1, 
    dtype=torch.float32  # 强制使用 FP32 避免数值不稳定
).to(query_states.dtype)
```

**5. 推理时的类型处理**

```python
# stage_block_common.py 中的处理
feeds = {
    "hidden_states": hidden,
    "attention_mask": attn,
    "position_ids": pos,  # 保持 int64 类型
    "past_key": past_key.astype(np.float32, copy=False),
    "past_value": past_value.astype(np.float32, copy=False),
}
```

### 精度验证

使用 `compare_results.py` 对比 PyTorch 原始输出和 ONNX 推理输出：

```bash
python compare_results.py \
  --runA ./runs_onnx \
  --runB ./runs_pytorch \
  --output compare_result.json
```

验证指标：
- **MSE (均方误差)**: 应接近 0
- **Cosine Similarity (余弦相似度)**: 应接近 1.0
- **Max Absolute Error (最大绝对误差)**: 应在可接受范围内

---

## 附录

### 文件结构

```
block/
├── convert_to_onnx.py          # ONNX 导出脚本
├── qwen3_custom_modules.py     # 自定义模型模块
├── config.json                 # 模型配置
├── simple_inference.py         # 简单推理脚本
├── run_full_model.py           # 完整模型推理
├── compare_results.py          # 结果对比工具
├── file_pipeline/
│   ├── stage_token_embed.py    # Token 嵌入阶段
│   ├── stage_block_common.py   # Block 通用逻辑
│   ├── stage_block{0-3}.py     # 各 Block 入口
│   ├── stage_output.py         # 输出采样阶段
│   ├── stage_utils.py          # 工具函数
│   └── run_pipeline_auto.py    # 自动化流水线
└── onnx_models/
    ├── embed.onnx
    ├── layers_0_6.onnx
    ├── layers_7_13.onnx
    ├── layers_14_20.onnx
    ├── layers_21_27.onnx
    ├── output.onnx
    └── config.json
```

### 依赖环境

```
transformers>=4.52.4
onnxruntime
numpy
torch
