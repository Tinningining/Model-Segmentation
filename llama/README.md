# Split Model Project

本项目用于 TinyLlama 模型的 ONNX 导出、切分以及基于 ONNX/ACL 的分布式流水线推理。

## 目录结构

- `export_llama.py`: 模型导出脚本，支持量化配置。
- `split_on_onnx.py`: ONNX 模型切分脚本。
- `inference_net/`: 推理相关代码，包含单机和分布式节点逻辑。
- `config/`: 量化配置文件 (如 `w8x8.py`, `w8.py` 等)。

## 使用指南

### 0. 环境配置 (重要)

在执行任何脚本之前，**必须**使用本项目提供的 `modeling_llama_4.35.py` 替换 `transformers` 库中的 `modeling_llama.py` 文件。这是为了修改模型的前向传播逻辑以支持导出。

```bash
# 1. 找到你的 transformers 库安装路径
# 例如: /home/yjr/miniconda3/envs/llama_export/lib/python3.9/site-packages/transformers
TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")

# 2. 备份原文件 (可选)
cp $TRANSFORMERS_PATH/models/llama/modeling_llama.py $TRANSFORMERS_PATH/models/llama/modeling_llama.py.bak

# 3. 替换文件
cp modeling_llama_4.35.py $TRANSFORMERS_PATH/models/llama/modeling_llama.py
```

### 1. 导出 ONNX 模型 (Export)

将 PyTorch 模型导出为 ONNX 格式，并应用指定的量化配置。

```bash
python export_llama.py \
    --model ../ascend-llm/export_llama/model/TinyLlama-1.1B-Chat-v1.0/ \
    --output ./tiny-llama.onnx \
    --act-path ./tiny-llama.pt \
    --quant ./config/w8x8.py
```

### 2. 切分 ONNX 模型 (Split)

将导出的 ONNX 模型切分为多个部分（如 M0, M1, M2...），以便进行流水线并行推理。

```bash
python split_on_onnx.py \
    --src ./tiny-llama.onnx \
    --out_dir ./split_models/
```

### 3. 模型转换 (Convert to OM)

将切分后的 ONNX 模型转换为 Ascend 的 OM 离线模型。请确保已安装 Ascend CANN 工具包并设置好环境变量，且进入存放 ONNX 模型的目录（如 `split_models`）。

```bash
# 假设你在 split_models 目录下，或者将 convert_to_om.sh 复制到该目录运行
# 记得修改脚本中的 --soc_version 为你的实际芯片型号 (如 Ascend310B1)
bash ../convert_to_om.sh
```

### 4. 推理运行 (Inference)

推理脚本位于 `inference_net` 目录下，请先进入该目录：

```bash
cd inference_net
```

#### 4.1 单机 ONNX 推理 (ONNX Engine)

验证切分后的 ONNX 模型是否能正常加载和推理。

```bash
python main.py \
    --hf-dir ../../ascend-llm/export_llama/model/TinyLlama-1.1B-Chat-v1.0/ \
    --engine onnx \
    --kv_size 1024 \
    --split-model-dir ../split_models/
```

#### 4.2 单机 ACL 推理 (Ascend Engine)

如果是已转换为 OM 格式的模型，使用 ACL 引擎运行（需在 Ascend 环境下）。

```bash
python main.py \
    --hf-dir ../export_llama/model/TinyLlama-1.1B-Chat-v1.0/ \
    --engine acl \
    --kv_size 1024 \
    --split-model-dir ../export_llama/model/export_out/om_model/
```

#### 4.3 分布式/流水线推理 (Net Mode)

模拟多节点流水线推理，需按顺序启动各个节点。

> **注意**: 请确保 `../export_llama/model/export_out/om_model/` 等路径存在且包含对应的 OM 模型文件。

**Step 1: 启动中间节点 Node 2 (Layers 5-10)**
```bash
python run_intermediate.py \
    --model ../export_llama/model/export_out/om_model/llama_m1_layers_5_10.om \
    --port 8001 \
    --next_ip 127.0.0.1 \
    --next_port 8002 \
    --hf_dir ../export_llama/model/TinyLlama-1.1B-Chat-v1.0/ \
    --n_layer 6 \
    --node_name "Node 2" \
    --device 0 \
    --kv_size 1024
```

**Step 2: 启动中间节点 Node 3 (Layers 11-16)**
```bash
python run_intermediate.py \
    --model ../export_llama/model/export_out/om_model/llama_m2_layers_11_16.om \
    --port 8002 \
    --next_ip 127.0.0.1 \
    --next_port 8003 \
    --hf_dir ../export_llama/model/TinyLlama-1.1B-Chat-v1.0/ \
    --n_layer 6 \
    --node_name "Node 3" \
    --device 0 \
    --kv_size 1024
```

**Step 3: 启动尾节点 Node 4 (Layers 17-21 & Head)**
```bash
python run_final.py \
    --model ../export_llama/model/export_out/om_model/llama_m3_layers_17_21_lmhead.om \
    --port 8003 \
    --head_ip 127.0.0.1 \
    --head_port 8004 \
    --hf_dir ../export_llama/model/TinyLlama-1.1B-Chat-v1.0/ \
    --n_layer 5 \
    --node_name "Node 4" \
    --device 0 \
    --kv_size 1024
```

**Step 4: 启动主节点/Client (Embedding & Layers 0-4)**
```bash
python main.py \
    --engine net \
    --split-model-dir ../export_llama/model/export_out/om_model \
    --hf-dir ../export_llama/model/TinyLlama-1.1B-Chat-v1.0/ \
    --next-ip 127.0.0.1 \
    --next-port 8001 \
    --listen-port 8004 \
    --kv_size 1024
```
