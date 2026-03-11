#!/bin/bash

# Qwen3-1.7B ONNX 转 OM 脚本
# 模型结构：28 层，切分为 5 个部分
# - embed.onnx: Token embedding
# - layers_0_6.onnx: Layers 0-6 (7 layers)
# - layers_7_13.onnx: Layers 7-13 (7 layers)
# - layers_14_20.onnx: Layers 14-20 (7 layers)
# - layers_21_27.onnx: Layers 21-27 (7 layers)
# - output.onnx: RMSNorm + LM Head

# 配置参数
ONNX_DIR="./onnx_models"
OM_DIR="./model_om"
SOC_VERSION="Ascend310B4"  # 根据实际硬件修改
SEQ_LEN=512                 # 序列长度
MAX_CACHE_LEN=1024          # KV Cache 最大长度

# Qwen3-1.7B 模型参数
HIDDEN_SIZE=2048
NUM_KV_HEADS=8
HEAD_DIM=128
VOCAB_SIZE=151936

# 确保输出目录存在
mkdir -p "$OM_DIR"

echo "=========================================="
echo "Qwen3-1.7B ONNX to OM Conversion"
echo "=========================================="
echo "ONNX Directory: $ONNX_DIR"
echo "OM Directory: $OM_DIR"
echo "SOC Version: $SOC_VERSION"
echo "Sequence Length: $SEQ_LEN"
echo "Max Cache Length: $MAX_CACHE_LEN"
echo "=========================================="

# ==========================================
# 1. Embedding Module
# Input: input_ids [B, T]
# Output: hidden_states [B, T, H]
# ==========================================
echo ""
echo "[1/6] Converting Embedding Module..."
atc --framework=5 \
    --model="$ONNX_DIR/embed.onnx" \
    --output="$OM_DIR/embed" \
    --input_format=ND \
    --input_shape="input_ids:1,$SEQ_LEN" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype

if [ $? -eq 0 ]; then
    echo "✓ Embedding module converted successfully"
else
    echo "✗ Failed to convert embedding module"
    exit 1
fi

# ==========================================
# 2. Layers 0-6 (7 layers)
# Inputs:
#   - hidden_states: [B, T, H]
#   - attention_mask: [B, 1, T, KV]
#   - position_ids: [B, T]
#   - past_key: [7, B, num_kv_heads, KV, head_dim]
#   - past_value: [7, B, num_kv_heads, KV, head_dim]
# Outputs:
#   - hidden_states_out: [B, T, H]
#   - present_key: [7, B, num_kv_heads, T, head_dim]
#   - present_value: [7, B, num_kv_heads, T, head_dim]
# ==========================================
echo ""
echo "[2/6] Converting Layers 0-6..."
atc --framework=5 \
    --model="$ONNX_DIR/layers_0_6.onnx" \
    --output="$OM_DIR/layers_0_6" \
    --input_format=ND \
    --input_shape="hidden_states:1,$SEQ_LEN,$HIDDEN_SIZE;attention_mask:1,1,$SEQ_LEN,$MAX_CACHE_LEN;position_ids:1,$SEQ_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype

if [ $? -eq 0 ]; then
    echo "✓ Layers 0-6 converted successfully"
else
    echo "✗ Failed to convert layers 0-6"
    exit 1
fi

# ==========================================
# 3. Layers 7-13 (7 layers)
# ==========================================
echo ""
echo "[3/6] Converting Layers 7-13..."
atc --framework=5 \
    --model="$ONNX_DIR/layers_7_13.onnx" \
    --output="$OM_DIR/layers_7_13" \
    --input_format=ND \
    --input_shape="hidden_states:1,$SEQ_LEN,$HIDDEN_SIZE;attention_mask:1,1,$SEQ_LEN,$MAX_CACHE_LEN;position_ids:1,$SEQ_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype

if [ $? -eq 0 ]; then
    echo "✓ Layers 7-13 converted successfully"
else
    echo "✗ Failed to convert layers 7-13"
    exit 1
fi

# ==========================================
# 4. Layers 14-20 (7 layers)
# ==========================================
echo ""
echo "[4/6] Converting Layers 14-20..."
atc --framework=5 \
    --model="$ONNX_DIR/layers_14_20.onnx" \
    --output="$OM_DIR/layers_14_20" \
    --input_format=ND \
    --input_shape="hidden_states:1,$SEQ_LEN,$HIDDEN_SIZE;attention_mask:1,1,$SEQ_LEN,$MAX_CACHE_LEN;position_ids:1,$SEQ_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype

if [ $? -eq 0 ]; then
    echo "✓ Layers 14-20 converted successfully"
else
    echo "✗ Failed to convert layers 14-20"
    exit 1
fi

# ==========================================
# 5. Layers 21-27 (7 layers)
# ==========================================
echo ""
echo "[5/6] Converting Layers 21-27..."
atc --framework=5 \
    --model="$ONNX_DIR/layers_21_27.onnx" \
    --output="$OM_DIR/layers_21_27" \
    --input_format=ND \
    --input_shape="hidden_states:1,$SEQ_LEN,$HIDDEN_SIZE;attention_mask:1,1,$SEQ_LEN,$MAX_CACHE_LEN;position_ids:1,$SEQ_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype

if [ $? -eq 0 ]; then
    echo "✓ Layers 21-27 converted successfully"
else
    echo "✗ Failed to convert layers 21-27"
    exit 1
fi

# ==========================================
# 6. Output Module (RMSNorm + LM Head)
# Input: hidden_states [B, T, H]
# Output: logits [B, T, V]
# ==========================================
echo ""
echo "[6/6] Converting Output Module..."
atc --framework=5 \
    --model="$ONNX_DIR/output.onnx" \
    --output="$OM_DIR/output" \
    --input_format=ND \
    --input_shape="hidden_states:1,$SEQ_LEN,$HIDDEN_SIZE" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype

if [ $? -eq 0 ]; then
    echo "✓ Output module converted successfully"
else
    echo "✗ Failed to convert output module"
    exit 1
fi

# ==========================================
# 完成
# ==========================================
echo ""
echo "=========================================="
echo "All conversions completed successfully!"
echo "=========================================="
echo "OM models saved to: $OM_DIR/"
echo ""
echo "Generated files:"
echo "  - embed.om"
echo "  - layers_0_6.om"
echo "  - layers_7_13.om"
echo "  - layers_14_20.om"
echo "  - layers_21_27.om"
echo "  - output.om"
echo ""
echo "Model specifications:"
echo "  - Total layers: 28 (4 blocks × 7 layers)"
echo "  - Hidden size: $HIDDEN_SIZE"
echo "  - KV heads: $NUM_KV_HEADS"
echo "  - Head dim: $HEAD_DIM"
echo "  - Vocab size: $VOCAB_SIZE"
echo "  - Sequence length: $SEQ_LEN"
echo "  - Max cache length: $MAX_CACHE_LEN"
echo ""
echo "Next steps:"
echo "  1. Verify OM files in $OM_DIR/"
echo "  2. Run distributed inference with:"
echo "     cd distributed_inference"
echo "     ./run_distributed.sh"
echo "=========================================="
