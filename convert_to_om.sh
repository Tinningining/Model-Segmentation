#!/bin/bash

# Qwen3-1.7B ONNX 转 OM 脚本（Prefill + Decode 双模型）
# Prefill: seq_len=512, 无 past KV
# Decode:  seq_len=1, 带 past KV

# 配置参数
ONNX_DIR="./onnx_models"
OM_DIR="./model_om"
SOC_VERSION="Ascend310B4"

# 模型参数
PREFILL_LEN=512
DECODE_LEN=1
MAX_CACHE_LEN=1024
HIDDEN_SIZE=2048
NUM_KV_HEADS=8
HEAD_DIM=128
VOCAB_SIZE=151936

# attention_mask 最后一维
PREFILL_MASK_KV=$PREFILL_LEN                          # 512 (无 past)
DECODE_MASK_KV=$(($MAX_CACHE_LEN + $DECODE_LEN))      # 1025 (past + 1)

mkdir -p "$OM_DIR/prefill"
mkdir -p "$OM_DIR/decode"

echo "=========================================="
echo "Qwen3-1.7B Prefill + Decode OM Conversion"
echo "=========================================="
echo "Prefill seq_len: $PREFILL_LEN"
echo "Decode seq_len:  $DECODE_LEN"
echo "Max cache len:   $MAX_CACHE_LEN"
echo "=========================================="

# ==========================================
# Prefill 组（6 个模型）
# ==========================================
echo ""
echo "===== Prefill Models ====="

# Prefill Embedding: input_ids [1, 512]
echo "[Prefill 1/6] Embedding..."
atc --framework=5 \
    --model="$ONNX_DIR/prefill/embed.onnx" \
    --output="$OM_DIR/prefill/embed" \
    --input_format=ND \
    --input_shape="input_ids:1,$PREFILL_LEN" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# Prefill Blocks: 无 past_key/past_value
# hidden_states [1, 512, 2048], attention_mask [1, 1, 512, 512], position_ids [1, 512]
for block in layers_0_6 layers_7_13 layers_14_20 layers_21_27; do
    idx=$((${block##*_} / 7))
    echo "[Prefill $((idx+2))/6] $block..."
    atc --framework=5 \
        --model="$ONNX_DIR/prefill/$block.onnx" \
        --output="$OM_DIR/prefill/$block" \
        --input_format=ND \
        --input_shape="hidden_states:1,$PREFILL_LEN,$HIDDEN_SIZE;attention_mask:1,1,$PREFILL_LEN,$PREFILL_MASK_KV;position_ids:1,$PREFILL_LEN" \
        --log=error \
        --soc_version=$SOC_VERSION \
        --precision_mode=must_keep_origin_dtype
    [ $? -ne 0 ] && echo "✗ Failed" && exit 1
    echo "✓ Done"
done

# Prefill Output: hidden_states [1, 512, 2048]
echo "[Prefill 6/6] Output..."
atc --framework=5 \
    --model="$ONNX_DIR/prefill/output.onnx" \
    --output="$OM_DIR/prefill/output" \
    --input_format=ND \
    --input_shape="hidden_states:1,$PREFILL_LEN,$HIDDEN_SIZE" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# ==========================================
# Decode 组（6 个模型）
# ==========================================
echo ""
echo "===== Decode Models ====="

# Decode Embedding: input_ids [1, 1]
echo "[Decode 1/6] Embedding..."
atc --framework=5 \
    --model="$ONNX_DIR/decode/embed.onnx" \
    --output="$OM_DIR/decode/embed" \
    --input_format=ND \
    --input_shape="input_ids:1,$DECODE_LEN" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# Decode Blocks: 带 past_key/past_value
# hidden_states [1, 1, 2048], attention_mask [1, 1, 1, 1025], position_ids [1, 1]
# past_key [7, 1, 8, 1024, 128], past_value [7, 1, 8, 1024, 128]
for block in layers_0_6 layers_7_13 layers_14_20 layers_21_27; do
    idx=$((${block##*_} / 7))
    echo "[Decode $((idx+2))/6] $block..."
    atc --framework=5 \
        --model="$ONNX_DIR/decode/$block.onnx" \
        --output="$OM_DIR/decode/$block" \
        --input_format=ND \
        --input_shape="hidden_states:1,$DECODE_LEN,$HIDDEN_SIZE;attention_mask:1,1,$DECODE_LEN,$DECODE_MASK_KV;position_ids:1,$DECODE_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
        --log=error \
        --soc_version=$SOC_VERSION \
        --precision_mode=must_keep_origin_dtype
    [ $? -ne 0 ] && echo "✗ Failed" && exit 1
    echo "✓ Done"
done

# Decode Output: hidden_states [1, 1, 2048]
echo "[Decode 6/6] Output..."
atc --framework=5 \
    --model="$ONNX_DIR/decode/output.onnx" \
    --output="$OM_DIR/decode/output" \
    --input_format=ND \
    --input_shape="hidden_states:1,$DECODE_LEN,$HIDDEN_SIZE" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# ==========================================
# 完成
# ==========================================
echo ""
echo "=========================================="
echo "All conversions completed!"
echo "=========================================="
echo ""
echo "Prefill models ($OM_DIR/prefill/):"
echo "  embed.om, layers_0_6.om, layers_7_13.om,"
echo "  layers_14_20.om, layers_21_27.om, output.om"
echo "  Shapes: seq_len=$PREFILL_LEN, no past KV"
echo ""
echo "Decode models ($OM_DIR/decode/):"
echo "  embed.om, layers_0_6.om, layers_7_13.om,"
echo "  layers_14_20.om, layers_21_27.om, output.om"
echo "  Shapes: seq_len=$DECODE_LEN, past KV cache=$MAX_CACHE_LEN"
echo "=========================================="
