#!/bin/bash

# Qwen3-1.7B ONNX 转 OM 脚本（System + Prefill + Decode 三组模型）
# System:  seq_len=SYSTEM_LEN, 无 past KV（预计算固定 system prompt）
# Prefill: seq_len=PREFILL_LEN, 带 past KV（处理用户输入，past KV 来自 system）
# Decode:  seq_len=1, 带 past KV

# 配置参数
ONNX_DIR="./onnx_models"
OM_DIR="./model_om"
SOC_VERSION="Ascend310B4"

# 模型参数
SYSTEM_LEN=256
PREFILL_LEN=512
DECODE_LEN=1
MAX_CACHE_LEN=1024
HIDDEN_SIZE=2048
NUM_KV_HEADS=8
HEAD_DIM=128
VOCAB_SIZE=151936

# attention_mask 最后一维
SYSTEM_MASK_KV=$SYSTEM_LEN                                    # 256 (无 past)
PREFILL_MASK_KV=$(($MAX_CACHE_LEN + $PREFILL_LEN))            # 1536 (past + prefill)
DECODE_MASK_KV=$(($MAX_CACHE_LEN + $DECODE_LEN))              # 1025 (past + 1)

mkdir -p "$OM_DIR/system"
mkdir -p "$OM_DIR/prefill"
mkdir -p "$OM_DIR/decode"

echo "=========================================="
echo "Qwen3-1.7B System+Prefill+Decode OM Conv"
echo "=========================================="
echo "System seq_len:  $SYSTEM_LEN (no past KV)"
echo "Prefill seq_len: $PREFILL_LEN (with past KV)"
echo "Decode seq_len:  $DECODE_LEN (with past KV)"
echo "Max cache len:   $MAX_CACHE_LEN"
echo "=========================================="

# ==========================================
# System 组（6 个模型，无 past KV）
# ==========================================
echo ""
echo "===== System Models ====="

# System Embedding: input_ids [1, SYSTEM_LEN]
echo "[System 1/6] Embedding..."
atc --framework=5 \
    --model="$ONNX_DIR/system/embed.onnx" \
    --output="$OM_DIR/system/embed" \
    --input_format=ND \
    --input_shape="input_ids:1,$SYSTEM_LEN" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# System Blocks: 无 past_key/past_value
# hidden_states [1, SYSTEM_LEN, 2048], attention_mask [1, 1, SYSTEM_LEN, SYSTEM_LEN], position_ids [1, SYSTEM_LEN]
for block in layers_0_6 layers_7_13 layers_14_20 layers_21_27; do
    idx=$((${block##*_} / 7))
    echo "[System $((idx+2))/6] $block..."
    atc --framework=5 \
        --model="$ONNX_DIR/system/$block.onnx" \
        --output="$OM_DIR/system/$block" \
        --input_format=ND \
        --input_shape="hidden_states:1,$SYSTEM_LEN,$HIDDEN_SIZE;attention_mask:1,1,$SYSTEM_LEN,$SYSTEM_MASK_KV;position_ids:1,$SYSTEM_LEN" \
        --log=error \
        --soc_version=$SOC_VERSION \
        --precision_mode=must_keep_origin_dtype
    [ $? -ne 0 ] && echo "✗ Failed" && exit 1
    echo "✓ Done"
done

# System Output: hidden_states [1, SYSTEM_LEN, 2048]
echo "[System 6/6] Output..."
atc --framework=5 \
    --model="$ONNX_DIR/system/output.onnx" \
    --output="$OM_DIR/system/output" \
    --input_format=ND \
    --input_shape="hidden_states:1,$SYSTEM_LEN,$HIDDEN_SIZE" \
    --log=error \
    --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# ==========================================
# Prefill 组（6 个模型，带 past KV）
# ==========================================
echo ""
echo "===== Prefill Models ====="

# Prefill Embedding: input_ids [1, PREFILL_LEN]
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

# Prefill Blocks: 带 past_key/past_value（来自 system 阶段）
# hidden_states [1, PREFILL_LEN, 2048], attention_mask [1, 1, PREFILL_LEN, MAX_CACHE_LEN+PREFILL_LEN]
# position_ids [1, PREFILL_LEN], past_key [7, 1, 8, MAX_CACHE_LEN, 128], past_value [7, 1, 8, MAX_CACHE_LEN, 128]
for block in layers_0_6 layers_7_13 layers_14_20 layers_21_27; do
    idx=$((${block##*_} / 7))
    echo "[Prefill $((idx+2))/6] $block..."
    atc --framework=5 \
        --model="$ONNX_DIR/prefill/$block.onnx" \
        --output="$OM_DIR/prefill/$block" \
        --input_format=ND \
        --input_shape="hidden_states:1,$PREFILL_LEN,$HIDDEN_SIZE;attention_mask:1,1,$PREFILL_LEN,$PREFILL_MASK_KV;position_ids:1,$PREFILL_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
        --log=error \
        --soc_version=$SOC_VERSION \
        --precision_mode=must_keep_origin_dtype
    [ $? -ne 0 ] && echo "✗ Failed" && exit 1
    echo "✓ Done"
done

# Prefill Output: hidden_states [1, PREFILL_LEN, 2048]
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
# Decode 组（6 个模型，带 past KV）
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
echo "System models ($OM_DIR/system/):"
echo "  embed.om, layers_0_6.om ~ layers_21_27.om, output.om"
echo "  Shapes: seq_len=$SYSTEM_LEN, no past KV"
echo ""
echo "Prefill models ($OM_DIR/prefill/):"
echo "  embed.om, layers_0_6.om ~ layers_21_27.om, output.om"
echo "  Shapes: seq_len=$PREFILL_LEN, past KV cache=$MAX_CACHE_LEN"
echo ""
echo "Decode models ($OM_DIR/decode/):"
echo "  embed.om, layers_0_6.om ~ layers_21_27.om, output.om"
echo "  Shapes: seq_len=$DECODE_LEN, past KV cache=$MAX_CACHE_LEN"
echo "=========================================="
