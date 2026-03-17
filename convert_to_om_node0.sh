#!/bin/bash
# Node 0: embed.om + layers_0_6.om (System + Prefill + Decode)

ONNX_DIR="./onnx_models"
OM_DIR="./model_om"
SOC_VERSION="Ascend310B4"

SYSTEM_LEN=256
PREFILL_LEN=512
DECODE_LEN=1
MAX_CACHE_LEN=1024
HIDDEN_SIZE=2048
NUM_KV_HEADS=8
HEAD_DIM=128

SYSTEM_MASK_KV=$SYSTEM_LEN
PREFILL_MASK_KV=$(($MAX_CACHE_LEN + $PREFILL_LEN))
DECODE_MASK_KV=$(($MAX_CACHE_LEN + $DECODE_LEN))

mkdir -p "$OM_DIR/system" "$OM_DIR/prefill" "$OM_DIR/decode"

echo "===== Node 0: embed + layers_0_6 ====="

# --- System (无 past KV) ---
echo "[System] Embedding..."
atc --framework=5 \
    --model="$ONNX_DIR/system/embed.onnx" \
    --output="$OM_DIR/system/embed" \
    --input_format=ND \
    --input_shape="input_ids:1,$SYSTEM_LEN" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

echo "[System] layers_0_6..."
atc --framework=5 \
    --model="$ONNX_DIR/system/layers_0_6.onnx" \
    --output="$OM_DIR/system/layers_0_6" \
    --input_format=ND \
    --input_shape="hidden_states:1,$SYSTEM_LEN,$HIDDEN_SIZE;attention_mask:1,1,$SYSTEM_LEN,$SYSTEM_MASK_KV;position_ids:1,$SYSTEM_LEN" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# --- Prefill (带 past KV) ---
echo "[Prefill] Embedding..."
atc --framework=5 \
    --model="$ONNX_DIR/prefill/embed.onnx" \
    --output="$OM_DIR/prefill/embed" \
    --input_format=ND \
    --input_shape="input_ids:1,$PREFILL_LEN" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

echo "[Prefill] layers_0_6..."
atc --framework=5 \
    --model="$ONNX_DIR/prefill/layers_0_6.onnx" \
    --output="$OM_DIR/prefill/layers_0_6" \
    --input_format=ND \
    --input_shape="hidden_states:1,$PREFILL_LEN,$HIDDEN_SIZE;attention_mask:1,1,$PREFILL_LEN,$PREFILL_MASK_KV;position_ids:1,$PREFILL_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# --- Decode (带 past KV) ---
echo "[Decode] Embedding..."
atc --framework=5 \
    --model="$ONNX_DIR/decode/embed.onnx" \
    --output="$OM_DIR/decode/embed" \
    --input_format=ND \
    --input_shape="input_ids:1,$DECODE_LEN" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

echo "[Decode] layers_0_6..."
atc --framework=5 \
    --model="$ONNX_DIR/decode/layers_0_6.onnx" \
    --output="$OM_DIR/decode/layers_0_6" \
    --input_format=ND \
    --input_shape="hidden_states:1,$DECODE_LEN,$HIDDEN_SIZE;attention_mask:1,1,$DECODE_LEN,$DECODE_MASK_KV;position_ids:1,$DECODE_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

echo "===== Node 0 Done ====="
