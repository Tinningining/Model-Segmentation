#!/bin/bash
# Node 2: layers_14_20.om (Prefill + Decode)

ONNX_DIR="./onnx_models"
OM_DIR="./model_om"
SOC_VERSION="Ascend310B4"

PREFILL_LEN=512
DECODE_LEN=1
MAX_CACHE_LEN=1024
HIDDEN_SIZE=2048
NUM_KV_HEADS=8
HEAD_DIM=128

PREFILL_MASK_KV=$PREFILL_LEN
DECODE_MASK_KV=$(($MAX_CACHE_LEN + $DECODE_LEN))

mkdir -p "$OM_DIR/prefill" "$OM_DIR/decode"

echo "===== Node 2: layers_14_20 ====="

# --- Prefill ---
echo "[Prefill] layers_14_20..."
atc --framework=5 \
    --model="$ONNX_DIR/prefill/layers_14_20.onnx" \
    --output="$OM_DIR/prefill/layers_14_20" \
    --input_format=ND \
    --input_shape="hidden_states:1,$PREFILL_LEN,$HIDDEN_SIZE;attention_mask:1,1,$PREFILL_LEN,$PREFILL_MASK_KV;position_ids:1,$PREFILL_LEN" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# --- Decode ---
echo "[Decode] layers_14_20..."
atc --framework=5 \
    --model="$ONNX_DIR/decode/layers_14_20.onnx" \
    --output="$OM_DIR/decode/layers_14_20" \
    --input_format=ND \
    --input_shape="hidden_states:1,$DECODE_LEN,$HIDDEN_SIZE;attention_mask:1,1,$DECODE_LEN,$DECODE_MASK_KV;position_ids:1,$DECODE_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

echo "===== Node 2 Done ====="
