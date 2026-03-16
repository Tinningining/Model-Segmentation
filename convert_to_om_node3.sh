#!/bin/bash
# Node 3: layers_21_27.om + output.om (Prefill + Decode)

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

echo "===== Node 3: layers_21_27 + output ====="

# --- Prefill ---
echo "[Prefill] layers_21_27..."
atc --framework=5 \
    --model="$ONNX_DIR/prefill/layers_21_27.onnx" \
    --output="$OM_DIR/prefill/layers_21_27" \
    --input_format=ND \
    --input_shape="hidden_states:1,$PREFILL_LEN,$HIDDEN_SIZE;attention_mask:1,1,$PREFILL_LEN,$PREFILL_MASK_KV;position_ids:1,$PREFILL_LEN" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

echo "[Prefill] output..."
atc --framework=5 \
    --model="$ONNX_DIR/prefill/output.onnx" \
    --output="$OM_DIR/prefill/output" \
    --input_format=ND \
    --input_shape="hidden_states:1,$PREFILL_LEN,$HIDDEN_SIZE" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

# --- Decode ---
echo "[Decode] layers_21_27..."
atc --framework=5 \
    --model="$ONNX_DIR/decode/layers_21_27.onnx" \
    --output="$OM_DIR/decode/layers_21_27" \
    --input_format=ND \
    --input_shape="hidden_states:1,$DECODE_LEN,$HIDDEN_SIZE;attention_mask:1,1,$DECODE_LEN,$DECODE_MASK_KV;position_ids:1,$DECODE_LEN;past_key:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM;past_value:7,1,$NUM_KV_HEADS,$MAX_CACHE_LEN,$HEAD_DIM" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

echo "[Decode] output..."
atc --framework=5 \
    --model="$ONNX_DIR/decode/output.onnx" \
    --output="$OM_DIR/decode/output" \
    --input_format=ND \
    --input_shape="hidden_states:1,$DECODE_LEN,$HIDDEN_SIZE" \
    --log=error --soc_version=$SOC_VERSION \
    --precision_mode=must_keep_origin_dtype
[ $? -ne 0 ] && echo "✗ Failed" && exit 1
echo "✓ Done"

echo "===== Node 3 Done ====="
