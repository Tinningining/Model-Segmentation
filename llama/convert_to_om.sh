#!/bin/bash

# 确保输出目录存在
mkdir -p om_model

# ==========================================
# M0: Embed + Layers 0-4
# Inputs: input_ids, position_ids, attention_mask, KV_0-4
# ==========================================
echo "Converting M0..."
atc --framework=5 --model="./llama_m0_embed_layers_0_4.onnx" \
    --output="./om_model/llama_m0_embed_layers_0_4" \
    --input_format=ND \
    --input_shape="input_ids:1,1;attention_mask:1,1025;position_ids:1,1;past_key_values_0:2,1,4,1024,64;past_key_values_1:2,1,4,1024,64;past_key_values_2:2,1,4,1024,64;past_key_values_3:2,1,4,1024,64;past_key_values_4:2,1,4,1024,64" \
    --log=error --soc_version=Ascend310B1 --precision_mode=must_keep_origin_dtype

# ==========================================
# M1: Layers 5-10
# Inputs: Hidden(from M0), input_ids, mask, pos, KV_0 (RoPE dependency), KV_5-10
# ==========================================
echo "Converting M1..."
atc --framework=5 --model="./llama_m1_layers_5_10.onnx" \
    --output="./om_model/llama_m1_layers_5_10" \
    --input_format=ND \
    --input_shape="/model/layers.4/Add_1_output_0:1,1,2048;input_ids:1,1;attention_mask:1,1025;position_ids:1,1;past_key_values_0:2,1,4,1024,64;past_key_values_5:2,1,4,1024,64;past_key_values_6:2,1,4,1024,64;past_key_values_7:2,1,4,1024,64;past_key_values_8:2,1,4,1024,64;past_key_values_9:2,1,4,1024,64;past_key_values_10:2,1,4,1024,64" \
    --log=error --soc_version=Ascend310B1 --precision_mode=must_keep_origin_dtype

# ==========================================
# M2: Layers 11-16
# Inputs: Hidden(from M1), input_ids, mask, pos, KV_0, KV_11-16
# ==========================================
echo "Converting M2..."
atc --framework=5 --model="./llama_m2_layers_11_16.onnx" \
    --output="./om_model/llama_m2_layers_11_16" \
    --input_format=ND \
    --input_shape="/model/layers.10/Add_1_output_0:1,1,2048;input_ids:1,1;attention_mask:1,1025;position_ids:1,1;past_key_values_0:2,1,4,1024,64;past_key_values_11:2,1,4,1024,64;past_key_values_12:2,1,4,1024,64;past_key_values_13:2,1,4,1024,64;past_key_values_14:2,1,4,1024,64;past_key_values_15:2,1,4,1024,64;past_key_values_16:2,1,4,1024,64" \
    --log=error --soc_version=Ascend310B1 --precision_mode=must_keep_origin_dtype

# ==========================================
# M3: Layers 17-21 + Head
# Inputs: Hidden(from M2), input_ids, mask, pos, KV_0, KV_17-21
# ==========================================
echo "Converting M3..."
atc --framework=5 --model="./llama_m3_layers_17_21_lmhead.onnx" \
    --output="./om_model/llama_m3_layers_17_21_lmhead" \
    --input_format=ND \
    --input_shape="/model/layers.16/Add_1_output_0:1,1,2048;input_ids:1,1;attention_mask:1,1025;position_ids:1,1;past_key_values_0:2,1,4,1024,64;past_key_values_17:2,1,4,1024,64;past_key_values_18:2,1,4,1024,64;past_key_values_19:2,1,4,1024,64;past_key_values_20:2,1,4,1024,64;past_key_values_21:2,1,4,1024,64" \
    --log=error --soc_version=Ascend310B1 --precision_mode=must_keep_origin_dtype

echo "All conversions finished."
