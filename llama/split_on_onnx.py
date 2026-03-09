import onnx
from onnx.utils import extract_model
from onnx import shape_inference
import os
import argparse

# =================================================
# 安全的 extract：补 shape 再保存
# =================================================
def safe_extract(input_path, output_path, input_names, output_names):
    # 1. 提取子图
    # check_model=False: 避免因中间节点转输入时暂时缺少 shape 而报错
    extract_model(
        input_path=input_path,
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
        check_model=False,
    )

    # 2. 补全 shape（关键）
    model = onnx.load(output_path)
    model = shape_inference.infer_shapes(model)

    # 3. 保存
    onnx.save(model, output_path)


# =================================================
# 命令行参数解析
# =================================================
parser = argparse.ArgumentParser(description="Split a large ONNX model into smaller parts.")
parser.add_argument('--src', type=str, required=True, help='Path to the source ONNX model to split')
parser.add_argument('--out_dir', type=str, required=True, help='Directory to save the split ONNX models')
args = parser.parse_args()

SRC_ONNX = args.src
OUT_DIR = args.out_dir

os.makedirs(OUT_DIR, exist_ok=True)

# =================================================
# 0. 预处理：对源模型进行 Shape Inference
#    (解决 extract_model 时因缺少 shape 报错的问题)
# =================================================
print(f"Running shape inference on source model: {SRC_ONNX}...")
inferred_model_path = os.path.join(OUT_DIR, "inferred_source.onnx")

# 使用 infer_shapes_path 处理大模型，避免内存溢出
# 如果模型非常大 (>2GB)，需确保 check_model=False 以跳过部分检查
from onnx import shape_inference
shape_inference.infer_shapes_path(SRC_ONNX, inferred_model_path)
SRC_ONNX = inferred_model_path
print(f"✓ Shape inference complete. Using: {SRC_ONNX}")

# =================================================
# 公共输入（除了 M0 以外，全部要带）
# =================================================
# 注意：现在的 past_key_values 已经不是一个大的 COMMON_INPUTS 了
# 而是每个 Block 只取自己需要的 past_key_values_i
# 所以这里移除 COMMON_INPUTS 中的 "past_key_values"
COMMON_INPUTS = [
    "input_ids",         # 某些层可能需要 input_ids (如 RoPE 或 shape 推导)
    "attention_mask",
    "position_ids",
    "past_key_values_0", # ！！！必须添加：很多层的 RoPE/Shape 推导依赖第0层的 KV 形作为基准！！！
]

# =================================================
# 切点定义 (TinyLlama-1.1B 为 22 层: 0-21)
# =================================================
# M0: Embed + Layers 0-4 (共5层: 0,1,2,3,4) -> 输出为 Layer.4 的输出
E0_out = "/model/layers.4/Add_1_output_0"

# M1: Layers 5-10 (共6层: 5,6,7,8,9,10) -> 输出为 Layer.10 的输出
E1_out = "/model/layers.10/Add_1_output_0"

# M2: Layers 11-16 (共6层: 11,12,13,14,15,16) -> 输出为 Layer.16 的输出
E2_out = "/model/layers.16/Add_1_output_0"

# M3: Layers 17-21 + Norm + LMHead (共5层: 17,18,19,20,21) -> 输出为 FINAL
E3_mid = "/model/layers.21/Add_1_output_0" # 用于 M3 内部连接到 Norm，但我们直接切最后一层输出即可
FINAL = "logits"

# Helper: 生成特定层范围的 kv input/output names
def get_kv_names(start_layer, end_layer_inclusive):
    pasts = [f"past_key_values_{i}" for i in range(start_layer, end_layer_inclusive + 1)]
    presents = [f"present_key_values_{i}" for i in range(start_layer, end_layer_inclusive + 1)]
    return pasts, presents

# =================================================
# M0: Embed + Layers 0-4
# =================================================
# Inputs: input_ids/mask/pos + past_kv_0..4
# Outputs: E0_out + present_kv_0..4
p0, o0 = get_kv_names(0, 4)
safe_extract(
    input_path=SRC_ONNX,
    output_path=os.path.join(OUT_DIR, "llama_m0_embed_layers_0_4.onnx"),
    input_names=[
        "input_ids",
        "attention_mask", # Embedding 本身可能不需要，但后续 Layer 需要
        "position_ids",
        *p0
    ],
    output_names=[
        E0_out,
        *o0
    ],
)
print("✓ M0 (Embed + Layers 0-4) extracted")

# =================================================
# M1: Layers 5-10
# =================================================
# Inputs: E0_out + input_ids/mask/pos + past_kv_5..10
# Outputs: E1_out + present_kv_5..10
p1, o1 = get_kv_names(5, 10)
safe_extract(
    input_path=SRC_ONNX,
    output_path=os.path.join(OUT_DIR, "llama_m1_layers_5_10.onnx"),
    input_names=[
        E0_out,
        *COMMON_INPUTS,
        *p1
    ],
    output_names=[
        E1_out,
        *o1
    ],
)
print("✓ M1 (Layers 5-10) extracted")

# =================================================
# M2: Layers 11-16
# =================================================
# Inputs: E1_out + input_ids/mask/pos + past_kv_11..16
# Outputs: E2_out + present_kv_11..16
p2, o2 = get_kv_names(11, 16)
safe_extract(
    input_path=SRC_ONNX,
    output_path=os.path.join(OUT_DIR, "llama_m2_layers_11_16.onnx"),
    input_names=[
        E1_out,
        *COMMON_INPUTS,
        *p2
    ],
    output_names=[
        E2_out,
        *o2
    ],
)
print("✓ M2 (Layers 11-16) extracted")

# =================================================
# M3: Layers 17-21 + Norm + LMHead
# =================================================
# Inputs: E2_out + input_ids/mask/pos + past_kv_17..21
# Outputs: FINAL (logits) + present_kv_17..21
p3, o3 = get_kv_names(17, 21)
safe_extract(
    input_path=SRC_ONNX,
    output_path=os.path.join(OUT_DIR, "llama_m3_layers_17_21_lmhead.onnx"),
    input_names=[
        E2_out,
        *COMMON_INPUTS,
        *p3
    ],
    output_names=[
        FINAL,
        *o3
    ],
)
print("✓ M3 (Layers 17-21 + Norm + LMHead) extracted")

print("\nAll ONNX models have been successfully split.")
