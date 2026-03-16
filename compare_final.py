# import sys
# import numpy as np

# sys.stdout.reconfigure(encoding='utf-8')

# a_raw = np.load('test_om_pre.npy')
# b = np.load('test_onnx.npy')

# print(f"om_pre raw: shape={a_raw.shape}, dtype={a_raw.dtype}")
# print(f"onnx:       shape={b.shape}, dtype={b.dtype}")

# # OM raw output: float32 values 0-255 = raw bytes from ACL
# # Correct interpretation: convert to uint8 bytes, then view as float32
# raw_bytes = a_raw.astype(np.uint8).tobytes()
# a_f32 = np.frombuffer(raw_bytes, dtype=np.float32)
# print(f"\nOM as float32: {a_f32.shape} elements, min={a_f32.min():.4f}, max={a_f32.max():.4f}")

# # ONNX is (7, 1, 8, 512, 128) float16
# target_shape = b.shape
# target_elements = int(np.prod(target_shape))
# print(f"ONNX target: {target_shape} = {target_elements} elements")
# print(f"OM float32 elements: {a_f32.shape[0]}")

# if a_f32.shape[0] == target_elements:
#     a_reshaped = a_f32.reshape(target_shape)
#     # Convert OM to float16 for fair comparison
#     a_f16 = a_reshaped.astype(np.float16)
    
#     diff = np.abs(a_f16.astype(np.float64) - b.astype(np.float64))
#     print(f"\n=== OM(f32->f16) vs ONNX(f16) ===")
#     print(f"allclose (atol=0.01): {np.allclose(a_f16, b, atol=0.01)}")
#     print(f"allclose (atol=0.1):  {np.allclose(a_f16, b, atol=0.1)}")
#     print(f"allclose (atol=1.0):  {np.allclose(a_f16, b, atol=1.0)}")
#     print(f"max abs diff:  {diff.max():.4f}")
#     print(f"mean abs diff: {diff.mean():.6f}")
    
#     mask = (np.abs(a_f16.astype(np.float64)) > 0.001) | (np.abs(b.astype(np.float64)) > 0.001)
#     if np.sum(mask) > 0:
#         corr = np.corrcoef(a_f16.flatten()[mask.flatten()].astype(np.float64),
#                            b.flatten()[mask.flatten()].astype(np.float64))[0, 1]
#         print(f"correlation (non-zero): {corr:.6f}")
    
#     # Padding boundary
#     print(f"\n=== Padding boundary ===")
#     for name, arr in [("OM", a_f16), ("ONNX", b)]:
#         layer0_head0 = arr[0, 0, 0, :, :]
#         padding_start = arr.shape[3]
#         for i in range(arr.shape[3] - 1):
#             if np.allclose(layer0_head0[i], layer0_head0[i+1], atol=1e-4):
#                 all_same = True
#                 for j in range(i+1, min(i+5, arr.shape[3])):
#                     if not np.allclose(layer0_head0[i], layer0_head0[j], atol=1e-4):
#                         all_same = False
#                         break
#                 if all_same:
#                     padding_start = i
#                     break
#         print(f"  {name}: unique positions = {padding_start}")
    
#     # Position-by-position
#     print(f"\n=== Position comparison (layer0, head0) ===")
#     for pos in [0, 1, 2, 5, 10, 50, 100, 196]:
#         if pos < target_shape[3]:
#             v1 = a_f16[0, 0, 0, pos, :5]
#             v2 = b[0, 0, 0, pos, :5]
#             d = np.max(np.abs(v1.astype(np.float32) - v2.astype(np.float32)))
#             print(f"  pos {pos:3d}: OM={v1}, ONNX={v2}, max_diff={d:.4f}")
    
#     # Layer-by-layer at first 197 positions
#     print(f"\n=== Layer-by-layer (first 197 positions) ===")
#     for layer in range(min(7, target_shape[0])):
#         a_sub = a_f16[layer, 0, :, :197, :].astype(np.float64)
#         b_sub = b[layer, 0, :, :197, :].astype(np.float64)
#         max_d = np.max(np.abs(a_sub - b_sub))
#         mean_d = np.mean(np.abs(a_sub - b_sub))
#         c = np.corrcoef(a_sub.flatten(), b_sub.flatten())[0, 1]
#         print(f"  Layer {layer}: max_diff={max_d:.4f}, mean_diff={mean_d:.6f}, corr={c:.6f}")

# elif a_f32.shape[0] > target_elements:
#     print(f"\nOM has more elements than target. Ratio: {a_f32.shape[0] / target_elements:.2f}")
#     # Try first target_elements
#     a_reshaped = a_f32[:target_elements].reshape(target_shape).astype(np.float16)
#     diff = np.abs(a_reshaped.astype(np.float64) - b.astype(np.float64))
#     print(f"First {target_elements} as {target_shape}: max_diff={diff.max():.4f}, mean_diff={diff.mean():.6f}")
# else:
#     print(f"\nOM has fewer elements than target!")


import numpy as np
import os

def compare_npy(file1, file2, atol=1e-5, rtol=1e-5):
    if not os.path.exists(file1):
        print(f"{file1} 不存在")
        return
    if not os.path.exists(file2):
        print(f"{file2} 不存在")
        return

    a = np.load(file1)
    b = np.load(file2)

    print(f"\nComparing: {file1}  vs  {file2}")
    print(f"Original shape: {a.shape}")

    if a.shape != b.shape:
        print(f"Shape 不一致: {a.shape} vs {b.shape}")
        return

    # 只比较 [:,:,:, :197, :]
    a_slice = a[:, :, :, :1, :]
    b_slice = b[:, :, :, :1, :]

    print(f"Compare slice shape: {a_slice.shape}")

    diff = np.abs(a_slice - b_slice)

    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Max abs diff : {max_diff}")
    print(f"Mean abs diff: {mean_diff}")

    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Max diff index (slice): {idx}")
    print(f"a value: {a_slice[idx]}")
    print(f"b value: {b_slice[idx]}")

    if np.allclose(a_slice, b_slice, atol=atol, rtol=rtol):
        print("Result: ✅ 在容差范围内一致")
    else:
        print("Result: ❌ 超出容差")


def main():
    for i in range(4):
        om_file = f"test_om_{i}.npy"
        onnx_file = f"test_onnx_{i}.npy"
        compare_npy(om_file, onnx_file)


if __name__ == "__main__":
    main()