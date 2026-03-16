# import numpy as np

# # ONNX decode output: already shaped (7,1,8,1,128) float16
# onnx = np.load('test_onnx_0.npy')
# print('ONNX shape:', onnx.shape, onnx.dtype)
# print('ONNX[0,0,0,0,:3]:', onnx[0,0,0,0,:3])

# # OM raw output: (29388800,) float32 = actually uint8 bytes
# om_raw = np.load('test_om_pre_0.npy')
# print('\nOM raw shape:', om_raw.shape, om_raw.dtype)
# print('OM raw first 10:', om_raw[:10])

# # Interpret as bytes then float32, reshape to (7,1,8,1025,128)
# om_bytes = om_raw.astype(np.uint8).tobytes()
# om_f32 = np.frombuffer(om_bytes, dtype=np.float32).reshape(7, 1, 8, 1025, 128)
# print('\nOM reshaped to (7,1,8,1025,128)')
# print('OM last token [0,0,0,-1,:3]:', om_f32[0, 0, 0, -1, :3])
# print('OM first token [0,0,0,0,:3]:', om_f32[0, 0, 0, 0, :3])

# # How many non-zero along seq dim for layer0, head0?
# nz_per_pos = np.count_nonzero(om_f32[0, 0, 0, :, :], axis=1)
# non_zero_positions = np.where(nz_per_pos > 0)[0]
# print(f'\nNon-zero positions count: {len(non_zero_positions)}')
# if len(non_zero_positions) > 0:
#     print(f'Non-zero positions range: {non_zero_positions[0]} to {non_zero_positions[-1]}')

# # test_om_0 is the already-processed version
# om_processed = np.load('test_om_0.npy')
# print('\nOM processed shape:', om_processed.shape, om_processed.dtype)
# print('OM processed [0,0,0,0,:3]:', om_processed[0, 0, 0, 0, :3])

# # Compare: onnx (float16) vs om
# onnx_f32 = onnx.astype(np.float32)
# print('\nONNX as f32 [0,0,0,0,:3]:', onnx_f32[0, 0, 0, 0, :3])

# # Compare with last position (should be the new token's KV)
# diff_last = np.max(np.abs(onnx_f32[0, 0, 0, 0, :] - om_f32[0, 0, 0, -1, :]))
# print(f'Max diff (onnx vs om LAST pos): {diff_last}')

# # Compare with first position
# diff_first = np.max(np.abs(onnx_f32[0, 0, 0, 0, :] - om_f32[0, 0, 0, 0, :]))
# print(f'Max diff (onnx vs om FIRST pos): {diff_first}')

# # Compare with processed
# diff_proc = np.max(np.abs(onnx_f32 - om_processed))
# print(f'Max diff (onnx vs om_processed): {diff_proc}')

# # Also check all 4 blocks
# print('\n=== All blocks ===')
# for i in range(4):
#     onnx_i = np.load(f'test_onnx_{i}.npy')
#     om_pre_i = np.load(f'test_om_pre_{i}.npy')
#     om_i = np.load(f'test_om_{i}.npy')
#     print(f'\nBlock {i}:')
#     print(f'  onnx: {onnx_i.shape} {onnx_i.dtype}')
#     print(f'  om_pre: {om_pre_i.shape} {om_pre_i.dtype}, size={om_pre_i.size}')
#     print(f'  om: {om_i.shape} {om_i.dtype}')
    
#     # Check if om_pre can reshape to (7,1,8,1025,128)
#     expected = 7 * 1 * 8 * 1025 * 128
#     if om_pre_i.size == expected * 4:  # float32 storing uint8 bytes
#         raw_bytes = om_pre_i.astype(np.uint8).tobytes()
#         kv = np.frombuffer(raw_bytes, dtype=np.float32)[:expected].reshape(7, 1, 8, 1025, 128)
#         new_kv = kv[:, :, :, -1:, :]
#         onnx_f = onnx_i.astype(np.float32)
#         diff = np.max(np.abs(onnx_f - new_kv))
#         print(f'  Diff (onnx vs om_pre LAST pos): {diff}')
#         diff_first = np.max(np.abs(onnx_f - kv[:, :, :, :1, :]))
#         print(f'  Diff (onnx vs om_pre FIRST pos): {diff_first}')
    
#     diff_proc = np.max(np.abs(onnx_i.astype(np.float32) - om_i.astype(np.float32)))
#     print(f'  Diff (onnx vs om_processed): {diff_proc}')


# import numpy as np

# # 载入原始 .npy 文件
# data = np.load('test_om_pre_0.npy')

# # 去除零值，只保留非零元素
# non_zero_data = data[data != 0]

# # 确保非零数据符合目标 shape (7, 1, 8, 1, 128)
# # 这里我们需要将数据切分并重新排列成新的形状
# reshaped_data = non_zero_data[:7 * 1 * 8 * 1 * 128].reshape(7, 1, 8, 1, 128)

# # 转换为 float16 类型
# reshaped_data = reshaped_data.astype(np.float16)

# # 保存为新的 .npy 文件
# np.save('compressed_test_om_pre_0.npy', reshaped_data)

# print("文件已保存为 'compressed_test_om_pre_0.npy'")


import numpy as np

# 加载两个 .npy 文件
file1 = np.load('compressed_test_om_pre_0.npy')
file2 = np.load('test_onnx_0.npy')

# 确保它们的形状和数据类型一致
if file1.shape != file2.shape:
    raise ValueError(f"文件形状不同：file1 shape = {file1.shape}, file2 shape = {file2.shape}")
if file1.dtype != file2.dtype:
    raise ValueError(f"文件数据类型不同：file1 dtype = {file1.dtype}, file2 dtype = {file2.dtype}")

# 计算差值
difference = file1 - file2

# 计算最大差值、最小差值、均方差等
max_diff = np.max(np.abs(difference))
min_diff = np.min(np.abs(difference))
mean_diff = np.mean(np.abs(difference))
mse = np.mean(difference ** 2)  # 均方误差

# 输出差值统计信息
print(f"最大差值: {max_diff}")
print(f"最小差值: {min_diff}")
print(f"均方误差 (MSE): {mse}")
print(f"平均差异: {mean_diff}")

# 如果你想查看差异矩阵，可以打印部分差异
print("差异的前 10 个元素：")
print(difference.flatten()[:10])

# 打印两个npy文件的前十个数据
print("文件1的前10个数据：")
print(file1.flatten()[:10])
print("文件2的前10个数据：")
print(file2.flatten()[:10])

import struct

# 四个 uint8 数字
# a = 191
# b = 68
# c = 254
# d = 4

a = 189
b = 24
c = 102
d = 154

# 将这四个整数转换为 8-bit 二进制数并拼接成 32-bit 二进制数
binary_str = f'{a:08b}{b:08b}{c:08b}{d:08b}'

# 将拼接的二进制字符串转为 32-bit 整数
binary_int = int(binary_str, 2)

# 使用 struct.unpack 来将 32-bit 整数解释为 float32 类型
float_value = struct.unpack('f', struct.pack('I', binary_int))[0]

# 输出结果
print(f"拼接后的 32-bit 二进制数：{binary_str}")
print(f"对应的 float32 数值：{float_value}")