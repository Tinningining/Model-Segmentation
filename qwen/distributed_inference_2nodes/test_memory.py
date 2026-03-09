"""测试内存使用情况"""
import gc
import sys
import numpy as np

def test_kvcache_memory(max_cache_len=1024):
    """测试 KV Cache 的 CPU 内存占用"""
    print(f"Testing KV Cache memory with max_cache_len={max_cache_len}")
    
    # 模拟 4 个 KV Cache (2 个节点，每个节点 2 个 block)
    # Shape: (7, 1, 8, max_cache_len, 128) × 2 (key + value) × float16
    shape = (7, 1, 8, max_cache_len, 128)
    size_per_cache = np.prod(shape) * 2  # float16 = 2 bytes
    total_size = size_per_cache * 2 * 2  # 2 caches per node, 2 nodes
    
    print(f"  Shape: {shape}")
    print(f"  Size per cache: {size_per_cache / 1024 / 1024:.2f} MB")
    print(f"  Total size (4 caches): {total_size / 1024 / 1024:.2f} MB")
    
    # 创建 KV Cache
    caches = []
    for i in range(4):
        key = np.zeros(shape, dtype=np.float16)
        value = np.zeros(shape, dtype=np.float16)
        caches.append((key, value))
        print(f"  Created cache {i+1}/4")
    
    print("  KV Cache created successfully!")
    
    # 清理
    del caches
    gc.collect()
    print("  KV Cache cleaned up")

def test_model_io_memory(max_cache_len=1024, max_input_len=16, hidden_size=2048):
    """测试模型输入输出的 CPU 内存占用"""
    print(f"\nTesting model I/O memory")
    
    # 模拟 block 模型的输入
    hidden = np.zeros((1, max_input_len, hidden_size), dtype=np.float32)
    attention_mask = np.zeros((1, 1, max_input_len, max_cache_len + max_input_len), dtype=np.float32)
    position_ids = np.zeros((1, max_input_len), dtype=np.int64)
    past_key = np.zeros((7, 1, 8, max_cache_len, 128), dtype=np.float32)
    past_value = np.zeros((7, 1, 8, max_cache_len, 128), dtype=np.float32)
    
    total_input = (hidden.nbytes + attention_mask.nbytes + position_ids.nbytes + 
                   past_key.nbytes + past_value.nbytes)
    
    print(f"  hidden: {hidden.nbytes / 1024 / 1024:.2f} MB")
    print(f"  attention_mask: {attention_mask.nbytes / 1024 / 1024:.2f} MB")
    print(f"  position_ids: {position_ids.nbytes / 1024 / 1024:.2f} MB")
    print(f"  past_key: {past_key.nbytes / 1024 / 1024:.2f} MB")
    print(f"  past_value: {past_value.nbytes / 1024 / 1024:.2f} MB")
    print(f"  Total input: {total_input / 1024 / 1024:.2f} MB")
    
    # 清理
    del hidden, attention_mask, position_ids, past_key, past_value
    gc.collect()
    print("  Model I/O memory cleaned up")

if __name__ == "__main__":
    max_cache_len = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    
    print("=" * 50)
    print("Memory Usage Test")
    print("=" * 50)
    
    test_kvcache_memory(max_cache_len)
    test_model_io_memory(max_cache_len)
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
