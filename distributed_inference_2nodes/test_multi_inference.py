"""测试多次推理循环"""
import acl
import numpy as np
import sys
import gc

# 全局 ACL 状态
_acl_initialized = False
_context = None
_stream = None

def init_acl():
    global _acl_initialized, _context, _stream
    if _acl_initialized:
        return
    acl.init()
    acl.rt.set_device(0)
    _context, _ = acl.rt.create_context(0)
    _stream, _ = acl.rt.create_stream()
    _acl_initialized = True

def run_single_inference(model_id, desc, inputs):
    """运行单次推理"""
    input_ds = acl.mdl.create_dataset()
    output_ds = acl.mdl.create_dataset()
    
    input_buffers = []
    input_data_buffers = []
    output_buffers = []
    output_data_buffers = []
    
    try:
        # 输入
        for arr in inputs:
            size = arr.nbytes
            dev, ret = acl.rt.malloc(size, 0)
            if ret != 0:
                raise RuntimeError(f"malloc failed: {ret}")
            input_buffers.append((dev, size))
            acl.rt.memcpy(dev, size, arr.ctypes.data, size, 1)
            buf = acl.create_data_buffer(dev, size)
            input_data_buffers.append(buf)
            acl.mdl.add_dataset_buffer(input_ds, buf)
        
        # 输出
        num_outputs = acl.mdl.get_num_outputs(desc)
        for i in range(num_outputs):
            size = acl.mdl.get_output_size_by_index(desc, i)
            dev, ret = acl.rt.malloc(size, 0)
            if ret != 0:
                raise RuntimeError(f"malloc failed: {ret}")
            output_buffers.append((dev, size))
            buf = acl.create_data_buffer(dev, size)
            output_data_buffers.append(buf)
            acl.mdl.add_dataset_buffer(output_ds, buf)
        
        # 执行
        ret = acl.mdl.execute(model_id, input_ds, output_ds)
        if ret != 0:
            raise RuntimeError(f"execute failed: {ret}")
        
        # 获取输出
        outputs = []
        for i in range(num_outputs):
            dev, size = output_buffers[i]
            host = np.empty(size, dtype=np.uint8)
            acl.rt.memcpy(host.ctypes.data, size, dev, size, 2)
            outputs.append(host)
        
        return outputs
        
    finally:
        for buf in input_data_buffers:
            acl.destroy_data_buffer(buf)
        for buf in output_data_buffers:
            acl.destroy_data_buffer(buf)
        for dev, _ in input_buffers:
            acl.rt.free(dev)
        for dev, _ in output_buffers:
            acl.rt.free(dev)
        acl.mdl.destroy_dataset(input_ds)
        acl.mdl.destroy_dataset(output_ds)

def test_multi_inference(om_path, num_iterations=10, max_cache_len=1024, max_input_len=16):
    """测试多次推理"""
    print(f"Testing: {om_path}")
    print(f"  iterations={num_iterations}, max_cache_len={max_cache_len}")
    
    init_acl()
    
    # 准备输入数据
    hidden = np.zeros((1, max_input_len, 2048), dtype=np.float32)
    attention_mask = np.zeros((1, 1, max_input_len, max_cache_len + max_input_len), dtype=np.float32)
    position_ids = np.zeros((1, max_input_len), dtype=np.int64)
    past_key = np.zeros((7, 1, 8, max_cache_len, 128), dtype=np.float32)
    past_value = np.zeros((7, 1, 8, max_cache_len, 128), dtype=np.float32)
    inputs = [hidden, attention_mask, position_ids, past_key, past_value]
    
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        
        # 加载模型
        print("  Loading model...")
        model_id, ret = acl.mdl.load_from_file(om_path)
        if ret != 0:
            print(f"  FAILED: ret={ret}")
            return
        
        desc = acl.mdl.create_desc()
        acl.mdl.get_desc(desc, model_id)
        
        # 执行推理
        print("  Executing inference...")
        try:
            outputs = run_single_inference(model_id, desc, inputs)
            print(f"  SUCCESS! Output sizes: {[len(o) for o in outputs]}")
        except Exception as e:
            print(f"  FAILED: {e}")
            return
        
        # 卸载模型
        print("  Unloading model...")
        # 同步 stream
        acl.rt.synchronize_stream(_stream)
        acl.mdl.unload(model_id)
        acl.mdl.destroy_desc(desc)
        
        # 强制 GC
        gc.collect()
    
    print(f"\n=== All {num_iterations} iterations completed successfully! ===")

if __name__ == "__main__":
    om_path = sys.argv[1] if len(sys.argv) > 1 else "/home/HwHiAiUser/qwen_distributed/models/layers_0_6.om"
    num_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_cache_len = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    test_multi_inference(om_path, num_iterations, max_cache_len)
