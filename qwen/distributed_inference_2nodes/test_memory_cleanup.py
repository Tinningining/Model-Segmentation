"""测试各种内存清理方法"""
import acl
import numpy as np
import sys
import gc
import time

def test_with_device_reset(om_path, num_iterations=10, max_cache_len=1024, max_input_len=16):
    """测试在每次卸载后重置设备"""
    print(f"Testing with device reset after each unload")
    print(f"  iterations={num_iterations}, max_cache_len={max_cache_len}")
    
    # 准备输入数据
    hidden = np.zeros((1, max_input_len, 2048), dtype=np.float32)
    attention_mask = np.zeros((1, 1, max_input_len, max_cache_len + max_input_len), dtype=np.float32)
    position_ids = np.zeros((1, max_input_len), dtype=np.int64)
    past_key = np.zeros((7, 1, 8, max_cache_len, 128), dtype=np.float32)
    past_value = np.zeros((7, 1, 8, max_cache_len, 128), dtype=np.float32)
    inputs = [hidden, attention_mask, position_ids, past_key, past_value]
    
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        
        # 每次迭代都重新初始化 ACL
        print("  Initializing ACL...")
        acl.init()
        acl.rt.set_device(0)
        context, _ = acl.rt.create_context(0)
        stream, _ = acl.rt.create_stream()
        
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
        input_ds = acl.mdl.create_dataset()
        output_ds = acl.mdl.create_dataset()
        
        input_buffers = []
        input_data_buffers = []
        output_buffers = []
        output_data_buffers = []
        
        try:
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
            
            num_outputs = acl.mdl.get_num_outputs(desc)
            for j in range(num_outputs):
                size = acl.mdl.get_output_size_by_index(desc, j)
                dev, ret = acl.rt.malloc(size, 0)
                if ret != 0:
                    raise RuntimeError(f"malloc failed: {ret}")
                output_buffers.append((dev, size))
                buf = acl.create_data_buffer(dev, size)
                output_data_buffers.append(buf)
                acl.mdl.add_dataset_buffer(output_ds, buf)
            
            ret = acl.mdl.execute(model_id, input_ds, output_ds)
            if ret != 0:
                raise RuntimeError(f"execute failed: {ret}")
            
            print(f"  SUCCESS!")
            
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
        
        # 完全清理
        print("  Cleaning up...")
        acl.rt.synchronize_stream(stream)
        acl.mdl.unload(model_id)
        acl.mdl.destroy_desc(desc)
        acl.rt.destroy_stream(stream)
        acl.rt.destroy_context(context)
        acl.rt.reset_device(0)
        acl.finalize()
        
        # 强制 GC
        gc.collect()
        time.sleep(0.5)  # 给系统一些时间释放资源
    
    print(f"\n=== All {num_iterations} iterations completed successfully! ===")

if __name__ == "__main__":
    om_path = sys.argv[1] if len(sys.argv) > 1 else "/home/HwHiAiUser/qwen_distributed/models/layers_0_6.om"
    num_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_cache_len = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    test_with_device_reset(om_path, num_iterations, max_cache_len)
