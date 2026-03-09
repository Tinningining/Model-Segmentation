"""测试 NPU 内存分配"""
import acl
import numpy as np
import sys

def test_npu_memory_allocation():
    """测试 NPU 内存分配"""
    print("Initializing ACL...")
    ret = acl.init()
    print(f"acl.init: ret={ret}")
    
    ret = acl.rt.set_device(0)
    print(f"set_device: ret={ret}")
    
    context, ret = acl.rt.create_context(0)
    print(f"create_context: ret={ret}")
    
    # 测试分配不同大小的内存
    sizes_mb = [10, 50, 100, 200, 500, 1000, 2000]
    
    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        dev, ret = acl.rt.malloc(size_bytes, 0)
        if ret == 0:
            print(f"  Allocated {size_mb} MB: SUCCESS")
            acl.rt.free(dev)
        else:
            print(f"  Allocated {size_mb} MB: FAILED (ret={ret})")
            break
    
    # 测试 KV Cache 大小的内存分配
    # past_key: (7, 1, 8, 1024, 128) × float32 = 28 MB
    kv_size = 7 * 1 * 8 * 1024 * 128 * 4
    print(f"\nKV Cache size: {kv_size / 1024 / 1024:.2f} MB")
    
    dev, ret = acl.rt.malloc(kv_size, 0)
    if ret == 0:
        print(f"  Allocated KV Cache: SUCCESS")
        acl.rt.free(dev)
    else:
        print(f"  Allocated KV Cache: FAILED (ret={ret})")
    
    # 清理
    acl.rt.destroy_context(context)
    acl.rt.reset_device(0)
    acl.finalize()
    print("\nTest completed!")

def test_model_with_small_input(om_path):
    """测试用小输入加载模型"""
    print(f"\nTesting model: {om_path}")
    
    ret = acl.init()
    ret = acl.rt.set_device(0)
    context, ret = acl.rt.create_context(0)
    
    # 加载模型
    print("Loading model...")
    model_id, ret = acl.mdl.load_from_file(om_path)
    if ret != 0:
        print(f"  Load model FAILED: ret={ret}")
        return
    print(f"  Load model SUCCESS: model_id={model_id}")
    
    # 获取模型描述
    desc = acl.mdl.create_desc()
    acl.mdl.get_desc(desc, model_id)
    
    num_inputs = acl.mdl.get_num_inputs(desc)
    num_outputs = acl.mdl.get_num_outputs(desc)
    print(f"  Inputs: {num_inputs}, Outputs: {num_outputs}")
    
    # 打印输入大小
    total_input_size = 0
    for i in range(num_inputs):
        size = acl.mdl.get_input_size_by_index(desc, i)
        total_input_size += size
        print(f"    Input {i}: {size / 1024 / 1024:.2f} MB")
    print(f"  Total input size: {total_input_size / 1024 / 1024:.2f} MB")
    
    # 卸载模型
    acl.mdl.unload(model_id)
    acl.mdl.destroy_desc(desc)
    print("  Model unloaded")
    
    acl.rt.destroy_context(context)
    acl.rt.reset_device(0)
    acl.finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_model_with_small_input(sys.argv[1])
    else:
        test_npu_memory_allocation()
