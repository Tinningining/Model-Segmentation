"""测试同时加载三个模型"""
import acl
import sys

def test_load_models(om_dir):
    print("Initializing ACL...")
    ret = acl.init()
    print(f"acl.init: ret={ret}")
    
    ret = acl.rt.set_device(0)
    print(f"set_device: ret={ret}")
    
    context, ret = acl.rt.create_context(0)
    print(f"create_context: ret={ret}")
    
    models = [
        f"{om_dir}/layers_14_20.om",
        f"{om_dir}/layers_21_27.om",
        f"{om_dir}/output.om",
    ]
    
    model_ids = []
    for path in models:
        print(f"\nLoading: {path}")
        model_id, ret = acl.mdl.load_from_file(path)
        print(f"  ret={ret}, model_id={model_id}")
        if ret != 0:
            print(f"  FAILED!")
            break
        model_ids.append(model_id)
        print(f"  SUCCESS!")
    
    print(f"\nLoaded {len(model_ids)} models successfully")
    
    # 清理
    for mid in model_ids:
        acl.mdl.unload(mid)
    
    acl.rt.destroy_context(context)
    acl.rt.reset_device(0)
    acl.finalize()

if __name__ == "__main__":
    om_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/HwHiAiUser/qwen_distributed/models"
    test_load_models(om_dir)
