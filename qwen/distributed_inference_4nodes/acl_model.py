"""
ACL 模型封装模块
封装华为昇腾 ACL 推理接口
4 节点版本：模型加载后保持常驻，无需频繁加载/卸载
"""
import acl
import numpy as np
from typing import List, Optional

# 全局 ACL 初始化状态
_acl_initialized = False
_shared_context = None
_shared_stream = None
_device_set = False


class ACLModel:
    """ACL 模型推理封装（常驻内存版本）"""
    
    def __init__(self, model_path: str, device_id: int = 0):
        """
        初始化 ACL 模型
        
        Args:
            model_path: OM 模型文件路径
            device_id: 设备 ID
        """
        self.model_path = model_path
        self.device_id = device_id
        self.model_id = None
        self.desc = None
        self.context = None
        self.stream = None
        self._initialized = False
    
    def init(self):
        """初始化 ACL 资源并加载模型"""
        global _acl_initialized, _shared_context, _shared_stream, _device_set
        
        if self._initialized:
            return
        
        # ACL 只能初始化一次
        if not _acl_initialized:
            ret = acl.init()
            if ret != 0:
                raise RuntimeError(f"acl.init failed: {ret}")
            _acl_initialized = True
        
        # 设备只需设置一次
        if not _device_set:
            ret = acl.rt.set_device(self.device_id)
            if ret != 0:
                raise RuntimeError(f"set_device failed: {ret}")
            _device_set = True
        
        # 共享 context 和 stream
        if _shared_context is None:
            _shared_context, ret = acl.rt.create_context(self.device_id)
            if ret != 0:
                raise RuntimeError(f"create_context failed: {ret}")
        self.context = _shared_context
        
        if _shared_stream is None:
            _shared_stream, ret = acl.rt.create_stream()
            if ret != 0:
                raise RuntimeError(f"create_stream failed: {ret}")
        self.stream = _shared_stream
        
        # 加载模型
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load model failed: {ret}, path={self.model_path}")
        
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)
        
        self._initialized = True
        print(f"[ACLModel] Loaded: {self.model_path}")
    
    def execute(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        执行推理
        
        Args:
            inputs: 输入数据列表
            
        Returns:
            输出数据列表
        """
        if not self._initialized:
            raise RuntimeError("ACLModel not initialized. Call init() first.")
        
        input_ds = acl.mdl.create_dataset()
        output_ds = acl.mdl.create_dataset()
        
        input_buffers = []
        output_buffers = []
        input_data_buffers = []
        output_data_buffers = []
        
        try:
            # 准备输入
            for arr in inputs:
                size = arr.nbytes
                dev, ret = acl.rt.malloc(size, 0)
                if ret != 0:
                    raise RuntimeError(f"malloc input failed: {ret}")
                input_buffers.append((dev, size))
                
                acl.rt.memcpy(dev, size, arr.ctypes.data, size, 1)  # H2D
                buf = acl.create_data_buffer(dev, size)
                input_data_buffers.append(buf)
                acl.mdl.add_dataset_buffer(input_ds, buf)
            
            # 准备输出
            num_outputs = acl.mdl.get_num_outputs(self.desc)
            for i in range(num_outputs):
                size = acl.mdl.get_output_size_by_index(self.desc, i)
                dev, ret = acl.rt.malloc(size, 0)
                if ret != 0:
                    raise RuntimeError(f"malloc output failed: {ret}")
                output_buffers.append((dev, size))
                
                buf = acl.create_data_buffer(dev, size)
                output_data_buffers.append(buf)
                acl.mdl.add_dataset_buffer(output_ds, buf)
            
            # 执行推理
            ret = acl.mdl.execute(self.model_id, input_ds, output_ds)
            if ret != 0:
                raise RuntimeError(f"execute failed: {ret}")
            
            # 获取输出
            outputs = []
            for i in range(num_outputs):
                dev, size = output_buffers[i]
                host = np.empty(size, dtype=np.uint8)
                acl.rt.memcpy(host.ctypes.data, size, dev, size, 2)  # D2H
                outputs.append(host)
            
            return outputs
            
        finally:
            # 清理资源
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
    
    def finalize(self):
        """释放 ACL 资源"""
        if not self._initialized:
            return
        
        # 同步 stream
        if self.stream is not None:
            acl.rt.synchronize_stream(self.stream)
        
        # 释放模型相关资源
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None
        if self.desc is not None:
            acl.mdl.destroy_desc(self.desc)
            self.desc = None
        
        self._initialized = False
        print(f"[ACLModel] Unloaded: {self.model_path}")
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._initialized
    
    def __del__(self):
        """析构函数"""
        self.finalize()


def finalize_acl():
    """完全释放 ACL 资源（程序退出时调用）"""
    global _acl_initialized, _shared_context, _shared_stream, _device_set
    
    if _shared_stream is not None:
        acl.rt.destroy_stream(_shared_stream)
        _shared_stream = None
    
    if _shared_context is not None:
        acl.rt.destroy_context(_shared_context)
        _shared_context = None
    
    if _device_set:
        acl.rt.reset_device(0)
        _device_set = False
    
    if _acl_initialized:
        acl.finalize()
        _acl_initialized = False
