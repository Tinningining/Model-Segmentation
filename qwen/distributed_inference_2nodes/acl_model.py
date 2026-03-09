"""
ACL 模型封装模块
封装华为昇腾 ACL 推理接口
"""
import acl
import sys
import numpy as np
from typing import List, Optional

# 全局 ACL 初始化状态
_acl_initialized = False
_shared_context = None
_shared_stream = None
_device_set = False


class ACLModel:
    """ACL 模型推理封装"""
    
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
        """初始化 ACL 资源"""
        global _acl_initialized, _shared_context, _shared_stream, _device_set
        
        if self._initialized:
            return
        
        # ACL 只能初始化一次
        if not _acl_initialized:
            ret = acl.init()
            if ret != 0:
                print(f"[ACLModel ERROR] acl.init, ret={ret}")
                raise RuntimeError(f"ACL operation failed: acl.init")
            _acl_initialized = True
        
        # 设备只需设置一次
        if not _device_set:
            self._check(acl.rt.set_device(self.device_id), "set_device")
            _device_set = True
        
        # 共享 context 和 stream
        if _shared_context is None:
            _shared_context, ret = acl.rt.create_context(self.device_id)
            self._check(ret, "create_context")
        self.context = _shared_context
        
        if _shared_stream is None:
            _shared_stream, ret = acl.rt.create_stream()
            self._check(ret, "create_stream")
        self.stream = _shared_stream
        
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        self._check(ret, f"load model from {self.model_path}")
        
        self.desc = acl.mdl.create_desc()
        self._check(acl.mdl.get_desc(self.desc, self.model_id), "get_desc")
        
        self._initialized = True
        print(f"[ACLModel] Loaded model: {self.model_path}")
    
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
        
        try:
            # 准备输入
            for arr in inputs:
                size = arr.nbytes
                dev, ret = acl.rt.malloc(size, 0)
                self._check(ret, "malloc input")
                input_buffers.append((dev, size))
                
                self._check(
                    acl.rt.memcpy(dev, size, arr.ctypes.data, size, 1),  # H2D
                    "memcpy H2D"
                )
                buf = acl.create_data_buffer(dev, size)
                acl.mdl.add_dataset_buffer(input_ds, buf)
            
            # 准备输出
            num_outputs = acl.mdl.get_num_outputs(self.desc)
            for i in range(num_outputs):
                size = acl.mdl.get_output_size_by_index(self.desc, i)
                dev, ret = acl.rt.malloc(size, 0)
                self._check(ret, "malloc output")
                output_buffers.append((dev, size))
                
                buf = acl.create_data_buffer(dev, size)
                acl.mdl.add_dataset_buffer(output_ds, buf)
            
            # 执行推理
            self._check(
                acl.mdl.execute(self.model_id, input_ds, output_ds),
                "execute"
            )
            
            # 获取输出
            outputs = []
            for i in range(num_outputs):
                dev, size = output_buffers[i]
                host = np.empty(size, dtype=np.uint8)
                self._check(
                    acl.rt.memcpy(host.ctypes.data, size, dev, size, 2),  # D2H
                    "memcpy D2H"
                )
                outputs.append(host)
            
            return outputs
            
        finally:
            # 清理资源
            for dev, size in input_buffers:
                acl.rt.free(dev)
            for dev, size in output_buffers:
                acl.rt.free(dev)
            
            acl.mdl.destroy_dataset(input_ds)
            acl.mdl.destroy_dataset(output_ds)
    
    def finalize(self):
        """释放 ACL 资源"""
        if not self._initialized:
            return
        
        # 只释放模型相关资源，不释放共享的 context 和 stream
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None
        if self.desc is not None:
            acl.mdl.destroy_desc(self.desc)
            self.desc = None
        
        # 注意：不释放共享的 context、stream，也不 reset device
        self._initialized = False
    
    def _check(self, ret: int, msg: str):
        """检查返回值"""
        if ret != 0:
            print(f"[ACLModel ERROR] {msg}, ret={ret}")
            err = acl.get_recent_err_msg()
            if err:
                print(f"[ACLModel] ACL Error: {err}")
            raise RuntimeError(f"ACL operation failed: {msg}")
    
    def __del__(self):
        """析构函数"""
        self.finalize()


class ACLModelPool:
    """ACL 模型池，用于管理多个模型"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.models = {}
        self._acl_initialized = False
    
    def init_acl(self):
        """初始化 ACL（只需调用一次）"""
        if self._acl_initialized:
            return
        
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"acl.init failed: {ret}")
        
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            raise RuntimeError(f"set_device failed: {ret}")
        
        self._acl_initialized = True
    
    def load_model(self, name: str, model_path: str) -> 'ACLModelSimple':
        """加载模型到池中"""
        if not self._acl_initialized:
            self.init_acl()
        
        model = ACLModelSimple(model_path, self.device_id)
        model.init()
        self.models[name] = model
        return model
    
    def get_model(self, name: str) -> Optional['ACLModelSimple']:
        """获取模型"""
        return self.models.get(name)
    
    def finalize(self):
        """释放所有资源"""
        for model in self.models.values():
            model.finalize()
        self.models.clear()
        
        if self._acl_initialized:
            acl.rt.reset_device(self.device_id)
            acl.finalize()
            self._acl_initialized = False


class ACLModelSimple:
    """简化版 ACL 模型（共享 ACL 初始化）"""
    
    def __init__(self, model_path: str, device_id: int = 0):
        self.model_path = model_path
        self.device_id = device_id
        self.model_id = None
        self.desc = None
        self.context = None
        self.stream = None
    
    def init(self):
        """初始化模型（假设 ACL 已初始化）"""
        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != 0:
            raise RuntimeError(f"create_context failed: {ret}")
        
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            raise RuntimeError(f"create_stream failed: {ret}")
        
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load model failed: {ret}")
        
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)
        
        print(f"[ACLModelSimple] Loaded: {self.model_path}")
    
    def execute(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """执行推理"""
        acl.rt.set_context(self.context)
        
        input_ds = acl.mdl.create_dataset()
        output_ds = acl.mdl.create_dataset()
        
        input_buffers = []
        output_buffers = []
        
        try:
            # 输入
            for arr in inputs:
                size = arr.nbytes
                dev, ret = acl.rt.malloc(size, 0)
                input_buffers.append((dev, size))
                acl.rt.memcpy(dev, size, arr.ctypes.data, size, 1)
                buf = acl.create_data_buffer(dev, size)
                acl.mdl.add_dataset_buffer(input_ds, buf)
            
            # 输出
            num_outputs = acl.mdl.get_num_outputs(self.desc)
            for i in range(num_outputs):
                size = acl.mdl.get_output_size_by_index(self.desc, i)
                dev, ret = acl.rt.malloc(size, 0)
                output_buffers.append((dev, size))
                buf = acl.create_data_buffer(dev, size)
                acl.mdl.add_dataset_buffer(output_ds, buf)
            
            # 执行
            acl.mdl.execute(self.model_id, input_ds, output_ds)
            
            # 获取输出
            outputs = []
            for i in range(num_outputs):
                dev, size = output_buffers[i]
                host = np.empty(size, dtype=np.uint8)
                acl.rt.memcpy(host.ctypes.data, size, dev, size, 2)
                outputs.append(host)
            
            return outputs
            
        finally:
            for dev, _ in input_buffers:
                acl.rt.free(dev)
            for dev, _ in output_buffers:
                acl.rt.free(dev)
            acl.mdl.destroy_dataset(input_ds)
            acl.mdl.destroy_dataset(output_ds)
    
    def finalize(self):
        """释放资源"""
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
        if self.desc is not None:
            acl.mdl.destroy_desc(self.desc)
        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)
        if self.context is not None:
            acl.rt.destroy_context(self.context)


class ACLModelLazy:
    """
    延迟加载的 ACL 模型
    支持按需加载和卸载，用于内存受限场景
    """
    
    def __init__(self, model_path: str, device_id: int = 0):
        self.model_path = model_path
        self.device_id = device_id
        self.model_id = None
        self.desc = None
        self._loaded = False
    
    def load(self):
        """加载模型到 NPU"""
        global _acl_initialized, _shared_context, _shared_stream, _device_set
        
        if self._loaded:
            return
        
        # 确保 ACL 已初始化
        if not _acl_initialized:
            ret = acl.init()
            if ret != 0:
                raise RuntimeError(f"acl.init failed: {ret}")
            _acl_initialized = True
        
        if not _device_set:
            ret = acl.rt.set_device(self.device_id)
            if ret != 0:
                raise RuntimeError(f"set_device failed: {ret}")
            _device_set = True
        
        if _shared_context is None:
            _shared_context, ret = acl.rt.create_context(self.device_id)
            if ret != 0:
                raise RuntimeError(f"create_context failed: {ret}")
        
        if _shared_stream is None:
            _shared_stream, ret = acl.rt.create_stream()
            if ret != 0:
                raise RuntimeError(f"create_stream failed: {ret}")
        
        # 加载模型
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load model failed: {ret}, path={self.model_path}")
        
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)
        
        self._loaded = True
        print(f"[ACLModelLazy] Loaded: {self.model_path}")
    
    def unload(self):
        """从 NPU 卸载模型"""
        global _shared_stream
        
        if not self._loaded:
            return
        
        # 同步 stream，确保所有操作完成
        if _shared_stream is not None:
            acl.rt.synchronize_stream(_shared_stream)
        
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None
        if self.desc is not None:
            acl.mdl.destroy_desc(self.desc)
            self.desc = None
        
        self._loaded = False
        print(f"[ACLModelLazy] Unloaded: {self.model_path}")
    
    def execute(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """执行推理"""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        input_ds = acl.mdl.create_dataset()
        output_ds = acl.mdl.create_dataset()
        
        input_buffers = []
        output_buffers = []
        input_data_buffers = []
        output_data_buffers = []
        
        try:
            # 输入
            for arr in inputs:
                size = arr.nbytes
                dev, ret = acl.rt.malloc(size, 0)
                if ret != 0:
                    raise RuntimeError(f"malloc input failed: {ret}")
                input_buffers.append((dev, size))
                acl.rt.memcpy(dev, size, arr.ctypes.data, size, 1)
                buf = acl.create_data_buffer(dev, size)
                input_data_buffers.append(buf)
                acl.mdl.add_dataset_buffer(input_ds, buf)
            
            # 输出
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
            
            # 执行
            ret = acl.mdl.execute(self.model_id, input_ds, output_ds)
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
            # 销毁 data buffer（必须在 destroy_dataset 之前）
            for buf in input_data_buffers:
                acl.destroy_data_buffer(buf)
            for buf in output_data_buffers:
                acl.destroy_data_buffer(buf)
            # 释放设备内存
            for dev, _ in input_buffers:
                acl.rt.free(dev)
            for dev, _ in output_buffers:
                acl.rt.free(dev)
            # 销毁 dataset
            acl.mdl.destroy_dataset(input_ds)
            acl.mdl.destroy_dataset(output_ds)
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded
