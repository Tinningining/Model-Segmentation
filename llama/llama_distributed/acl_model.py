"""
ACL 模型封装
简化的 ACL 模型接口
"""
import sys
import os

# 添加 llama/inference_net 到路径以复用 engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llama', 'inference_net'))

from engine import ACLModel as BaseACLModel, initResource, destroyResource


class ACLModel:
    """ACL 模型封装类"""
    
    def __init__(self, model_path: str, device_id: int, context=None):
        """
        初始化 ACL 模型
        
        Args:
            model_path: 模型文件路径
            device_id: 设备 ID
            context: ACL 上下文（如果为 None 则创建新的）
        """
        self.model_path = model_path
        self.device_id = device_id
        self.context = context
        self.model = None
        self.own_context = False
    
    def init(self):
        """初始化模型"""
        # 如果没有提供 context，创建新的
        if self.context is None:
            self.context = initResource(self.device_id)
            self.own_context = True
        
        # 加载模型
        self.model = BaseACLModel(self.model_path, context=self.context, mode='rc')
        print(f"[ACLModel] Loaded model: {self.model_path}")
    
    def execute(self, inputs: list) -> list:
        """
        执行推理
        
        Args:
            inputs: 输入列表
            
        Returns:
            输出列表
        """
        return self.model.inference(inputs)
    
    def finalize(self):
        """清理资源"""
        if self.model:
            self.model.unload()
        
        if self.own_context and self.context:
            destroyResource(self.device_id, self.context)
