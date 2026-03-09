from abc import ABC, abstractmethod
from typing import Dict, Any


class ToolState:
    UNLOADED = "unloaded"
    LOADED = "loaded"
    RUNNING = "running"


class MCPTool(ABC):
    """
    MCP Tool 抽象基类
    """

    name: str = ""
    description: str = ""
    schema: Dict[str, Any] = {}

    def __init__(self):
        self.state = ToolState.UNLOADED

    def load(self):
        """
        加载工具所需资源（模型、连接、文件、动态库等）
        """
        if self.state != ToolState.UNLOADED:
            return
        self._load()
        self.state = ToolState.LOADED

    def unload(self):
        """
        释放所有资源
        """
        if self.state == ToolState.UNLOADED:
            return
        self._unload()
        self.state = ToolState.UNLOADED

    def run(self, **kwargs) -> Any:
        """
        执行工具
        """
        if self.state != ToolState.LOADED:
            raise RuntimeError(f"Tool {self.name} not loaded")
        self.state = ToolState.RUNNING
        try:
            return self._run(**kwargs)
        finally:
            self.state = ToolState.LOADED

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def _run(self, **kwargs):
        pass

    @abstractmethod
    def _unload(self):
        pass