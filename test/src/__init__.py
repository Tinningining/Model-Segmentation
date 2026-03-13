"""
Qwen MCP 工具调用测试 - 核心模块
"""

from .llm_offline import chat, chat_with_tools, get_llm
from .config import SAMPLE_TOOLS, MCP_TOOLS, TEST_CASES

__all__ = ['chat', 'chat_with_tools', 'get_llm', 'SAMPLE_TOOLS', 'MCP_TOOLS', 'TEST_CASES']
