"""
离线模式 LLM 模块
直接加载模型，不需要启动 API 服务
"""

import json
from vllm import LLM, SamplingParams

# 模型路径
MODEL_PATH = "./models/Qwen3-1.7B"

# 全局 LLM 实例
_llm_instance = None

def get_llm():
    """获取或创建 LLM 实例（单例模式）"""
    global _llm_instance
    if _llm_instance is None:
        print("加载模型中...")
        _llm_instance = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.7
        )
        print("✓ 模型加载完成")
    return _llm_instance

def build_chat_prompt(system: str, user: str) -> str:
    """构建 Qwen 聊天格式的提示"""
    return f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
"""

def chat(system_prompt: str, user_message: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    """发送消息并获取回复"""
    llm = get_llm()
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],  # 只使用标准停止符
        skip_special_tokens=True
    )
    prompt = build_chat_prompt(system_prompt, user_message)
    outputs = llm.generate([prompt], sampling_params)
    result = outputs[0].outputs[0].text.strip()
    
    # 后处理：移除 think 标签内容
    if "<think>" in result and "</think>" in result:
        # 移除完整的 think 块
        import re
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    elif "<think>" in result:
        # 如果只有开始标签，取之前的内容
        result = result.split("<think>")[0].strip()
    
    return result

def chat_with_tools(user_message: str, tools: list, temperature: float = 0.3) -> str:
    """带工具定义的对话"""
    tool_desc = json.dumps(tools, ensure_ascii=False, indent=2)
    system_prompt = f"""你是一个AI助手，可以使用以下工具：

{tool_desc}

当需要使用工具时，请严格按照以下JSON格式输出：
```json
{{"tool_name": "工具名称", "arguments": {{"参数名": "参数值"}}}}
```
只输出JSON，不要输出其他内容。"""
    
    return chat(system_prompt, user_message, temperature)
