"""
JSON格式输出测试脚本 - 离线模式
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_offline import chat, get_llm

def extract_json(text: str):
    """从文本中提取 JSON"""
    try:
        if "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
        else:
            json_str = text
        return json.loads(json_str), None
    except Exception as e:
        return None, str(e)

def run_tests():
    """运行测试，返回 (passed, total)"""
    tests = [
        {
            "name": "简单JSON对象",
            "prompt": "输出JSON：{\"name\": \"测试\", \"value\": 42}",
            "required_keys": ["name", "value"]
        },
        {
            "name": "嵌套JSON对象",
            "prompt": "输出JSON：{\"user\": {\"name\": \"张三\"}, \"config\": {\"theme\": \"dark\"}}",
            "required_keys": ["user", "config"]
        },
        {
            "name": "工具调用格式",
            "prompt": "输出工具调用JSON：{\"tool_name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}",
            "required_keys": ["tool_name", "arguments"]
        },
        {
            "name": "MCP工具定义",
            "prompt": "输出MCP工具定义：{\"name\": \"read_file\", \"description\": \"读取文件\", \"inputSchema\": {\"type\": \"object\"}}",
            "required_keys": ["name", "description", "inputSchema"]
        }
    ]
    
    system = "你是一个只输出JSON的助手。不要输出任何其他内容，只输出JSON。"
    passed = 0
    
    for test in tests:
        print(f"\n测试: {test['name']}")
        response = chat(system, test['prompt'])
        print(f"  输出: {response[:60]}...")
        
        data, err = extract_json(response)
        if data:
            missing = [k for k in test['required_keys'] if k not in data]
            if not missing:
                print("  结果: ✓ 通过")
                passed += 1
            else:
                print(f"  结果: ✗ 缺少字段 {missing}")
        else:
            print(f"  结果: ✗ JSON解析失败")
    
    return passed, len(tests)

if __name__ == "__main__":
    print("=" * 50)
    print("JSON格式输出测试")
    print("=" * 50)
    get_llm()
    p, t = run_tests()
    print(f"\n结果: {p}/{t}")
