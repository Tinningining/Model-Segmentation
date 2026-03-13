"""
指令跟随能力测试脚本 - 离线模式
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_offline import chat, get_llm

def validate_json(response: str) -> tuple:
    """验证 JSON 格式"""
    try:
        if "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
        else:
            json_str = response
        data = json.loads(json_str)
        return True, data
    except:
        return False, None

def run_tests():
    """运行测试，返回 (passed, total)"""
    tests = [
        {
            "name": "JSON对象输出",
            "system": "你是一个只输出JSON的助手。不要输出任何其他内容。",
            "user": "请以JSON格式输出：姓名张三，年龄25，城市北京",
        },
        {
            "name": "工具调用JSON格式",
            "system": "当需要调用工具时，请输出JSON：{\"tool_name\": \"工具名\", \"arguments\": {}}",
            "user": "调用get_weather工具查询北京天气",
        },
        {
            "name": "MCP工具描述格式",
            "system": "请输出MCP工具定义JSON，包含name、description、inputSchema字段",
            "user": "定义一个read_file工具，参数为path",
        }
    ]
    
    passed = 0
    for test in tests:
        print(f"\n测试: {test['name']}")
        response = chat(test['system'], test['user'])
        print(f"  输出: {response[:80]}...")
        
        valid, data = validate_json(response)
        if valid:
            print("  结果: ✓ 通过")
            passed += 1
        else:
            print("  结果: ✗ 格式错误")
    
    return passed, len(tests)

if __name__ == "__main__":
    print("=" * 50)
    print("指令跟随能力测试")
    print("=" * 50)
    get_llm()
    p, t = run_tests()
    print(f"\n结果: {p}/{t}")
