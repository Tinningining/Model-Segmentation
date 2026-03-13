"""
工具调用测试脚本 - 离线模式
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_offline import chat_with_tools, get_llm
from config import SAMPLE_TOOLS, TEST_CASES

def extract_json(response: str) -> dict:
    """从响应中提取 JSON"""
    if "```json" in response:
        json_str = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        json_str = response.split("```")[1].split("```")[0].strip()
    else:
        json_str = response
    return json.loads(json_str)

def run_tests():
    """运行测试，返回 (passed, total)"""
    test_cases = TEST_CASES["tool_calling"]
    passed = 0
    
    for case in test_cases:
        print(f"\n测试: {case['name']}")
        print(f"  用户: {case['user_message']}")
        print(f"  期望: {case['expected_tool']}")
        
        response = chat_with_tools(case['user_message'], SAMPLE_TOOLS)
        print(f"  输出: {response[:100]}...")
        
        try:
            data = extract_json(response)
            tool_name = data.get("tool_name") or data.get("name", "")
            
            if case['expected_tool'] in tool_name.lower() or tool_name.lower() in case['expected_tool']:
                print("  结果: ✓ 通过")
                passed += 1
            else:
                print(f"  结果: ✗ 工具不匹配 ({tool_name})")
        except Exception as e:
            print(f"  结果: ✗ JSON解析失败 ({e})")
    
    return passed, len(test_cases)

if __name__ == "__main__":
    print("=" * 50)
    print("工具调用测试")
    print("=" * 50)
    get_llm()
    p, t = run_tests()
    print(f"\n结果: {p}/{t}")
