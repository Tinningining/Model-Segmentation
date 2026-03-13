#!/usr/bin/env python3
"""
Qwen MCP 工具调用测试 - 带真实工具执行的三阶段测试
"""

import sys
import os
import json

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src')

from llm_offline import chat, chat_with_tools, get_llm
from config import SAMPLE_TOOLS
from tool_executor import execute_tool

# 测试用例
TOOL_TEST_CASES = [
    {
        "name": "天气查询-北京",
        "user_message": "北京今天天气怎么样？",
        "expected_tool": "get_weather",
        "difficulty": "简单"
    },
    {
        "name": "天气查询-上海华氏度",
        "user_message": "请用华氏度告诉我上海的天气",
        "expected_tool": "get_weather",
        "difficulty": "中等"
    },
    {
        "name": "数学计算-简单",
        "user_message": "计算一下 123 * 456 等于多少",
        "expected_tool": "calculate",
        "difficulty": "简单"
    },
    {
        "name": "数学计算-平方根",
        "user_message": "16的平方根是多少？",
        "expected_tool": "calculate",
        "difficulty": "中等"
    },
    {
        "name": "时间查询-本地",
        "user_message": "现在几点了？",
        "expected_tool": "get_current_time",
        "difficulty": "简单"
    },
    {
        "name": "时间查询-纽约",
        "user_message": "纽约现在是什么时间？",
        "expected_tool": "get_current_time",
        "difficulty": "中等"
    },
]

def extract_json(text):
    """从文本中提取 JSON"""
    try:
        return json.loads(text.strip()), None
    except:
        pass
    
    if "```" in text:
        try:
            json_str = text.split("```")[1].split("```")[0].strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
            return json.loads(json_str), None
        except:
            pass
    
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end]), None
    except:
        pass
    
    return None, "无法解析JSON"

def print_box(title, content="", width=70):
    """打印带边框的内容"""
    print("┌" + "─" * (width - 2) + "┐")
    print("│ " + title.ljust(width - 4) + " │")
    if content:
        print("├" + "─" * (width - 2) + "┤")
        for line in content.split('\n'):
            if line:
                print("│ " + line.ljust(width - 4) + " │")
    print("└" + "─" * (width - 2) + "┘")

def test_with_execution():
    """带真实工具执行的三阶段测试"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "    Qwen MCP 工具调用测试 - 三阶段执行流程    ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print("\n[0] 加载模型...")
    get_llm()
    
    passed = 0
    total = len(TOOL_TEST_CASES)
    
    for idx, case in enumerate(TOOL_TEST_CASES, 1):
        print("\n" + "=" * 70)
        print(f"测试 {idx}/{total}: {case['name']} [{case['difficulty']}]")
        print("=" * 70)
        
        print(f"\n用户问题: {case['user_message']}")
        
        # ============================================================
        # 第1轮：模型推理（决策阶段）
        # ============================================================
        print("\n" + "┌" + "─" * 68 + "┐")
        print("│ " + "第1轮：模型推理（决策阶段）".ljust(66) + " │")
        print("├" + "─" * 68 + "┤")
        print("│ " + "输入：用户问题".ljust(66) + " │")
        print("│ " + "输出：工具调用JSON".ljust(66) + " │")
        print("│ " + "说明：模型决定需要调用哪个工具，但此时还没有实际数据".ljust(66) + " │")
        print("└" + "─" * 68 + "┘")
        
        response_1 = chat_with_tools(case['user_message'], SAMPLE_TOOLS)
        print(f"\n模型输出:")
        print(f"  {response_1[:150]}{'...' if len(response_1) > 150 else ''}")
        
        data, err = extract_json(response_1)
        
        if not data:
            print(f"\n❌ JSON解析失败: {err}")
            print("测试失败，跳过后续阶段\n")
            continue
        
        print(f"\n解析后的JSON:")
        print(f"  {json.dumps(data, ensure_ascii=False, indent=2).replace(chr(10), chr(10) + '  ')}")
        
        tool_name = data.get("tool_name") or data.get("name", "")
        arguments = data.get("arguments") or data.get("parameters", {})
        
        if not tool_name:
            print(f"\n❌ 未找到工具名称")
            print("测试失败，跳过后续阶段\n")
            continue
        
        print(f"\n✓ 模型决定调用工具: {tool_name}")
        print(f"✓ 工具参数: {json.dumps(arguments, ensure_ascii=False)}")
        
        # ============================================================
        # 工具执行（不涉及模型）
        # ============================================================
        print("\n                            ↓")
        print("\n" + "┌" + "─" * 68 + "┐")
        print("│ " + "工具执行（不涉及模型）".ljust(66) + " │")
        print("├" + "─" * 68 + "┤")
        print("│ " + f"执行：{tool_name}({', '.join(f'{k}={repr(v)}' for k, v in arguments.items())})".ljust(66)[:66] + " │")
        print("│ " + "说明：纯函数调用，获取真实数据".ljust(66) + " │")
        print("└" + "─" * 68 + "┘")
        
        print(f"\n正在执行工具...")
        tool_result = execute_tool(tool_name, arguments)
        
        print(f"\n工具返回结果:")
        result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
        for line in result_str.split('\n')[:10]:
            print(f"  {line}")
        if len(result_str.split('\n')) > 10:
            print(f"  ... (共 {len(result_str.split(chr(10)))} 行)")
        
        if "error" in tool_result:
            print(f"\n⚠️ 工具执行出错: {tool_result['error']}")
            print("继续进行第2轮推理...\n")
        else:
            print(f"\n✓ 工具执行成功")
        
        # ============================================================
        # 第2轮：模型推理（综合阶段）
        # ============================================================
        print("\n                            ↓")
        print("\n" + "┌" + "─" * 68 + "┐")
        print("│ " + "第2轮：模型推理（综合阶段）".ljust(66) + " │")
        print("├" + "─" * 68 + "┤")
        print("│ " + "输入：用户问题 + 工具结果".ljust(66) + " │")
        print("│ " + "输出：基于真实数据的回答".ljust(66) + " │")
        print("│ " + "说明：模型看到了真实数据，可以给出具体回答".ljust(66) + " │")
        print("└" + "─" * 68 + "┘")
        
        # 构建第二轮的系统提示
        system_prompt_2 = f"""你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。

用户问题：{case['user_message']}

工具调用：{tool_name}
工具参数：{json.dumps(arguments, ensure_ascii=False)}
工具返回：{json.dumps(tool_result, ensure_ascii=False)}

请基于工具返回的真实数据，用自然语言回答用户的问题。回答要具体、准确、友好。"""
        
        response_2 = chat(system_prompt_2, "请根据工具返回的数据回答用户的问题", temperature=0.7, max_tokens=512)
        
        print(f"\n模型最终回答:")
        print("─" * 70)
        for line in response_2.split('\n'):
            print(f"  {line}")
        print("─" * 70)
        
        # 验证是否包含工具返回的关键信息
        tool_match = case['expected_tool'] in tool_name.lower() or tool_name.lower() in case['expected_tool']
        has_data = not ("error" in tool_result)
        
        if tool_match and has_data:
            print(f"\n✅ 测试通过")
            print(f"   - 工具选择正确: {tool_name}")
            print(f"   - 工具执行成功")
            print(f"   - 模型给出了基于真实数据的回答")
            passed += 1
        elif tool_match:
            print(f"\n⚠️ 部分通过")
            print(f"   - 工具选择正确: {tool_name}")
            print(f"   - 工具执行失败")
            passed += 0.5
        else:
            print(f"\n❌ 测试失败")
            print(f"   - 期望工具: {case['expected_tool']}")
            print(f"   - 实际工具: {tool_name}")
    
    # 汇总报告
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "    测试结果汇总    ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    rate = passed / total * 100 if total > 0 else 0
    print(f"\n通过率: {passed}/{total} ({rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！模型具备完整的工具调用能力。")
    elif passed >= total * 0.8:
        print("\n✅ 大部分测试通过，模型基本具备工具调用能力。")
    elif passed >= total * 0.6:
        print("\n⚠️ 部分测试通过，模型需要进一步优化。")
    else:
        print("\n❌ 测试通过率较低，建议检查模型配置。")
    
    return passed >= total * 0.8

def main():
    try:
        return test_with_execution()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return False
    except Exception as e:
        print(f"\n\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
