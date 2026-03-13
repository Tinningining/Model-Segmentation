#!/usr/bin/env python3
"""
Qwen MCP 多工具调用测试 - 测试模型同时调用多个工具的能力
"""

import sys
import os
import json

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src')

from llm_offline import chat, get_llm
from config import SAMPLE_TOOLS
from tool_executor import execute_tool

# 多工具调用测试用例
MULTI_TOOL_TEST_CASES = [
    {
        "name": "天气+时间组合",
        "user_message": "告诉我北京现在的天气和时间",
        "expected_tools": ["get_weather", "get_current_time"],
        "difficulty": "中等"
    },
    {
        "name": "计算+时间组合",
        "user_message": "现在几点了？另外帮我算一下 25 * 48 和 8 + 19",
        "expected_tools": ["get_current_time", "calculate", "calculate"],
        "difficulty": "中等"
    },
    {
        "name": "多城市天气对比",
        "user_message": "对比一下北京和上海的天气",
        "expected_tools": ["get_weather", "get_weather"],
        "difficulty": "困难"
    },
    {
        "name": "多个计算任务",
        "user_message": "帮我算两个数：123 * 456 和 sqrt(144)",
        "expected_tools": ["calculate", "calculate"],
        "difficulty": "困难"
    },
    {
        "name": "多时区时间查询",
        "user_message": "告诉我北京和纽约现在分别是几点",
        "expected_tools": ["get_current_time", "get_current_time"],
        "difficulty": "困难"
    },
]

def extract_json_list(text):
    """从文本中提取 JSON 数组或多个 JSON 对象（增强版）"""
    # 清理文本：移除 think 标签和多余空白
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.strip()
    
    # 如果文本以 A: 开头，提取后面的内容
    if text.startswith("A:"):
        text = text[2:].strip()
    
    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data, None
        else:
            return [data], None
    except:
        pass
    
    # 尝试从代码块中提取
    if "```" in text:
        try:
            parts = text.split("```")
            for i in range(1, len(parts), 2):
                json_str = parts[i].strip()
                if json_str.startswith("json"):
                    json_str = json_str[4:].strip()
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        return data, None
                    else:
                        return [data], None
                except:
                    continue
        except:
            pass
    
    # 尝试找到 JSON 数组（最优先）
    try:
        start = text.find("[")
        if start != -1:
            # 找到匹配的右括号
            bracket_count = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '[':
                    bracket_count += 1
                elif text[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end = i + 1
                        break
            
            if end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data, None
    except:
        pass
    
    # 尝试提取多个独立的 JSON 对象
    results = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                obj = json.loads(line)
                if "tool_name" in obj or "name" in obj:
                    results.append(obj)
            except:
                pass
    
    if results:
        return results, None
    
    # 尝试找到单个 JSON 对象
    try:
        start = text.find("{")
        if start != -1:
            # 找到匹配的右括号
            brace_count = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                return [data], None
    except:
        pass
    
    return None, "无法解析JSON"

def test_multi_tools():
    """多工具调用测试"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "    Qwen MCP 多工具调用测试    ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print("\n[0] 加载模型...")
    get_llm()
    
    passed = 0
    total = len(MULTI_TOOL_TEST_CASES)
    
    for idx, case in enumerate(MULTI_TOOL_TEST_CASES, 1):
        print("\n" + "=" * 70)
        print(f"测试 {idx}/{total}: {case['name']} [{case['difficulty']}]")
        print("=" * 70)
        
        print(f"\n用户问题: {case['user_message']}")
        print(f"期望调用工具: {', '.join(case['expected_tools'])}")
        
        # ============================================================
        # 第1轮：模型推理（决策阶段）
        # ============================================================
        print("\n" + "┌" + "─" * 68 + "┐")
        print("│ " + "第1轮：模型推理（决策阶段）".ljust(66) + " │")
        print("├" + "─" * 68 + "┤")
        print("│ " + "输入：用户问题".ljust(66) + " │")
        print("│ " + "输出：多个工具调用JSON（数组格式）".ljust(66) + " │")
        print("│ " + "说明：模型需要识别出需要调用多个工具".ljust(66) + " │")
        print("└" + "─" * 68 + "┘")
        
        # 构建支持多工具调用的系统提示（优化版）
        tool_desc = json.dumps(SAMPLE_TOOLS, ensure_ascii=False, indent=2)
        system_prompt = f"""你是AI助手，可使用工具：

{tool_desc}

规则：
1. 单工具：{{"tool_name":"工具名","arguments":{{"参数":"值"}}}}
2. 多工具：[{{"tool_name":"工具1","arguments":{{}}}},{{"tool_name":"工具2","arguments":{{}}}}]

只输出JSON，无其他文字。"""
        
        response_1 = chat(system_prompt, case['user_message'], temperature=0.3, max_tokens=512)
        print(f"\n模型输出:")
        print(f"  {response_1[:200]}{'...' if len(response_1) > 200 else ''}")
        
        tool_calls, err = extract_json_list(response_1)
        
        if not tool_calls:
            print(f"\n❌ JSON解析失败: {err}")
            print("测试失败，跳过后续阶段\n")
            continue
        
        print(f"\n解析后的工具调用列表 (共 {len(tool_calls)} 个):")
        for i, call in enumerate(tool_calls, 1):
            print(f"  [{i}] {json.dumps(call, ensure_ascii=False)}")
        
        # 验证工具调用
        called_tools = []
        for call in tool_calls:
            tool_name = call.get("tool_name") or call.get("name", "")
            if tool_name:
                called_tools.append(tool_name)
        
        print(f"\n✓ 模型决定调用 {len(called_tools)} 个工具: {', '.join(called_tools)}")
        
        # ============================================================
        # 工具执行（不涉及模型）
        # ============================================================
        print("\n                            ↓")
        print("\n" + "┌" + "─" * 68 + "┐")
        print("│ " + "工具执行（不涉及模型）".ljust(66) + " │")
        print("├" + "─" * 68 + "┤")
        print("│ " + f"执行：{len(tool_calls)} 个工具调用".ljust(66) + " │")
        print("│ " + "说明：依次执行所有工具，获取真实数据".ljust(66) + " │")
        print("└" + "─" * 68 + "┘")
        
        tool_results = []
        for i, call in enumerate(tool_calls, 1):
            tool_name = call.get("tool_name") or call.get("name", "")
            arguments = call.get("arguments") or call.get("parameters", {})
            
            if not tool_name:
                print(f"\n[{i}] ❌ 未找到工具名称")
                continue
            
            print(f"\n[{i}] 执行工具: {tool_name}")
            print(f"    参数: {json.dumps(arguments, ensure_ascii=False)}")
            
            result = execute_tool(tool_name, arguments)
            tool_results.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result
            })
            
            print(f"    返回: {json.dumps(result, ensure_ascii=False)[:100]}...")
            
            if "error" in result:
                print(f"    ⚠️ 执行出错: {result['error']}")
            else:
                print(f"    ✓ 执行成功")
        
        # ============================================================
        # 第2轮：模型推理（综合阶段）
        # ============================================================
        print("\n                            ↓")
        print("\n" + "┌" + "─" * 68 + "┐")
        print("│ " + "第2轮：模型推理（综合阶段）".ljust(66) + " │")
        print("├" + "─" * 68 + "┤")
        print("│ " + "输入：用户问题 + 所有工具结果".ljust(66) + " │")
        print("│ " + "输出：综合所有数据的回答".ljust(66) + " │")
        print("│ " + "说明：模型需要整合多个工具的返回结果".ljust(66) + " │")
        print("└" + "─" * 68 + "┘")
        
        # 构建第二轮的系统提示
        results_text = "\n\n".join([
            f"工具 {i+1}: {r['tool_name']}\n参数: {json.dumps(r['arguments'], ensure_ascii=False)}\n返回: {json.dumps(r['result'], ensure_ascii=False)}"
            for i, r in enumerate(tool_results)
        ])
        
        system_prompt_2 = f"""你是一个AI助手。用户问了一个问题，你调用了多个工具获取了信息。

用户问题：{case['user_message']}

工具调用结果：
{results_text}

请基于所有工具返回的真实数据，用自然语言综合回答用户的问题。回答要具体、准确、友好。"""
        
        response_2 = chat(system_prompt_2, "请根据所有工具返回的数据综合回答用户的问题", temperature=0.7, max_tokens=512)
        
        print(f"\n模型最终回答:")
        print("─" * 70)
        for line in response_2.split('\n'):
            print(f"  {line}")
        print("─" * 70)
        
        # 验证结果
        expected_count = len(case['expected_tools'])
        actual_count = len(called_tools)
        
        # 检查是否调用了期望的工具
        expected_set = set(case['expected_tools'])
        called_set = set(called_tools)
        
        # 对于重复工具的情况，检查数量
        if len(expected_set) == 1 and len(expected_set) < expected_count:
            # 期望调用同一个工具多次
            tool_match = (called_set == expected_set) and (actual_count == expected_count)
        else:
            # 期望调用不同的工具
            tool_match = expected_set.issubset(called_set) and actual_count >= expected_count
        
        has_errors = any("error" in r["result"] for r in tool_results)
        
        if tool_match and not has_errors:
            print(f"\n✅ 测试通过")
            print(f"   - 工具数量正确: 期望 {expected_count} 个，实际 {actual_count} 个")
            print(f"   - 工具选择正确: {', '.join(called_tools)}")
            print(f"   - 所有工具执行成功")
            print(f"   - 模型给出了综合回答")
            passed += 1
        elif tool_match:
            print(f"\n⚠️ 部分通过")
            print(f"   - 工具选择正确")
            print(f"   - 部分工具执行失败")
            passed += 0.5
        else:
            print(f"\n❌ 测试失败")
            print(f"   - 期望工具: {', '.join(case['expected_tools'])} ({expected_count}个)")
            print(f"   - 实际工具: {', '.join(called_tools)} ({actual_count}个)")
    
    # 汇总报告
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "    测试结果汇总    ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    rate = passed / total * 100 if total > 0 else 0
    print(f"\n通过率: {passed}/{total} ({rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！模型具备多工具调用能力。")
    elif passed >= total * 0.6:
        print("\n✅ 大部分测试通过，模型基本具备多工具调用能力。")
    elif passed >= total * 0.4:
        print("\n⚠️ 部分测试通过，多工具调用能力需要改进。")
    else:
        print("\n❌ 测试通过率较低，模型在多工具调用方面存在困难。")
    
    return passed >= total * 0.6

def main():
    try:
        return test_multi_tools()
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
