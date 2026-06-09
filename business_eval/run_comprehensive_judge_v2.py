"""
综合 LLM-as-Judge 评测 v2

改进：
  1. 包含完整产品资料（不截断）
  2. 调整幻觉定义：温馨提示、安全提示、公共固有知识、帮助引导不计入幻觉

对比：
  1. 业务模型（Expected）vs Baseline
  2. 业务模型（Expected）vs DoLa-Low

评测维度：
  - 幻觉检测：是否包含产品资料/对话历史中没有的专业技术信息
  - 参考文档正确性：是否正确引用了产品资料
  - 连贯性：回答是否逻辑清晰、条理分明
  - 客服规范：是否符合客服身份、用语礼貌
  - 系统指令遵循：是否遵循了系统提示中的各项规则
"""

import json
import csv
import re
import time
from openai import OpenAI

# DeepSeek API client
client = OpenAI(api_key="sk-691de00b229546a78be33c9d5f4fbd1c", base_url="https://api.deepseek.com")


def extract_product_docs(user_msg):
    """Extract product documents from user message."""
    idx = user_msg.find('当前问题产品资料：')
    if idx < 0:
        return []
    
    docs_section = user_msg[idx + len('当前问题产品资料：'):]
    end_idx = docs_section.find('##对话历史')
    if end_idx >= 0:
        docs_str = docs_section[:end_idx].strip()
    else:
        docs_str = docs_section.strip()
    
    if docs_str.startswith('{[') and docs_str.endswith(']}'):
        docs_str = docs_str[1:-1]
    
    try:
        import ast
        docs = ast.literal_eval(docs_str)
        return docs
    except:
        return []


def extract_chat_history(user_msg):
    """Extract chat history from user message."""
    idx = user_msg.find('##对话历史')
    if idx < 0:
        return []
    
    chat_str = user_msg[idx + len('##对话历史'):].strip()
    if chat_str.startswith(':'):
        chat_str = chat_str[1:].strip()
    
    bracket_count = 0
    start_idx = -1
    for i, c in enumerate(chat_str):
        if c == '[':
            if bracket_count == 0:
                start_idx = i
            bracket_count += 1
        elif c == ']':
            bracket_count -= 1
            if bracket_count == 0 and start_idx >= 0:
                array_str = chat_str[start_idx:i+1]
                try:
                    chat = json.loads(array_str)
                    return chat
                except:
                    return []
    return []


def extract_system_prompt(messages):
    """Extract system prompt from messages."""
    for msg in messages:
        if msg.get('role') == 'system':
            return msg.get('content', '')
    return ''


def judge_comparison(product_docs, chat_history, system_prompt, answer_ref, answer_test, ref_name, test_name):
    """
    Compare two answers using DeepSeek API.
    """
    
    # 完整的产品资料，不截断
    docs_text = "\n\n".join([doc.get('content', '').strip() for doc in product_docs])
    
    # Format chat history（完整，不截断）
    chat_text = ""
    if chat_history:
        for msg in chat_history:
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except:
                    continue
            if isinstance(msg, dict):
                role = msg.get('role', '')
                text = msg.get('text', '')
                chat_text += f"{role}: {text}\n"
    
    prompt = f"""你是一位专业的客服质检专家。你的任务是对比两个AI客服回答的质量。

## 评测背景
我们有一个业务模型生成的标准回答，现在需要评估另一个模型的回答质量是否优于、等于或劣于业务模型。

## 产品资料（完整）
{docs_text}

## 对话历史（完整）
{chat_text if chat_text else "（无）"}

## 系统指令（完整）
{system_prompt}

## 业务模型回答（{ref_name}）
{answer_ref}

## 待评测回答（{test_name}）
{answer_test}

## 评测维度

请从以下5个维度进行评分（每项1-5分，5分最高）：

### 1. 幻觉检测（hallucination）
**重要说明**：以下内容**不计入幻觉扣分**：
- 温馨提示、安全提示（如"自行安装需注意安全"）
- 公共固有知识（如常识性操作、通用礼貌用语）
- 帮助引导语（如"还有什么可以帮助您的吗"、"欢迎继续咨询"）
- 客服安抚话语、礼貌用语

幻觉仅指：回答中包含了**产品资料或对话历史中没有的专业/技术信息**，且不属于上述豁免类别。

评分标准：
- 5分：无幻觉（或仅有豁免内容），所有专业技术信息都有据可查
- 4分：轻微问题，有不影响核心内容的技术推断
- 3分：中等问题，有少量编造但不影响主要结论
- 2分：较严重，编造了关键技术细节
- 1分：严重幻觉，大量编造专业技术信息

### 2. 参考文档正确性（doc_accuracy）
- 5分：完全正确引用产品资料，无遗漏无曲解
- 4分：基本正确，有小瑕疵（如遗漏非核心信息）
- 3分：部分正确，有遗漏或轻微曲解
- 2分：较多错误或遗漏重要信息
- 1分：严重错误或完全未参考产品资料

### 3. 连贯性（coherence）
- 5分：逻辑清晰，结构完整，条理分明
- 4分：基本连贯，有轻微不足
- 3分：一般，有逻辑跳跃但不影响理解
- 2分：较差，结构混乱
- 1分：极差，无法理解

### 4. 客服规范（customer_service）
- 5分：完全符合客服身份，用语礼貌专业，有适当的关怀引导
- 4分：基本符合，有轻微不足
- 3分：一般，有不够专业的地方
- 2分：较差，用语不当或态度问题
- 1分：极差，严重违反客服规范

### 5. 系统指令遵循（instruction_following）
**重要说明**：以下内容**不计入扣分**：
- 添加温馨提示、安全提示（即使资料中未提及，只要是合理的安全提醒）
- 添加帮助引导语（如"还有什么可以帮助您的吗"）
- 使用礼貌用语、客服安抚话语

评分标准：
- 5分：完全遵循所有系统指令，或仅有上述豁免内容
- 4分：基本遵循，有轻微遗漏但不影响核心
- 3分：部分遵循，有较明显遗漏
- 2分：较多违反系统指令
- 1分：严重违反系统指令

## 输出格式

请以JSON格式输出：
{{
    "scores": {{
        "{ref_name}": {{
            "hallucination": 1-5,
            "doc_accuracy": 1-5,
            "coherence": 1-5,
            "customer_service": 1-5,
            "instruction_following": 1-5,
            "total": 总分
        }},
        "{test_name}": {{
            "hallucination": 1-5,
            "doc_accuracy": 1-5,
            "coherence": 1-5,
            "customer_service": 1-5,
            "instruction_following": 1-5,
            "total": 总分
        }}
    }},
    "winner": "{ref_name}" 或 "{test_name}" 或 "tie",
    "reason": "判断理由，说明为什么这个回答更好（或为什么平局）",
    "key_differences": [
        "差异1：...",
        "差异2：...",
        "差异3：..."
    ]
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": "你是一位专业的客服质检专家，需要客观、公正地评测两个AI客服回答的质量。请严格按照评测维度打分，特别注意：温馨提示、安全提示、公共固有知识、帮助引导语不计入幻觉扣分和指令遵循扣分。"
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=3000
        )
        
        content = response.choices[0].message.content
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return {
                "error": "无法解析API响应",
                "raw_content": content[:500]
            }
    except Exception as e:
        return {
            "error": f"API错误: {str(e)}"
        }


def main():
    # Read data
    print("读取数据...")
    with open('/workspace/datasets/data.csv', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Read merged results
    with open('/workspace/datasets/merged_results.json') as f:
        merged = json.load(f)
    
    # Results storage
    results = {
        "config": {
            "ref_model": "业务模型(Expected)",
            "test_models": ["Baseline", "DoLa-Low"],
            "total_samples": len(merged['samples']),
            "version": "v2",
            "improvements": ["完整产品资料", "温馨提示/安全提示/帮助引导不计入幻觉"]
        },
        "stats": {
            "Baseline_vs_Expected": {"baseline_wins": 0, "expected_wins": 0, "tie": 0},
            "DoLa_vs_Expected": {"dola_wins": 0, "expected_wins": 0, "tie": 0}
        },
        "score_summary": {
            "Baseline": {"hallucination": [], "doc_accuracy": [], "coherence": [], "customer_service": [], "instruction_following": [], "total": []},
            "DoLa-Low": {"hallucination": [], "doc_accuracy": [], "coherence": [], "customer_service": [], "instruction_following": [], "total": []},
            "Expected": {"hallucination": [], "doc_accuracy": [], "coherence": [], "customer_service": [], "instruction_following": [], "total": []}
        },
        "details": []
    }
    
    print(f"共 {len(merged['samples'])} 个样本")
    
    for i, sample in enumerate(merged['samples']):
        print(f"\n处理样本 {i+1}/200...")
        
        # Get context
        row = rows[i]
        inputs = json.loads(row['inputs'])
        user_msg = inputs[1]['content'] if len(inputs) > 1 else ""
        system_prompt = extract_system_prompt(inputs)
        
        product_docs = extract_product_docs(user_msg)
        chat_history = extract_chat_history(user_msg)
        
        expected = sample['expected']
        baseline = sample['baseline_generated']
        dola = sample['dola_generated']
        
        detail = {
            "index": i,
            "title": sample['title']
        }
        
        # Comparison 1: Expected vs Baseline
        print(f"  评测 Expected vs Baseline...")
        result1 = judge_comparison(
            product_docs, chat_history, system_prompt,
            expected, baseline,
            "Expected", "Baseline"
        )
        detail["expected_vs_baseline"] = result1
        
        # Update stats
        if "winner" in result1:
            if result1["winner"] == "Baseline":
                results["stats"]["Baseline_vs_Expected"]["baseline_wins"] += 1
            elif result1["winner"] == "Expected":
                results["stats"]["Baseline_vs_Expected"]["expected_wins"] += 1
            else:
                results["stats"]["Baseline_vs_Expected"]["tie"] += 1
        
        # Update scores for result1 (Expected and Baseline)
        if "scores" in result1:
            for model in ["Expected", "Baseline"]:
                if model in result1["scores"]:
                    for dim in ["hallucination", "doc_accuracy", "coherence", "customer_service", "instruction_following", "total"]:
                        if dim in result1["scores"][model]:
                            results["score_summary"][model][dim].append(result1["scores"][model][dim])
        
        time.sleep(0.5)  # Rate limiting
        
        # Comparison 2: Expected vs DoLa-Low
        print(f"  评测 Expected vs DoLa-Low...")
        result2 = judge_comparison(
            product_docs, chat_history, system_prompt,
            expected, dola,
            "Expected", "DoLa-Low"
        )
        detail["expected_vs_dola"] = result2
        
        # Update stats
        if "winner" in result2:
            if result2["winner"] == "DoLa-Low":
                results["stats"]["DoLa_vs_Expected"]["dola_wins"] += 1
            elif result2["winner"] == "Expected":
                results["stats"]["DoLa_vs_Expected"]["expected_wins"] += 1
            else:
                results["stats"]["DoLa_vs_Expected"]["tie"] += 1
        
        # Update scores for result2 (Expected and DoLa-Low)
        if "scores" in result2:
            for model in ["Expected", "DoLa-Low"]:
                if model in result2["scores"]:
                    for dim in ["hallucination", "doc_accuracy", "coherence", "customer_service", "instruction_following", "total"]:
                        if dim in result2["scores"][model]:
                            results["score_summary"][model][dim].append(result2["scores"][model][dim])
        
        results["details"].append(detail)
        
        # Save after each sample
        with open('/workspace/datasets/comprehensive_judge_results_v2.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"  完成! Baseline胜={results['stats']['Baseline_vs_Expected']['baseline_wins']}, DoLa胜={results['stats']['DoLa_vs_Expected']['dola_wins']}")
        
        time.sleep(0.5)  # Rate limiting
    
    # Calculate final averages
    print("\n计算最终统计...")
    for model in ["Baseline", "DoLa-Low", "Expected"]:
        for dim in results["score_summary"][model]:
            scores = results["score_summary"][model][dim]
            if scores:
                results["score_summary"][model][dim] = {
                    "mean": sum(scores) / len(scores),
                    "count": len(scores)
                }
    
    # Save final results
    with open('/workspace/datasets/comprehensive_judge_results_v2.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("评测完成!")
    print("="*80)
    print(f"\nExpected vs Baseline:")
    print(f"  Baseline 胜: {results['stats']['Baseline_vs_Expected']['baseline_wins']}")
    print(f"  Expected 胜: {results['stats']['Baseline_vs_Expected']['expected_wins']}")
    print(f"  平局: {results['stats']['Baseline_vs_Expected']['tie']}")
    
    print(f"\nExpected vs DoLa-Low:")
    print(f"  DoLa-Low 胜: {results['stats']['DoLa_vs_Expected']['dola_wins']}")
    print(f"  Expected 胜: {results['stats']['DoLa_vs_Expected']['expected_wins']}")
    print(f"  平局: {results['stats']['DoLa_vs_Expected']['tie']}")
    
    print(f"\n结果已保存到: /workspace/datasets/comprehensive_judge_results_v2.json")


if __name__ == "__main__":
    main()
