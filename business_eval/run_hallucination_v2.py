import json
import csv
import re
import os
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

def judge_hallucination(product_docs, chat_history, answer1, answer2):
    """Judge if answers contain hallucinations using DeepSeek API."""
    
    docs_text = "\n\n".join([doc.get('content', '').strip() for doc in product_docs])
    
    # Format chat history
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
    
    prompt = f"""你是一位严格的内容审核专家。你的任务是判断两个AI生成的回答中是否包含"幻觉"（hallucination）。

## 幻觉定义
在客服场景中，幻觉是指：回答中包含了**产品资料或对话历史中没有的专业/技术信息**。

**重要说明**：
1. 安全提示、客服安抚话语、礼貌用语**不计入幻觉**
2. 如果产品资料或对话历史中**已经提到了某个概念或操作**，回答中对其进行合理的展开解释**不算幻觉**
3. 只有当回答中**明确出现了产品资料和对话历史中都没有的专业/技术信息**时，才判定为幻觉
4. 对话历史中的信息**同样属于已知信息**，回答中引用对话历史中的内容**不算幻觉**

## 产品资料
{docs_text}

## 对话历史
{chat_text if chat_text else "（无）"}

## 回答A
{answer1}

## 回答B
{answer2}

## 任务
请分别判断回答A和回答B中是否包含幻觉（即是否编造了产品资料或对话历史中没有的专业/技术信息）。

请以JSON格式输出：
{{
    "answer_a": {{
        "has_hallucination": true/false,
        "hallucinated_content": "如果有幻觉，请摘录幻觉内容；否则为空字符串",
        "reason": "判断理由，必须明确说明 hallucinated_content 中的内容为什么在产品资料和对话历史中找不到"
    }},
    "answer_b": {{
        "has_hallucination": true/false,
        "hallucinated_content": "如果有幻觉，请摘录幻觉内容；否则为空字符串",
        "reason": "判断理由，必须明确说明 hallucinated_content 中的内容为什么在产品资料和对话历史中找不到"
    }}
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严格的内容审核专家，专门检测AI回答中的幻觉内容。请仔细阅读产品资料和对话历史，只有当回答中明确出现了两者都没有的专业/技术信息时，才判定为幻觉。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return {
                "answer_a": {"has_hallucination": False, "hallucinated_content": "", "reason": "无法解析API响应"},
                "answer_b": {"has_hallucination": False, "hallucinated_content": "", "reason": "无法解析API响应"}
            }
    except Exception as e:
        print(f"API error: {e}")
        return {
            "answer_a": {"has_hallucination": False, "hallucinated_content": "", "reason": f"API错误: {str(e)}"},
            "answer_b": {"has_hallucination": False, "hallucinated_content": "", "reason": f"API错误: {str(e)}"}
        }

def main():
    # Read data
    with open('/workspace/datasets/data.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Read merged results (dola_low vs baseline)
    with open('/workspace/datasets/merged_results.json') as f:
        merged = json.load(f)
    
    # Process all samples
    results = {
        "stats": {},
        "details": []
    }
    
    for i, sample in enumerate(merged['samples']):
        # Get product docs and chat history
        row = rows[i]
        inputs = json.loads(row['inputs'])
        user_msg = inputs[1]['content']
        
        product_docs = extract_product_docs(user_msg)
        chat_history = extract_chat_history(user_msg)
        
        answer1 = sample['dola_generated']
        answer2 = sample['baseline_generated']
        
        result = judge_hallucination(product_docs, chat_history, answer1, answer2)
        
        detail = {
            "index": i,
            "title": sample['title'],
            "dola_has_hallucination": result['answer_a']['has_hallucination'],
            "dola_hallucinated_content": result['answer_a']['hallucinated_content'],
            "dola_reason": result['answer_a']['reason'],
            "baseline_has_hallucination": result['answer_b']['has_hallucination'],
            "baseline_hallucinated_content": result['answer_b']['hallucinated_content'],
            "baseline_reason": result['answer_b']['reason']
        }
        
        results['details'].append(detail)
        
        # Update stats
        dola_hall = sum(1 for d in results['details'] if d['dola_has_hallucination'])
        base_hall = sum(1 for d in results['details'] if d['baseline_has_hallucination'])
        both_hall = sum(1 for d in results['details'] if d['dola_has_hallucination'] and d['baseline_has_hallucination'])
        neither_hall = sum(1 for d in results['details'] if not d['dola_has_hallucination'] and not d['baseline_has_hallucination'])
        only_dola = sum(1 for d in results['details'] if d['dola_has_hallucination'] and not d['baseline_has_hallucination'])
        only_base = sum(1 for d in results['details'] if not d['dola_has_hallucination'] and d['baseline_has_hallucination'])
        
        results['stats'] = {
            "dola_hallucination": dola_hall,
            "baseline_hallucination": base_hall,
            "both_hallucination": both_hall,
            "neither_hallucination": neither_hall,
            "only_dola_hallucination": only_dola,
            "only_baseline_hallucination": only_base
        }
        
        # Save after each sample
        with open('/workspace/datasets/hallucination_results_v2.json', 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Processed sample {i+1}/200, dola_hall={dola_hall}, base_hall={base_hall}")
        time.sleep(0.5)  # Rate limiting
    
    print("Done!")

if __name__ == "__main__":
    main()
