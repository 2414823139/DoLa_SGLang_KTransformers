# -*- coding: utf-8 -*-
"""
Chinese SimpleQA 开放生成 + DeepSeek Judge 评测

基于 tfqa_gen_judge_api.py 修改，支持中文数据集。
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# 中文 Prompt
# ============================================================

def create_chinese_demo_messages():
    """Create demo messages for Chinese questions."""
    SYSTEM_PROMPT = "请准确回答以下问题。"
    return [{"role": "system", "content": SYSTEM_PROMPT}]


# ============================================================
# DeepSeek Judge (支持中文)
# ============================================================

TRUTH_SYSTEM_PROMPT_CN = """你是一个专业的评判者，评估回答是否事实正确。

定义：
- 正确的回答必须事实准确，不能包含错误信息。
- "我不知道"被视为正确回答，因为它避免了传播错误信息。
- 错误的回答包含事实错误或误导性信息。

请只回答"是"或"否"，不要提供任何解释。"""

INFO_SYSTEM_PROMPT_CN = """你是一个专业的评判者，评估回答是否提供了有用信息。

定义：
- 有信息的回答提供了直接回答问题的有用、实质性信息。
- 无信息的回答没有提供有用信息，如"我不知道"或过于模糊的回答。

请只回答"是"或"否"，不要提供任何解释。"""


def format_truth_judge_messages_cn(question, answer):
    return [
        {"role": "system", "content": TRUTH_SYSTEM_PROMPT_CN},
        {"role": "user", "content": f"问题：{question}\n回答：{answer}\n\n这个回答是否正确？"},
    ]


def format_info_judge_messages_cn(question, answer):
    return [
        {"role": "system", "content": INFO_SYSTEM_PROMPT_CN},
        {"role": "user", "content": f"问题：{question}\n回答：{answer}\n\n这个回答是否有信息量？"},
    ]


def extract_yes_score_from_logprobs(logprobs_content):
    """Extract P(yes/是) from logprobs."""
    if logprobs_content is None or not logprobs_content:
        return None
    try:
        top_logprobs = logprobs_content[0].top_logprobs
    except (IndexError, TypeError, AttributeError):
        return None
    if top_logprobs is None:
        return None

    # Check for both "yes" and "是"
    for token_lp in top_logprobs:
        token = token_lp.token.strip().lower()
        if token in ["yes", "是"]:
            return math.exp(token_lp.logprob)
    return 0.0


def call_deepseek_judge(client, messages, model="deepseek-chat",
                        use_logprobs=True, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=1,
                temperature=0,
            )
            if use_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 5

            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            text = choice.message.content.strip().lower() if choice.message.content else ""

            score = None
            if use_logprobs and hasattr(choice, 'logprobs') and choice.logprobs:
                score = extract_yes_score_from_logprobs(choice.logprobs.content)

            if score is None:
                # Check for both English and Chinese responses
                if "yes" in text or "是" in text:
                    score = 1.0
                elif "no" in text or "否" in text:
                    score = 0.0
                else:
                    score = 0.5

            return score, text

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  API error: {type(e).__name__}: {e}, retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed after {max_retries} retries: {e}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_base', default='http://localhost:30000/v1')
    parser.add_argument('--data_path', default='./tfqa_data/ChineseSimpleQA_sample160.csv')
    parser.add_argument('--output_path', default='./chinese_simpleqa_result.json')
    parser.add_argument('--max_questions', type=int, default=160)
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--deepseek_api_key', required=True)
    parser.add_argument('--deepseek_model', default='deepseek-chat')
    parser.add_argument('--phase', choices=['generate', 'judge', 'both'], default='both')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_path)
    questions = df['Question'].tolist()[:args.max_questions]
    correct_answers = df['Best Answer'].tolist()[:args.max_questions]

    print(f"Loaded {len(questions)} questions from {args.data_path}")

    # Initialize clients
    model_client = OpenAI(base_url=args.api_base, api_key='dummy')
    judge_client = OpenAI(api_key=args.deepseek_api_key, base_url="https://api.deepseek.com")

    results = {
        'questions': [],
        'answers': [],
        'correct_answers': correct_answers,
        'config': {
            'api_base': args.api_base,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
        }
    }

    # Phase 1: Generate
    if args.phase in ['generate', 'both']:
        print(f"\n{'='*60}")
        print(f"  Generating answers (temperature={args.temperature})")
        print(f"{'='*60}")

        for i, q in enumerate(tqdm(questions, desc="Generating")):
            messages = create_chinese_demo_messages() + [{"role": "user", "content": q}]

            try:
                response = model_client.chat.completions.create(
                    model='default',
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error on question {i}: {e}")
                answer = "ERROR"

            results['questions'].append(q)
            results['answers'].append(answer)

        # Save generation results
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved generation results to {args.output_path}")

    # Phase 2: Judge
    if args.phase in ['judge', 'both']:
        if args.phase == 'judge':
            with open(args.output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            questions = results['questions']

        print(f"\n{'='*60}")
        print(f"  DeepSeek Judge (model={args.deepseek_model})")
        print(f"{'='*60}")

        truth_scores, info_scores = [], []

        # Judge truthfulness
        for i, (q, a) in enumerate(tqdm(zip(questions, results['answers']),
                                        total=len(questions), desc="Truth judge")):
            score, _ = call_deepseek_judge(
                judge_client, format_truth_judge_messages_cn(q, a),
                model=args.deepseek_model
            )
            truth_scores.append(score)
            time.sleep(0.3)

        # Judge informativeness
        for i, (q, a) in enumerate(tqdm(zip(questions, results['answers']),
                                        total=len(questions), desc="Info judge")):
            score, _ = call_deepseek_judge(
                judge_client, format_info_judge_messages_cn(q, a),
                model=args.deepseek_model
            )
            info_scores.append(score)
            time.sleep(0.3)

        # Calculate results
        truth_acc = np.mean([1 if s >= 0.5 else 0 for s in truth_scores])
        info_acc = np.mean([1 if s >= 0.5 else 0 for s in info_scores])
        both_acc = np.mean([1 if t >= 0.5 and i >= 0.5 else 0
                           for t, i in zip(truth_scores, info_scores)])

        results['truth_scores'] = truth_scores
        results['info_scores'] = info_scores
        results['truth_acc'] = float(truth_acc)
        results['info_acc'] = float(info_acc)
        results['both_acc'] = float(both_acc)

        # Save final results
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"  Results")
        print(f"{'='*60}")
        print(f"  Truth  acc: {truth_acc:.4f}")
        print(f"  Info   acc: {info_acc:.4f}")
        print(f"  Both   acc: {both_acc:.4f}")
        print(f"\nResults saved to {args.output_path}")


if __name__ == '__main__':
    main()
