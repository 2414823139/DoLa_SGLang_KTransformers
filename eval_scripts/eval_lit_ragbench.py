# -*- coding: utf-8 -*-
"""
LIT-RAGBench 评测脚本

测试模型在 RAG 场景下的能力：
1. 从 positive chunks 中提取正确信息
2. 不被 negative chunks 干扰
3. 进行多步推理、计算等复杂任务
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from openai import OpenAI


def create_rag_prompt(chunks, question):
    """Create RAG prompt with retrieved chunks."""
    # Combine chunks into context
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"Document {i+1}:\n"
        context += f"Title: {chunk.get('title', '')}\n"
        context += f"Content: {chunk['content']}\n\n"

    system_prompt = """You are a helpful assistant that answers questions based on the provided documents.

Rules:
1. Answer the question using ONLY information from the provided documents
2. If the documents don't contain enough information, say "I cannot answer based on the provided documents"
3. Be precise and concise
4. Show your reasoning when needed"""

    user_prompt = f"""Documents:
{context}

Question: {question}

Answer:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def judge_answer(client, question, model_answer, correct_answer, judge_model="deepseek-chat"):
    """Judge if the answer is correct."""
    judge_prompt = """You are a judge evaluating whether a model's answer is correct.

Rules:
1. The answer should contain the key information from the reference answer
2. Minor wording differences are acceptable
3. Numerical answers should match exactly or be equivalent

Answer only "Yes" or "No"."""

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": f"""Question: {question}
Reference Answer: {correct_answer}
Model Answer: {model_answer}

Is the model's answer correct?"""}
    ]

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=messages,
            max_tokens=10,
            temperature=0,
        )
        text = response.choices[0].message.content.strip().lower()
        if "yes" in text:
            return 1.0
        elif "no" in text:
            return 0.0
        return 0.5
    except Exception as e:
        print(f"Judge error: {e}")
        return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_base', default='http://localhost:30000/v1')
    parser.add_argument('--data_path', default='/workspace/LIT-RAGBench/datasets/en.jsonl')
    parser.add_argument('--output_path', default='./lit_ragbench_result.json')
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--deepseek_api_key', required=True)
    parser.add_argument('--use_negative', action='store_true', help='Include negative chunks as distractors')
    parser.add_argument('--phase', choices=['generate', 'judge', 'both'], default='both')
    args = parser.parse_args()

    # Load data
    data = []
    with open(args.data_path) as f:
        for line in f:
            data.append(json.loads(line))

    if args.max_samples > 0 and args.max_samples < len(data):
        np.random.seed(42)
        indices = np.random.choice(len(data), args.max_samples, replace=False)
        data = [data[i] for i in sorted(indices)]

    print(f"Loaded {len(data)} samples from {args.data_path}")
    print(f"Use negative chunks: {args.use_negative}")

    # Initialize clients
    model_client = OpenAI(base_url=args.api_base, api_key='dummy')
    judge_client = OpenAI(api_key=args.deepseek_api_key, base_url="https://api.deepseek.com")

    results = {
        'config': {
            'api_base': args.api_base,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'use_negative': args.use_negative,
        },
        'samples': [],
    }

    # Phase 1: Generate
    if args.phase in ['generate', 'both']:
        print(f"\n{'='*60}")
        print(f"  Generating answers (temperature={args.temperature})")
        print(f"{'='*60}")

        for i, sample in enumerate(tqdm(data, desc="Generating")):
            # Prepare chunks: positive only or positive + negative
            if args.use_negative:
                chunks = sample['positive_chunk_list'] + sample['negative_chunk_list']
            else:
                chunks = sample['positive_chunk_list']

            messages = create_rag_prompt(chunks, sample['question'])

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
                print(f"Error on sample {i}: {e}")
                answer = "ERROR"

            results['samples'].append({
                'index': i,
                'question': sample['question'],
                'qa_type': sample['qa_type'],
                'answer': sample['answer'],
                'model_answer': answer,
                'reasoning_content': sample.get('reasoning_content', ''),
                'positive_chunks': len(sample['positive_chunk_list']),
                'negative_chunks': len(sample['negative_chunk_list']),
            })

        # Save generation results
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved generation results to {args.output_path}")

    # Phase 2: Judge
    if args.phase in ['judge', 'both']:
        if args.phase == 'judge':
            with open(args.output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

        print(f"\n{'='*60}")
        print(f"  Judging answers")
        print(f"{'='*60}")

        scores = []
        for sample in tqdm(results['samples'], desc="Judging"):
            score = judge_answer(judge_client, sample['question'], sample['model_answer'], sample['answer'])
            sample['score'] = score
            scores.append(score)

        # Calculate results by qa_type
        qa_type_scores = {}
        for sample in results['samples']:
            qa_type = str(sample['qa_type'])
            if qa_type not in qa_type_scores:
                qa_type_scores[qa_type] = []
            qa_type_scores[qa_type].append(sample['score'])

        results['overall_acc'] = float(np.mean([1 if s >= 0.5 else 0 for s in scores]))
        results['avg_score'] = float(np.mean(scores))
        results['qa_type_results'] = {k: float(np.mean(v)) for k, v in qa_type_scores.items()}

        # Save final results
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"  Results")
        print(f"{'='*60}")
        print(f"  Overall Accuracy: {results['overall_acc']:.4f} ({int(results['overall_acc']*len(scores))}/{len(scores)})")
        print(f"  Average Score: {results['avg_score']:.4f}")
        print(f"\nBy QA Type:")
        for qa_type, score in sorted(results['qa_type_results'].items()):
            print(f"  {qa_type}: {score:.4f}")
        print(f"\nResults saved to {args.output_path}")


if __name__ == '__main__':
    main()