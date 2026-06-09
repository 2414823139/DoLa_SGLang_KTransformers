"""
RAGAS Official Evaluation v2 - Improved with actual user questions
- Uses actual user question (##当前用户问题) instead of title
- Supports DoLa-low, DoLa-mid, and Baseline comparison
- ragas 0.2.14 with DeepSeek API + bge-small-zh-v1.5 embeddings
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_URL'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')

import json
import csv
import ast
import traceback
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

def extract_product_docs(user_msg):
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
        docs = ast.literal_eval(docs_str)
        return [doc.get('content', '').strip() for doc in docs]
    except:
        return []

def extract_user_question(user_msg):
    """Extract actual user question from ##当前用户问题:"""
    idx = user_msg.find('##当前用户问题:')
    if idx < 0:
        return ''
    question_section = user_msg[idx + len('##当前用户问题:'):]
    end_idx = question_section.find('\n##')
    if end_idx >= 0:
        question = question_section[:end_idx].strip()
    else:
        question = question_section.strip()
    return question

def prepare_dataset(merged_path, data_csv_path, answer_key):
    with open(data_csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    with open(merged_path) as f:
        merged = json.load(f)
    
    questions, answers, contexts_list, references = [], [], [], []
    skipped = 0
    for i, sample in enumerate(merged['samples']):
        row = rows[i]
        inputs = json.loads(row['inputs'])
        user_msg = inputs[1]['content']
        ctxs = extract_product_docs(user_msg)
        answer = sample[answer_key]
        reference = sample.get('expected', '')
        # Use actual user question instead of title
        question = extract_user_question(user_msg)
        if not question:
            question = sample.get('title', '')
        if not ctxs or not answer or not reference:
            skipped += 1
            continue
        questions.append(question)
        answers.append(answer)
        contexts_list.append(ctxs)
        references.append(reference)
    print(f"Prepared {len(questions)} samples (skipped {skipped})")
    if questions:
        print(f"Sample question: {questions[0][:80]}")
    return Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts': contexts_list,
        'reference': references,
    })

def extract_metric_value(result, metric_name):
    """ragas 0.2.x returns a list of per-sample scores for each metric.
    The printed result shows the mean, so we compute the mean here."""
    val = result[metric_name]
    if isinstance(val, list):
        return sum(val) / len(val) if val else 0.0
    elif hasattr(val, 'mean'):  # pandas Series
        return float(val.mean())
    elif hasattr(val, 'item'):  # numpy scalar
        return float(val.item())
    else:
        return float(val)

def main():
    import sys
    # Accept mode: dola_low, dola_mid, baseline, all
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key="sk-691de00b229546a78be33c9d5f4fbd1c",
        base_url="https://api.deepseek.com",
        temperature=0.0,
        max_tokens=4096,
    )
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-zh-v1.5')
    
    results = {}
    
    # Load merged results for DoLa-low (original)
    if mode in ('dola_low', 'all'):
        print("Preparing DoLa-low dataset...")
        dola_low_dataset = prepare_dataset(
            '/workspace/datasets/merged_results.json',
            '/workspace/datasets/data.csv',
            'dola_generated'
        )
        print(f"\nDoLa-low samples: {len(dola_low_dataset)}")
        print("=== Evaluating DoLa-low ===")
        dola_low_result = evaluate(
            dola_low_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )
        print(f"DoLa-low results: {dola_low_result}")
        results['dola_low'] = {
            'faithfulness': extract_metric_value(dola_low_result, 'faithfulness'),
            'answer_relevancy': extract_metric_value(dola_low_result, 'answer_relevancy'),
        }
    
    # Load DoLa-mid results (uses merged_dola_mid_results.json, field is dola_generated)
    if mode in ('dola_mid', 'all'):
        print("\nPreparing DoLa-mid dataset...")
        dola_mid_dataset = prepare_dataset(
            '/workspace/datasets/merged_dola_mid_results.json',
            '/workspace/datasets/data.csv',
            'dola_generated'  # DoLa-mid results use same field name
        )
        print(f"\nDoLa-mid samples: {len(dola_mid_dataset)}")
        print("=== Evaluating DoLa-mid ===")
        dola_mid_result = evaluate(
            dola_mid_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )
        print(f"DoLa-mid results: {dola_mid_result}")
        results['dola_mid'] = {
            'faithfulness': extract_metric_value(dola_mid_result, 'faithfulness'),
            'answer_relevancy': extract_metric_value(dola_mid_result, 'answer_relevancy'),
        }
    
    # Load Baseline results
    if mode in ('baseline', 'all'):
        print("\nPreparing Baseline dataset...")
        baseline_dataset = prepare_dataset(
            '/workspace/datasets/merged_results.json',
            '/workspace/datasets/data.csv',
            'baseline_generated'
        )
        print(f"\nBaseline samples: {len(baseline_dataset)}")
        print("=== Evaluating Baseline ===")
        baseline_result = evaluate(
            baseline_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )
        print(f"Baseline results: {baseline_result}")
        results['baseline'] = {
            'faithfulness': extract_metric_value(baseline_result, 'faithfulness'),
            'answer_relevancy': extract_metric_value(baseline_result, 'answer_relevancy'),
        }
    
    # Save results
    output_path = '/workspace/datasets/ragas_official_results_v2.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("=== RAGAS Official Results (v2 - with actual user questions) ===")
    for key, vals in results.items():
        print(f"\n{key}:")
        print(f"  Faithfulness:      {vals['faithfulness']:.4f}")
        print(f"  Answer Relevancy:  {vals['answer_relevancy']:.4f}")
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
