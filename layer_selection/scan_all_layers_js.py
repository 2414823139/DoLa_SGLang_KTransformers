"""
扫描模型每一层与 final layer 的 JS 散度
找到最佳对比层，而不局限于 DoLa 默认的 high/low 候选层
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace")


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def compute_js_per_layer(model, tokenizer, questions, device, n_questions=50):
    """对每个问题，计算每一层 hidden state 经 lm_head 后与 final layer 的 JS 散度"""
    final_layer = model.config.get_text_config().num_hidden_layers
    lm_head = model.get_output_embeddings()
    print(f"Model: {model.config.model_type}, layers: {final_layer}")

    # shape: (n_questions, final_layer+1)  — +1 因为 hidden_states 包含 embedding 层 (index 0)
    all_js = []

    for qi, question in enumerate(questions[:n_questions]):
        prompt = f"Q: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, return_dict=True, output_hidden_states=True)

        final_logits = outputs.logits[:, -1, :].float()
        softmax_final = F.softmax(final_logits, dim=-1)
        log_softmax_final = F.log_softmax(final_logits, dim=-1)

        js_per_layer = []
        # hidden_states[0] = embedding, hidden_states[1..N] = transformer layers
        for layer_idx in range(len(outputs.hidden_states)):
            h = outputs.hidden_states[layer_idx][:, -1, :]
            layer_logits = lm_head(h).to(final_logits.device).float()
            softmax_layer = F.softmax(layer_logits, dim=-1)
            log_softmax_layer = F.log_softmax(layer_logits, dim=-1)

            avg_dist = 0.5 * (softmax_final + softmax_layer)
            kl1 = F.kl_div(log_softmax_final, avg_dist, reduction="none").mean(-1)
            kl2 = F.kl_div(log_softmax_layer, avg_dist, reduction="none").mean(-1)
            js = 0.5 * (kl1 + kl2)
            js_per_layer.append(js.item())

        all_js.append(js_per_layer)

        if (qi + 1) % 10 == 0:
            print(f"  Processed {qi+1}/{n_questions} questions")

    all_js = np.array(all_js)  # (n_questions, n_layers+1)
    return all_js


def print_js_report(model_name, all_js, final_layer):
    """打印 JS 散度报告"""
    mean_js = all_js.mean(axis=0)  # (n_layers+1,)
    std_js = all_js.std(axis=0)

    # 隐藏 embedding 层 (index 0) 和 final layer (index final_layer)，它们对比无意义
    print(f"\n{'='*75}")
    print(f"  JS Divergence Report: {model_name}")
    print(f"  (Layer N vs Final Layer {final_layer})")
    print(f"{'='*75}")
    print(f"  {'Layer':>6s}  {'Mean JS':>12s}  {'Std JS':>10s}  Distribution")
    print(f"  {'-'*60}")

    max_js = mean_js[1:final_layer].max()  # 排除 embedding 和 final layer
    for i in range(1, final_layer):  # 跳过 embedding (0) 和 final layer
        bar_len = int(mean_js[i] / max_js * 40) if max_js > 0 else 0
        bar = "#" * bar_len
        print(f"  {i:>6d}  {mean_js[i]:>12.8f}  {std_js[i]:>10.8f}  {bar}")

    # Top-K layers
    sorted_indices = np.argsort(mean_js[1:final_layer])[::-1] + 1  # +1 因为跳过了 index 0
    print(f"\n  Top-10 layers by JS divergence:")
    for rank, idx in enumerate(sorted_indices[:10]):
        print(f"    #{rank+1}: Layer {idx}  (JS = {mean_js[idx]:.8f})")

    # DoLa high/low candidate layers 的平均 JS
    if final_layer <= 40:
        high_layers = list(range(final_layer // 2, final_layer, 2))
        low_layers = list(range(2, final_layer // 2, 2))
    else:
        high_layers = list(range(final_layer - 20, final_layer, 2))
        low_layers = list(range(2, 20, 2))

    high_mean = np.mean([mean_js[i] for i in high_layers if i < final_layer])
    low_mean = np.mean([mean_js[i] for i in low_layers if i < final_layer])

    # 中间层
    mid_start = final_layer // 3
    mid_end = 2 * final_layer // 3
    mid_layers = list(range(mid_start, mid_end, 2))
    mid_mean = np.mean([mean_js[i] for i in mid_layers if i < final_layer])

    print(f"\n  Average JS by region:")
    print(f"    Low  layers {low_layers[:3]}...{low_layers[-1]}:  {low_mean:.8f}")
    print(f"    Mid  layers {mid_layers[:3]}...{mid_layers[-1]}:  {mid_mean:.8f}")
    print(f"    High layers {high_layers[:3]}...{high_layers[-1]}:  {high_mean:.8f}")

    # 最佳区域建议
    best_layer = sorted_indices[0]
    if best_layer < final_layer // 3:
        suggestion = "low"
    elif best_layer < 2 * final_layer // 3:
        suggestion = "mid"
    else:
        suggestion = "high"

    print(f"\n  Suggestion: best single contrast layer = {best_layer} (region: {suggestion})")
    print(f"  Best 5 layers: {sorted_indices[:5].tolist()}")

    return mean_js


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--save", type=str, default=None, help="Save JS data to npy")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    final_layer = model.config.get_text_config().num_hidden_layers

    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    questions = [item["question"] for item in dataset.select(range(min(args.n_questions, len(dataset))))]

    print(f"\nScanning JS divergence for ALL {final_layer} layers on {args.n_questions} questions...")
    all_js = compute_js_per_layer(model, tokenizer, questions, args.device, args.n_questions)

    mean_js = print_js_report(args.model, all_js, final_layer)

    if args.save:
        np.save(args.save, all_js)
        print(f"\nJS data saved to {args.save}")


if __name__ == "__main__":
    main()
