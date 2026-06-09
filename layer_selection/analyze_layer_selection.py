"""
分析 DoLa 对比过程中，某个 token 在不同层的 logit 变化趋势

这个脚本需要修改 logits_processor.py 来输出中间层的 logits，
或者通过分析 DOLA_JS 日志来推断。

更简单的方法：直接修改 sglang 的 DoLa 实现，输出候选层的 top token logits。
"""

import os
import sys
import json
import argparse
import re
from collections import defaultdict

def parse_dola_js_log(log_file):
    """解析 DOLA_JS 日志，提取每一步的 JS 散度"""
    results = []

    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('DOLA_JS:'):
                # 解析格式: DOLA_JS: 0:0.000023,2:0.000023,...
                parts = line.strip().split(':')[1:]  # 去掉 "DOLA_JS"
                js_values = {}
                for i in range(0, len(parts)-1, 2):
                    layer = int(parts[i].strip())
                    js_val = float(parts[i+1].strip())
                    js_values[layer] = js_val
                results.append(js_values)

    return results


def parse_layer_selection_log(log_file):
    """解析 DOLA_LAYERS_SELECTED 日志"""
    results = []

    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('DOLA_LAYERS_SELECTED:'):
                # 解析格式: DOLA_LAYERS_SELECTED: 18,18,22,22,...
                parts = line.strip().split(':')[1]
                layers = [int(x.strip()) for x in parts.split(',') if x.strip()]
                results.append(layers)

    return results


def analyze_layer_selection(log_file):
    """分析层选择统计"""
    js_data = parse_dola_js_log(log_file)
    layer_data = parse_layer_selection_log(log_file)

    print(f"\n{'='*60}")
    print("DoLa Layer Selection Analysis")
    print(f"{'='*60}")

    print(f"\nTotal steps analyzed: {len(layer_data)}")

    # 统计每个层被选中的次数
    layer_counts = defaultdict(int)
    for layers in layer_data:
        for l in layers:
            layer_counts[l] += 1

    total_selections = sum(layer_counts.values())
    print(f"\nLayer selection frequency:")
    for l, count in sorted(layer_counts.items()):
        pct = count / total_selections * 100
        print(f"  Layer {l}: {count} times ({pct:.2f}%)")

    # 分析 JS 散度统计
    if js_data:
        print(f"\nJS divergence stats per layer:")
        all_layers = set()
        for js in js_data:
            all_layers.update(js.keys())

        for l in sorted(all_layers):
            js_values = [js[l] for js in js_data if l in js]
            if js_values:
                mean_js = sum(js_values) / len(js_values)
                min_js = min(js_values)
                max_js = max(js_values)
                print(f"  Layer {l}: mean={mean_js:.6f}, min={min_js:.6f}, max={max_js:.6f}")

    return layer_counts, js_data


def main():
    parser = argparse.ArgumentParser(description="Analyze DoLa layer selection")
    parser.add_argument("--log_file", type=str, required=True, help="Log file with DOLA_JS output")
    args = parser.parse_args()

    analyze_layer_selection(args.log_file)


if __name__ == "__main__":
    main()