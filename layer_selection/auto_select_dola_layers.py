#!/usr/bin/env python3
"""
DoLa 候选层自动选择工具

功能：
1. 使用模型 + 抽样数据，在所有候选层上计算 JS 散度
2. 使用滑动窗口分析 JS 散度分布
3. 根据窗口内 JS 散度的方差（一致性）选择最优层区间

用法：
    # 一键运行：启动服务 + 收集数据 + 计算最优区间
    python auto_select_dola_layers.py --model_path <model_path> --data_path <data_path> --sample_size 20

    # 从已有日志分析
    python auto_select_dola_layers.py --skip_scan --log_file <log_file>
"""

import os
import sys
import json
import argparse
import subprocess
import time
import re
import signal
import numpy as np
from typing import List, Dict, Tuple, Optional


def parse_js_from_log(log_content: str) -> Dict[int, List[float]]:
    """从日志中解析 JS 散度信息

    Returns:
        {layer_idx: [js_values]}
    """
    js_pattern = r"DOLA_JS: (.+)"
    js_data = {}

    for line in log_content.split('\n'):
        match = re.search(js_pattern, line)
        if match:
            # 解析格式: "2:0.000021,4:0.000021,..."
            pairs = match.group(1).split(',')
            for pair in pairs:
                if ':' in pair:
                    layer_str, js_str = pair.split(':')
                    layer = int(layer_str.strip())
                    js = float(js_str.strip())
                    if layer not in js_data:
                        js_data[layer] = []
                    js_data[layer].append(js)

    return js_data


def compute_js_statistics(js_data: Dict[int, List[float]]) -> Dict[int, Dict]:
    """计算每层 JS 散度的统计信息"""
    stats = {}
    for layer, values in js_data.items():
        if values:
            stats[layer] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    return stats


def sliding_window_analysis(
    stats: Dict[int, Dict],
    window_size: int = 6,
    step: int = 2
) -> List[Dict]:
    """滑动窗口分析，找出最优层区间

    Args:
        stats: 每层的 JS 统计信息
        window_size: 窗口大小（层数）
        step: 滑动步长

    Returns:
        窗口分析结果列表，按方差排序
    """
    layers = sorted(stats.keys())
    if len(layers) < window_size:
        print(f"警告: 候选层数量 ({len(layers)}) 小于窗口大小 ({window_size})")
        window_size = len(layers)

    results = []

    for i in range(0, len(layers) - window_size + 1, step):
        window_layers = layers[i:i + window_size]
        window_means = [stats[l]['mean'] for l in window_layers]
        window_stds = [stats[l]['std'] for l in window_layers]

        # 计算窗口内的统计指标
        mean_js = np.mean(window_means)
        variance_of_means = np.var(window_means)  # 窗口内均值的方差
        mean_std = np.mean(window_stds)  # 窗口内标准差的均值

        results.append({
            'layers': window_layers,
            'layer_range': f"{window_layers[0]}-{window_layers[-1]}",
            'mean_js': mean_js,
            'variance_of_means': variance_of_means,
            'mean_std': mean_std,
            'score': variance_of_means,  # 方差越小越好（一致性高）
        })

    # 按方差排序（方差越小越好）
    results.sort(key=lambda x: x['score'])

    return results


def generate_dola_layers_config(
    best_window: Dict,
    mode: str = 'consistency'
) -> str:
    """生成 DOLA_LAYERS 环境变量配置"""
    layers = best_window['layers']
    return ','.join(map(str, layers))


def run_evaluation_with_dola(
    model_path: str,
    data_path: str,
    dola_layers: str,
    sample_size: int,
    port: int = 30000,
    gpu_ids: str = "0,1",
    tensor_parallel_size: int = 2,
    log_file: str = None
) -> Tuple[Dict[int, List[float]], str]:
    """启动带 DoLa 的服务并收集 JS 散度数据

    Returns:
        {layer_idx: [js_values]}
    """
    if log_file is None:
        log_file = f"/tmp/dola_auto_select_{port}.log"

    # 构建启动命令
    cmd = f"""
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate serve && \
export SGLANG_DISABLE_CUDNN_CHECK=1 && \
export DOLA_LAYERS={dola_layers} && \
export DOLA_RELATIVE_TOP=0 && \
CUDA_VISIBLE_DEVICES={gpu_ids} python -m sglang.launch_server \
  --model-path {model_path} \
  --tensor-parallel-size {tensor_parallel_size} \
  --host 0.0.0.0 \
  --port {port} \
  --trust-remote-code \
  --attention-backend flashinfer \
  --mem-fraction-static 0.88 \
  --max-running-requests 2 \
  --max-total-tokens 16384 \
  --disable-cuda-graph \
  --kt-method BF16 \
  --kt-cpuinfer 64 \
  --kt-weight-path {model_path} \
  --kt-num-gpu-experts 1 \
  > {log_file} 2>&1 &
"""

    print(f"启动服务: DOLA_LAYERS={dola_layers}")
    subprocess.run(cmd, shell=True, executable='/bin/bash')

    # 等待服务启动
    print(f"等待服务启动 (120秒)...")
    time.sleep(120)

    # 检查服务是否启动
    import urllib.request
    try:
        url = f"http://localhost:{port}/v1/models"
        with urllib.request.urlopen(url, timeout=10) as resp:
            print("服务启动成功")
    except Exception as e:
        print(f"服务启动失败: {e}")
        return {}

    # 运行评测脚本收集 JS 散度
    eval_script = f"""
import json
import urllib.request
import pandas as pd
from transformers import AutoTokenizer

# 加载数据
df = pd.read_csv('{data_path}')
questions = df['Question'].tolist()[:{sample_size}]

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained('{model_path}', trust_remote_code=True, local_files_only=True)

# 构建 prompt 并发送请求
for i, q in enumerate(questions):
    prompt = f"Q: {{q}}\\nA:"
    url = f"http://localhost:{port}/v1/completions"
    payload = {{
        'prompt': prompt,
        'max_tokens': 10,
        'temperature': 0,
    }}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={{'Content-Type': 'application/json'}})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        print(f"[{{i+1}}/{{len(questions)}}] 完成")
    except Exception as e:
        print(f"[{{i+1}}/{{len(questions)}}] Error: {{e}}")
"""

    eval_file = f"/tmp/dola_eval_{port}.py"
    with open(eval_file, 'w') as f:
        f.write(eval_script)

    print(f"运行评测收集 JS 散度 ({sample_size} 条数据)...")
    subprocess.run(f"python {eval_file}", shell=True)

    # 从日志中解析 JS 散度
    with open(log_file, 'r') as f:
        log_content = f.read()

    js_data = parse_js_from_log(log_content)

    # 停止服务
    print("停止服务...")
    subprocess.run(f"pkill -f 'sglang.launch_server.*--port {port}'", shell=True)
    time.sleep(5)

    return js_data, log_file


def scan_all_layers(
    model_path: str,
    data_path: str,
    sample_size: int,
    num_layers: int,
    layer_step: int = 2,
    port: int = 30000,
    gpu_ids: str = "0,1",
    tensor_parallel_size: int = 2,
    log_file: str = None
) -> Tuple[Dict[int, List[float]], str]:
    """扫描所有候选层，收集 JS 散度数据

    Args:
        model_path: 模型路径
        data_path: 数据路径
        sample_size: 抽样数据量
        num_layers: 模型总层数
        layer_step: 层间隔
        port: 服务端口
        gpu_ids: GPU ID
        log_file: 日志文件路径

    Returns:
        {layer_idx: [js_values]}
    """
    # 构建所有候选层
    # 根据论文，跳过第 0 层（embedding）和第 1 层
    start_layer = 2
    all_layers = list(range(start_layer, num_layers, layer_step))

    # 添加最后一层
    if num_layers - 1 not in all_layers:
        all_layers.append(num_layers - 1)

    dola_layers_str = ','.join(map(str, all_layers))

    print(f"\n{'='*70}")
    print(f"扫描所有候选层: {len(all_layers)} 层")
    print(f"候选层: {all_layers[:5]} ... {all_layers[-5:]}")
    print(f"{'='*70}\n")

    js_data, actual_log_file = run_evaluation_with_dola(
        model_path=model_path,
        data_path=data_path,
        dola_layers=dola_layers_str,
        sample_size=sample_size,
        port=port,
        gpu_ids=gpu_ids,
        tensor_parallel_size=tensor_parallel_size,
        log_file=log_file
    )

    return js_data, actual_log_file


def print_layer_statistics(stats: Dict[int, Dict]):
    """打印各层 JS 散度统计信息（高精度）"""
    print(f"\n各层 JS 散度统计:")
    print(f"{'层':<6} {'均值':<20} {'标准差':<20} {'最小值':<20} {'最大值':<20}")
    print("-" * 90)
    for layer in sorted(stats.keys()):
        s = stats[layer]
        print(f"{layer:<6} {s['mean']:.15e}  {s['std']:.15e}  {s['min']:.15e}  {s['max']:.15e}")


def print_window_results(window_results: List[Dict], top_n: int = 10):
    """打印窗口分析结果（高精度）"""
    print(f"\n窗口分析结果 (按一致性排序):")
    print(f"{'层区间':<15} {'均值JS':<20} {'方差':<25} {'评分':<25}")
    print("-" * 90)
    for i, w in enumerate(window_results[:top_n]):
        print(f"{w['layer_range']:<15} {w['mean_js']:.15e}  {w['variance_of_means']:.20e}  {w['score']:.20e}")


def main():
    parser = argparse.ArgumentParser(description="DoLa 候选层自动选择工具")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="TruthfulQA 数据路径")
    parser.add_argument("--sample_size", type=int, default=20, help="抽样数据量")
    parser.add_argument("--num_layers", type=int, default=50, help="模型总层数")
    parser.add_argument("--layer_step", type=int, default=2, help="层间隔")
    parser.add_argument("--window_size", type=int, default=6, help="滑动窗口大小")
    parser.add_argument("--window_step", type=int, default=2, help="滑动窗口步长")
    parser.add_argument("--port", type=int, default=30000, help="服务端口")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="GPU IDs")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--output", type=str, default="dola_layer_selection.json", help="输出文件")
    parser.add_argument("--log_file", type=str, help="日志文件路径")
    parser.add_argument("--skip_scan", action="store_true", help="跳过扫描，使用已有日志文件")
    parser.add_argument("--log_files", type=str, nargs='+', help="多个日志文件路径（合并分析）")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("DoLa 候选层自动选择工具")
    print(f"{'='*70}")
    print(f"模型: {args.model_path}")
    print(f"数据: {args.data_path}")
    print(f"抽样量: {args.sample_size}")
    print(f"窗口大小: {args.window_size}")
    print(f"窗口步长: {args.window_step}")
    print(f"{'='*70}\n")

    # 步骤 1: 扫描所有层或从已有日志加载
    if args.skip_scan:
        js_data = {}
        actual_log_file = None
        if args.log_files:
            # 合并多个日志文件
            print(f"从多个日志文件加载:")
            for log_file in args.log_files:
                print(f"  - {log_file}")
                with open(log_file, 'r') as f:
                    log_content = f.read()
                file_js_data = parse_js_from_log(log_content)
                # 合并数据
                for layer, values in file_js_data.items():
                    if layer not in js_data:
                        js_data[layer] = []
                    js_data[layer].extend(values)
        elif args.log_file:
            print(f"从日志文件加载: {args.log_file}")
            with open(args.log_file, 'r') as f:
                log_content = f.read()
            js_data = parse_js_from_log(log_content)
            actual_log_file = args.log_file
        else:
            print("错误: 需要指定 --log_file 或 --log_files")
            sys.exit(1)
    else:
        js_data, actual_log_file = scan_all_layers(
            model_path=args.model_path,
            data_path=args.data_path,
            sample_size=args.sample_size,
            num_layers=args.num_layers,
            layer_step=args.layer_step,
            port=args.port,
            gpu_ids=args.gpu_ids,
            tensor_parallel_size=args.tensor_parallel_size,
            log_file=args.log_file
        )

    if not js_data:
        print("错误: 未能获取 JS 散度数据")
        sys.exit(1)

    # 步骤 2: 计算统计信息
    print("\n步骤 2: 计算 JS 散度统计信息...")
    stats = compute_js_statistics(js_data)
    print_layer_statistics(stats)

    # 步骤 3: 滑动窗口分析
    print(f"\n步骤 3: 滑动窗口分析 (窗口大小={args.window_size}, 步长={args.window_step})...")
    window_results = sliding_window_analysis(
        stats=stats,
        window_size=args.window_size,
        step=args.window_step
    )
    print_window_results(window_results)

    # 步骤 4: 推荐最优配置
    best_window = window_results[0]
    recommended_layers = generate_dola_layers_config(best_window)

    print(f"\n{'='*70}")
    print("推荐配置")
    print(f"{'='*70}")
    print(f"最优层区间: {best_window['layer_range']}")
    print(f"候选层: {best_window['layers']}")
    print(f"平均 JS 散度: {best_window['mean_js']:.15e}")
    print(f"方差 (一致性指标): {best_window['variance_of_means']:.20e}")
    print(f"\nDOLA_LAYERS 环境变量:")
    print(f"export DOLA_LAYERS={recommended_layers}")
    print(f"{'='*70}")

    # 保存结果
    result = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'sample_size': args.sample_size,
        'window_size': args.window_size,
        'window_step': args.window_step,
        'log_file': actual_log_file,
        'layer_statistics': {str(k): v for k, v in stats.items()},
        'window_analysis': window_results,
        'recommendation': {
            'layers': best_window['layers'],
            'layer_range': best_window['layer_range'],
            'mean_js': best_window['mean_js'],
            'variance_of_means': best_window['variance_of_means'],
            'dola_layers_env': recommended_layers,
        }
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n结果已保存到: {args.output}")

    return recommended_layers


if __name__ == "__main__":
    main()
