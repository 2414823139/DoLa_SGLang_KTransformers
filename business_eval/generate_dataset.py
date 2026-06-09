"""
开放生成测试脚本
用法:
    # Baseline (不设置 DOLA_LAYERS)
    python generate_dataset.py --output /workspace/datasets/baseline_results.json

    # DoLa (设置环境变量)
    DOLA_LAYERS=0,2,4,6,8,10,12,14,16 python generate_dataset.py --output /workspace/datasets/dola_results.json
"""

import os
import csv
import json
import argparse
import urllib.request
import urllib.error
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="开放生成测试")
    parser.add_argument("--api_base", type=str, default="http://localhost:30000/v1")
    parser.add_argument("--data_path", type=str, default="/workspace/datasets/data.csv")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--tokenizer_path", type=str,
                        default="/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    args = parser.parse_args()

    # 加载 tokenizer
    print(f"加载 tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True
    )

    # 读取数据
    print(f"读取数据: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        samples = list(reader)
    print(f"共 {len(samples)} 条数据")

    # 检查 DoLa 状态
    dola_layers = os.environ.get("DOLA_LAYERS", "")
    mode = f"DoLa ({dola_layers})" if dola_layers else "Baseline"
    print(f"运行模式: {mode}")

    # 检查服务
    try:
        with urllib.request.urlopen(f"{args.api_base}/models", timeout=10) as resp:
            models = json.loads(resp.read().decode("utf-8"))
            if models.get("data"):
                print(f"模型: {models['data'][0]['id']}")
    except Exception as e:
        print(f"错误: 无法连接服务 {args.api_base}: {e}")
        return

    # 生成
    results = []
    start_time = datetime.now()

    for i, sample in enumerate(tqdm(samples, desc=f"生成中 [{mode}]")):
        inputs = json.loads(sample['inputs'])
        outputs = json.loads(sample['outputs'])

        # 格式化 prompt，禁用思考模式
        prompt = tokenizer.apply_chat_template(
            inputs,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        payload = {
            "model": "default",
            "prompt": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }

        try:
            url = f"{args.api_base}/completions"
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            if "choices" in result and len(result["choices"]) > 0:
                generated = result["choices"][0]["text"]
                results.append({
                    "index": i,
                    "title": sample['title'],
                    "generated": generated,
                    "expected": outputs['text'],
                    "generated_len": len(generated),
                    "expected_len": len(outputs['text']),
                    "success": True,
                })
            else:
                results.append({"index": i, "error": "No choices", "success": False})

        except Exception as e:
            print(f"\n错误 样本 {i}: {e}")
            results.append({"index": i, "error": str(e), "success": False})

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    # 统计
    success_count = sum(1 for r in results if r.get('success'))
    failed_count = len(results) - success_count

    print(f"\n{'='*60}")
    print(f"生成完成")
    print(f"{'='*60}")
    print(f"模式: {mode}")
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {failed_count}")
    print(f"耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"平均: {elapsed/len(results):.2f} 秒/条")

    if failed_count > 0:
        failed_indices = [r['index'] for r in results if not r.get('success')]
        print(f"失败索引: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")

    # 保存结果
    output_data = {
        "config": {
            "mode": mode,
            "api_base": args.api_base,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "dola_layers": dola_layers if dola_layers else None,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_seconds": elapsed,
            "total_samples": len(samples),
            "success_count": success_count,
            "failed_count": failed_count,
        },
        "samples": results,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
