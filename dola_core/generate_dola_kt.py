#!/usr/bin/env python
"""
使用 ktransformers + DoLa 生成文本（侵入性最小方案）

特点：
- 不修改 ktransformers 任何源代码
- 复用 local_chat 模型加载逻辑
- 独立运行，传入 output_hidden_states=True 做 DoLa decode

用法示例：
    python generate_dola_kt.py \
        --model_path /path/to/Qwen3.5-122B-A10B \
        --gguf_path /path/to/model.q4_k_m.gguf \
        --optimize_config_path /path/to/Qwen3Moe-serve.yaml \
        --prompt "什么是量子计算？" \
        --dola_layers high \
        --relative_top 0.1 \
        --max_new_tokens 200
"""

import os
import sys
import time
import argparse
import logging

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)

from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.util.utils import (
    get_device,
    get_all_used_cuda_device,
    sync_all_device,
    tf_logits_warper,
    torch_device_mapping,
)
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.textstream import TextStreamer
from ktransformers.util.globals import GLOBAL_CONFIG
from ktransformers.util.vendors import device_manager, GPUVendor, get_compute_capability
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled

warm_uped = False


def get_candidate_premature_layers(config, dola_layers="high"):
    """获取 DoLa 候选 premature layers，与 eval_truthfulqa_dola.py 对齐"""
    final_layer = config.num_hidden_layers

    if not getattr(config, "tie_word_embeddings", True):
        start_layer = 0
    elif final_layer > 2:
        start_layer = 2
    elif final_layer == 2:
        start_layer = 1
    else:
        start_layer = 0

    if isinstance(dola_layers, str) and dola_layers == "low":
        if start_layer == final_layer // 2:
            candidate_premature_layers = [start_layer]
        else:
            candidate_premature_layers = (
                list(range(start_layer, final_layer // 2, 2))
                if final_layer <= 40
                else list(range(start_layer, 20, 2))
            )
    elif isinstance(dola_layers, str) and dola_layers == "high":
        candidate_premature_layers = (
            list(range(final_layer // 2, final_layer, 2))
            if final_layer <= 40
            else list(range(final_layer - 20, final_layer, 2))
        )
    elif isinstance(dola_layers, list):
        candidate_premature_layers = [i for i in dola_layers if i < final_layer]
    else:
        raise ValueError("dola_layers must be 'low', 'high', or a list of ints")

    return candidate_premature_layers


def dola_decode_one_tokens(
    model,
    cur_token,
    position_ids,
    cache_position,
    past_key_values,
    torch_device,
    all_cuda_device,
    candidate_premature_layers,
    mature_layer,
    relative_top,
    generation_config,
    inputs,
):
    """DoLa decode: forward 取中间层 hidden_states，JS divergence 选层，对比解码后采样"""
    # ktransformers 的 embed_tokens 在 CPU 上，需要手动移设备
    inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to(torch_device)

    outputs = model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=past_key_values,
        output_hidden_states=True,
        return_dict=True,
        use_cache=True,
    )

    # 更新 StaticCache
    if past_key_values is not None and isinstance(past_key_values, StaticCache):
        past_key_values.change_seq_length(1)
    sync_all_device(all_cuda_device)

    # hidden_states tuple 结构（Qwen3 / KQwen3MoeModel）：
    #   hidden_states[0] = inputs_embeds (embed 输出，layer 0 输入)
    #   hidden_states[i] = layer i-1 的输出 (i >= 1)
    #   hidden_states[num_layers] = final norm 后的输出 (= outputs[0])
    hidden_states = outputs.hidden_states

    # mature_logits: 用 final norm 后的 hidden state (和模型正常输出一致)
    final_logits = model.lm_head(outputs[0])[:, -1, :].float()  # [batch, vocab]

    # candidate premature layers logits
    # 与 eval_truthfulqa_dola.py 保持一致：直接用 hidden_states[layer] 过 lm_head，不做额外 norm
    # 注意对齐设备：lm_head 通常在 cuda 上，hidden_states 可能来自不同设备
    candidate_logits_list = []
    for l in candidate_premature_layers:
        hs = hidden_states[l].to(torch_device)
        premature_logits = model.lm_head(hs)[:, -1, :].float()
        candidate_logits_list.append(premature_logits)

    stacked_premature = torch.stack(
        candidate_logits_list, dim=0
    )  # [num_candidates, batch, vocab]

    # JS divergence 选最优 premature layer
    softmax_mature = F.softmax(final_logits, dim=-1)  # [batch, vocab]
    softmax_premature = F.softmax(
        stacked_premature, dim=-1
    )  # [num_candidates, batch, vocab]

    M = 0.5 * (softmax_mature.unsqueeze(0) + softmax_premature)

    log_softmax_mature = F.log_softmax(final_logits, dim=-1).unsqueeze(0)
    log_softmax_premature = F.log_softmax(stacked_premature, dim=-1)

    kl1 = F.kl_div(log_softmax_mature, M, reduction="none").mean(-1)  # [num_candidates, batch]
    kl2 = F.kl_div(log_softmax_premature, M, reduction="none").mean(-1)
    js_divs = 0.5 * (kl1 + kl2).mean(-1)  # [num_candidates]

    picked_idx = int(js_divs.argmax().cpu().item())
    picked_layer = candidate_premature_layers[picked_idx]

    # Contrastive logits: mature_log_softmax - premature_log_softmax
    base_logits = candidate_logits_list[picked_idx]

    final_log_sm = final_logits.log_softmax(dim=-1)
    base_log_sm = base_logits.log_softmax(dim=-1)
    diff_logits = final_log_sm - base_log_sm
    diff_logits = diff_logits.log_softmax(dim=-1)

    # Relative top filtering
    if relative_top > 0.0:
        probs_max = final_log_sm.max(dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        mask = final_log_sm < probs_thresh.unsqueeze(-1)
        diff_logits = torch.where(
            mask,
            torch.tensor(-1000.0, device=diff_logits.device, dtype=diff_logits.dtype),
            diff_logits,
        )

    # 应用 sampling (temperature / top_p / top_k)
    logits_warper = tf_logits_warper(generation_config)
    next_token_scores = logits_warper(inputs, diff_logits)

    if generation_config.do_sample:
        probs = F.softmax(next_token_scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_token = torch.argmax(next_token_scores, dim=-1)

    return next_token, picked_layer


def baseline_decode_one_tokens(
    model,
    cur_token,
    position_ids,
    cache_position,
    past_key_values,
    torch_device,
    all_cuda_device,
    generation_config,
    inputs,
):
    """Baseline decode: 正常 forward，不做 DoLa"""
    inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to(torch_device)
    logits = model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True,
    )[0]

    if past_key_values is not None and isinstance(past_key_values, StaticCache):
        past_key_values.change_seq_length(1)
    sync_all_device(all_cuda_device)

    logits_warper = tf_logits_warper(generation_config)
    next_token_scores = logits_warper(inputs, logits[:, -1, :])

    if generation_config.do_sample:
        probs = F.softmax(next_token_scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_token = torch.argmax(next_token_scores, dim=-1)

    return next_token


def prefill_and_generate(
    model,
    tokenizer,
    inputs,
    max_new_tokens=1000,
    use_cuda_graph=False,
    mode="normal",
    chunk_size=16384,
    dola_layers="high",
    mature_layer=None,
    relative_top=0.1,
    stop_strings=None,
):
    """
    完整的 prefill + generate 循环，支持 baseline 和 dola 两种 decode 模式。
    复用 ktransformers local_chat 的核心逻辑，但独立封装。
    """
    global warm_uped

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True

    batch_size, seq_length = inputs.shape
    device_map = model.gguf_loader.tensor_device_map
    torch_device = get_device("model.layers.0.self_attn", device_map)
    torch_device = torch_device_mapping.get(torch_device, torch_device)
    inputs = inputs.to(torch_device)
    all_cuda_device = get_all_used_cuda_device(device_map)

    # DoLa 配置
    use_dola = mode in ("dola", "dola-static")
    if use_dola:
        candidate_premature_layers = get_candidate_premature_layers(
            model.config, dola_layers
        )
        if mature_layer is None:
            mature_layer = model.config.num_hidden_layers
        print(f"[DoLa] candidate_premature_layers: {candidate_premature_layers}")
        print(f"[DoLa] mature_layer: {mature_layer}")
        picked_layer_dist = {l: 0 for l in candidate_premature_layers}
    else:
        candidate_premature_layers = None
        picked_layer_dist = None

    tokens = []

    # chunk prefill (复用 local_chat 逻辑)
    def chunk_prefill(inputs_chunk, cache_position, past_key_values):
        inputs_embeds = model.model.embed_tokens(inputs_chunk.to("cpu")).to(
            torch_device
        )
        logits = model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        )[0][:, -1, :].unsqueeze(0).clone().to(torch_device)
        return logits

    with torch.no_grad():
        stream = TextStreamer(tokenizer)

        past_key_values = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=seq_length + max_new_tokens,
            device=device_map,
            dtype=model.dtype,
        )

        generation_config, _ = model._prepare_generation_config(
            None, do_sample=True
        )

        cache_position = torch.arange(
            seq_length, device=torch_device, dtype=torch.int32
        )
        generated_ids = torch.zeros(
            batch_size,
            seq_length + max_new_tokens + 1,
            dtype=torch.int,
            device=torch_device,
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)

        start_time = time.time()

        # Prefill: 逐 chunk 处理
        chunk_start = 0
        while chunk_start < seq_length:
            chunk_end = min(chunk_start + chunk_size, seq_length)
            past_key_values.cur_idx = cache_position[chunk_start:chunk_end]
            logits = chunk_prefill(
                inputs[:, chunk_start:chunk_end],
                cache_position[chunk_start:chunk_end],
                past_key_values,
            )
            chunk_start += chunk_size

        logits_warper = tf_logits_warper(generation_config)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)

        first_token_time = time.time() - start_time

        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, seq_length] = next_token
        tokens.append(int(next_token))
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor(
            [seq_length], device=torch_device, dtype=torch.int32
        )
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1

        cuda_graph_runner = None

        # Decode loop
        decode_start = time.time()
        for i in range(1, max_new_tokens):
            if use_dola:
                next_token, picked_layer = dola_decode_one_tokens(
                    model,
                    next_token.unsqueeze(0),
                    position_ids,
                    cache_position,
                    past_key_values,
                    torch_device,
                    all_cuda_device,
                    candidate_premature_layers,
                    mature_layer,
                    relative_top,
                    generation_config,
                    inputs,
                )
                picked_layer_dist[picked_layer] += 1
            else:
                next_token = baseline_decode_one_tokens(
                    model,
                    next_token.unsqueeze(0),
                    position_ids,
                    cache_position,
                    past_key_values,
                    torch_device,
                    all_cuda_device,
                    generation_config,
                    inputs,
                )

            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(int(next_token))
            seq_length += 1

            # 检查停止条件
            stop_hit = False
            if stop_strings:
                current_text = tokenizer.decode(tokens, skip_special_tokens=True)
                for s in stop_strings:
                    if current_text.endswith(s):
                        stop_hit = True
                        break

            if (
                next_token[0].item() == tokenizer.eos_token_id
                or tokenizer.decode(next_token.tolist()) == "<|im_end|>"
                or stop_hit
            ):
                stream.end()
                break
            else:
                print(stream.put(next_token.item()), end="", flush=True)

            cache_position += 1
            position_ids = cache_position.unsqueeze(0)

    total_decode_time = time.time() - decode_start
    tokens_generated = len(tokens)
    tps = tokens_generated / total_decode_time if total_decode_time > 0 else 0

    print("")
    if use_dola:
        print(f"[DoLa] picked_layer_dist: {picked_layer_dist}")
    print(f"[Speed] Prefill: {first_token_time:.2f}s")
    print(
        f"[Speed] Decode: {tokens_generated} tokens in {total_decode_time:.2f}s = {tps:.2f} tokens/s"
    )

    return tokens, picked_layer_dist


def main():
    parser = argparse.ArgumentParser(
        description="ktransformers + DoLa 生成（侵入性最小）"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="模型目录路径（含 config.json）"
    )
    parser.add_argument(
        "--gguf_path", type=str, required=True, help="GGUF 文件路径"
    )
    parser.add_argument(
        "--optimize_config_path",
        type=str,
        default=None,
        help="optimize yaml 路径，默认自动查找",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请介绍一下量子计算的基本原理。",
        help="生成 prompt",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="最大生成 token 数"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dola",
        choices=["baseline", "dola", "dola-static"],
        help="生成模式: baseline=普通, dola=动态选层, dola-static=固定层",
    )
    parser.add_argument(
        "--dola_layers",
        type=str,
        default="high",
        help="DoLa 候选层范围: 'high', 'low', 或层索引列表如 '[40,42,44]'",
    )
    parser.add_argument(
        "--mature_layer", type=int, default=None, help="mature layer 索引，默认最后一层"
    )
    parser.add_argument(
        "--relative_top",
        type=float,
        default=0.1,
        help="relative top filter 阈值 (0=禁用)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="采样温度"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
    parser.add_argument("--top_k", type=int, default=0, help="top_k (0=禁用)")
    parser.add_argument(
        "--cpu_infer", type=int, default=32, help="CPU 推理线程数"
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
        help="是否使用 CUDA Graph (DoLa 模式下建议关闭)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16384, help="prefill chunk 大小"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="主设备"
    )
    parser.add_argument(
        "--chat_template",
        action="store_true",
        default=True,
        help="是否使用 chat template",
    )
    args = parser.parse_args()

    # 解析 dola_layers 参数
    dola_layers = args.dola_layers
    if dola_layers.startswith("[") and dola_layers.endswith("]"):
        dola_layers = [int(x.strip()) for x in dola_layers[1:-1].split(",")]

    torch.set_grad_enabled(False)
    GLOBAL_CONFIG._config["mod"] = "infer"

    from ktransformers.server.config.config import Config

    Config().cpu_infer = args.cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    if config.architectures[0] == "Qwen2MoeForCausalLM":
        config._attn_implementation = "flash_attention_2"

    torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in [
            "Qwen2MoeForCausalLM",
            "Qwen3MoeForCausalLM",
        ]:
            # 直接 import 对应的 modeling 文件
            from ktransformers.models.modeling_qwen3_moe import Qwen3MoeForCausalLM
            from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM

            custom_models_local = {
                "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
                "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
            }
            model = custom_models_local[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    optimize_config_path = args.optimize_config_path
    if optimize_config_path is None:
        # 尝试自动查找默认 rule
        rules_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ktransformers",
            "kt-sft",
            "ktransformers",
            "optimize",
            "optimize_rules",
        )
        # 常见 Qwen3 规则文件名
        candidates = [
            "Qwen3Moe-serve.yaml",
            "Qwen3Moe-Chat.yaml",
            "Qwen3Moe-sft-amx.yaml",
        ]
        for c in candidates:
            p = os.path.join(rules_dir, c)
            if os.path.exists(p):
                optimize_config_path = p
                break
        if optimize_config_path is None:
            raise ValueError(
                f"未找到默认 optimize rule，请手动指定 --optimize_config_path"
            )

    print(f"[Load] 使用 optimize config: {optimize_config_path}")
    optimize_and_load_gguf(model, optimize_config_path, args.gguf_path, config)

    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True,
    )
    model.generation_config = gen_config
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()

    print(f"[Load] 模型加载完成: {config.architectures[0]}")
    print(f"[Load] num_hidden_layers: {config.num_hidden_layers}")
    print(f"[Load] hidden_size: {config.hidden_size}")
    print(f"[Load] vocab_size: {config.vocab_size}")

    # 构建输入
    if args.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        input_tensor = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        input_tensor = tokenizer(args.prompt, return_tensors="pt").input_ids

    print(f"\n[Prompt] {args.prompt}")
    print("=" * 60)

    tokens, dist = prefill_and_generate(
        model,
        tokenizer,
        input_tensor.to(args.device),
        max_new_tokens=args.max_new_tokens,
        use_cuda_graph=args.use_cuda_graph,
        mode=args.mode,
        chunk_size=args.chunk_size,
        dola_layers=dola_layers,
        mature_layer=args.mature_layer,
        relative_top=args.relative_top,
    )

    print("=" * 60)
    print("\n[Final Output]")
    print(tokenizer.decode(tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
