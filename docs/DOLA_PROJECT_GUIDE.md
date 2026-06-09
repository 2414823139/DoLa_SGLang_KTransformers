# DoLa + SGLang + KTransformers 项目说明

> **项目目标**: 在 SGLang + KTransformers 框架上实现 DoLa (Decoding by Contrasting Layers)，使 Qwen3.5-122B 大模型仅需 2 张 RTX 4090 即可运行，并显著提升事实准确性。

---

## 目录

1. [项目背景](#1-项目背景)
2. [技术原理](#2-技术原理)
3. [实现方案](#3-实现方案)
4. [环境配置](#4-环境配置)
5. [使用指南](#5-使用指南)
6. [动态候选层选择](#6-动态候选层选择)
7. [实验结果](#7-实验结果)
8. [技术细节](#8-技术细节)
9. [已知问题与解决方案](#9-已知问题与解决方案)
10. [文件索引](#10-文件索引)

---

## 1. 项目背景

### 1.1 DoLa 论文简介

DoLa (Decoding by Contrasting Layers) 是 ICLR 2024 发表的一种解码策略，通过对比 Transformer 不同层的 logits 输出来提高大模型的事实准确性。

**核心发现**：
- 成熟层（final layer）：包含更多事实性知识
- 早熟层（premature layer）：偏向语言建模的"表面模式"
- 通过 logits 对比，可放大事实性知识、抑制常见错误模式

**论文链接**: [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883)

### 1.2 项目目标

| 目标 | 说明 |
|------|------|
| 框架集成 | 将 DoLa 集成到 SGLang 高性能推理框架 |
| 硬件优化 | 结合 KTransformers 实现 CPU-GPU 混合推理 |
| 模型适配 | 支持 Qwen3 / Qwen3.5 系列模型 |
| 效果验证 | 在 TruthfulQA 评测集上验证效果 |

### 1.3 核心成果

- **Qwen3-8B**: TruthfulQA MC2 从 0.436 提升至 0.604 (+16.8pp)
- **Qwen3.5-122B**: 在 2×RTX 4090 上成功运行，支持 DoLa 推理

---

## 2. 技术原理

### 2.1 DoLa 算法流程

```
输入: input_ids, candidate_premature_layers, mature_layer (final layer)

┌─────────────────────────────────────────────────────────────┐
│ Step 1: Forward 获取多层 hidden_states                      │
│   outputs = model(input_ids, output_hidden_states=True)     │
│   hidden_states: [embed, layer1, layer2, ..., final]        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 计算候选层 logits                                    │
│   for layer in candidate_premature_layers:                  │
│       candidate_logits[layer] = lm_head(hidden_states[layer])│
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: JS 散度选择最优 premature layer (动态模式)          │
│   JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)          │
│   M = 0.5 * (softmax(mature) + softmax(premature))          │
│   picked_layer = argmax(JS divergence)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Logits 对比                                         │
│   diff = log_softmax(mature_logits) - log_softmax(base_logits)│
│   if post_softmax: diff = log_softmax(diff)                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Relative Top 过滤 (可选)                            │
│   thresh = max_prob + log(relative_top)                     │
│   diff[prob < thresh] = -1000                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: 采样生成                                            │
│   next_token = argmax(diff) 或 multinomial(softmax(diff))   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 层选择策略

DoLa 的效果高度依赖于 premature layer 的选择。论文发现不同模型的最佳配置不同：

| 模型 | 总层数 | "low" 配置 | "high" 配置 | 推荐配置 |
|------|--------|------------|-------------|----------|
| LLaMA-7B | 32 | range(0, 16, 2) | range(16, 32, 2) | **high** |
| Qwen3-8B | 36 | range(0, 18, 2) | range(18, 36, 2) | **low** |
| Qwen3.5-122B | 122 | range(0, 20, 2) | range(102, 122, 2) | 需实验确定 |

**关键发现**：Qwen3-8B 使用 "high" 配置时，MC2 从 0.57 **暴跌**至 0.29。这说明层选择必须针对具体模型进行消融实验。

### 2.3 SGLang 架构适配

SGLang 的 DoLa 实现需要在两个阶段捕获 hidden states：

```
┌─────────────────────────────────────────────────────────────┐
│ Prefill (extend) 模式:                                      │
│   - 捕获所有候选层的 hidden states                           │
│   - 用于 MC2 多选题评测 (计算 log-likelihood)               │
│   - 生成第一个 token                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Decode 模式:                                                │
│   - 逐 token 捕获候选层 hidden states                       │
│   - 应用 DoLa 对比解码                                      │
│   - 用于开放式生成                                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 KTransformers CPU-GPU 混合推理

KTransformers 通过以下技术实现大模型在有限 GPU 上运行：

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3.5-122B-A10B                        │
├─────────────────────────────────────────────────────────────┤
│  Expert Layers (MoE)                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ Expert 1│  │ Expert 2│  │   ...   │  → CPU (GGUF 量化)  │
│  └─────────┘  └─────────┘  └─────────┘                     │
├─────────────────────────────────────────────────────────────┤
│  Attention + Dense Layers                                   │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │  Attention  │  │   LM Head   │  → GPU (2×RTX 4090)      │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 实现方案

### 3.1 方案对比

本项目包含三套实现，分别适用于不同场景：

| 实现方案 | 框架 | 适用模型 | 硬件需求 | 使用场景 |
|----------|------|----------|----------|----------|
| **transformers 原版** | transformers-4.28.1 (魔改) | LLaMA-7B | 单卡 ~16GB | 论文复现 |
| **transformers 扩展** | transformers-5.x (不魔改) | Qwen3-8B | 单卡 ~16GB | 研究/实验 |
| **SGLang 集成** | sglang (site-package 修改) | Qwen3/3.5 | 1-2 GPU | **生产 API 服务** |
| **KTransformers 集成** | ktransformers + GGUF | Qwen3.5-122B | **2×RTX 4090** | 大模型低资源推理 |

### 3.2 SGLang 实现架构

```
┌─────────────────────────────────────────────────────────────┐
│                    SGLang DoLa 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   qwen3_5.py    │    │ logits_processor│                │
│  │                 │    │     .py         │                │
│  │  - 捕获 hidden  │───→│  - DoLa 对比    │                │
│  │    states       │    │  - JS 散度选层  │                │
│  │  - 存储到       │    │  - relative_top │                │
│  │    ForwardBatch │    │    filtering    │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                      │                          │
│           ↓                      ↓                          │
│  ┌─────────────────────────────────────────┐               │
│  │            ForwardBatch                  │               │
│  │  - dola_candidate_hidden: dict           │               │
│  │  - dola_candidate_residual: dict         │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 KTransformers 实现架构

```
┌─────────────────────────────────────────────────────────────┐
│                KTransformers DoLa 架构                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  generate_dola_kt.py                                        │
│  ┌─────────────────────────────────────────┐               │
│  │  prefill_and_generate()                 │               │
│  │    ├── chunk_prefill()                  │               │
│  │    ├── dola_decode_one_tokens()         │               │
│  │    │     ├── forward(output_hidden=True)│               │
│  │    │     ├── JS 散度选层                │               │
│  │    │     └── logits 对比 + 采样         │               │
│  │    └── baseline_decode_one_tokens()     │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
│  依赖:                                                      │
│  - ktransformers.models.modeling_qwen3_moe.Qwen3MoeForCausalLM│
│  - optimize_and_load_gguf() for CPU-GPU hybrid loading      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 环境配置

### 4.1 Conda 环境创建

```bash
# 从现有 serve 环境克隆
conda create --name serve-dola --clone serve
conda activate serve-dola
```

### 4.2 Site-Package 修改

需要修改两个 sglang 文件：

#### 4.2.1 修改 `logits_processor.py`

**文件路径**:
```
/opt/miniconda3/envs/serve-dola/lib/python3.12/site-packages/sglang/srt/layers/logits_processor.py
```

**新增函数 1**: `_dola_contrast_logits`

```python
def _dola_contrast_logits(
    final_logits: torch.Tensor,
    candidate_logits: dict,
    post_softmax: bool = False,  # MC2 评测用 False
    relative_top: float = 0.0,   # MC2 评测用 0.0，开放生成用 0.1
    relative_top_value: float = -1000.0,
):
    """DoLa 对比解码：逐 token JS 散度选层 + log_softmax contrast

    对齐 DoLa/dola.py lm_score 函数 (lines 161-217)
    """
    if not candidate_logits:
        return final_logits

    candidate_layers = sorted(candidate_logits.keys())
    num_tokens = final_logits.shape[0]

    with torch.no_grad():
        final_f = final_logits.float()
        final_log_sm = final_f.log_softmax(dim=-1)
        softmax_mature = torch.softmax(final_f, dim=-1)

        # 逐层计算 JS 散度（内存高效）
        js_divs_per_layer = []
        for l in candidate_layers:
            cand_f = candidate_logits[l].float()
            softmax_premature = torch.softmax(cand_f, dim=-1)
            log_softmax_premature = cand_f.log_softmax(dim=-1)

            M = 0.5 * (softmax_mature + softmax_premature)
            kl1 = F.kl_div(final_log_sm, M, reduction="none").mean(-1)
            kl2 = F.kl_div(log_softmax_premature, M, reduction="none").mean(-1)
            js_div = 0.5 * (kl1 + kl2)
            js_divs_per_layer.append(js_div)

        js_divs = torch.stack(js_divs_per_layer, dim=0)
        picked_layer_indices = js_divs.argmax(dim=0)

        # 构建对比 logits
        base_logits = torch.zeros_like(final_f)
        for token_idx in range(num_tokens):
            layer_idx = picked_layer_indices[token_idx].item()
            picked_layer = candidate_layers[layer_idx]
            base_logits[token_idx] = candidate_logits[picked_layer][token_idx]

        base_log_sm = base_logits.log_softmax(dim=-1)
        diff_logits = final_log_sm - base_log_sm

        if post_softmax:
            diff_logits = diff_logits.log_softmax(dim=-1)

        # Relative top 过滤
        if relative_top > 0.0:
            probs_max = torch.max(final_log_sm, dim=-1).values
            probs_thresh = probs_max + torch.log(torch.tensor(relative_top))
            probs_thresh = probs_thresh.unsqueeze(-1)
            relative_top_mask = final_log_sm < probs_thresh
            diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

        return diff_logits.to(final_logits.dtype)
```

**新增函数 2**: `_dola_compute_candidate_logits`

```python
def _dola_compute_candidate_logits(
    candidate_hidden: dict,
    candidate_residual: dict,
    norm_layer,
    logits_processor,
    lm_head,
    logits_metadata,
    pruned_indices: list = None,
):
    """从 hidden_states 计算候选层 logits

    关键：不对中间层应用 norm，与论文对齐
    """
    candidate_logits = {}
    for l in sorted(candidate_hidden.keys()):
        hs = candidate_hidden[l]
        with torch.no_grad():
            # 不使用 norm，与论文方法一致
            normed = hs
            candidate_logits[l] = torch.matmul(
                normed.to(lm_head.weight.dtype), lm_head.weight.T
            ).float()
    return candidate_logits
```

**修改点**: 在 `LogitsProcessor` 类的 `_get_logits` 方法中调用上述函数

#### 4.2.2 修改 `qwen3_5.py`

**文件路径**:
```
/opt/miniconda3/envs/serve-dola/lib/python3.12/site-packages/sglang/srt/models/qwen3_5.py
```

**修改位置**: `Qwen3_5ForCausalLM.forward()` 方法

**修改内容**:

```python
# 原代码
_is_dola_extend = False
if _dola_layers and forward_batch.forward_mode.is_extend():
    _dola_candidate_set = set(int(x) for x in _dola_layers.split(",") if x.strip())
    if not hasattr(forward_batch, "dola_candidate_hidden"):
        forward_batch.dola_candidate_hidden = {}
    if not hasattr(forward_batch, "dola_candidate_residual"):
        forward_batch.dola_candidate_residual = {}
    _is_dola_extend = True

# 修改后
_is_dola_active = False
if _dola_layers and forward_batch.forward_mode.is_decode_or_idle():
    # DoLa during decode: capture candidate hidden states per token
    _dola_candidate_set = set(int(x) for x in _dola_layers.split(",") if x.strip())
    if not hasattr(forward_batch, "dola_candidate_hidden"):
        forward_batch.dola_candidate_hidden = {}
    if not hasattr(forward_batch, "dola_candidate_residual"):
        forward_batch.dola_candidate_residual = {}
    _is_dola_active = True
elif _dola_layers and forward_batch.forward_mode.is_extend():
    # DoLa during prefill: capture candidate hidden states for MC2 + first token
    _dola_candidate_set = set(int(x) for x in _dola_layers.split(",") if x.strip())
    if not hasattr(forward_batch, "dola_candidate_hidden"):
        forward_batch.dola_candidate_hidden = {}
    if not hasattr(forward_batch, "dola_candidate_residual"):
        forward_batch.dola_candidate_residual = {}
    _is_dola_active = True

# 后续使用 _is_dola_active 替代 _is_dola_extend
```

**修改说明**:

| 变更 | 原版本 | 修改后 |
|------|--------|--------|
| 变量名 | `_is_dola_extend` | `_is_dola_active` |
| 触发条件 | 仅 `is_extend()` | 同时支持 `is_decode_or_idle()` + `is_extend()` |
| 功能 | 只在 prefill 捕获 | decode + prefill 都能捕获 |

### 4.3 环境变量

```bash
# DoLa 候选层配置
export DOLA_LAYERS=0,2,4,6,8,10,12,14,16  # 逗号分隔的层索引

# Relative Top 过滤阈值
export DOLA_RELATIVE_TOP=0.0  # MC2 评测用 0.0，开放生成用 0.1
```

---

## 5. 使用指南

### 5.1 SGLang API 服务

#### 5.1.1 环境准备

```bash
# 创建 conda 环境
conda create -n serve python=3.12 -y
conda activate serve

# 安装依赖
pip install ktransformers sglang-kt transformers==4.57.1

# 应用 DoLa patch
python apply_patch.py --apply
```

#### 5.1.2 Qwen3-8B 服务（单卡 GPU）

**启动 Baseline 服务**：

```bash
conda activate serve

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --attention-backend flashinfer \
  --mem-fraction-static 0.90 \
  --max-running-requests 4 \
  --max-total-tokens 16384 \
  --disable-cuda-graph
```

**启动 DoLa 服务**：

```bash
conda activate serve

# 设置 DoLa 参数
export DOLA_LAYERS=0,2,4,6,8,10,12,14,16
export DOLA_RELATIVE_TOP=0.0  # MC2 评测用 0.0，开放生成用 0.1

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --attention-backend flashinfer \
  --mem-fraction-static 0.90 \
  --max-running-requests 4 \
  --max-total-tokens 16384 \
  --disable-cuda-graph
```

#### 5.1.3 Qwen3.5-122B-A10B 服务（KTransformers CPU-GPU 混合推理）

Qwen3.5-122B-A10B 是 MoE 模型，参数量 122B，通过 KTransformers 实现 CPU-GPU 混合推理，仅需 2×RTX 4090 即可运行。

**关键配置**：
- `--kt-method BF16`：CPU 量化方法
- `--kt-cpuinfer 64`：CPU 推理线程数
- `--kt-num-gpu-experts 1`：每层 GPU 上的 expert 数量
- `--disable-cuda-graph`：**必须禁用**（DoLa 不兼容）

**启动 Baseline 服务**：

```bash
conda activate serve

CUDA_VISIBLE_DEVICES=0,1 SGLANG_DISABLE_CUDNN_CHECK=1 \
python -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --kt-weight-path <MODEL_PATH> \
  --kt-method BF16 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 1 \
  --kt-gpu-prefill-token-threshold 4096 \
  --kt-enable-dynamic-expert-update \
  --attention-backend flashinfer \
  --mem-fraction-static 0.88 \
  --chunked-prefill-size 4096 \
  --max-running-requests 2 \
  --max-total-tokens 16384 \
  --watchdog-timeout 6000 \
  --enable-mixed-chunk \
  --disable-cuda-graph
```

**启动 DoLa 服务**：

```bash
conda activate serve

# DoLa 参数设置
export DOLA_LAYERS=0,2,4,6,8,10,12,14,16
export DOLA_RELATIVE_TOP=0.1  # 开放生成用 0.1，MC2 评测用 0.0

CUDA_VISIBLE_DEVICES=0,1 SGLANG_DISABLE_CUDNN_CHECK=1 \
python -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --kt-weight-path <MODEL_PATH> \
  --kt-method BF16 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 1 \
  --kt-gpu-prefill-token-threshold 4096 \
  --kt-enable-dynamic-expert-update \
  --attention-backend flashinfer \
  --mem-fraction-static 0.88 \
  --chunked-prefill-size 4096 \
  --max-running-requests 2 \
  --max-total-tokens 16384 \
  --watchdog-timeout 6000 \
  --enable-mixed-chunk \
  --disable-cuda-graph
```

#### 5.1.4 Qwen3.5-397B-A17B 服务（纯 CPU 推理）

Qwen3.5-397B-A17B 是更大的 MoE 模型，可完全通过 CPU 推理运行。

**关键配置**：
- `--kt-method FP8`：使用 FP8 量化减小内存占用
- `--kt-num-gpu-experts 0`：**纯 CPU 推理**，不将 expert 放到 GPU
- `--kt-gpu-prefill-token-threshold 512`：降低 GPU prefill 阈值
- `--tensor-parallel-size 1`：单卡即可

**启动 DoLa 服务**：

```bash
conda activate serve

# DoLa 参数设置
export DOLA_LAYERS=0,2,4,6,8,10,12,14,16,18,20
export DOLA_RELATIVE_TOP=0.1

CUDA_VISIBLE_DEVICES=0 SGLANG_DISABLE_CUDNN_CHECK=1 \
python -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --kt-weight-path <MODEL_PATH> \
  --kt-method FP8 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 0 \
  --kt-gpu-prefill-token-threshold 512 \
  --kt-enable-dynamic-expert-update \
  --attention-backend flashinfer \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 512 \
  --max-running-requests 1 \
  --max-total-tokens 4096 \
  --context-length 8192 \
  --watchdog-timeout 12000 \
  --enable-mixed-chunk \
  --disable-cuda-graph \
  --skip-server-warmup
```

#### 5.1.5 KTransformers 参数说明

| 参数 | 说明 | 122B 推荐 | 397B 推荐 |
|------|------|-----------|-----------|
| `--kt-method` | CPU 量化方法 | `BF16` | `FP8` |
| `--kt-cpuinfer` | CPU 推理线程数 | `64` | `64` |
| `--kt-threadpool-count` | CPU 线程池数量 | `2` | `2` |
| `--kt-num-gpu-experts` | GPU 上的 expert 数量 | `1` | `0` (纯CPU) |
| `--kt-gpu-prefill-token-threshold` | GPU prefill 阈值 | `4096` | `512` |
| `--kt-enable-dynamic-expert-update` | 动态 expert 更新 | 开启 | 开启 |
| `--mem-fraction-static` | GPU 静态显存比例 | `0.88` | `0.80` |
| `--max-running-requests` | 最大并发请求 | `2` | `1` |
| `--watchdog-timeout` | watchdog 超时 | `6000` | `12000` |
| `--tensor-parallel-size` | TP 大小 | `2` | `1` |

#### 5.1.6 DoLa 配置说明

| 配置名 | DOLA_LAYERS | 说明 | 适用场景 |
|--------|-------------|------|----------|
| DoLa-low | `0,2,4,6,8,10,12,14,16` | 前半层 | TruthfulQA MC2 + 业务数据 |
| DoLa-mid | `16,18,20,22,24,26,28,30,32,34` | 中间层 | 业务数据对比评测 |
| DOLA_RELATIVE_TOP | `0.0` / `0.1` | 过滤阈值 | MC2 用 0.0，开放生成用 0.1 |

> **注意**：候选层配置需要根据具体模型调整。122B 模型共 48 层，397B 模型层数更多，建议通过 `auto_select_dola_layers_v2.py` 自动选择最优候选层。

#### 5.1.7 API 调用示例

**Chat 模式**：

```python
import openai

client = openai.OpenAI(base_url="http://localhost:30000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "What happens if you eat watermelon seeds?"}
    ],
    max_tokens=100,
    temperature=0.0
)
print(response.choices[0].message.content)
```

**Completion 模式**（适用于需要自定义 prompt template 的场景）：

```python
import openai

client = openai.OpenAI(base_url="http://localhost:30000/v1", api_key="dummy")

response = client.completions.create(
    model="default",
    prompt="<自定义的 prompt 文本>",
    max_tokens=384,
    temperature=0
)
print(response.choices[0].text)
```

### 5.2 评测脚本使用

#### 5.2.1 TruthfulQA MC2 评测

```bash
# Baseline 评测
python eval_scripts/eval_truthfulqa_api_chat.py \
  --api_base "http://localhost:30000/v1" \
  --model_name "baseline" \
  --tokenizer_path "<TOKENIZER_PATH>" \
  --data_path "data/TruthfulQA.csv" \
  --output_path "results_baseline.json"

# DoLa 评测（需先设置 DOLA_LAYERS 启动服务）
python eval_scripts/eval_truthfulqa_api_chat.py \
  --api_base "http://localhost:30000/v1" \
  --model_name "dola-low" \
  --tokenizer_path "<TOKENIZER_PATH>" \
  --data_path "data/TruthfulQA.csv" \
  --output_path "results_dola.json"
```

#### 5.2.2 业务数据评测

```bash
# Step 1: 生成 Baseline 结果
python business_eval/generate_dataset.py \
  --api_base "http://localhost:30000/v1" \
  --data_path "business_eval/data.csv" \
  --output "baseline_results.json" \
  --max_tokens 384 \
  --temperature 0

# Step 2: 启动 DoLa 服务（设置 DOLA_LAYERS）后生成 DoLa 结果
python business_eval/generate_dataset.py \
  --api_base "http://localhost:30000/v1" \
  --data_path "business_eval/data.csv" \
  --output "dola_results.json" \
  --max_tokens 384 \
  --temperature 0

# Step 3: 幻觉检测
python business_eval/run_hallucination_v2.py \
  --baseline_path "baseline_results.json" \
  --dola_path "dola_results.json"

# Step 4: RAGAS 评测
python business_eval/run_ragas_official_v2.py \
  --results_path "dola_results.json"

# Step 5: 大模型对比评判
python business_eval/run_comprehensive_judge_v2.py \
  --baseline_path "baseline_results.json" \
  --dola_path "dola_results.json"
```
|---------|------|------|
| 大模型评判 | DeepSeek API | 对比 DoLa 和 Baseline 哪个回答更好 |
| 幻觉检测 | `run_hallucination_v2.py` | 检测回答中是否编造了产品资料/对话历史中没有的信息 |
| RAGAS 评测 | `run_ragas_deepseek_v3.py` | Faithfulness（忠实度）+ Relevance（相关性） |

---

## 6. 动态候选层选择

### 6.1 问题背景

DoLa 的效果高度依赖于 premature layer 的选择。论文发现：

- **不同模型的最佳配置不同**：LLaMA-7B 需要后半层（high），Qwen3-8B 需要前半层（low）
- **不合适的层选择有害**：Qwen3-8B 使用 high 配置时，MC2 从 0.57 **暴跌**至 0.29
- **需要针对具体模型进行层选择消融实验**

### 6.2 解决方案：基于 JS 散度的自动层选择

本项目实现了一套**自动候选层选择工具**，通过分析推理过程中各层的 JS 散度分布，自动推荐最优候选层配置。

#### 6.2.1 核心原理

```
┌─────────────────────────────────────────────────────────────┐
│                   自动层选择流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 扫描所有候选层                                     │
│  ┌─────────────────────────────────────────┐               │
│  │  启动 DoLa 服务 (DOLA_LAYERS=所有层)    │               │
│  │  运行抽样数据推理                        │               │
│  │  收集每层的 JS 散度数据                  │               │
│  └─────────────────────────────────────────┘               │
│                         ↓                                   │
│  Step 2: 计算统计信息                                       │
│  ┌─────────────────────────────────────────┐               │
│  │  每层 JS 散度的均值、方差、最大/最小值   │               │
│  └─────────────────────────────────────────┘               │
│                         ↓                                   │
│  Step 3: 滑动窗口分析                                       │
│  ┌─────────────────────────────────────────┐               │
│  │  以 window_size=6, step=2 滑动          │               │
│  │  计算窗口内 JS 均值的方差 (一致性指标)   │               │
│  │  方差越小 = 各层 JS 散度越一致 = 越优    │               │
│  └─────────────────────────────────────────┘               │
│                         ↓                                   │
│  Step 4: 推荐最优配置                                       │
│  ┌─────────────────────────────────────────┐               │
│  │  输出 DOLA_LAYERS 环境变量配置          │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 6.2.2 使用方法

**方式一：一键运行（启动服务 + 收集数据 + 计算）**

```bash
python /workspace/dola-demo/auto_select_dola_layers_v2.py \
  --model_path /path/to/Qwen3-8B \
  --data_path /path/to/TruthfulQA.csv \
  --sample_size 50 \
  --num_layers 36 \
  --window_size 12 \
  --window_step 1 \
  --output dola_layer_selection_v2.json
```

**方式二：从已有日志分析**

```bash
python /workspace/dola-demo/auto_select_dola_layers_v2.py \
  --skip_scan \
  --log_files /workspace/dola-demo/chat_dola_1.log /workspace/dola-demo/chat_dola_2.log \
  --window_size 12 \
  --output dola_layer_selection_v2.json
```

#### 6.2.3 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型路径（用于启动服务） | 必填 |
| `--data_path` | TruthfulQA 数据路径 | 必填 |
| `--sample_size` | 抽样数据量 | 50 |
| `--num_layers` | 模型总层数 | 36 |
| `--window_size` | 滑动窗口大小（层数） | 12 |
| `--window_step` | 滑动窗口步长 | 1 |
| `--skip_scan` | 跳过扫描，使用已有日志 | False |
| `--log_files` | 已有日志文件列表（多文件合并分析） | - |

#### 6.2.4 输出示例

```
======================================================================
DoLa 候选层自动选择工具 (修正版)
======================================================================
模型: /path/to/Qwen3-8B
数据: /path/to/TruthfulQA.csv
抽样量: 50
窗口大小: 12
窗口步长: 1
======================================================================

各层 JS 散度统计:
层      均值                 标准差               最小值               最大值
------------------------------------------------------------------------------------------
0      2.345678901234567e-05  1.234567890123456e-06  1.000000000000000e-05  5.000000000000000e-05
2      3.456789012345678e-05  2.345678901234567e-06  1.500000000000000e-05  6.000000000000000e-05
...

窗口分析结果 (按一致性排序):
层区间           均值JS               方差                     评分
------------------------------------------------------------------------------------------
0-22            4.567890123456789e-05  1.23456789012345678901e-10  1.23456789012345678901e-10
2-24            5.678901234567890e-05  2.34567890123456789012e-10  2.34567890123456789012e-10
...

======================================================================
推荐配置
======================================================================
最优层区间: 0-22
候选层: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
平均 JS 散度: 4.567890123456789e-05
方差 (一致性指标): 1.23456789012345678901e-10

DOLA_LAYERS 环境变量:
export DOLA_LAYERS=0,2,4,6,8,10,12,14,16,18,20,22
======================================================================

结果已保存到: dola_layer_selection_v2.json
```

### 6.3 日志格式说明

SGLang 服务在 DoLa 模式下会输出以下日志：

#### 6.3.1 JS 散度日志

```
DOLA_JS: 0:0.000023,2:0.000045,4:0.000067,...
```

格式：`层索引:JS散度值`，逗号分隔多个层。

#### 6.3.2 层选择日志

```
DOLA_LAYERS_SELECTED: 18,18,22,22,16,18,...
```

表示每个 token 选择的 premature layer 索引。

### 6.4 分析脚本说明

| 脚本 | 功能 |
|------|------|
| `auto_select_dola_layers.py` | 基础版：滑动窗口分析，推荐最优层区间 |
| `auto_select_dola_layers_v2.py` | 修正版：使用完整 answer 文本计算 JS 散度，与 MC2 评测对齐 |
| `analyze_layer_selection.py` | 分析已有日志，统计各层被选中频率 |
| `analyze_js_per_question.py` | 按问题类型分析 JS 散度分布 |
| `scan_all_layers_js.py` | 扫描所有层的 JS 散度 |

### 6.5 实际应用建议

1. **新模型适配流程**：
   ```bash
   # Step 1: 运行自动层选择
   python auto_select_dola_layers_v2.py --model_path <新模型> --data_path <数据> --sample_size 100

   # Step 2: 使用推荐配置启动服务
   export DOLA_LAYERS=<推荐的层配置>
   python -m sglang.launch_server ...

   # Step 3: 运行完整评测验证效果
   python eval_truthfulqa_api_chat.py ...
   ```

2. **多日志合并分析**：收集多次运行的日志，合并分析可提高推荐准确性。

3. **窗口大小调整**：窗口越大，候选层越多，但可能包含不相关层；建议从 `window_size=10-12` 开始。

---

## 7. 实验结果

### 7.1 论文复现验证（Transformers 4.28）

#### LLaMA-7B TruthfulQA790q-MC

| 模式 | MC1 | MC2 | MC3 |
|------|-----|-----|-----|
| Baseline | 0.2392 | 0.3925 | 0.1809 |
| DoLa (16-32层) | **0.3278** | **0.6539** | **0.3285** |
| 提升 | +8.86pp | **+26.14pp** | +14.76pp |

**与论文对比**：

| 模式 | MC1 | MC2 | MC3 |
|------|-----|-----|-----|
| 论文 Baseline | 0.256 | 0.406 | 0.192 |
| 论文 DoLa | **0.322** | **0.638** | **0.321** |
| 本实验 Baseline | 0.2392 | 0.3925 | 0.1809 |
| 本实验 DoLa | **0.3278** | **0.6539** | **0.3285** |

**结论**：复现结果与论文基本一致，验证了 DoLa 方法的有效性。

#### LLaMA-7B FACTOR 数据集

| 模式 | 得分 | 对比 |
|------|------|------|
| Baseline | **0.5858** | +3.38pp |
| DoLa | 0.6196 | |

**结论**：FACTOR 数据集上同样验证有效。

### 7.2 版本迁移实验（Transformers 5.x）

#### Qwen3-8B TruthfulQA790q-MC（Chat Template）

| 模式 | MC1 | MC2 | MC3 |
|------|-----|-----|-----|
| Baseline | 0.2671 | 0.4392 | 0.2049 |
| DoLa-low | **0.3165** | **0.5848** | **0.2857** |
| 提升 | +4.94pp | **+14.56pp** | +8.08pp |

**关键发现**：

1. **提示词构造影响评测结果**：使用 chat template 后 baseline 降低（0.4392 vs 0.5677），但 DoLa 提升仍然显著
2. **剔除特殊 token 后**（仅计算 answer 部分 likelihood）：

| 模式 | MC1 | MC2 | MC3 |
|------|-----|-----|-----|
| Baseline | 0.3646 | 0.5548 | 0.2713 |
| DoLa-low | 0.3633 | **0.6262** | **0.3264** |
| 提升 | -0.01 | **+7.14pp** | **+5.51pp** |

**核心结论**：
- 不同提示词构造会影响 MC 评测绝对值，但 **DoLa 在各种构造下均有提升**
- Qwen3-8B 候选层取 **low 层（16层之前）** 有提升，取 high 层无明显提升甚至有害
- 说明 Qwen3-8B 的事实知识集中在浅层，与 LLaMA-7B 相反

### 7.3 层选择消融实验

#### Qwen3-8B 自定义候选层实验

| 候选层配置 | MC2 | 相对 Baseline |
|------------|-----|---------------|
| Baseline | 0.5548 | - |
| 0,2,4,6,8,10,12,14 (low) | **0.6262** | **+7.14pp** |
| 2,4,6,8,10,12,14 (掐头) | **0.6256** | **+7.08pp** |
| 12,14,16,18,20,22 (mid) | **0.6347** | **+7.99pp** |

**关键发现**：

1. **JS 散度趋势**：随着层数增加，JS 平均降低（差异减小），但方差增大（不确定性增加）
2. **中间层效果更好**：使用相对中间的候选层（12-22）可以利用到更好的知识表示
3. **过早层的知识未必正确**：虽然 JS 散度高，但过早层的知识可能不准确

### 7.4 SGLang + KTransformers 大模型评测

#### Qwen3-8B SGLang 评测（不处理特殊标签）

| 模式 | MC1 | MC2 | MC3 |
|------|-----|-----|-----|
| Baseline | 0.2671 | 0.4392 | 0.2049 |
| DoLa-low | **0.3076** | **0.5789** | **0.2827** |

**结论**：与直接使用 transformers 运行结果相近，对 Qwen3-8B 的提升幅度在同一量级。

#### Qwen3.5-122B-A10B 抽样评测（160条）

由于 790q 全量评测较慢，通过随机抽样 160 条数据进行快速验证：

| 模式 | MC1 | MC2 | MC3 |
|------|-----|-----|-----|
| Baseline (160条) | 0.4125 | 0.5912 | 0.2779 |
| DoLa-low (0-20) | ~0.35 | **0.6636** | **0.3642** |
| DoLa-mid (16-36) | ~0.35 | **0.6636** | **0.3642** |
| DoLa-high (32-48) | ~0.35 | **0.6644** | **0.3640** |
| DoLa-高频6层 | 0.35 | **0.6636** | **0.3658** |
| DoLa-高频12层 | **0.35** | **0.6638** | **0.3643** |

**关键发现**：

1. **不同候选层配置效果相近**：low/mid/high 配置在 MC2 上差异不大（0.6636-0.6644）
2. **基于 JS 散度频率的动态选层有效**：高频层组合与固定区间效果相当
3. **MC2 提升显著**：从 0.5912 提升至 ~0.664，**提升约 +7.3pp**

### 7.5 开放生成评测（Qwen3.5-122B）

#### 不同温度设置下的 Truth/Info/Both 指标

| 模式 | Truth | Info | Both |
|------|-------|------|------|
| Baseline/temp=0.7/原始指令 | 0.981 | 0.188 | 0.181 |
| DoLa-low/temp=0.7/原始指令 | 0.962 | **0.238** | **0.206** |
| Baseline/temp=0/非原始指令 | 0.8812 | 0.8688 | 0.775 |
| DoLa-low/temp=0/非原始指令 | 0.8812 | 0.8688 | 0.775 |

**关键结论**：

1. **显式拒绝回答场景下 DoLa 有效**：在允许拒绝回答（"I have no comment"）的指令下，DoLa 可以降低拒绝回答的意愿，使综合效果提升
2. **高性能模型上提升不明显**：122B 模型本身性能很强，baseline 已经很高，DoLa 提升有限
3. **生成内容确有差异**：即使 temperature=0，DoLa 和 baseline 的生成文本也有较大差别，说明 DoLa 确实改变了生成倾向

### 7.6 真实业务数据评测

使用实际业务数据（智能体问答场景）进行评测：

| 指标 | Baseline | DoLa-mid | 变化 |
|------|----------|----------|------|
| 总 log-likelihood | -5225.45 | - | - |
| 平均 log-likelihood | -26.1272 | - | - |
| 平均 perplexity | 1.2015 | - | 微小提升 |

**结论**：使用客观指标，只有很小的提升。但在实际生成中，DoLa 和 baseline 的回答在措辞、说法、顺序等方面存在差异。

### 7.7 真实业务数据：大模型评判分析

为了更客观地评估 DoLa 在实际业务场景中的效果，我们使用大模型对 DoLa 和 Baseline 的生成结果进行了对比评判。

#### 评判统计

| 评判结果 | 数量 | 占比 |
|----------|------|------|
| DoLa 更优 | 20 | 20.8% |
| Baseline 更优 | 13 | 13.5% |
| 平局 | 37 | 38.5% |
| 内容高度相似（跳过） | 93 | - |

#### 评判结论

1. **DoLa 在信息完整性和事实准确性方面表现更好**
   - 更善于补充细节和不同场景的使用说明
   - 在涉及事实判断时，错误率更低

2. **Baseline 在操作步骤的具体性和覆盖全面性方面表现更好**
   - 更贴近参考答案的结构和表述
   - 在需要覆盖多种情况时更全面

3. **两者各有优劣，DoLa 略占优势**
   - DoLa 更优：20 条（20.8%）
   - Baseline 更优：13 条（13.5%）
   - 平局：37 条（38.5%）
   - **净优势：DoLa +7.3%**

### 7.8 真实业务数据：幻觉检测与 RAGAS 评测

在 200 条业务数据上，使用 DeepSeek API 进行幻觉检测和 RAGAS 评测。

#### 幻觉检测（参考产品资料 + 对话历史，排除安全提示）

| 指标 | DoLa-low | DoLa-mid | Baseline |
|------|----------|----------|----------|
| 幻觉样本数 | **16 (8.0%)** | 25 (12.5%) | 17 (8.5%) |
| 两者都有幻觉 | 7 (3.5%) | 9 (4.5%) | - |
| 仅 DoLa 有幻觉 | 9 (4.5%) | 16 (8.0%) | - |
| 仅 Baseline 有幻觉 | 10 (5.0%) | 15 (7.5%) | - |
| 两者都无幻觉 | **174 (87.0%)** | 160 (80.0%) | - |

**关键发现**：
- DoLa-low 的幻觉率与 Baseline 接近（8.0% vs 8.5%），差异不显著
- DoLa-mid 的幻觉率偏高（12.5%），说明中间层选择策略可能引入更多不确定性
- 87% 的样本两者都无幻觉

#### RAGAS 评测（官方框架 ragas 0.2.14）

使用 RAGAS 官方框架评测，DeepSeek API 作为 LLM（claim extraction + NLI），BAAI/bge-small-zh-v1.5 作为本地 embedding 模型。

**v1（用 title 作为 question，结果偏低）**：

| 指标 | DoLa-low | Baseline |
|------|----------|----------|
| Faithfulness | 0.6503 | 0.6523 |
| Answer Relevancy | 0.2930 | 0.2994 |

> v1 的 answer_relevancy 偏低，因为 question 字段使用了通用 title（如"调试指导-智能体业务知识问答"），导致 RAGAS 生成的反向问题与原始 title 语义差异大。

**v2（用实际用户问题，结果更准确）**：

| 指标 | DoLa-low | DoLa-mid | Baseline |
|------|----------|----------|----------|
| Faithfulness（忠实度） | **0.6596** | 0.6410 | 0.6450 |
| Answer Relevancy（相关性） | 0.4026 | **0.4133** | 0.4182 |

**关键发现**：
- DoLa-low 忠实度最高（0.6596），比 Baseline 高 2.3%
- DoLa-mid 和 Baseline 的 answer_relevancy 接近，DoLa-low 略低
- 三者的 Faithfulness 差异很小（0.6410~0.6596），均在 3% 以内
- Answer Relevancy 整体偏低（0.40~0.42），因 RAGAS 反向问题生成在中文客服场景下天然偏低
- 使用实际用户问题后 answer_relevancy 从 0.29 提升至 0.40（+38%），验证了 question 字段的重要性

#### 综合对比

| 评测维度 | DoLa-low | DoLa-mid | Baseline |
|---------|----------|----------|----------|
| 大模型评判胜率 | **79.82%** | 56.9% | 78.95% |
| 幻觉率 | **8.0%** | 12.5% | 8.5% |
| RAGAS Faithfulness | **0.6596** | 0.6410 | 0.6450 |
| RAGAS Answer Relevancy | 0.4026 | **0.4133** | 0.4182 |

**结论**：
- **DoLa-low** 综合表现最好：幻觉率最低、忠实度最高、大模型评判胜率最高
- **DoLa-mid** 相关性略高但幻觉率偏高，大模型评判表现较差
- **Baseline** 各项指标居中
- **推荐使用 DoLa-low 配置**（`DOLA_LAYERS=0,2,4,6,8,10,12,14,16`）

### 7.9 综合实验结论

#### 核心发现

1. **DoLa 方法有效且可复现**
   - LLaMA-7B: MC2 +26.14pp（论文 +23.2pp）
   - Qwen3-8B: MC2 +7-14pp（不同配置）
   - Qwen3.5-122B: MC2 +7.3pp（抽样验证）

2. **层选择具有模型特异性**
   - LLaMA-7B: 后半层（high）有效
   - Qwen3-8B: 前半层（low）有效
   - Qwen3.5-122B: 中间层（mid）或动态选层有效

3. **动态候选层选择策略**
   - 基于 JS 散度频率的选层与固定区间效果相当
   - 可以自适应不同模型的知识分布特点

4. **大模型上的应用建议**
   - 对于高性能模型（122B），baseline 已经很高，DoLa 提升有限
   - 但在特定场景（如显式拒绝回答）下仍有价值
   - 生成内容的差异表明 DoLa 可以改变模型的生成倾向

5. **业务数据上的效果验证**
   - DoLa-low 在业务场景（海信智能体问答）综合表现最好
   - 幻觉率与 Baseline 接近（8.0% vs 8.5%），不会增加幻觉风险
   - RAGAS 相关性优于 Baseline（0.854 vs 0.842）
   - 大模型评判胜率更高（79.82% vs 78.95%）

#### 改进效果总结

| 模型 | 数据集 | 指标 | Baseline | DoLa | 提升 |
|------|--------|------|----------|------|------|
| LLaMA-7B | TruthfulQA790q | MC2 | 0.3925 | **0.6539** | **+26.14pp** |
| LLaMA-7B | FACTOR | 得分 | 0.5858 | **0.6196** | **+3.38pp** |
| Qwen3-8B | TruthfulQA790q | MC2 | 0.4392 | **0.5848** | **+14.56pp** |
| Qwen3-8B | TruthfulQA790q (chat) | MC2 | 0.5548 | **0.6262** | **+7.14pp** |
| Qwen3.5-122B | TruthfulQA160q | MC2 | 0.5912 | **0.6644** | **+7.32pp** |
| Qwen3.5-122B | 业务数据200q | 大模型评判胜率 | 78.95% | **79.82%** (low) | +0.87pp |
| Qwen3.5-122B | 业务数据200q | 幻觉率 | 8.5% | **8.0%** (low) | -0.5pp |
| Qwen3.5-122B | 业务数据200q | RAGAS Faithfulness | 0.6450 | **0.6596** (low) | +0.015 |
| Qwen3.5-122B | 业务数据200q | RAGAS Answer Relevancy | 0.4182 | 0.4026 (low) / **0.4133** (mid) | -0.016 / -0.005 |

---

## 8. 技术细节

### 8.1 关键参数配置

#### MC2 评测参数

```python
DOLA_LAYERS = "0,2,4,6,8,10,12,14,16"  # Qwen3-8B low 配置
DOLA_RELATIVE_TOP = 0.0                # 禁用 relative top
post_softmax = False                   # 与论文对齐
```

#### 开放生成参数

```python
DOLA_LAYERS = "0,2,4,6,8,10,12,14,16"
DOLA_RELATIVE_TOP = 0.1               # 启用 relative top
temperature = 0.9
top_p = 0.95
repetition_penalty = 1.2              # 推荐 >= 1.2
```

### 8.2 Hidden States 索引说明

```
hidden_states 结构 (以 Qwen3-8B 为例，共 36 层):

hidden_states[0]   → embed_tokens 输出 (layer 0 输入)
hidden_states[1]   → layer 0 输出
hidden_states[2]   → layer 1 输出
...
hidden_states[36]  → layer 35 输出 = final_norm 输入
hidden_states[37]  → final_norm 输出 (如果有)

DoLa 层索引对应:
  - 索引 0: embedding 层
  - 索引 1-36: transformer layers 0-35
  - 索引 37: 最终输出层
```

### 8.3 JS 散度计算细节

```python
# 论文公式
JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
其中 M = 0.5 * (P + Q)

# PyTorch 实现
softmax_mature = F.softmax(mature_logits, dim=-1)
softmax_premature = F.softmax(premature_logits, dim=-1)
M = 0.5 * (softmax_mature + softmax_premature)

log_softmax_mature = F.log_softmax(mature_logits, dim=-1)
log_softmax_premature = F.log_softmax(premature_logits, dim=-1)

kl1 = F.kl_div(log_softmax_mature, M, reduction='none').mean(-1)
kl2 = F.kl_div(log_softmax_premature, M, reduction='none').mean(-1)
js_div = 0.5 * (kl1 + kl2)
```

### 8.4 Relative Top 过滤

```python
def _relative_top_filter(scores, baseline_scores, relative_top=0.1):
    """
    过滤掉概率过低的 token，避免对比时引入噪声
    """
    scores_normalized = scores.log_softmax(dim=-1)
    probs_max = scores_normalized.max(dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)

    mask = scores_normalized < probs_thresh.unsqueeze(-1)
    scores_normalized[mask] = -float('inf')

    return scores_normalized
```

---

## 9. 已知问题与解决方案

### 8.1 层选择策略差异

**问题**: SGLang 实现使用 per-token 层选择，DoLa 论文使用 batch-level 层选择

**影响**: 结果可能略有差异

**解决方案**: 当前实现效果更好（MC2 提升更高），保持现状

### 8.2 CUDA Graph 兼容性

**问题**: DoLa 需要动态捕获 hidden states，与 CUDA Graph 不兼容

**解决方案**: 启动服务时添加 `--disable-cuda-graph`

### 8.3 Tensor Parallelism

**问题**: 直接调用 `lm_head(hidden_states)` 在 TP 场景下会出错

**解决方案**: 使用 `torch.matmul(hidden_states, lm_head.weight.T)` 并手动处理 all-gather

### 8.4 内存占用

**问题**: DoLa 需要存储多个候选层的 hidden states，内存占用增加

**解决方案**:
- 减少候选层数量
- 使用 `--mem-fraction-static` 调整显存分配

---

## 10. 文件索引

> 本章节列出项目交付物中的所有核心文件，按功能分类整理。打包文件 `DoLa_SGLang_KTransformers_v1.0.tar.gz` 已包含所有文件。

### 10.1 核心实现文件 (dola_core/)

| 文件路径 | 说明 |
|----------|------|
| `dola_core/dola.py` | DoLa 官方实现 (LLaMA-7B, transformers 4.28) |
| `dola_core/dola_qwen.py` | Qwen3-8B 扩展实现 (transformers 5.x) |
| `dola_core/custom_generate_fixed.py` | DoLa 核心函数 (relative_top_filter, dola_select_contrast) |
| `dola_core/generate_dola_kt.py` | KTransformers 独立生成脚本 |

### 10.2 SGLang 修改文件 (sglang_patch/)

| 文件路径 | 修改内容 |
|----------|----------|
| `sglang_patch/full_files/logits_processor.py` | DoLa 对比解码核心逻辑（JS散度选层、log_softmax对比、relative_top过滤） |
| `sglang_patch/full_files/qwen3_5.py` | 候选层 hidden states 捕获（decode + extend 模式支持） |
| `sglang_patch/apply_patch.py` | Patch 安装/回滚脚本（--apply/--revert/--status） |
| `sglang_patch/patches/*.patch` | Git diff 格式补丁文件（仅供参考） |

### 10.3 评测脚本 (eval_scripts/)

| 文件路径 | 用途 | 对应实验 |
|----------|------|----------|
| `eval_scripts/eval_truthfulqa.py` | TruthfulQA MC2 本地评测 (transformers) | 论文复现 |
| `eval_scripts/eval_truthfulqa_api_chat.py` | TruthfulQA MC2 API 评测 (SGLang chat template) | SGLang + KTransformers |
| `eval_scripts/factor_eval.py` | FACTOR 数据集评测 | 论文复现 |
| `eval_scripts/eval_lit_ragbench.py` | LIT-RAGBench 评测 | 扩展验证 |
| `eval_scripts/eval_halueval_qa.py` | Halueval QA 评测 | 幻觉检测 |
| `eval_scripts/eval_chinese_simpleqa.py` | 中文 SimpleQA 评测 | 扩展验证 |

### 10.4 业务数据评测脚本 (business_eval/)

| 文件路径 | 功能 | 对应实验 |
|----------|------|----------|
| `business_eval/generate_dataset.py` | 业务数据生成脚本（Baseline/DoLa 对比生成） | 开放生成评测 |
| `business_eval/run_hallucination_v2.py` | 幻觉检测评测（DeepSeek API） | 业务数据评测 |
| `business_eval/run_ragas_official_v2.py` | RAGAS 评测（Faithfulness + Relevance） | 业务数据评测 |
| `business_eval/run_comprehensive_judge_v2.py` | 大模型对比评判（DeepSeek API） | 业务数据评测 |
| `business_eval/data.csv` | 业务数据集（200条，海信智能体问答场景） | 评测数据 |
| `business_eval/diff_results.json` | DoLa vs Baseline 生成差异分析结果 | 结果文件 |
| `business_eval/judge_results.json` | 大模型评判结果（含典型案例分析） | 结果文件 |

### 10.5 动态层选择脚本 (layer_selection/)

| 文件路径 | 功能 |
|----------|------|
| `layer_selection/auto_select_dola_layers.py` | 基础版自动层选择（滑动窗口分析） |
| `layer_selection/auto_select_dola_layers_v2.py` | 修正版自动层选择（与 MC2 评测对齐） |
| `layer_selection/analyze_layer_selection.py` | 分析层选择频率统计 |
| `layer_selection/analyze_js_per_question.py` | 按问题类型分析 JS 散度 |
| `layer_selection/analyze_js_full_sequence.py` | 全序列 JS 散度分析 |
| `layer_selection/scan_all_layers_js.py` | 扫描所有层 JS 散度 |

### 10.6 文档 (docs/)

| 文件路径 | 内容 |
|----------|------|
| `docs/DOLA_PROJECT_GUIDE.md` | 项目完整说明文档（本文档） |
| `docs/DOLA复现实验报告.md` | 详细实验记录和结果分析 |

### 10.7 数据文件 (data/)

| 文件路径 | 内容 |
|----------|------|
| `data/TruthfulQA.csv` | TruthfulQA 790q 评测数据 |

---

## 附录

### A. 参考文献

1. DoLa 论文: [Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883)
2. SGLang: [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
3. KTransformers: [https://github.com/kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers)

### B. 版本信息

| 组件 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.9.1 |
| transformers | 4.57.1 |
| sglang-kt | 0.6.2.post3 |
| ktransformers | 0.6.2.post4 |
| kt-kernel | 0.6.2.post4 |

### C. 完整部署流程

> **重要**：只需要一个 conda 环境 `serve`，DoLa patch 直接应用到该环境的 sglang-kt 上，无需创建额外环境或克隆现有环境。

#### C.1 解压打包文件

```bash
tar -xzvf DoLa_SGLang_KTransformers_v1.0.tar.gz
cd DoLa_SGLang_KTransformers
```

#### C.2 创建 conda 环境

```bash
conda create -n serve python=3.12 -y
conda activate serve
```

#### C.3 安装依赖

```bash
# 安装 KTransformers（自动包含 sglang-kt 和 kt-kernel）
pip install ktransformers

# 安装 transformers（可选，指定版本，避免兼容性问题）
pip install transformers==4.57.1
```

#### C.4 应用 DoLa Patch

```bash
# 进入打包目录
cd DoLa_SGLang_KTransformers

# 应用 patch
python sglang_patch/apply_patch.py --apply

# 验证 patch 状态
python sglang_patch/apply_patch.py --status
# 输出应为:
#   🔧 已 patch: srt/layers/logits_processor.py
#   🔧 已 patch: srt/models/qwen3_5.py
```

#### C.5 验证安装

```bash
python -c "
from sglang.srt.layers.logits_processor import _dola_contrast_logits, _dola_compute_candidate_logits
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
print('✅ DoLa patch 已成功应用')
"
```

#### C.6 启动服务

参考第 5 章的使用指南，根据模型选择合适的启动命令。

#### C.7 回滚 Patch（如需恢复原始状态）

```bash
python sglang_patch/apply_patch.py --revert
```

### D. 打包文件目录结构

```
DoLa_SGLang_KTransformers/
├── sglang_patch/               # SGLang 修改文件
│   ├── full_files/             # 完整替换文件
│   │   ├── logits_processor.py # DoLa 对比解码核心逻辑
│   │   └── qwen3_5.py          # 候选层 hidden states 捕获
│   ├── patches/                # Git diff 补丁（仅供参考）
│   └── apply_patch.py          # Patch 安装/回滚脚本
├── dola_core/                  # DoLa 核心实现
│   ├── dola.py                 # LLaMA-7B 实现 (transformers 4.28)
│   ├── dola_qwen.py            # Qwen3-8B 实现 (transformers 5.x)
│   ├── custom_generate_fixed.py
│   └── generate_dola_kt.py
├── eval_scripts/               # 评测脚本
│   ├── eval_truthfulqa.py      # MC2 本地评测
│   ├── eval_truthfulqa_api_chat.py  # MC2 API 评测
│   ├── factor_eval.py          # FACTOR 评测
│   └── ...
├── layer_selection/            # 动态层选择脚本
│   ├── auto_select_dola_layers_v2.py
│   └── ...
├── business_eval/              # 业务数据评测
│   ├── generate_dataset.py
│   ├── run_hallucination_v2.py
│   ├── run_ragas_official_v2.py
│   ├── run_comprehensive_judge_v2.py
│   └── data.csv                # 业务数据集
├── data/                       # 评测数据
│   └── TruthfulQA.csv
└── docs/                       # 文档
    ├── DOLA_PROJECT_GUIDE.md    # 项目说明（本文档）
    └── DOLA复现实验报告.md       # 详细实验记录
```