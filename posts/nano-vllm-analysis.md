# Nano-vLLM: 轻量级 vLLM 实现分析

📅 2026-04-18 · 🏷️ vLLM · 👤 mm

---

## 📦 项目概述

**Nano-vLLM** 是一个轻量级的 vLLM 从零实现，仅用约 **1400 行** Python 代码就实现了完整的离线推理能力。性能甚至略优于原生 vLLM，非常适合作为教学用例。

| 特性 | 说明 |
|------|------|
| 代码量 | 1434 行 Python |
| 性能 | 与 vLLM 持平（测试中略快） |
| 模型支持 | 目前主要支持 Qwen3 |
| 仓库 | [GitHub](https://github.com/GeeeekExplorer/nano-vllm) |

---

## 🎓 核心架构

项目采用了经典的 **Prefill-Decode 分离** 架构：

```
llm.py (5行)
    ↓
llm_engine.py (90行) ← 核心调度
    ↓
scheduler.py (84行) ← Prefill/Decode 调度
    ↓
model_runner.py (258行) ← 模型执行 + CUDA Graph
```

### 各模块职责

- **llm_engine.py** - 顶层引擎，协调调度器和模型运行器
- **scheduler.py** - 负责请求调度，决定哪些序列进行 prefill 或 decode
- **model_runner.py** - 模型推理执行，含 CUDA Graph 优化
- **block_manager.py** - KV Cache 分页管理 + 前缀缓存

---

## ⚡ 关键优化技术

### 1. Prefix Caching（前缀缓存）

利用 xxhash 快速计算 block 哈希，相同前缀的请求可以复用已计算的 KV Cache：

```python
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 链式哈希
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

### 2. CUDA Graph 优化

预捕获不同 batch size 的 CUDA Graph，推理时直接 replay，避免 kernel launch 开销：

```python
def capture_cudagraph(self):
    for bs in self.graph_bs:  # 预定义 batch size 档位
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        self.graphs[bs] = graph
```

### 3. Flash Attention

直接调用 flash-attn 的 varlen 变体，支持变长序列：

```python
o = flash_attn_varlen_func(q, k, v,
                           max_seqlen_q=context.max_seqlen_q, 
                           cu_seqlens_q=context.cu_seqlens_q,
                           max_seqlen_k=context.max_seqlen_k, 
                           cu_seqlens_k=context.cu_seqlens_k,
                           softmax_scale=self.scale, 
                           causal=True, 
                           block_table=context.block_tables)
```

### 4. Tensor Parallelism

通过 NCCL + SharedMemory 实现多卡并行推理。

---

## 📊 调度策略

调度器采用 **Continuous Batching** 策略：

1. **Prefill 阶段** - 处理新请求的输入 token，可并行
2. **Decode 阶段** - 逐 token 生成，每次只处理 1 个新 token
3. **Preemption** - 当 KV Cache 不足时，置换低优先级序列

```python
# Prefill: 处理新请求
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    if not seq.block_table:
        self.block_manager.allocate(seq)
    scheduled_seqs.append(seq)

# Decode: 逐 token 生成  
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    seq.num_scheduled_tokens = 1
    self.block_manager.may_append(seq)
```

---

## 💡 教学价值

这个项目非常适合用于教学：

1. **代码量适中** - 1400 行 vs vLLM 上万行，学生可以完整读完
2. **架构清晰** - 模块化设计，容易理解各组件职责
3. **优化点全面** - 覆盖了现代推理引擎的核心技术
4. **可直接运行** - 有完整的 example.py 和 bench.py

---

## 🚀 快速开始

```bash
# 安装
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git

# 下载模型
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
 --local-dir ~/huggingface/Qwen3-0.6B/ \
 --local-dir-use-symlinks False

# 运行示例
python example.py
```

---

## 📈 性能对比

| 推理引擎 | Output Tokens | Time (s) | Throughput (tokens/s) |
|---------|---------------|----------|----------------------|
| vLLM    | 133,966       | 98.37    | 1361.84              |
| **Nano-vLLM** | 133,966 | 93.41    | **1434.13**          |

> 测试环境: RTX 4070 Laptop (8GB), Qwen3-0.6B, 256 requests, 输入/输出长度 100-1024 tokens
