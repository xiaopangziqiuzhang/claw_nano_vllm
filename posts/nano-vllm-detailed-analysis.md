# Nano-vLLM 详细代码分析

> 一个仅 1434 行的轻量级 vLLM 实现

---

## 📁 项目结构总览

```
nano-vllm/
├── nanovllm/
│   ├── __init__.py           # 导出 LLM, SamplingParams
│   ├── llm.py                 # LLM 类（5行，仅继承）
│   ├── config.py              # 配置类
│   ├── sampling_params.py     # 采样参数
│   ├── engine/
│   │   ├── llm_engine.py      # 核心引擎（90行）
│   │   ├── scheduler.py       # 调度器（84行）
│   │   ├── model_runner.py    # 模型运行器（258行）
│   │   ├── sequence.py        # 序列对象（82行）
│   │   └── block_manager.py  # KV Cache 管理（112行）
│   ├── layers/
│   │   ├── linear.py          # 分布式线性层（156行）
│   │   ├── attention.py       # 注意力机制（75行）
│   │   ├── rotary_embedding.py # RoPE 位置编码（59行）
│   │   ├── layernorm.py       # LayerNorm（50行）
│   │   ├── sampler.py         # 采样器（12行）
│   │   ├── embed_head.py      # 嵌入 & 输出头（66行）
│   │   └── activation.py      # 激活函数（11行）
│   ├── models/
│   │   └── qwen3.py           # Qwen3 模型定义（216行）
│   └── utils/
│       ├── loader.py          # 模型加载
│       └── context.py          # 推理上下文
├── example.py                  # 使用示例
└── bench.py                     # 基准测试
```

---

## 🔥 核心执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM.generate()                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. add_request()                                           │
│     - tokenize prompt                                       │
│     - 创建 Sequence 对象                                     │
│     - 加入 scheduler.waiting 队列                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. step() 循环                                             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Scheduler.schedule()                                │   │
│  │   ├─ Prefill: 从 waiting 取请求，批量处理输入       │   │
│  │   └─ Decode: 从 running 取请求，逐 token 生成       │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ModelRunner.run()                                   │   │
│  │   ├─ prepare_prefill() / prepare_decode()          │   │
│  │   ├─ run_model() → 前向计算                         │   │
│  │   └─ Sampler() → 采样下一个 token                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Scheduler.postprocess()                             │   │
│  │   - 更新 Sequence 状态                              │   │
│  │   - 检查是否完成                                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 模块详细分析

### 1. 配置系统 (`config.py`)

```python
@dataclass(slots=True)
class Config:
    model: str                              # 模型路径
    max_num_batched_tokens: int = 16384     # 最大批处理 token 数
    max_num_seqs: int = 512                 # 最大序列数
    max_model_len: int = 4096                # 最大模型长度
    gpu_memory_utilization: float = 0.9      # GPU 显存利用率
    tensor_parallel_size: int = 1           # 张量并行度
    enforce_eager: bool = False              # 禁用 CUDA Graph
    kvcache_block_size: int = 256            # KV Cache 块大小
```

**设计亮点**：
- `slots=True` 减少内存开销
- 自动从 HuggingFace 加载模型配置
- 自动限制 max_model_len 不超过模型支持的最大长度

---

### 2. 序列管理 (`sequence.py`)

**Sequence 类**是推理的核心数据结构：

```python
class Sequence:
    block_size = 256  # 每个 block 存储 256 个 token
    
    def __init__(self, token_ids, sampling_params):
        self.seq_id = next(Sequence.counter)  # 全局唯一 ID
        self.token_ids = token_ids            # 完整 token 列表
        self.last_token = token_ids[-1]       # 最后一个 token（decode 用）
        self.num_tokens = len(token_ids)      # 总 token 数
        self.num_prompt_tokens = len(token_ids)  # prompt token 数
        self.num_cached_tokens = 0            # 已缓存的 token（前缀缓存）
        self.block_table = []                 # KV Cache 块表
```

**关键属性**：

| 属性 | 说明 |
|------|------|
| `num_cached_tokens` | 已经过 prefill 的 token，用于前缀缓存 |
| `num_scheduled_tokens` | 本轮调度的 token 数 |
| `block_table` | 存储 KV Cache 的物理块地址 |
| `completion_token_ids` | 生成的 token（排除 prompt） |

**序列状态机**：

```
WAITING → RUNNING → FINISHED
  ↑___________|___________|
   (preemption)
```

---

### 3. 调度器 (`scheduler.py`)

**核心逻辑：Continuous Batching**

```python
class Scheduler:
    def __init__(self, config):
        self.waiting: deque[Sequence] = deque()  # 新请求队列
        self.running: deque[Sequence] = deque()  # 正在推理的队列
    
    def schedule(self) -> tuple[list[Sequence], bool]:
        # 第一阶段：Prefill（处理新请求的输入）
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq):
                self.block_manager.allocate(seq)  # 分配 KV Cache
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
        
        # 第二阶段：Decode（逐 token 生成）
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            if self.block_manager.can_append(seq):  # 检查能否扩展
                self.block_manager.may_append(seq)
                seq.num_scheduled_tokens = 1  # 每次只处理 1 个
                scheduled_seqs.append(seq)
```

**调度策略详解**：

1. **Prefill 优先**：尽可能处理更多新请求的输入
2. **Chunked Prefill**：当 batch 空间不足时，分块处理长序列
3. **Preemption**：当 KV Cache 不足时，置换低优先级序列回 waiting

---

### 4. KV Cache 管理 (`block_manager.py`)

**这是项目最精彩的部分！**

#### 4.1 Block 结构

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0          # 引用计数（多序列共享）
        self.hash = -1              # block 内容哈希
        self.token_ids = []         # 存储的 token
```

#### 4.2 前缀缓存（Prefix Caching）

**核心思想**：相同前缀的序列可以复用 KV Cache

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 链式哈希
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**哈希设计**：
- 使用 xxhash64（极快）
- 链式计算：block_i 的哈希 = hash(block_i, block_{i-1})
- 相同前缀的序列会命中相同的 block 哈希

#### 4.3 分配逻辑

```python
def allocate(self, seq: Sequence):
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        
        # 计算当前 block 的哈希
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        
        # 尝试从缓存中查找
        block_id = self.hash_to_block_id.get(h, -1)
        
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            # Cache miss：从空闲块分配
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # Cache hit：复用已有块
            seq.num_cached_tokens += self.block_size
            block.ref_count += 1
        
        # 更新哈希表
        if h != -1:
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

#### 4.4 内存布局

```
┌─────────────────────────────────────┐
│           KV Cache                  │
├──────────┬──────────┬───────────────┤
│ Block 0  │ Block 1  │ Block 2  ... │
│ [0-255]  │[256-511] │[512-767]     │
│ (key)    │ (key)    │ (key)        │
├──────────┼──────────┼───────────────┤
│ Block 0  │ Block 1  │ Block 2  ... │
│ [0-255]  │[256-511] │[512-767]     │
│ (value)  │ (value)  │ (value)      │
└──────────┴──────────┴───────────────┘
```

---

### 5. 模型运行器 (`model_runner.py`)

#### 5.1 初始化

```python
def __init__(self, config, rank, event):
    # 初始化分布式（Tensor Parallelism）
    dist.init_process_group("nccl", "tcp://localhost:2333", 
                           world_size=self.world_size, rank=rank)
    torch.cuda.set_device(rank)
    
    # 加载模型
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)
    
    # 预热
    self.warmup_model()
    
    # 分配 KV Cache
    self.allocate_kv_cache()
    
    # 捕获 CUDA Graph（可选）
    if not self.enforce_eager:
        self.capture_cudagraph()
```

#### 5.2 KV Cache 分配

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    
    # 计算每个 block 的大小
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype
    
    # 计算可用的 block 数量
    num_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
    
    # 预分配 KV Cache
    self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, 
                                 num_kv_heads, head_dim)  # [2, L, N, B, H, D]
```

**内存布局**：
```
kv_cache[0] = K cache  [layer, block, head, dim]
kv_cache[1] = V cache
```

#### 5.3 CUDA Graph 优化

```python
def capture_cudagraph(self):
    max_bs = min(self.config.max_num_seqs, 512)
    
    # 预定义 batch size 档位
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    
    for bs in self.graph_bs:
        graph = torch.cuda.CUDAGraph()
        
        # 设置虚拟输入
        set_context(False, slot_mapping=slot_mapping[:bs], ...)
        
        # 捕获
        with torch.cuda.graph(graph):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        self.graphs[bs] = graph
```

**为什么需要多档位？**
- CUDA Graph 只能处理固定 shape
- 预定义 1, 2, 4, 8, 16, 32, ... 档位
- 运行时选择最接近的档位

#### 5.4 推理执行

```python
def run(self, seqs, is_prefill):
    # 准备输入
    if is_prefill:
        input_ids, positions = self.prepare_prefill(seqs)
    else:
        input_ids, positions = self.prepare_decode(seqs)
    
    # 前向计算
    logits = self.run_model(input_ids, positions, is_prefill)
    
    # 采样
    token_ids = self.sampler(logits, temperatures).tolist()
    
    # 重置上下文
    reset_context()
    return token_ids
```

---

### 6. 注意力机制 (`attention.py`)

#### 6.1 Flash Attention 集成

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        # 存储 KV 到 Cache
        if k_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # Prefill 阶段：使用 flash_attn_varlen_func
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables  # PagedAttention
            )
        else:
            # Decode 阶段：使用 flash_attn_with_kvcache
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
        return o
```

#### 6.2 KV Cache 存储（自定义 Triton Kernel）

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)  # 虚拟地址 → 物理地址
    if slot == -1: return
    
    key = tl.load(key_ptr + idx * key_stride)
    value = tl.load(value_ptr + idx * value_stride)
    
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

---

### 7. 分布式线性层 (`linear.py`)

**支持 Tensor Parallelism**：

```python
class ColumnParallelLinear(LinearBase):
    """列并行：用于 QKV 投影"""
    
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        # 输出维度按 tp_size 切分
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)
    
    def forward(self, x):
        # 每个 rank 只计算输出的 1/tp_size
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(LinearBase):
    """行并行：用于输出投影"""
    
    def __init__(self, input_size, output_size, bias=False):
        super().__init__(input_size, output_size, bias, 1)
    
    def forward(self, x):
        # 输入是 all-reduce 后的结果
        return F.linear(x, self.weight, self.bias)
```

---

### 8. 采样器 (`sampler.py`)

**简洁但高效的采样实现**：

```python
@torch.compile
def forward(self, logits, temperatures):
    # 温度缩放
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    
    # 计算概率
    probs = torch.softmax(logits, dim=-1)
    
    # Gumbel-Max 采样（无需 random 值）
    sample_tokens = probs.div_(
        torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    ).argmax(dim=-1)
    
    return sample_tokens
```

**原理**：对于 Gumbel 分布 `G = -log(-log(U))`，`argmax(logits + G)` 等价于从概率分布采样。

---

### 9. 推理上下文 (`context.py`)

**在 kernel 间传递信息的全局状态**：

```python
@dataclass
class Context:
    is_prefill: bool
    cu_seqlens_q: torch.Tensor   # Query 的序列长度累积
    cu_seqlens_k: torch.Tensor   # Key 的序列长度累积
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: torch.Tensor   # 虚拟地址 → 物理地址
    block_tables: torch.Tensor   # PagedAttention 块表
    context_lens: torch.Tensor   # 每个序列的上下文长度
```

---

## 🔄 多进程架构

```
主进程 (Rank 0)                      子进程 (Rank 1, 2, ...)
┌─────────────────────┐             ┌─────────────────────┐
│ LLMEngine           │             │ ModelRunner         │
│  ├─ Scheduler       │             │  └─ loop()         │
│  └─ ModelRunner     │──Shared──→  │     ├─ read_shm    │
│      (主模型)        │   Memory    │     ├─ run()        │
│                     │             │     └─ write_shm    │
└─────────────────────┘             └─────────────────────┘
```

**Tensor Parallelism 通信**：
- NCCL 用于梯度同步（训练）
- 这里只做前向，无需 all-reduce
- 通过 SharedMemory 传递请求

---

## 📊 性能对比

| 指标 | vLLM | Nano-vLLM |
|------|------|-----------|
| 代码量 | ~10000+ 行 | 1434 行 |
| Throughput | 1361 tok/s | 1434 tok/s |
| 加速比 | 1x | 1.05x |

**为什么更快？**
- 代码量少，减少了抽象开销
- 针对性的优化（CUDA Graph、Flash Attention）
- 更少的锁和分支

---

## 💡 关键技术点总结

1. **PagedAttention**：分页管理 KV Cache，避免连续内存要求
2. **Prefix Caching**：xxhash 快速哈希，相同前缀复用计算
3. **CUDA Graph**：预捕获计算图，消除 kernel launch 开销
4. **Flash Attention**：算子融合，O(N) 显存复杂度
5. **Continuous Batching**：动态批处理，提高 GPU 利用率
6. **Tensor Parallelism**：多卡并行，支持大模型

---

## 🎓 教学价值

这个项目是学习推理引擎的最佳入门材料：

1. **完整性**：从调度到 kernel，覆盖推理全栈
2. **简洁性**：1400 行 vs 上万行，容易掌握
3. **可运行**：有完整 example.py 和 bench.py
4. **优化全**：包含主流推理优化技术
