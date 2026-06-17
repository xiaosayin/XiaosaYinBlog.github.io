---
layout:     post
title:      Flash Attention 2 Chapter3
subtitle:   Kernel 1 基础实现
date:       2026-04-17
author:     BY
header-img: img/post-bg-2015.jpg
catalog: true
auto_heading_numbering: true
tags:
    - Cuda, Flash Attention
---

## 简介

在 [Part 2](https://lubits.ch/flash/Part-2) 里，我们已经讨论了 Flash Attention 的基础 CUDA building blocks：Tensor Core `mma` 与高效内存传输（`cp.async`、`ldmatrix`）。这一章进入“组装阶段”：把这些低层组件拼成第一个完整、可运行的 Flash Attention kernel（Kernel 1）。

我们会按 3 个步骤推进：

1. 先确定 CTAs / warps / threads 的分工；
2. 基于底层指令构建更高层操作；
3. 把所有模块拼进完整 kernel。

本文目标配置：

- forward pass only
- non-causal attention
- `d_head = 128`
- no dropout / no KV caching
- Q/K/V 序列长度相同
- 序列长度可被 block size 整除（通常 `64~128`）
- 输入输出 `bf16/fp16`，softmax 用 `fp32`

在该配置下，Kernel 1 在 RTX 3090 上可达到约 `49%` reference 性能。

## Kernel Architecture Overview

Flash Attention kernel 遵循经典三段式：

### Kernel Phases

1. **Prologue（初始化）**
   - 计算 tensor/warp 地址
   - 执行一次性加载（例如 `Q: GMEM -> SMEM -> RF`）
   - 初始化 softmax 统计量（`m, l`）和输出累加器
2. **Mainloop（迭代计算）**
   - `K: GMEM -> SMEM -> RF`
   - 计算 attention score
   - 执行 online softmax 并更新统计量
   - `V: GMEM -> SMEM -> RF`
   - 计算输出增量
3. **Epilogue（收尾）**
   - 完成 softmax 归一化
   - `fp32 -> fp16/bf16`
   - `O: RF -> SMEM -> GMEM`

### Implementation Challenges

落地时的三大技术点：

1. **Data Movement**：跨层搬运（GMEM -> SMEM -> RF）与不同 layout 协调
2. **Math Ops**：`mma` + warp primitives 的 GEMM / online softmax
3. **Synchronization**：线程与 warp/CTA 的同步边界

我们实现中的主要复杂度来自**数据搬运**。不同张量（$Q$、$O$、$K$、$V$）在数据布局与调度上具有不同需求：  
$Q$ 和 $O$ 可以在每个 warp 内独立处理，而 $K^{(j)}$ 与 $V^{(j)}$ 则需要多个 warp 协同完成。  
有些张量在整个存储层级中始终保持行主序（row-major），另一些则必须执行转置。  
当数据被放置到正确的位置后，实际的数学计算过程反而相对直接。

接下来我们将逐项展开，先从决定整体实现形态的几个基础设计选择开始。

## Kernel Configuration

在明确整体结构之后，我们进一步固定决定 kernel 行为的关键参数。

从现在到 kernel 7，我们将 block size 设为 $B_r = 64$、$B_c = 64$。这一组尺寸匹配良好：它们都是 `m16n8k16` 指令粒度的整数倍，同时又足够小，使各个 tile 能够较紧凑地放入 SMEM 与 RF。

此外，每个 CTA 使用 128 个线程（4 个 warps），遵循 CUDA 给出的推荐起始配置。

在 kernel 7（第 5 部分）中，我们将对不同配置进行测试与基准评估，以确定性能最优方案。

### CTA 级别的工作分配
我们需要在三个层级上设计工作分配：CTA、warp 和 thread。先从最高层开始，即在 grid 中如何将任务分配到各个 CTA。

FlashAttention kernel 处理的输入张量形状为 `(batch_size, seq_len, n_heads, d_head)`。为适配 SM（Streaming Multiprocessor）内存容量，需要先将其切分为更小的 tile。基于当前配置，划分方式如下：

- **$Q$ 与 $O$**：划分为 $(B_r, d_{\text{head}}) = (64, 128)$ 的 tile  
- **$K$ 与 $V$**：划分为 $(B_c, d_{\text{head}}) = (64, 128)$ 的 tile，分别记作 $K^{(j)}$ 与 $V^{(j)}$

对于给定的 `(sample, head)` 对，(固定 (sample, head), 还剩(seq_len, d_head) 数据需要处理)每个 CTA 负责一个特定的 $Q^{(i)}$ 与 $O^{(i)}$ block，并需要与该 `(sample, head)` 下全部 $T_c$ 个 $K^{(j)}$、$V^{(j)}$ block 交互完成计算，其中：
$$
T_c = \frac{\text{seq\_len}}{B_c}
$$

因此，总共需要处理的 query block 数量为 `n_samples * n_heads * T_r`，其中：
$$
T_r = \frac{\text{seq\_len}}{B_r}
$$
我们将启动与该数量完全一致的 CTA，以覆盖全部计算任务。

#### Kernel Arguments

`forward_kernel.cuh`

```cpp
// (batch_size, seq_len, n_heads, d_head)
struct ForwardKernelArgs {
    using index_t = int64_t;
 
    void *__restrict__ Q;
    void *__restrict__ K;
    void *__restrict__ V;
    void *__restrict__ O;
    
    // 比较好理解,就是总体数据都是按一维来存储。
    const index_t batch_stride; // seq_len * n_heads * d_head
    const index_t seq_stride; // n_heads * d_head
    const index_t head_stride; // d_head
 
    const index_t seq_len; // 4096
    const index_t n_heads; // 16
 
    const int n_Q_blocks;
    const int n_KV_blocks;
};
```

### Grid Mapping

每个 CTA 会为一个 `(sample, head)` 对处理一个 `(64, 128)` 的 query tile，因此我们将 kernel 的 grid 设为形状 `(sample, query_block, head)` 的某种排列。那么，CTA 应该如何映射到 query blocks 呢？

先考虑某个特定 head 下的单个 sample。该 sample 对应的多个 CTA 会读取不同的 $Q$ 分块、写回不同的 $O$ 分块，但它们都会加载相同的 $K^{(j)}$ 与 $V^{(j)}$ 分块。

关键洞察在于：我们希望处理同一 `(sample, head)` 的 CTA 在时间上尽量相邻启动，从而复用缓存中的 $K/V$ 数据。第一个 CTA 加载某个 $K^{(j)}$ 分块后，该分块会进入 L2；如果同一 `(sample, head)` 的其他 CTA 紧随其后启动，就更可能命中缓存，而不是回退到 DRAM 访问。

CTA 按其 ID 顺序启动：
`blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z`

由于 `y`、`z` 相同而 `x` 不同的 CTA 具有连续 ID，它们会被集中调度启动。因此我们将 query block 映射到 `x` 维，即：
`(x, y, z) -> (Q_block, head, batch)`。

`forward_kernel.cuh`

```cpp
const int sample = blockIdx.z;
const int head = blockIdx.y;
const int q_seq_block = blockIdx.x;
```

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/grid-mapping-unoptimal.svg" alt="grid-mapping-unoptimal" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：利于 L2 复用的 CTA 映射(按 Q_block 连续)
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/grid-mapping-optimal.svg" alt="grid-mapping-optimal" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：不利于命中 KV 缓存的 CTA 映射
  </figcaption>
</figure>

因此，我们将变量设置为如下形式：

```cpp
// ...
const int sample = blockIdx.z;
const int head = blockIdx.y;
const int q_seq_block = blockIdx.x;
// ...
``` 

这种 grid 映射会如何影响性能？每个 $K^{(j)}$ 与 $V^{(j)}$ tile 都会被多个 CTA 读取；在每个 tile 约 16KiB 的情况下，L2 可以容纳多个 tile。以 A100 为例，L2 命中延迟约为 200-400 cycles，而 DRAM 访问延迟约为 550 cycles（参考 Luo et al.）。对于 FlashAttention 这类计算密集型负载，这种缓存复用会带来小幅但可观测的性能收益。

下面进一步说明：在 FlashAttention 工作负载下，grid 映射如何影响缓存性能。

| GPU | L2 Size | Unoptimized Hit Rate | Optimized Hit Rate | Perf Impact |
| --- | --- | --- | --- | --- |
| RTX 3090 | 6MB | ~2% | ~98% | ~3% TFLOPs 提升 |
| A100 | 40MB | ~25.6% | ~92.6% | ~1% TFLOPs 提升 |

收益幅度虽然有限，但由于正确设置这种映射几乎没有额外成本，因此仍然值得在实现中采用。

### 配置模板参数

虽然我们目前已经有一套固定的 kernel 配置，但为了便于后续泛化并测试不同配置（这部分会在 kernel 7 中展开），我们希望把配置参数结构化。  
具体做法是引入一个 `struct` 类型的模板参数，用于统一描述不同的 block size、每个 CTA 的线程数，以及不同的数据类型。下面是该 kernel 函数对应的模板参数定义：

`flash_attention.cuh`

```cpp
struct FlashForwardKernelConfig {
    // 该类型会在编译期静态映射为 half 或 nv_bfloat16
    const torch::ScalarType dtype;

    const int d_head;  // [64, 128]
    const int B_r;     // [64, 128]
    const int B_c;     // [32, 64, 128]
    const int n_warps; // [4, 8]，当 B_r = 128 时只能取 8
};
```

## Warp-Level Work Distribution

现在我们已经明确了 CTA 如何分配工作，接下来把视角投放到 warp 级别。  
从这里开始，我们关注 `warp-to-CTA` 与 `thread-to-warp` 两类交互，因此为简化记号，我们将从 $Q^{(i)}$ 和 $O^{(i)}$ 中省略上标 $(i)$。

回顾第 2 部分：tensor 操作要求同一 warp 内所有线程严格 lock step 执行。  
我们的 `mma` 指令以 $16 \times 16$ tile 为基本计算单元，因此后续会以它作为 warp 间工作划分的基础。

先做一下尺寸计算：当前每个 block 有 64 行、每个 CTA 包含 4 个 warp，因此每个 warp 分到 $64 \div 4 = 16$ 行。  
再结合 head 维度为 128 列，可得每个 warp 负责一个 $(16, 128)$ 的子块（sub-tile）。

我们会对 $Q$、$O$ 的加载方式，与 $K^{(j)}$、$V^{(j)}$ 采用不同策略。  
为什么 $Q/O$ 与 $K/V$ 的策略不同？虽然每个 warp 仅负责 $Q$ 和 $O$ tensor 的一个独立切片，但为了计算该切片对应的 attention score，它必须访问 $K^{(j)}$ 和 $V^{(j)}$ 的整个 block。(因为 Q 要和 warp 内的所有 K(64,128) 做矩阵乘法)    
这意味着：$Q$ 和 $O$ 基本可以按 warp 独立处理；而 $K^{(j)}$ 与 $V^{(j)}$ 则需要 CTA 内所有 warp 协同完成。
### `Q/O`（Independent per Warp）

- 64 行被分为 4 个独立 slice
- 每个 warp 对其 slice 独立完成加载/存储与 GEMM

### `K/V`（Cooperative across Warps）

- **Loading**：每个 warp 先搬自己的 `(16,128)` slice 到 SMEM(搬到 SMEM 后, 整个 CTA 共享一整个 (64, 128) K, V 块)
- **Sync**：全 CTA 等待完整 `(64,128)` block 就绪
- **Copy**：每个 warp 再从 SMEM 读整块(64,128) KV 到 RF

### Per-Warp 工作负载总结

每个 CTA 中的 4 个 warp 分别处理如下任务：

**独立操作（每个 warp 内）**：

- 加载 $Q$ 的一个 $(16, 128)$ 切片：`GMEM -> SMEM -> RF`
- 在 `RF` 中计算  
  $$
  S^{(j)} = Q\left(K^{(j)}\right)^\mathsf{T}, \quad \text{shape}=(16, 64)
  $$
- 在 `RF` 中计算  
  $$
  \tilde{P}^{(j)} = \mathrm{softmax}\!\left(S^{(j)}\right)
  $$
- 在 `RF` 中计算  
  $$
  \tilde{O}^{(j)} = \tilde{P}^{(j)}V^{(j)}
  $$
- 存储 $O$ 的一个 $(16, 128)$ 切片：先在 `RF` 中累积，再执行 `RF -> SMEM -> GMEM`

**协同操作（跨所有 warp）**：

- 加载 $K^{(j)}$ 与 $V^{(j)}$ 的 $(64, 128)$ block：先由所有 warp 协同完成 `GMEM -> SMEM`，再由各 warp 独立执行 `SMEM -> RF`

下图展示了每个 warp 如何处理其对应的工作切片。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/warp-workload.svg" alt="warp-workload" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：CTA 内 4 个 warp 的分工示意
  </figcaption>
</figure>

每个带有背景色的 tile 都存储在 `RF` 中。

> [!TIP]
> **后续优化（Future optimization）**  
> 实际上，在加载 $Q$ 和 $O$ block 时，让多个 warp 协同处理会略微更高效。  
> 这部分我们会在第 7 部分（kernel 9）中详细展开。

## Data Movement

现在我们已经理解了工作分配策略，接下来进入实现部分。真正复杂、也最有意思的地方从这里开始。  
在这个 kernel 中，高效搬运数据是最棘手的问题之一：我们需要同时处理不同的布局（layout）、访问模式（access pattern）以及同步需求（synchronization）。

不同 tensor 的处理需求并不相同：$Q$ 和 $O$ 主要按 warp 独立处理，而 $K^{(j)}$ 与 $V^{(j)}$ 需要跨 warp 协作。  
此外，有些 tensor 在整个 memory hierarchy 中保持 row-major，不变形；另一些则需要转置（transposition）。这些因素叠加在一起，复杂度会迅速上升。

我们的策略是分层构建这套复杂系统：从底层内存操作开始，逐步抽象到一个干净的接口，把实现细节封装起来：

1. **核心内存操作（Core memory operations）**：  
   提供通用的 `GMEM <-> SMEM` 传输函数，以及负责转置的专用 `SMEM -> RF` 函数。  

2. **地址管理（Address management）**：  
   为每个 tensor、每个 warp 计算正确的访问指针。  

3. **统一接口（Unified interface）**：  
   通过 `MatrixLDST` 类封装上述逻辑，暴露简洁且按 tensor 定制的 API。  

下面我们一步一步来构建它。

### Configuration Structs
为了在 memory hierarchy（`GMEM -> SMEM -> RF`）中高效搬运数据，我们需要一组能够适配不同 tensor 需求的操作。  
下面给出用于编码全部 LD/ST 需求的配置结构体（configuration structs）：


`load_store.cuh`

```cpp
struct TileLayout {
    const int row_fragments;
    const int col_fragments;
};
 
struct TensorLDSTConfig {
    const TileLayout GSM;
    const TileLayout RF;
 
    const bool transposed;
    const int block_size;
    const int smem_cols;
    const int warp_ldst_rows;
    const bool compute_over_entire_block;
};
```

### Storage Layout 存储布局

针对不同的内存层级与 tensor，我们采用了不同的存储布局（storage layout）：

- **SMEM**：所有 tensor 都按 row-major 存储，以便高效完成从 GMEM 的加载。  
- **RF**：大多数 tensor 保持 row-major，但 $V^{(j)}$ 在 RF 中采用转置存储。  

下面这张表有助于快速梳理：我们需要搬运哪些数据，以及它们在各个内存层级上的形状（shape）。

| Tensor | Element Size (bytes) | GMEM+SMEM Majorness | GMEM↔SMEM Shape | RF Majorness | SMEM Shape (SMEM → RF) | RF Shape (Registers) |
|---|---:|---|---|---|---|---|
| $Q$ | 2 | Row major | $(16, 128)$ | Row major | $(16, 128)$ | $(2, 16)$ |
| $K^{(j)}$ | 2 | Row major | $(64, 128)$ | Row major | $(64, 128)$ | $(8, 16)$ |
| $V^{(j)}$ | 2 | Row major | $(64, 128)$ | Column major | $(64, 128)$ | $(16, 8)$ |
| $S^{(j)}$ | 4 | N/A | N/A | Row major | N/A | $(2, 16)$ |
| $\tilde{P}^{(j)}$ | 2 | N/A | N/A | Row major | N/A | $(2, 8)$ |
| $\tilde{O}^{(j)}$ | 4 | N/A | N/A | Row major | N/A | $(2, 32)$ |
| $O$ | 2 | Row major | $(16, 128)$ | Row major | $(16, 128)$ | $(2, 16)$ |

*Tensor transfer shapes across memory hierarchy for block configuration $(B_r = 64,\ B_c = 64,\ d_{\text{head}} = 128,\ n_{\text{warps}} = 4)$*

下面给出我们将用于完成这些数据传输的 LD/ST 操作。

| From | To | Blocks | PTX Instr. / C++ | Warp-Wide Op Size | Thread Op Size | Thread Mapping | Register Shape | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMEM | SMEM | $Q,\ K^{(j)},\ V^{(j)}$ | `cp.async` | $(4,64)$ | $(1,8)$ | row-major |  |  |
| SMEM | RF | $Q,\ K^{(j)},\ V^{(j)}$ | `ldmatrix.x4` | $(16,16)$ | $(1,8)$ | column-major | $(2,2)$ | $V^{(j)}$ transpose |
| RF | SMEM | $O$ | standard (4B) | $(8,8)$ | $(1,2)$ | row-major | $(1,1)$ |  |
| SMEM | GMEM | $O$ | standard (16B) | $(4,64)$ | $(1,8)$ | row-major |  |  |


### Copying Between GMEM ↔ SMEM

现在我们开始实现实际的数据搬运。回顾一下：每次 warp-wide 操作会访问一个 $(4, 64)$ 的数据块，而我们的 tile 大小是 $(16, 128)$，因此需要执行 $(4, 2)$ 次 `cp.async()` 指令，才能完整拷贝一个 block。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/g2smem_warp2thr.svg" alt="g2smem_warp2thr" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 4：GMEM -> SMEM 的 warp/thread 映射
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/gmem_smem_all_ops.svg" alt="gmem_smem_all_ops" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 5：内存层级中的主要 load/store 操作
  </figcaption>
</figure>

下面是 GMEM 与 SMEM 之间拷贝代码的设计要点：

- **双向支持（Bidirectional）**：通过传入不同的 operation functor，同时支持 `GMEM -> SMEM` 与 `SMEM -> GMEM`。  
- **线程协同（Thread coordination）**：每个线程按 row-major 顺序计算自身 offset，以避免访问冲突。  
- **模板化设计（Template-based）**：通过 `TensorLDSTConfig` 模板参数，统一适配不同 tensor 的布局需求。  

这些 operation functor 定义了实际的数据拷贝行为。下面给出的实现通过模板化 operation functor 同时覆盖 `GMEM -> SMEM` 和 `SMEM -> GMEM` 两种传输路径。

```cpp
#define ROWS_PER_FRAGMENT 8
#define COLS_PER_FRAGMENT 8
#define GSM_LDST_ROWS_PER_ITER 4
#define BYTES_PER_VEC4_ACCESS 16
 
template <typename T>
struct GM2SM_async {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem);
    }
};
 
template <typename T>
struct SM2GM {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(gmem)[0] = reinterpret_cast<uint4 *>(smem)[0];
    }
};
 
template <typename op, /* either GM2SM_async or SM2GM */
          TensorLDSTConfig CFG,
          typename value_t,
          typename index_t = int64_t>
__forceinline__ __device__ constexpr void copy_block_GSM(
	value_t *gmem,
	value_t *smem,
    index_t gmem_seq_stride,
    const int lane_id
) {
    // 搬运 GM 的 (16,128) 到 SM
    // row_fragments = 2
    // GSM_LDST_ROWS_PER_ITER = 4, 是因为 warp 每次搬 (4,64), 搬 (4,2) 次
    // 因为要满足, 一次搬内存一行 128B( 64 个 BP_16), 每个线程搬 16B(8个元素), 一行 128B 是 8 个线程处理
    constexpr int n_row_iters =
        CFG.GSM.row_fragments * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER; // 2*8/ 4 = 4 // 一共 16 行, 一轮搬 4 行, 那就搬 n_row_iters 轮
 
    constexpr int col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    // 32 / 4 = 8, 每一轮搬 col_fragments, 其实是因为每轮搬 4 行, 一个 warp 32 个线程;
    constexpr int col_fragments_per_row = CFG.smem_cols / COLS_PER_FRAGMENT; 
    // smem_cols = 128, COLS_PER_FRAGMENT= 8, 每一行有 16 个 col_fragments

    // 线程行地址 32/8
    const int thread_row = lane_id / col_fragments_per_iter;
    // 线程列地址 32 % 8
    const int thread_col_fragment = lane_id % col_fragments_per_iter;
 
    #pragma unroll
    // r 代表现在进行到 (4,2) 的第几轮了
    for (int r = 0; r < n_row_iters; ++r) {
        const int cur_row = r * GSM_LDST_ROWS_PER_ITER + thread_row;
        #pragma unroll
        // 每行 16 列 col_fragments_per_row, 每轮处理 col_fragments_per_iter(8) 列
        for (int c = 0; c < col_fragments_per_row;
             c += col_fragments_per_iter) {
            const int col_fragment = c + thread_col_fragment;
 
            op()(&gmem[cur_row * gmem_seq_stride +
                       col_fragment * COLS_PER_FRAGMENT],
                 &smem[cur_row * CFG.smem_cols +
                       col_fragment * COLS_PER_FRAGMENT]);
        }
    }
}
```

<div style="margin: 18px 0; padding: 16px 18px; border-radius: 12px; border: 1px solid rgba(59,130,246,.35); background: linear-gradient(180deg, rgba(30,41,59,.78) 0%, rgba(15,23,42,.78) 100%);">
  <div style="font-weight: 700; color: #93c5fd; margin-bottom: 10px;">
    ✎ A Note on <code>reinterpret_cast</code>
  </div>
  <div style="color: #e5e7eb; line-height: 1.8;">
    你会发现 <code>reinterpret_cast</code> 在这里被频繁使用。这是高性能 CUDA 中常见且安全的向量化访存模式：
    通过把指针重解释为更大的类型（例如 <code>uint4</code>），我们可以在单条指令中完成 16B 的 load/store。
    前提是内存地址满足对齐要求，而当前实现是满足该约束的。
  </div>
</div>

<div style="margin: 18px 0 24px; padding: 16px 18px; border-radius: 12px; border: 1px solid rgba(251,146,60,.35); background: linear-gradient(180deg, rgba(55,36,21,.72) 0%, rgba(41,26,16,.72) 100%);">
  <div style="font-weight: 700; color: #fdba74; margin-bottom: 10px;">
    ⚠ Additional Requirements for <code>cp.async</code>
  </div>
  <div style="color: #f3f4f6; line-height: 1.8;">
    使用 <code>cp.async()</code> 时，需要通过 <code>cp.commit()</code> 提交，并通过 <code>cp.wait()</code> 等待完成。
    更完整的同步细节会在后续 <strong>Synchronization</strong> 一节展开。
  </div>
</div>

### SMEM → RF 

你可能还记得我们在 `How Fragments Are Laid Out Across Threads` 一节中提到过：在 RF 中，$Q$ 和 $K^{(j)}$ 的存取方式与 $V^{(j)}$ 不同。关键差异在于，虽然所有 tensor 在 SMEM 中都按行主序（row-major）存储，但 $V^{(j)}$ 在 RF 中按列主序（col-major）存储。

这种布局差异要求我们在 SMEM $\rightarrow$ RF 的拷贝阶段执行转置：即将 SMEM 中的 $(\text{row}, \text{col})$ 转换为 RF 中的 $(\text{col}, \text{row})$。该转置同时影响 fragment 的排布方式和线程级元素存储方式，这也是为什么对于 $V^{(j)}$，我们使用 `ldmatrix_transpose()` 而不是 `ldmatrix()`。为处理这些差异，我们将实现两个独立的 helper 函数。

$V^{(j)}$ 的一次迭代过程如下：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/e2e_initial_qk.svg" alt="e2e_initial_qk" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 6：Q/K 的加载路径（SMEM -> RF）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/e2e_initial_v.svg" alt="e2e_initial_v" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 7：V 的 transpose 加载路径
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/ldmatrix_single_fragment_accum.svg" alt="ldmatrix_single_fragment_accum" style="width: 90%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 8：ldmatrix 单 fragment 在线程寄存器中的分布
  </figcaption>
</figure>

#### $Q$ 和 $K^{(j)}$
>load_store.cuh

```cpp
#define ROWS_PER_FRAGMENT 8  

#define COLS_PER_FRAGMENT 8  

#define ELEMS_PER_VEC4_ACCESS 8   
// ELEMS_PER_VEC4_ACCESS = 8
// 表示一次 16B 对齐访问，正好覆盖 8 个 half/bf16 元素。
 
template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int col_fragment_offset = 0
) 
// 每个 regs 的形状是 [2][16] 见下图, (16,128) 每个 regs[x][y], 可以放 2 个 bp16 数据, 也就是对应 ldmatrix_x4,
// 因为是 ldmatrix_x4,所以每个线程需要提供 4 个 寄存器地址, 每个寄存器位宽 32bit, ldmatrix 直接从 SMEM 中按 tensor core 要求的方式加载 2 个元素到对应 regs 地址.    
// ldmatrix 每个线程提供一个 SMEM 一行 8 元素的地址.  
{
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter; // 16
    // 因为一个 warp 从 SM 加载到 RF, 形状就是 16x16

    // col_fragments = 128/8 = 16
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    // col_fragments_per_iter = 32 / 2 = 16,每轮处理 16 个
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    // 32 % 2 
    const int thread_row = lane_id % rows_per_iter;
    // 32 / 2
    const int thread_col_fragment = lane_id / rows_per_iter;

    // CFG.RF.row_fragments = 2 的倍数
    #pragma unroll
    for (int r = 0; r < CFG.RF.row_fragments; r += row_fragments_per_iter) {
        const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = thread_col_fragment + c + col_fragment_offset;
 
            ldmatrix_x4(&smem[cur_row * CFG.smem_cols +
                              smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                        regs[r][c], regs[r + 1][c], regs[r][c + 1],
                        regs[r + 1][c + 1]);
        }
    }
}
```

#### $V^{(j)}$
- 我们在 `SMEM` 中按 row $\rightarrow$ col 的顺序遍历；在 `RF` 中则采用相反方向（col $\rightarrow$ row）。
- 与其通过交换 `SMEM` 指针来实现 fragment 转置，我们将改为交换 `RF` 索引。
  - 这样可以让 kernel 与其他 tile 复用相同的 `SMEM` offset 计算逻辑。

>load_store.cuh

```cpp
template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int row_fragment_offset = 0
) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;
    // 16 = 8*2 每轮处理 16 行
 
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    // col_fragments = 128/ 8 = 16
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;
    // col_fragments_per_iter = 32 / 2 = 16, 每轮处理 col_fragments_per_iter
 
    const int thread_row = lane_id % rows_per_iter; // 和上面加载 Q, K 没什么不同
    const int thread_col_fragment = lane_id / rows_per_iter;

    // 突然理解 为什么 r <  CFG.RF.col_fragments, 却按照 row_fragments_per_iter 去增长?
    // 因为, r 沿着 SMEM 的行去遍历,但是在 RF 里面则是遍历 列
    // 所以正常版本,是 col_fragment_offset, 在这里变成了 row_fragment_offset
    #pragma unroll
    for (int r = 0; r < CFG.RF.col_fragments; r += row_fragments_per_iter) {
        const int cur_row =
            thread_row + (r + row_fragment_offset) * ROWS_PER_FRAGMENT;
        #pragma unroll
        for (int c = 0; c < CFG.RF.row_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = thread_col_fragment + c;
 
            ldmatrix_x4_transpose(
                &smem[cur_row * CFG.smem_cols +
                      smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                regs[c][r], regs[c][r + 1], regs[c + 1][r], regs[c + 1][r + 1]);
        }
    }
}
```

##### 用一个具体的 shape 看,就明白了
假设转置后 RF 是：
```cpp
regs[16][2]
```
也就是  
```cpp
CFG.RF.row_fragments = 16
CFG.RF.col_fragments = 2
```

但是 $V$ 在 `SMEM` 中的形状仍然是 $(16, 128)$，因此这个循环在 `SMEM` 上的寻址是正确的，并且与常规版本一致。

那么循环可写为：

```cpp
for (r = 0; r < 2; r += 2)      // 只执行一次
for (c = 0; c < 16; c += 2)     // c = 0,2,4,...,14
```
每次写入：

``` cpp
regs[c][0] 
regs[c][1]. 
regs[c+1][0]
regs[c+1][1]
```
这很合理，因为：

r 这一维长度只有 $2$，所以一次即可处理完。
c 这一维长度是 $16$，每次处理 $2$ 个元素，共需要 $8$ 次。
所以虽然看起来像：

- `c` 这维长度是 16，所以每次处理两个，共 8 次。

所以虽然看着像：

> “为什么 `r` 遍历 `col_fragments` 却用 `row_fragments_per_iter` 递增？”

但实际上这里就是：

> 在 RF 的第 2 维（也就是转置后的 `col-fragment` 维）上，每次处理两个元素，完全合理。

### RF → SMEM

我们从 `RF` $\rightarrow$ `SMEM` 回写的唯一 block 是 $O$。这里不使用 `ldmatrix`，而是采用标准的 4B store：

`smem[dst] = rf[src];`

循环每次迭代会将一个 $(8,8)$ tile 写回 `SMEM`。由于单个 warp 对应的 $O$ 尺寸为 $(16,128)$，因此完整拷贝 $O$ 需要执行 $(2,16)$ 次迭代。

`load_store.cuh`

```cpp
template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_RF2SM(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id
) {
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT; // 8
    constexpr int col_fragments_per_iter = 1; // 一轮只处理 8 列, 因为一次只处理一个 8x8 tile
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS; // 128/8 = 16
 
    constexpr int elems_per_store = 2; // 每个线程从 RF 存 4B 到 SMEM, 那就是 2 个 BP16
    // 线程排布是 8x4, 每行 4 个线程, 每个线程存 4B
    // The thread to address mapping will follow the mma layout format.
    // Inside a fragment, we have 8 rows with 4 threads each.
    // Each thread stores 2 values / 4B
    // So we can store a single fragment per instruction
    // 这么存回去主要是因为和 MMA 的 layout 有关系
    const int thread_row = lane_id / 4;
    const int thread_inner_col = (lane_id % 4) * elems_per_store;
 
    #pragma unroll
    for (int r = 0; r < CFG.RF.row_fragments; ++r) {
        const int cur_row = thread_row + r * rows_per_iter;
        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = c;
 
            reinterpret_cast<uint32_t *>(
                &smem[cur_row * CFG.smem_cols +
                      (smem_col_fragment * ELEMS_PER_VEC4_ACCESS +
                       thread_inner_col)])[0] = regs[r][c];
        }
    }
}
```

### GMEM 地址计算

每个 block 的地址计算相对直接。下面这段代码会为特定的 `sample`、`head` 和 `query block` 计算其在对应 block 中的指针位置。由于不同 tensor 的 `warp` 级地址计算方式并不相同，我们会将这部分逻辑放入一个 tensor 类中，并由该类统一封装后续所需的各类操作。

`forward_kernel.cuh`

`张量 shape 以及对应的一些参数`

shape: (`batch_size = 16`, `seq_len = 4096`, `n_heads = 16`, `d_head = 128`)

```cpp
TQ.shape = [batch_size, seq_len, n_heads, d_head]
         = [16, 4096, 16, 128]
```

`stride`

```cpp
const auto batch_stride = TQ.stride(0);
const auto seq_stride   = TQ.stride(1);
const auto head_stride  = TQ.stride(2);
```

1. batch_stride = 8388608 = 4096 × 16 × 128，从一个 batch 样本跳到下一个 batch 样本，要跨过整个 [seq_len, n_heads, d_head] 块。
2. seq_stride = 2048 = 16 × 128，在同一个 batch 下，从一个 token 位置移动到下一个 token 位置，要跨过所有 head 的数据（[n_heads × d_head]）。
3. head_stride = 128。
这正好和一个连续张量 [16, 4096, 16, 128] 的默认布局一致。

```cpp
    const int sample = ...; // 16
    const int head = ...; // 16 
    const int q_seq_block = blockIdx.x;
 
	// ....
 
	using value_t = nv_bfloat16; // or half
 
    const index_t gmem_seq_stride = args.seq_stride;
 
    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    // 定位到当前 batch 和 head
    // We only read/write one block for Q and O.
    // These offsets are the same for the whole thread-block.
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * 64 * gmem_seq_stride;
    // 定位到当前 是哪个 Q(64,128) 切片
    // We read the entire key sequence.
    const index_t KV_gmem_block_offset = sample_head_offset;
 
    value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_gmem_block_offset];
    value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_gmem_block_offset];
```

## Tensor Abstraction Layer 

### Matrix LDST Class

`MatrixLDST` 类封装了所有 load 与 store 操作，覆盖内存层级的各个层次。该抽象将不同 tensor 布局（layout）与访问模式（access pattern）的复杂性统一集中在一处处理。

**核心特性：**

- GMEM $\leftrightarrow$ SMEM $\leftrightarrow$ RF 操作的统一接口
- Warp 级专属地址计算
- 同时支持独立（independent）与协同（cooperative）加载模式
- 自动处理转置布局（适用于 $\mathbf{V}^{(j)}$）

`tensor.cuh`

```cpp
template <TensorLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    // Static configuration
    using matrix_storage_t =
        RFMatrix<value_t, ldst.mma_load_stages, 
		        ldst.RF.row_fragments,ldst.RF.col_fragments>;
    using GM2SM_op = std::conditional_t<ldst.Common.async_copy,
                                        GM2SM_async<value_t>,
                                        GM2SM<value_t>>;
 
    using SM2GM_op = SM2GM<value_t>;
    static constexpr int mma_load_stages = ldst.mma_load_stages;
    static constexpr bool load_entire_block_into_rf =
        ldst.load_entire_block_into_rf;
    static constexpr bool transposed = ldst.transposed;
 
    // Runtime properties
    value_t *gmem_ptr;
    index_t gmem_seq_stride;
    // The location in memory used to load fragments from SMEM → RF.
    value_t *smem_srm_ptr;
    // The location in memory that the warp writes to for Q, K, V from GMEM to
    // smem and O for SMEM to GMEM.
    value_t *smem_gsm_ptr;
 
    const int lane_id;
 
    matrix_storage_t storage;
 
    __forceinline__ __device__ MatrixLDST(
	    value_t *gmem_block_ptr,
	    index_t _gmem_seq_stride,
        value_t *_smem_ptr
)
        : lane_id(threadIdx.x % WARP_SIZE) {
        const int warp_rank = threadIdx.x / WARP_SIZE;
 
        const index_t warp_seq = ldst.warp_ldst_rows * warp_rank;
 
        gmem_seq_stride = _gmem_seq_stride;
        gmem_ptr = gmem_block_ptr + warp_seq * gmem_seq_stride;
 
        smem_gsm_ptr = _smem_ptr + warp_seq * ldst.smem_cols;
        smem_srm_ptr =
            ldst.compute_over_entire_block ? _smem_ptr : smem_gsm_ptr;
    }
 
    __forceinline__ __device__ constexpr void zero() { storage.zero(); }
 
    __forceinline__ __device__ constexpr typename matrix_storage_t::storage_t (&data(
        const int stage = 0))[matrix_storage_t::rows][matrix_storage_t::cols] {
        return storage.data(stage);
    }
 
    __forceinline__ __device__ constexpr void advance_gmem_block() {
        gmem_ptr += ldst.block_size * gmem_seq_stride;
    }
 
    __forceinline__ __device__ constexpr void copy_GM2SM() {
        copy_block_GSM<GM2SM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }
 
    __forceinline__ __device__ constexpr void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }
 
    __forceinline__ __device__ constexpr void copy_SM2RF(int stage = 0, int tile_offset = 0) {
        if constexpr (!transposed) {
            copy_warp_fragment_SM2RF<ldst, value_t>(
                storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        } else {
            copy_warp_fragment_transposed_SM2RF<ldst, value_t>(
                storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        }
    }
 
    __forceinline__ __device__ constexpr void copy_RF2SM() {
        copy_warp_fragment_RF2SM<ldst, value_t>(data(), smem_srm_ptr, lane_id);
    }
};
 
```

### 寄存器存储类 Register Storage Classes

`tensor.cuh`

```cpp
template <typename value_t, int N>
struct RFVector {
    static constexpr int size = N;
    value_t regs[N]; // 每个线程都持有一个这样的结构体, 用来存 fragment 的一个碎片
 
    __forceinline__ __device__ constexpr value_t &operator[](int idx) { return regs[idx]; }
};
 
template <typename value_t, int row_fragments, int col_fragments>
struct RFMatrix {
    using storage_t = std::conditional_t<sizeof(value_t) == 4, float, uint32_t>;
    static constexpr int regs_per_fragment = sizeof(value_t) / 2;
    static constexpr int rows = row_fragments;
    static constexpr int cols = col_fragments * regs_per_fragment;
 
    storage_t regs[rows][cols];
 
    __forceinline__ __device__ constexpr storage_t (&data(const int stage = 0))[rows][cols] {
        return reinterpret_cast<storage_t(&)[rows][cols]>(regs[stage]);
    }
 
    __forceinline__ __device__ constexpr void zero() {
		#pragma unroll       

		for (int j = 0; j < rows; ++j) {
			#pragma unroll     

			for (int k = 0; k < cols; ++k) {
				regs[j][k] = 0;
			}
		}
    }
};
```

### 类型转换

有几处张量 tile 需要从 32 位转换为 16 位。我们需要在每次迭代中将 attention 矩阵 $\tilde{P}^{(j)}$ 转换一次，用于与 value 向量做矩阵乘法；并在最终写回 GMEM 之前，将 $O$ 转换一次。32 位与 16 位之间的转换本身并不复杂，但数据类型的对应处理需要一些额外的记录工作：

```cpp
template <typename value_t, int M_fragments, int N_fragments>
__forceinline__ __device__ constexpr void
convert_to_16_bit_dtype(
	float (&src_float)[M_fragments][N_fragments * 2],
    uint32_t (&dest_uint)[M_fragments][N_fragments]
) {
    using value2_t =
        std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;
 
    float2(&src)[M_fragments][N_fragments] =
        reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);
    value2_t(&dest)[M_fragments][N_fragments] =
        reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);
    #pragma unroll

    for (int m = 0; m < M_fragments; ++m) {
        #pragma unroll

        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (std::is_same_v<value_t, half>) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}
```

## 计算操作 Computing Operations

在数据搬运基础设施就绪之后，我们将目光转向数学计算部分。我们需要的两个核心操作是 GEMM（用于计算 $\mathbf{Q}\mathbf{K}^\top$ 和 $\tilde{P}V$）和 softmax。GEMM 直接构建在 `mma` 原语之上，而 online softmax 则涉及一些值得仔细推敲的线程协同技巧与统计量更新策略。

### GEMM 实现
#### MMA 操作概览
一个 GEMM 由覆盖所有 fragment 的 `mma` 指令组合而成。

回顾第 2 部分，每条 `mma` 指令执行的计算形式为 $D = AB^\top + C$，对于我们所使用的具体指令，各操作数的形状如下：

- $A$ 的形状为 $(m,\ k) = (16,\ 16)$
- $B$ 的形状为 $(n,\ k) = (8,\ 16)$
- $C$ 与 $D$ 的形状为 $(m,\ n) = (16,\ 8)$

下面给出各操作数的形状与迭代模式：(warp 级别和线程级别都可以推导迭代数的,以 QK 为例, 因为一个 warp 处理 Q(16,128), 那么一个线程的寄存器形状就是 (2,16), 每做一次矩阵运算, 每个线程消耗(2,2) 个元素, 所以 k 维是 (16/2)=(128/16) = 8 次迭代, m 维迭代是 (2/2) = (16/16) = 1, n 维是(8/1)=(64/8) = 8)

| $A$ | $A$ Shape（寄存器）| $B$ | $B$ Shape（寄存器）| 迭代形状 $(k,\ m,\ n)$ |
|---|---|---|---|---|
| $Q$ | $(2,\ 16)$ | $K^\top$ | $(8,\ 16)$ | $(8,\ 1,\ 8)$ |
| $\tilde{P}$ | $(2,\ 8)$ | $V^{(j)}$ | $(16,\ 8)$ | $(4,\ 1,\ 16)$ |

*GEMM 操作数形状与迭代模式*

每次 GEMM 迭代在 $(k,\ m,\ n)$ 三个维度上各覆盖 $(2,\ 2,\ 1)$ 个寄存器，操作数的寄存器形状为：$A$ 矩阵 $(2,\ 2)$，$B$ 矩阵 $(1,\ 2)$。

下图展示单次迭代的示意：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/mma_rf_view.svg" alt="mma_rf_view" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 9：mma 操作在 RF 视角下的 fragment 组织
  </figcaption>
</figure>

#### 实现
为了对所有 fragment 完成计算，我们将以 `K -> M -> N` 的顺序构造一个三重嵌套循环。

```cpp
// 这个函数从线程级别的调用来看,线程只需要提供寄存器地址,提供 fragment 碎片对应的地址就可以了
template <typename value_t, const int M_fragments, const int N_fragments,
          const int K_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments],
    uint32_t (&regs_B)[N_fragments][K_fragments],
    accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT]
) {
    #pragma unroll

    for (int k = 0; k < K_fragments; k += MMA_K_FRAGMENTS_PER_ITER) {
        #pragma unroll

        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            #pragma unroll

            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k],
                    regs_A[m + 1][k],
                    regs_A[m][k + 1],
                    regs_A[m + 1][k + 1],
                    regs_B[n][k],
                    regs_B[n][k + 1],
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}
```



## Softmax

### Softmax 数学回顾

对给定向量 $x = (x_1, x_2, \ldots, x_N)$，softmax 的定义为：

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_{k=1}^{N} e^{x_k}}
$$

在实际实现中，为避免指数溢出（`fp16` 的上限约为 $65504$），通常采用 **safe softmax**：先减去行内最大值 $m = \max_k x_k$，再做指数归一化：

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{k=1}^{N} e^{x_k - m}}
$$

这样分子最大值被控制在 $e^{0}=1$，数值稳定性大幅提升。

### FlashAttention 的分块 Softmax（Online Softmax）

标准 softmax 需要遍历整行 $S = QK^\top \in \mathbb{R}^{B_r \times N}$ 两次：一次求最大值 $m$，一次求归一化分母 $\ell = \sum_k e^{S_k - m}$。这要求在 SRAM 中一次性装下整行 $S$，对长序列显然不可行。

FlashAttention 的核心技巧是将 softmax 沿 $K/V$ 序列维度 **切成若干块增量计算**：主循环每迭代一次，处理一块 $K^{(j)}, V^{(j)}$，在线维护每行的两个统计量：

- 当前已见最大值 $m^{(j)}$
- 当前已见指数求和 $\ell^{(j)}$

设第 $j$ 块内的局部最大值为 $\tilde{m}^{(j)} = \max_k S^{(j)}_k$，将全局最大值更新为：

$$
m^{(j)} = \max\!\left(m^{(j-1)},\ \tilde{m}^{(j)}\right)
$$

由于最大值发生了变化，上一轮累积的 $\ell^{(j-1)}$ 与部分输出 $O^{(j-1)}$ 需要按下列比例 **重标定（rescale）**：

$$
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot \ell^{(j-1)} \;+\; \sum_k e^{S^{(j)}_k - m^{(j)}}
$$

$$
O^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot O^{(j-1)} \;+\; \tilde{P}^{(j)} V^{(j)},\quad \tilde{P}^{(j)} = e^{S^{(j)} - m^{(j)}}
$$

当主循环遍历完全部 $T_c$ 个 KV 块后，在 epilogue 中用最终 $\ell$ 做一次归一化即可得到最终输出：

$$
O = \frac{O^{(T_c)}}{\ell^{(T_c)}}
$$

这样，FlashAttention 将原本需要全行可见的 softmax 改写为 **一次遍历、块内增量更新** 的形式，使得整行 $S$ 无需驻留在片上内存中，从而实现显存线性、计算高效的 attention。


**softmax 是对 $QK^\top$ 的计算结果做的**。完整的 attention 计算分三步：

$$
S = QK^\top \quad\Longrightarrow\quad P = \mathrm{softmax}(S) \quad\Longrightarrow\quad O = PV
$$

具体来说：

**第 1 步：$S = QK^\top$**

得到一个形状为 `(seq_len, seq_len)` 的 **attention score 矩阵**，$S_{ij}$ 表示 query 位置 $i$ 对 key 位置 $j$ 的原始相关性打分。

**第 2 步：$P = \mathrm{softmax}(S)$**

对 $S$ 的**每一行**独立做 softmax，把这些打分归一化成一组概率分布（每行之和为 1）。$P_{ij}$ 表示位置 $i$ 关注位置 $j$ 的"注意力权重"。

> ⚠️ softmax 是**行内**（row-wise）操作，不是整个矩阵一起做。

**第 3 步：$O = PV$**

用注意力权重对 value 做加权平均，得到最终输出。

---

**回到 FlashAttention 的切块：**

问题在于第 1 步的 $S$ 矩阵太大了（$N \times N$，当 $N=4096$ 时约 64 MB），无法整块放进 SRAM。所以 FlashAttention 不会一次性算出整行 $S$ 再做 softmax，而是：

- 每次只算 $S$ 的一小段 $S^{(j)} = Q (K^{(j)})^\top$（形状 $(B_r,\ B_c)$，比如 $(64,\ 64)$）
- 立刻在这一小段上做"部分 softmax"，同时用 $V^{(j)}$ 累加到输出 $O$ 中
- 用前面推导的公式**在线修正**累积的最大值 $m$ 和分母 $\ell$
- 整个主循环结束后再做一次最终归一化

这样 **$S$ 矩阵从头到尾没有完整存在过**，只有 $(B_r,\ B_c)$ 那一小块驻留在寄存器中。

---

所以那段代码里把 `m` 初始化为 $-\infty$、`l` 初始化为 $0$，就是在为第 2 步的 softmax 做准备——每行各自维护一个在线的 $m$ 与 $\ell$。

### 看一下源代码
```cpp
template <bool is_first, bool optimized_softmax, typename S_accum_untiled_t,
          typename O_accum_untiled_t, typename row_statistics_t,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void local_softmax(S_accum_untiled_t &S_accum_untiled,   // 当前块的 attention 分数 S^{(j)} = Q(K^{(j)})^T（也是输出 P^{(j)} 的载体）
                                       O_accum_untiled_t &O_accum_untiled,   // 跨块累积的输出 O^{(j-1)}（会被原地更新为 O^{(j)}）
                                       row_statistics_t &m, row_statistics_t &l,  // 行统计量：最大值 m、指数求和 l（每行一个）
                                       const accum_t &softmax_scale) {        // softmax 缩放因子 1/sqrt(d_head)
    if constexpr (is_first && optimized_softmax) {
        // ============== 第一次迭代的快速路径 ==============
        // 因为 m_prev = -inf、l_prev = 0、O_prev = 0，所以 rescale 项恒为 0，可以直接跳过

        calc_row_max<is_first>(S_accum_untiled, m);
        // 步骤①：计算当前块的行最大值 m̃^{(j)}，直接作为全局 m（因为是第一块，不需要和 m_prev 比较）

        exponentiate_tensor(S_accum_untiled, m, softmax_scale);
        // 步骤④：就地将 S 转为 P = exp(S * scale - m)，即未归一化的 softmax

        update_row_exp_sum<is_first>(S_accum_untiled, l);
        // 步骤⑤：把刚算出来的 exp 值按行求和，累加到 l 中（第一次没有旧值，直接覆写）

    } else {
        // ============== 常规路径（第 2 块及以后，或未启用优化） ==============

        row_statistics_t m_prev;
        m_prev.copy(m);
        // 步骤②a：保存上一轮的全局最大值 m^{(j-1)}，后面 rescale 要用

        calc_row_max<is_first>(S_accum_untiled, m);
        // 步骤②b：用当前块的局部最大值更新 m → m^{(j)} = max(m^{(j-1)}, m̃^{(j)})

        scale_l_O(m, m_prev, l, O_accum_untiled, softmax_scale);
        // 步骤③：因为 m 变了，旧的 l^{(j-1)} 和 O^{(j-1)} 都要乘上 exp(m^{(j-1)} - m^{(j)}) 做 rescale
        //        对应公式：l ← l * exp(m_prev - m)， O ← O * exp(m_prev - m)

        exponentiate_tensor(S_accum_untiled, m, softmax_scale);
        // 步骤④：把 S^{(j)} 就地转为 P^{(j)} = exp(S * scale - m^{(j)})

        update_row_exp_sum<is_first>(S_accum_untiled, l);
        // 步骤⑤：把本块的行 exp 求和累加到已 rescale 过的 l 上
        //        对应公式：l^{(j)} = exp(m_prev - m) * l^{(j-1)} + Σ exp(S^{(j)} - m^{(j)})
    }
}
// 备注：此函数执行完后，S_accum_untiled 里保存的是 P^{(j)}，
//       接下来主循环会做 O_accum += P^{(j)} * V^{(j)}，完成一轮 online softmax。
```

**与前文数学推导的对应关系速查表：**

| 代码调用 | 对应公式 | 含义 |
|---|---|---|
| `calc_row_max` | $m^{(j)} = \max(m^{(j-1)},\ \tilde{m}^{(j)})$ | 更新全局行最大值 |
| `scale_l_O` | $l \leftarrow e^{m_{prev}-m}\,l$，$O \leftarrow e^{m_{prev}-m}\,O$ | 用新最大值重标定旧累积量 |
| `exponentiate_tensor` | $\tilde{P}^{(j)} = e^{S^{(j)}-m^{(j)}}$ | 把 $S$ 原地转成未归一化 $P$ |
| `update_row_exp_sum` | $l^{(j)} \mathrel{+}= \sum_k e^{S^{(j)}_k-m^{(j)}}$ | 累加本块行指数求和 |

`is_first && optimized_softmax` 分支是**优化**：第一块时 $m_{prev}=-\infty$、$l_{prev}=0$、$O_{prev}=0$，rescale 的 $e^{-\infty}=0$ 会让旧值清零——既然结果必然为 $0$，直接跳过 `scale_l_O` 这一步，省一次读写。


很好的问题！**$O / \ell$ 这个除法只在所有 KV 块都处理完之后，在 epilogue（收尾阶段）做一次**，而不是每块都做。

---

**原因：**

在主循环内维护的 $O_{\text{accum}}$ 并不是最终的 $O$，而是**未归一化的加权和**：

$$
O_{\text{accum}}^{(j)} \;=\; \sum_{k=1}^{j} e^{m^{(k-1)} - m^{(j)}} \cdot \tilde{P}^{(k)} V^{(k)}
$$

可以证明，当遍历完所有 $T_c$ 个块后：

$$
O_{\text{accum}}^{(T_c)} \;=\; \sum_{k=1}^{N} e^{S_k - m_{\text{final}}} \cdot V_k,\qquad \ell^{(T_c)} \;=\; \sum_{k=1}^{N} e^{S_k - m_{\text{final}}}
$$

此时才能做归一化：

$$
O_{\text{final}} \;=\; \frac{O_{\text{accum}}^{(T_c)}}{\ell^{(T_c)}}
$$

---

**为什么不在每块都除？**

如果每块都做 `O /= l`，下一轮 rescale（$O \leftarrow e^{m_{prev}-m} \cdot O$）之后还得乘回旧的 $\ell_{prev}$、再用新分母除，会让在线更新公式变复杂且数值上累积误差。**推迟到最后做一次除法是最简洁且数值最稳定的做法。**

---

**对应到代码位置：**

主循环只维护 $(m,\ \ell,\ O_{\text{accum}})$，写回前才归一化。Epilogue 的第一步正是这件事：

```cpp
// Epilogue
final_softmax_normalization(O_accum, l);   // ← 这里就是 O_accum /= l
// 然后 fp32 → bf16
// 然后 RF → SMEM → GMEM
```

`final_softmax_normalization` 函数内部大致做：

```cpp
for (; block >= 0; --block) {
        process_kv_block<false, Kernel::optimized_softmax, Q_t, K_t, V_t,
                         S_accum_t, P_t, O_accum_t, row_statistics_t, GEMM_QK,
                         GEMM_PV>(Q, K, V, O_accum, m, l, softmax_scale, block);
    }

    final_softmax_normalization(O_accum_no_op_tiling, l);
```

---

**一句话总结：**

> 主循环只累加分子 $O_{\text{accum}}$ 和分母 $\ell$，**除法 $O = O_{\text{accum}} / \ell$ 延迟到 epilogue 做一次**，再把 fp32 结果转成 bf16 写回 GMEM。具体做 O/l 的尺寸,以下章节会做说明.

### Thread-Level 视角

在此前介绍的所有操作中，我们始终将每个 warp 视为一个独立的整体单元，并关注各 warp 之间如何相互协作。现在，我们将把视角切换到线程级别，分析 softmax 操作，以及同一 warp 内各线程之间如何协同工作。

每个 warp 和线程都直接在自己持有的数据上进行计算，工作负载均匀分布，无需额外的 LD/ST 操作即可开始执行。

softmax 在 32 位精度下计算，包含 element-wise 与 row-wise 两类操作。

#### 初始化行信息
行最大值 $m$ 与行指数求和 $\ell$ 各占一个 32 位寄存器，分别初始化为 $-\infty$ 和 $0.0$。

`forward_kernel.cuh`

```cpp
    // ...
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();
    accum_t m[N::QO_fragments_per_warp];
    accum_t l[N::QO_fragments_per_warp];
    #pragma unroll

    for (int q = 0; q < N::QO_fragments_per_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0;
    }
    // ...
```

#### 点乘 Scaling
我们将对 $S = \mathbf{Q}\mathbf{K}^\top$ 执行点积缩放（dot-product scaling），即对每个元素乘以 $1/\sqrt{d_{\text{head}}}$。

`softmax.cuh`

```cpp
   // ...
const float softmax_scale = rsqrt(static_cast<float>(CFG.d_head));
   // ...
 
template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void
scale_S_accum(
	accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    const accum_t &softmax_scale
) {
    #pragma unroll

    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll

        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] *= softmax_scale;
        }
    }
}
```

#### 使用 warp shuffle 指令来做归约

由于每个线程持有 fragment 的不同部分，我们使用 *warp shuffle* 在 warp 内线程间传递数据，从而在执行归约时避免借助共享内存通信。每条 shuffle 指令允许一个线程向同 warp 内的另一个线程发送值，同时接收对方的值。

为了让每个线程在归约结束时都能持有相同的最终值，我们使用的 warp shuffle 指令为 `__shfl_xor_sync(WARP_MASK, val_to_share, xor_offset)`：

- 该指令会发送 `val_to_share` 的值
- 同时从线程 `tid ^ xor_offset` 处读取值，其中 `tid` 为调用线程的 ID

我们先在线程内部对持有的值做归约，再通过两次 warp shuffle 在同一 thread row quartet 的线程之间完成跨线程归约。

下面是行最大值归约的实现代码：

`softmax.cuh`
```cpp
// This mask indicates that every thread in the warp participates in the shuffle
#define SHFL_ENTIRE_WARP_MASK 0xffffffff
 
template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void
calc_row_max(
	accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m_next)[QO_fragments],
    accum_t (&m_cur)[QO_fragments]
) {
    #pragma unroll

    for (int q = 0; q < QO_fragments; ++q) {
        m_next[q] = m_cur[q];
 
        // Calculate max for row across all in-thread registers.
        #pragma unroll

        for (int k = 0; k < KV_accum_fragments; ++k) {
            m_next[q] = max(m_next[q], S_accum[q][k]);
        }
 
        // Group reduction
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2),
                        m_next[q]);
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1),
                        m_next[q]);
    }
}
```

在归约过程中，每次调用 `__shfl_xor_sync()` 后，每个线程所掌握的信息量都会翻倍。

- 第一次以 offset 2 做 xor

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/xor_offset_2.svg" alt="xor_offset_2" style="width: 90%; max-width: 700px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 11：`__shfl_xor_sync` 第1步归约（offset=2）
  </figcaption>
</figure>

- 第二次以 offset 1 做 xor

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/xor_offset_1.svg" alt="xor_offset_1" style="width: 90%; max-width: 700px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 10：`__shfl_xor_sync` 第2步归约（offset=1）
  </figcaption>
</figure>

这里还需要看看总体的项目代码, 从更高的层面来看, 这里是怎么个事;

#### $\ell$ 与 $\tilde{O}^{(j)}$ 的 Rescaling

该函数将上一轮迭代的累加器 $\ell^{(j-1)}$ 和 $\tilde{O}^{(j-1)}$ 重标定到当前块的行最大值：

`softmax.cuh`
```cpp
template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void
scale_l_O(
	accum_t (&m_next)[QO_fragments],
	accum_t (&m_cur)[QO_fragments],
    accum_t (&l)[QO_fragments],
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments]
) {
    #pragma unroll

    for (int q = 0; q < QO_fragments; ++q) {
        accum_t scale;
        scale = expf(m_cur[q] - m_next[q]);
        m_cur[q] = m_next[q];
        l[q] *= scale;
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= scale;
        }
    }
}
```

#### 对 $S^{(j)}$ 做指数化（Exponentiating $S^{(j)}$）

`softmax.cuh`

```cpp
template <int QO_fragments, int KV_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void
exponentiate_tensor(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m)[QO_fragments]
) {
    #pragma unroll

    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll

        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] = expf(S_accum[q][k] - m[q]);
        }
    }
}
```

#### $\ell$ 的部分归约（$\ell$ Partial Reduction）

我们将当前块的 $\tilde{P}^{(j)}$ 按行求和，累加到 $\ell$ 中。由于 $\ell$ 实际上要到最后归一化 $O$ 时才会用到，在此之前并不需要执行 warp shuffle。考虑到 warp shuffle 是相对昂贵的操作，这里选择跳过。整体逻辑与 `calc_row_max` 基本相同，区别在于将取最大值改为求和，并省略 shuffle 步骤。

注意：`P_accum` 与 `S_accum` 共用同一块底层存储。

`softmax.cuh`

```cpp
template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void
update_row_exp_sum(
    accum_t (&P_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]
) {
    #pragma unroll

    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll

        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            l[q] += P_accum[q][d_head];
        }
    }
}
```

#### Softmax Epilogue

在 epilogue（主循环结束后），我们通过 warp shuffle 在线程间交换数据，得到 $\ell$ 的最终值，然后对 $O$ 进行归一化：

`softmax.cuh`

```cpp
template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void final_softmax_normalization(
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]
) {
    // 完成同一行内所有线程的行求和归约
    #pragma unroll

    for (int q = 0; q < QO_fragments; ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
    }

    // 最终按行对 O 做 softmax 归一化
    #pragma unroll

    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll

        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] /= l[q];
        }
    }
}
```

## 同步 Synchronization
数据搬运与计算部分我们已经介绍完毕，但还有一个关键问题需要处理：确保线程之间不会相互干扰。若缺少正确的 barrier，就会出现竞态条件——线程读到过期数据，或覆写了其他线程仍在使用的数据。

### 必要的同步点（Required Barriers）

我们来思考一下，为避免竞态条件，单个 tensor 所需的最少同步操作。暂且假设每个 block 都拥有独立的 SMEM 切片。

在 warp 级别：

- 每次 GMEM $\leftrightarrow$ SMEM 操作传输一个 $(4,\ 64)$ 的元素 tile；
- 每次 SMEM $\rightarrow$ RF 操作拷贝一个 $(16,\ 16)$ 的元素 tile；
- 每次 RF $\rightarrow$ SMEM 操作拷贝一个 $(8,\ 8)$ 的元素 tile。

关键同步点发生在上述内存传输之间。在 GMEM $\rightarrow$ SMEM 与 SMEM $\rightarrow$ RF 操作之间，同一 warp 内的线程会访问其兄弟线程写入的数据。对于 $K^{(j)}$ 和 $V^{(j)}$，情况更加复杂——warp 还会访问其他 warp 写入的数据。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/e2e_sync_overlap.svg" alt="e2e_sync_overlap" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 12：主循环中搬运与计算的同步/覆盖关系
  </figcaption>
</figure>

为处理这些依赖关系，我们需要根据不同 tensor 选用不同类型的 barrier：对 $Q^1$ 使用 *warp-wide* barrier（`__syncwarp()`），对 $K^{(j)}$ 和 $V^{(j)}$ 使用 *CTA-wide* barrier（`__syncthreads()`）。

此外，在任何 barrier 之前，都必须先调用 `cp.wait`，以确保异步操作中的数据已真正完成拷贝。

最后，在 RF $\rightarrow$ SMEM 与 SMEM $\rightarrow$ GMEM 操作之间，同一 warp 内的线程会访问兄弟线程写入的数据（对应输出张量 $O$），因此还需要为输出 tensor 额外插入一次 `__syncwarp()`。

### SMEM $\rightarrow$ RF 通信

一旦 tensor 进入 SMEM，线程之间只会与*同一* warp 内的其他线程进行通信。

所执行的指令要么是 warp 同步的（如 `ldmatrix`、`mma` 及 warp shuffle），要么不涉及线程间通信（如不含 warp shuffle 的 softmax 操作）。

### 循环迭代间的同步（Synchronization Between Loop Iterations）

由于我们会将多个 $K^{(j)}$ 和 $V^{(j)}$ tile 拷贝到 SMEM 中的同一位置，必须确保所有 warp 的 `ldmatrix()` 操作全部执行完毕后，才能覆写 SMEM 中的任何 tensor。因此，在每次迭代之间需要插入一个 `__syncthreads()` barrier。

### 小结（Summary）

| Tensor | GMEM $\rightarrow$ SMEM 与 SMEM $\rightarrow$ RF 之间所需 Barrier | 迭代间所需 Barrier |
|---|---|---|
| $Q$ | `__syncwarp()` | N/A，只拷贝一次 |
| $K^{(j)}$ | `__syncthreads()` | `__syncthreads()` |
| $V^{(j)}$ | `__syncthreads()` | `__syncthreads()` |
| $O$ | `__syncwarp()`（RF $\rightarrow$ SMEM 之后）| N/A，只拷贝一次 |

<div style="margin: 18px 0; padding: 16px 18px; border-radius: 12px; border: 1px solid rgba(251,146,60,.35); background: linear-gradient(180deg, rgba(55,36,21,.72) 0%, rgba(41,26,16,.72) 100%);">
  <div style="font-weight: 700; color: #fdba74; margin-bottom: 10px;">
    ⚠ 注意：Q 与 O 的同步粒度
  </div>
  <div style="color: #f3f4f6; line-height: 1.8;">
    在当前实现中，使用更细粒度的同步范围实际上收益有限，原因有两点：
    <ol>
      <li>目前 $Q$ 和 $O$ 只会与 SMEM 之间各拷贝一次，而 $K^{(j)}$ 和 $V^{(j)}$ 则会被多次拷贝。</li>
      <li>在第一轮迭代中，瓶颈已经来自 $K^{(0)}$ 所需的 <code>__syncthreads()</code>。</li>
    </ol>
    在 kernel 8 中，我们将对布局进行调整，届时也需要为 $Q$ 和 $O$ 引入 CTA-wide 同步。
  </div>
</div>


## Kernel 1 组装
至此，所有基础模块——数据搬运、GEMM 操作、softmax 计算以及同步要求——均已介绍完毕，下面将它们组装成第一个完整的 Flash Attention kernel。

### 代码结构（Code Structure）

我们按照 Cutlass 的标准术语，将 kernel 划分为以下三个部分：

1. **Prologue（初始化）**：包含模版化的初始化工作，如计算正确的内存地址，以及将 $Q$ 从 GMEM 拷贝到 SMEM。
2. **Mainloop（主循环）**：承载绝大部分逻辑，负责处理迭代式的 attention 计算。
3. **Epilogue（收尾）**：完成 softmax 归一化，并将 $O$ 写回 GMEM。

### Prologue

Prologue 包含大量模版化的初始化代码。

`forward_kernel.cuh`

```cpp
template <typename Kernel>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {
    using accum_t = float;
    using index_t = int64_t;
    using value_t = typename Kernel::value_t;
 
	// ...
    // We initialize a CTA for each sample, seq tile, and head.
    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;
 
    const index_t gmem_seq_stride = args.seq_stride;
 
    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    // We only read/write one block for Q and O.
    // These offsets are the same for the whole thread-block.
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * CFG.B_r * gmem_seq_stride;
    // We read the entire key sequence.
    const index_t KV_gmem_block_offset = sample_head_offset;
 
    value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_gmem_block_offset];
    value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_gmem_block_offset];
 
    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_O = smem_Q;
    value_t *smem_K = smem_Q;
    value_t *smem_V = smem_K;	
 
	// MatrixLDST types
    Q_t Q(gmem_Q, gmem_seq_stride, smem_Q);
    K_t K(gmem_K, gmem_seq_stride, smem_K);
    V_t V(gmem_V, gmem_seq_stride, smem_V);
    // S is only stored in registers.
    S_accum_t S_accum(nullptr, -1, nullptr);
    // P is only stored in registers.
    P_value_t P_b16(nullptr, -1, nullptr);
    // The accumulator for O is only kept in registers. At the end of the kernel, it is then converted into a 16-bit type and then copied into gmem.
    O_accum_t O_accum(nullptr, -1, nullptr);
    O_value_t O_b16(gmem_O, gmem_seq_stride, smem_O);
 
	// ...
 
    // Start the async copy of the Q and K tiles.
    Q.copy_GM2SM();
    cp_async_commit();
    O_accum.zero();
 
    // Initialize softmax_scale, m, and l.
    const accum_t softmax_scale = rsqrt(static_cast<accum_t>(CFG.d_head));
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();
    accum_t m[N::QO_fragments_per_warp];
    accum_t l[N::QO_fragments_per_warp];
    #pragma unroll
    for (int q = 0; q < N::QO_fragments_per_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0;
    }
 
	cp_async_wait<0>();
	__syncwarp();
	Q.copy_SM2RF();
	
	// ...
```

### Mainloop

主循环实现了算法的核心部分，依次完成以下操作：

1. 将 $K^{(j)}$ 从 GMEM $\rightarrow$ SMEM $\rightarrow$ RF
2. 计算 $S^{(j)} = Q(K^{(j)})^\top$
3. 计算 softmax 并对 $\ell^{(j-1)}$ 和 $\tilde{O}^{(j-1)}$ 进行 rescale
4. 将 $V^{(j)}$ 从 GMEM $\rightarrow$ SMEM $\rightarrow$ RF
5. 计算 $\tilde{O}^{(j)}$

这里的 barrier 承担了双重职责：

- **Barrier 1**：确保 $K^{(j)}$ 的 `GMEM->SMEM` 拷贝完成后，任何 warp 才能开始执行 `SMEM->RF` 拷贝；同时确保上一轮迭代中 $V^{(j-1)}$ 的 `SMEM->RF` 拷贝已完成，其 SMEM 空间才能被新的 $K^{(j)}$ tile 覆写。
- **Barrier 2**：对另一侧 tensor 执行相同的保护：确保 $V^{(j)}$ 完整进入 SMEM 后再读入 RF，同时确保 $K^{(j)}$ 已完整加载到 RF 后，其 SMEM 切片才能被下一个 $K^{(j+1)}$ tile 覆写。

`forward_kernel.cuh`

```cpp
	// ...
	
	for (int j = 0; j < args.n_KV_blocks; ++j) {
		K.copy_GM2SM();
		K.advance_gmem_block();
		cp_async_commit();
        S_accum.zero();
        cp_async_wait<0>();
        __syncthreads(); // <---- Barrier 1
        
		K.copy_SM2RF();
 
        matmul<Kernel::S_QK_GEMM>(Q, K, S_accum);
 
        // Online softmax
        accum_t m_next[N::QO_fragments_per_warp];
		scale_S_accum(S_accum.data(), softmax_scale);
        calc_row_max(S_accum.data(), m_next, m);
        scale_l_O(m_next, m, l, O_accum.data());
        exponentiate_tensor(S_accum.data(), m_next);
        update_row_exp_sum(S_accum.data(), l);
 
        // Convert the S accumulator block into P bf16/fp16 input block.
        convert_to_16_bit_dtype<value_t>(S_accum.data(), P_b16.data());
 
		V.copy_GM2SM();
		V.advance_gmem_block();
		cp_async_commit();
		cp_async_wait<0>();
		__syncthreads(); // <---- Barrier 2
		V.copy_SM2RF();
 
        matmul<typename Kernel::O_PV_GEMM>(P_b16, V, O_accum);
    }
    
	// ...
```

### Epilogue

Epilogue 负责处理最后几个步骤：用 softmax 的最终统计量归一化输出，将 $O$ 从 fp32 转换为 16 位输出数据类型，并将其从 SMEM $\rightarrow$ GMEM 写回。

`forward_kernel.cuh`

```cpp
    // ...

    final_softmax_normalization(O_accum.data(), l);

    convert_to_16_bit_dtype<value_t>(O_accum.data(), O_b16.data());

    O_b16.copy_RF2SM();

    __syncwarp();

    // Copy the final O tile from SMEM to GMEM.
    O_b16.copy_SM2GM();
}
```


## Occupancy 与资源预算

当前 block 配置为 $(B_r = 64,\ B_c = 64,\ d_{\text{head}} = 128)$，每个 CTA 使用 4 个 warp，共 128 个线程。下面来确定同一 SM 上能同时驻留多少个 warp。

### 每 CTA 的线程数（Threads per CTA）

`SM_86` 每个 SM 最多支持 1536 个 resident thread。以每 CTA 128 个线程计算，单个 SM 最多可容纳 $1536 / 128 = 12$ 个 CTA。线程数不会成为瓶颈。

### 每线程的寄存器数（Registers per Thread）

在 RTX 3090（`SM_86`）上，每个 SM 拥有 65536 个寄存器，每个线程最多可使用 255 个，超出部分会 spill 到 local memory。你可能会认为我们离 255 还差得远——在很多情况下确实如此——但矩阵乘法是典型的寄存器密集型操作。

为了直观感受寄存器压力，我们来看一下各 tensor 所需的 RF 存储量。

| Tensor | Element Size (bytes) | mma matrix variable | Storage Shape | Register Count |
|---|---|---|---|---|
| $Q$ | 2 | A | $(2, 16)$ | 32 |
| $\tilde{O}^{(j)}$ | 4 | C/D | $(2, 32)$ | 64 |
| $K^{(j)}$ | 2 | B | $(8, 16)$ | 128 |
| $V^{(j)}$ | 2 | B | $(16, 8)$ | 128 |
| $S^{(j)}$ | 4 | C/D | $(2, 16)$ | 32 |
| $\tilde{P}^{(j)}$ | 2 | A | $(2, 8)$ | 16 |
| $m$ | 4 | | $(2, 1)$ | 2 |
| $\ell$ | 4 | | $(2, 1)$ | 2 |

*Tensor register file storage requirements*

合计 404 个寄存器！这还不包括内存访问等其他用途所占用的寄存器。好在我们并不需要在同一时刻持有所有这些数据。实际上，在整个主循环迭代过程中始终驻留寄存器的 tensor（不考虑 register spill）只有 $Q$、$\tilde{O}^{(j)}$、$m$ 和 $\ell$，合计 100 个寄存器，其余 tensor 均按需加载。

编译后，该 kernel 每线程最多使用 202 个寄存器，这将我们限制在：

$$\frac{65536}{202} \approx 324 \text{ threads} \approx 8 \text{ warps} = 2 \text{ CTAs per SM}$$

> **每 SM 12 个 Warp**
>
> 达到 12 warps/SM 的寄存器阈值为 168 个。`nvcc` 提供了人为限制每线程寄存器数量的选项，但从 202 降到 168 会引发大量 register spill。这是一个经典的性能权衡：限制寄存器数可以提升 occupancy（更多并发 warp），有助于隐藏内存延迟；但若编译器被迫将寄存器 spill 到较慢的 local memory，单线程性能则会下降。对于像我们这样的 compute-bound kernel，spill 的代价超过了高 occupancy 带来的收益。我们将在 kernel 3 中重新审视寄存器压力问题。


### 每 CTA 的共享内存（Shared Memory per CTA）

`SM_86` 每个 CTA 最多支持 99KiB 的 SMEM。在当前 block 配置 $(B_r = 64,\ B_c = 64,\ d_{\text{head}} = 128)$ 下：

- $Q$ 和 $O$ 各占 $(B_r \times d_{\text{head}} \times \texttt{sizeof(value\_t)}) = 64 \times 128 \times 2 = 16\text{KiB}$
- $K^{(j)}$ 和 $V^{(j)}$ 各占 $(B_c \times d_{\text{head}} \times \texttt{sizeof(value\_t)}) = 64 \times 128 \times 2 = 16\text{KiB}$

若每个 tensor 独占一块 SMEM 切片，合计需要 64KiB。而为了实现每 SM 2 个 CTA 的目标，我们需要将其压缩到 48KiB 以内。

好在，$Q$ 和 $O$ 的访问时间窗口不重叠，因此可以让它们共享同一块内存空间。这样恰好将用量降至 48KiB 的阈值——完美！

更一般地说，每 CTA 所需的 SMEM 用量与我们如何排序和同步各 SMEM 切片的访问密切相关。这里不深入展开，但其他 tensor 同样可以通过增加同步 barrier 来共享 SMEM 切片。例如，$K^{(j)}$ 可以与 $V^{(j)}$ 共用同一切片；$Q$ 也有共享的潜力，因为在当前实现中，$Q$ 的完整数据已全部加载到 RF 中。

### 小结（Summary）

每线程 202 个寄存器，每 CTA 48KB SMEM，我们的 kernel 最终可在每个 SM 上驻留 8 个 warp。

## 性能与展望（Performance and What's Next）

铺垫了这么多，终于到了大家最期待的问题：kernel 跑得有多快？

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/RTX3090_tflops_1_all.svg" alt="RTX3090_tflops_1_all" style="width: 90%; max-width: 700px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 13：Kernel 1 在 RTX 3090 上的性能（seq_len=4096）
  </figcaption>
</figure>

当前达到 33.28 TFLOPS，而参考实现为 67.29 TFLOPS，约为参考性能的 49%。对于一个零优化的初版 kernel 来说，这个成绩相当不错！核心功能已全部正确运行，并且已经逼近高度优化实现性能的一半。当然，提升空间依然明显。

在下一章中，我们将开始诊断当前 kernel 的性能瓶颈，并通过 swizzling、double buffering、instruction fusion 等技术对其进行迭代优化，逐步缩小与 RTX 3090 上参考实现之间的性能差距。
