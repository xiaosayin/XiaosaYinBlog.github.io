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

复杂度主要来自数据搬运：不同 tensor 对布局、访问模式、同步范围的要求并不一致。

## Kernel Configuration

从 Kernel 1 到 Kernel 7，固定配置为：

- `B_r = 64`
- `B_c = 64`
- `d_head = 128`
- `n_warps = 4`（每 CTA 128 线程）

这些参数与 `m16n8k16` 对齐，且能较稳地放进 SMEM / RF 预算。

## CTA Work Distribution

输入 tensor 形状：

`(batch_size, seq_len, n_heads, d_head)`

切块后：

- `Q/O`：按 query 维切成 `B_r` 行块
- `K/V`：按 key/value 维切成 `B_c` 行块

每个 CTA 负责一个 `(sample, head, q_block)`，并遍历该 `(sample, head)` 对应全部 `K/V` blocks。  
总 CTA 数为 `n_samples * n_heads * T_r`，其中 `T_r = seq_len / B_r`。

### Kernel Arguments

`forward_kernel.cuh`

```cpp
struct ForwardKernelArgs {
    using index_t = int64_t;
 
    void *__restrict__ Q;
    void *__restrict__ K;
    void *__restrict__ V;
    void *__restrict__ O;
 
    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;
 
    const index_t seq_len;
    const index_t n_heads;
 
    const int n_Q_blocks;
    const int n_KV_blocks;
};
```

### Grid Mapping

每个 CTA 处理一个 `(64,128)` query tile，grid 可看成 `(sample, query_block, head)` 的某种排列。核心问题：怎么映射 CTA 才更利于 cache？

同一 `(sample, head)` 下，不同 CTA 会读不同 `Q` / 写不同 `O`，但会共享同一批 `K^{(j)}` 与 `V^{(j)}` blocks。  
因此希望这批 CTA 启动时间靠近，便于复用先前 CTA 放入 L2 的 K/V 数据。

CTA 线性启动顺序：

`blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z`

固定 `y,z` 时，不同 `x` 的 CTA 在线性 ID 上连续，通常会“更一起”被调度。所以采用：

`(x, y, z) -> (Q_block, head, batch)`

`forward_kernel.cuh`

```cpp
const int sample = blockIdx.z;
const int head = blockIdx.y;
const int q_seq_block = blockIdx.x;
```

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/grid-mapping-unoptimal.svg" alt="grid-mapping-unoptimal" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：不利于 L2 复用的 CTA 映射
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/grid-mapping-optimal.svg" alt="grid-mapping-optimal" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：优化后的 CTA 映射（按 Q_block 连续）
  </figcaption>
</figure>

| GPU | L2 Size | Unoptimized Hit Rate | Optimized Hit Rate | Perf Impact |
| --- | --- | --- | --- | --- |
| RTX 3090 | 6MB | ~2% | ~98% | ~3% TFLOPs 提升 |
| A100 | 40MB | ~25.6% | ~92.6% | ~1% TFLOPs 提升 |

## Warp-Level Work Distribution

进入 CTA 内部：`B_r=64` 且每 CTA 4 warps，所以每 warp 分到 `64/4=16` 行，处理子块 `(16,128)`。

关键分工：

- `Q/O`：warp 内独立处理
- `K/V`：跨 warp 协作加载（每 warp 先各自加载 slice 到 SMEM，随后每个 warp 都要读完整块进 RF）

### `Q/O`（Independent per Warp）

- 64 行被分为 4 个独立 slice
- 每个 warp 对其 slice 独立完成加载/存储与 GEMM

### `K/V`（Cooperative across Warps）

- **Loading**：每个 warp 先搬自己的 `(16,128)` slice 到 SMEM
- **Sync**：全 CTA 等待完整 `(64,128)` block 就绪
- **Copy**：每个 warp 再从 SMEM 读整块到 RF

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/warp-workload.svg" alt="warp-workload" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：CTA 内 4 个 warp 的分工示意
  </figcaption>
</figure>

> 后续（Kernel 9）会进一步优化到 `Q` 相关加载也能更协作。

## Data Movement

数据搬运是本章最复杂部分。实现策略分三层：

1. **Core ops**：GMEM↔SMEM、SMEM->RF 原语
2. **Addressing**：各 tensor / warp 的偏移计算
3. **Abstraction**：`MatrixLDST` 封装统一接口

### Configuration Structs

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

### Storage Layout

- **SMEM**：统一 row-major（便于 GMEM 加载）
- **RF**：大多 row-major；`V` 在 RF 侧使用 transpose 语义

### LD/ST 操作规格

| From | To | Blocks | PTX Instr. / C++ | Warp-Wide Op Size | Thread Op Size | Thread Mapping | Register Shape | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMEM | SMEM | `Q, K^{(j)}, V^{(j)}` | `cp.async` | $(4,64)$ | $(1,8)$ | row-major |  |  |
| SMEM | RF | `Q, K^{(j)}, V^{(j)}` | `ldmatrix.x4` | $(16,16)$ | $(1,8)$ | column-major | $(2,2)$ | `V^{(j)}` transpose |
| RF | SMEM | `O` | standard (4B) | $(8,8)$ | $(1,2)$ | row-major | $(1,1)$ |  |
| SMEM | GMEM | `O` | standard (16B) | $(4,64)$ | $(1,8)$ | row-major |  |  |

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

### Copying Between GMEM ↔ SMEM

`cp.async` 每次搬 16B；按 `(4,64)` warp 宽块组织。  
实现做成双向模板（GM2SM 与 SM2GM）：

```cpp
template <typename T>
struct GM2SM_async {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        cp_async<16>(smem, gmem);
    }
};
 
template <typename T>
struct SM2GM {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(gmem)[0] = reinterpret_cast<uint4 *>(smem)[0];
    }
};
```

`reinterpret_cast<uint4*>` 是常见向量化访问写法，前提是对齐满足。

### SMEM → RF

`Q/K` 与 `V` 的 RF 布局不同，因此拆成两类 helper：

- `copy_warp_fragment_SM2RF()`：普通 `ldmatrix_x4`
- `copy_warp_fragment_transposed_SM2RF()`：`ldmatrix_x4_transpose`

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

### RF → SMEM

`O` 用标准 4B stores 从 RF 写回 SMEM，再由 SMEM 向量化写回 GMEM。

### GMEM 地址计算

`forward_kernel.cuh`

```cpp
const index_t sample_head_offset = sample * args.batch_stride + head * args.head_stride;
const index_t QO_gmem_block_offset = sample_head_offset + q_seq_block * CFG.B_r * gmem_seq_stride;
const index_t KV_gmem_block_offset = sample_head_offset;
```

## Tensor Abstraction Layer

`MatrixLDST` 封装 GMEM↔SMEM↔RF 全流程，统一不同 tensor 的地址与 layout 差异。

`tensor.cuh`

```cpp
template <TensorLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    // ...
    __forceinline__ __device__ constexpr void copy_GM2SM();
    __forceinline__ __device__ constexpr void copy_SM2RF(int stage = 0, int tile_offset = 0);
    __forceinline__ __device__ constexpr void copy_RF2SM();
    __forceinline__ __device__ constexpr void copy_SM2GM();
};
```

## Computing Operations

### GEMM

基于 `mma` 的核心形式：

$$
D = AB^{\top} + C
$$

实现按 `K -> M -> N` 三层展开：

```cpp
template <typename value_t, const int M_fragments, const int N_fragments,
          const int K_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void warp_fragment_mma_f32_accum(...) {
    // loop over k,m,n
    // mma_m16n8k16_f32_accum(...)
}
```

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/mma_rf_view.svg" alt="mma_rf_view" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 9：mma 操作在 RF 视角下的 fragment 组织
  </figcaption>
</figure>

### Softmax（Thread-Level 视角）

softmax 在 `fp32` 中执行，包含 element-wise 与 row-wise 两类操作。

关键步骤：

1. 初始化行统计：`m = -inf`，`l = 0`
2. `S` 乘 `1/sqrt(d_head)`（dot-product scaling）
3. 用 warp shuffle 做行最大值归约
4. 重标定 `l` 与 `O_accum`
5. `exp(S - m)`
6. 更新行和 `l`（主循环内先不做跨线程归并）
7. epilogue 再完成 `l` 的最终跨线程归约并归一化 `O`

`__shfl_xor_sync` 的两次归约（offset 2 与 1）用于同一 row quartet 聚合。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/xor_offset_1.svg" alt="xor_offset_1" style="width: 90%; max-width: 700px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 10：`__shfl_xor_sync` 第一步归约（offset=1）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/xor_offset_2.svg" alt="xor_offset_2" style="width: 90%; max-width: 700px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 11：`__shfl_xor_sync` 第二步归约（offset=2）
  </figcaption>
</figure>

### Type Conversion

需要两处 `fp32 -> 16-bit`：

- 每轮把 `S_accum` 转为 `P`（给 `PV`）
- epilogue 把 `O_accum` 转回输出 dtype

使用 paired conversion：`__float22half2_rn` / `__float22bfloat162_rn`。

## Synchronization

不同 tensor 的同步范围不同：

- `Q/O` 多为 warp 内依赖，可用 `__syncwarp()`
- `K/V` 涉及跨 warp 协作，需 `__syncthreads()`
- `cp.async` 数据可见前必须 `cp_async_wait`
- 迭代间必须确保上一轮 `ldmatrix` 全部完成，才允许覆写 SMEM

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/e2e_sync_overlap.svg" alt="e2e_sync_overlap" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 12：主循环中搬运与计算的同步/覆盖关系
  </figcaption>
</figure>

## Kernel 1 组装

### Prologue

- 计算 `(sample, head, q_seq_block)` 与各 tensor gmem 偏移
- 构造 `Q/K/V/O` 的 `MatrixLDST`
- 启动 `Q.copy_GM2SM()`，初始化 `O_accum` / `m` / `l`
- `cp_async_wait<0>() + __syncwarp()` 后把 `Q` 装入 RF

### Mainloop

每轮 `j`：

1. `K.copy_GM2SM()` 并前移 gmem 指针
2. `cp_async_wait + __syncthreads`（Barrier 1）
3. `K.copy_SM2RF()`
4. `S = QK^T`
5. online softmax：`scale -> row_max -> rescale(l/O) -> exp -> row_sum`
6. `S_accum -> P_b16`
7. `V.copy_GM2SM()` 并前移 gmem 指针
8. `cp_async_wait + __syncthreads`（Barrier 2）
9. `V.copy_SM2RF()`
10. `O_accum += PV`

### Epilogue

- `final_softmax_normalization(O_accum, l)`
- `O_accum -> O_b16`
- `O_b16.copy_RF2SM()`
- `__syncwarp()`
- `O_b16.copy_SM2GM()`

## Occupancy 与资源预算

以 `SM_86`（RTX 3090）为例：

- 每 SM 最多 1536 resident threads；128 threads/CTA 本身不是限制项
- 编译后约 `202 regs/thread`
- 目标是 2 CTAs/SM，因此需把 SMEM 控制在约 `48KB/CTA`

通过让访问时序不重叠的 tensor 共享 SMEM 切片，可从更高需求降到可接受范围。

最终得到约 `8 resident warps/SM`。

## 性能结果

在该 baseline 配置下：

- Kernel 1：约 `33.28 TFLOPS`
- Reference：约 `67.29 TFLOPS`
- 相对性能：约 `49%`

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/RTX3090_tflops_1_all.svg" alt="RTX3090_tflops_1_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 13：Kernel 1 在 RTX 3090 上的性能结果
  </figcaption>
</figure>

## 小结

Chapter3 的关键意义是搭起第一版完整链路：  
从数据搬运、GEMM、online softmax 到同步与写回全部打通。尽管只是 baseline，但它定义了后续优化（swizzling、double buffering、instruction fusion、auto-tuning）的结构基础。
