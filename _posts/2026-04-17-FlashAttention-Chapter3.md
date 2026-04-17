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

在 [Part 2](https://lubits.ch/flash/Part-2) 里，我们已经梳理了 Flash Attention 的底层 CUDA building blocks：Tensor Core `mma` 指令与高效内存搬运（`cp.async`、`ldmatrix`）。  
这一章进入真正的组装阶段：把这些零件拼成第一个可运行的完整 kernel（Kernel 1）。

这一版是整个系列的基线实现（base implementation）：先保证 correctness，再逐步迭代优化性能。

本章目标：

- 仅 forward pass
- non-causal attention
- `d_head = 128`
- 不含 dropout 与 KV caching
- `Q/K/V` 序列长度相同
- 序列长度可被 block size 整除（本文配置通常为 `64~128`）
- 输入输出用 `bf16/fp16`，softmax 在 `fp32` 中计算

按该配置，Kernel 1 在 RTX 3090 上可达到官方实现约 `49%` 的性能。

## Kernel 总体结构

Flash Attention kernel 在实现上遵循标准三段式：

1. **Prologue**
   - 计算各 tensor 的地址偏移
   - 完成一次性加载（例如 `Q: GMEM -> SMEM -> RF`）
   - 初始化 softmax 统计量（`m`、`l`）与输出累加器
2. **Mainloop**
   - 迭代每个 `K/V` block
   - `K: GMEM -> SMEM -> RF`
   - 计算分数矩阵并执行 online softmax
   - `V: GMEM -> SMEM -> RF`
   - 计算输出贡献并累计到 `O_accum`
3. **Epilogue**
   - 完成最终 softmax 归一化
   - `fp32 -> fp16/bf16` 类型转换
   - `O: RF -> SMEM -> GMEM`

实现层面的核心难点主要有三个：

1. **Data Movement**：跨层内存搬运（GMEM/SMEM/RF）和不同 layout 协同
2. **Math Operations**：用 `mma` + warp primitives 实现 GEMM 与 online softmax
3. **Synchronization**：线程与 warp/CTA 级同步，避免 race condition

## Kernel 配置

从 Kernel 1 到 Kernel 7，我们采用固定配置：

- `B_r = 64`
- `B_c = 64`
- `d_head = 128`
- 每个 CTA 使用 `128` 线程（`4` 个 warps）

该组合与 `m16n8k16` 对齐良好，且 tile 大小可以比较紧凑地放进 SMEM 与 RF。

## CTA 级工作划分

输入张量形状记为：

`(batch_size, seq_len, n_heads, d_head)`

分块后：

- `Q` 与 `O` 按 query 维切成 `B_r` 行块
- `K` 与 `V` 按 key/value 维切成 `B_c` 行块

每个 CTA 负责一个 `(sample, head, q_block)`，并遍历该 `(sample, head)` 对应的全部 `K/V` blocks。

总 CTA 数：

`n_samples * n_heads * T_r`，其中 `T_r = seq_len / B_r`。

### Kernel 参数

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

我们采用 `(x, y, z) -> (Q_block, head, batch)`。  
原因：同一 `(sample, head)` 下，不同 `Q_block` CTA 会复用相同 `K/V` block，若调度时间接近，可提升 L2 cache 命中率。

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

## Warp 级工作划分

在一个 CTA 中，`64` 行块由 `4` 个 warps 均分，每个 warp 负责 `16` 行，因此 warp 处理的子块是 `(16, 128)`。

- `Q/O`：每个 warp 独立处理自己的 slice
- `K/V`：需要跨 warp 协作加载（先协作写入 SMEM，再各自从 SMEM 拷到 RF）

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/warp-workload.svg" alt="warp-workload" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：CTA 内 4 个 warp 的分工示意
  </figcaption>
</figure>

## Data Movement

这一部分是 Kernel 1 最复杂的部分。我们按“低层原语 -> 地址管理 -> 统一封装”三层来组织：

1. **Core memory ops**：`GMEM <-> SMEM` 与 `SMEM -> RF` 原语
2. **Addressing**：按 tensor/warp 计算正确偏移
3. **Abstraction**：`MatrixLDST` 封装统一接口

### 配置结构

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

### 跨层搬运规格

| From | To | Blocks | PTX Instr. / C++ | Warp-Wide Op Size | Thread Op Size | Thread Mapping | Register Shape | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMEM | SMEM | `Q, K^{(j)}, V^{(j)}` | `cp.async` | $(4,64)$ | $(1,8)$ | row-major |  |  |
| SMEM | RF | `Q, K^{(j)}, V^{(j)}` | `ldmatrix.x4` | $(16,16)$ | $(1,8)$ | column-major | $(2,2)$ | `V^{(j)}` 走 transpose |
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

### GMEM <-> SMEM

`load_store.cuh`

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

### SMEM -> RF（含 transpose）

`Q/K` 使用普通 `ldmatrix_x4`；`V` 使用 `ldmatrix_x4_transpose`，因为 `V` 在 RF 中以等效 column-major 组织，以匹配后续 `mma` 语义。

`load_store.cuh`

```cpp
template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int col_fragment_offset = 0
) {
    // ... ldmatrix_x4(...)
}

template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int row_fragment_offset = 0
) {
    // ... ldmatrix_x4_transpose(...)
}
```

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

### RF -> SMEM

`O` 写回使用标准 4B store（`smem[dst] = rf[src]`），逐 fragment 落到 SMEM，再 vectorized 写回 GMEM。

## Tensor Abstraction：`MatrixLDST`

我们用 `MatrixLDST` 统一封装 GMEM/SMEM/RF 全路径搬运与地址计算，避免在 kernel 主循环里塞满重复偏移逻辑。

`tensor.cuh`

```cpp
template <TensorLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    // ...
    __forceinline__ __device__ constexpr void copy_GM2SM();
    __forceinline__ __device__ constexpr void copy_SM2RF(int stage = 0, int tile_offset = 0);
    __forceinline__ __device__ constexpr void copy_RF2SM();
    __forceinline__ __device__ constexpr void copy_SM2GM();
    // ...
};
```

## Computing Operations

### GEMM

核心运算是 warp 级 `mma`：

$$
D = AB^{\top} + C
$$

在 `m16n8k16` 下，按 `K -> M -> N` 三重循环展开 fragment 级计算。

`gemm.cuh`

```cpp
template <typename value_t, const int M_fragments, const int N_fragments,
          const int K_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments],
    uint32_t (&regs_B)[N_fragments][K_fragments],
    accum_t (&regs_C)[M_fragments][N_fragments * 2]
) {
    // ... nested loops over k,m,n ...
    // ... mma_m16n8k16_f32_accum(...)
}
```

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/mma_rf_view.svg" alt="mma_rf_view" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 9：mma 操作在 RF 视角下的 fragment 组织
  </figcaption>
</figure>

### Online Softmax

softmax 在 `fp32` 中进行，关键步骤：

1. `S` 缩放：乘以 `1/sqrt(d_head)`
2. 计算每行 `row max`（warp 内用 `__shfl_xor_sync` 做归约）
3. 用新 `m` 重标定 `l` 与 `O_accum`
4. 对 `S` 做 `exp(S - m)`
5. 更新行和 `l`
6. epilogue 中完成跨线程最终 `l` 归约并归一化 `O`

`softmax.cuh`

```cpp
template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void calc_row_max(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m_next)[QO_fragments],
    accum_t (&m_cur)[QO_fragments]
) {
    // ... in-thread max ...
    // ... __shfl_xor_sync(..., 2) ...
    // ... __shfl_xor_sync(..., 1) ...
}
```

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

### 类型转换

需要两处 `fp32 -> 16-bit`：

- 每次迭代将 `S_accum` 转成 `P`（`bf16/fp16`）参与 `PV`
- epilogue 将 `O_accum` 转成输出类型

`convert_to_16_bit_dtype()` 使用 `float2 -> half2/bfloat162` 的 paired conversion（`__float22half2_rn` / `__float22bfloat162_rn`）。

## Synchronization

为了避免覆盖未消费数据，需要在关键阶段同步：

- `Q/O` 相关多数场景可用 warp 级 `__syncwarp()`
- `K/V` 由于涉及跨 warp 协作，需 CTA 级 `__syncthreads()`
- `cp.async` 数据可见前必须先 `cp_async_wait`
- 主循环迭代之间，需要确保上一轮 `SMEM -> RF` 已完成再覆写 SMEM

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/e2e_sync_overlap.svg" alt="e2e_sync_overlap" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 12：主循环中搬运与计算的同步/覆盖关系
  </figcaption>
</figure>

## Kernel 1 组装

### Prologue

- 建立 `gmem/smem` 指针与 block 偏移
- 构造 `Q/K/V/O` 对应 `MatrixLDST`
- 启动 `Q` 异步加载并初始化 `O_accum`、`m`、`l`
- `cp_async_wait + __syncwarp` 后，将 `Q` 载入 RF

### Mainloop

每轮 `j`：

1. 加载 `K`（`GMEM -> SMEM -> RF`）
2. 计算 `S = QK^T`
3. 执行 online softmax，并更新 `m`/`l`/`O_accum` 的重标定
4. 将 `S_accum` 转成 `P`（16-bit）
5. 加载 `V`（`GMEM -> SMEM -> RF`）
6. 计算 `O_accum += PV`

### Epilogue

- `final_softmax_normalization`
- `O_accum` 转 `16-bit`
- `O: RF -> SMEM -> GMEM`

## Occupancy 与性能

在 `SM_86`（RTX 3090）下：

- 配置：`B_r=64, B_c=64, d_head=128, 4 warps/CTA`
- 编译后寄存器使用约 `202 regs/thread`
- SMEM 通过共享切片后控制在 `48KB/CTA`

最终可达到约 `8 resident warps/SM`。  
实测性能约 `33.28 TFLOPS`，官方参考约 `67.29 TFLOPS`，即约 `49%`。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter3/RTX3090_tflops_1_all.svg" alt="RTX3090_tflops_1_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 13：Kernel 1 在 RTX 3090 上的性能结果
  </figcaption>
</figure>

## 小结

这一章完成了第一个可运行的 Flash Attention kernel：数据搬运、GEMM、online softmax、同步与写回链路都已打通。  
虽然当前仍是 baseline，但结构完整且可迭代，为后续 swizzling、double buffering、instruction fusion 等优化打下基础。
