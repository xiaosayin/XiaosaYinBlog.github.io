---
layout:     post
title:      Flash Attention 2 Chapter6
subtitle:   FP 指令融合与 Auto-Tuning
date:       2026-04-18
author:     BY
header-img: img/post-bg-2015.jpg
catalog: true
auto_heading_numbering: true
tags:
    - Cuda, Flash Attention
---

## 简介

在 Chapter5 中，我们已经完成了 Cutlass 风格的三项核心优化（eager loading、fragment interleaving、double buffering），在 RTX 3090 上达到接近 reference 的性能。  
这一章继续做最后两步：

1. **Kernel 6：FP instruction fusion**（优化 softmax 的 FP32 指令开销）
2. **Kernel 7：Auto-tuning**（系统搜索最优配置）

最终目标是：在 RTX 3090 上小幅超过 reference。

## Kernel 6：提升 FP32 吞吐

此前优化主要集中在 Tensor Core 路径，现在 matmul 已接近峰值，下一步应关注 FP32 侧（尤其 softmax）的指令效率。

### Roofline 视角

对于这个 kernel，`L2` arithmetic intensity 更能代表真实瓶颈；`L1` 会受 `cp.async(.cg)` 的 bypass 行为影响而失真，`DRAM` 只体现主存与回写的“外层”流量。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter6/rtx3090_roofline_tensor_full.svg" alt="rtx3090_roofline_tensor_full" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：Roofline 视角下的瓶颈定位（Tensor core 接近饱和）
  </figcaption>
</figure>

### Softmax 中做 FMA 融合

思路：把原本分离的 scale 与减 max 操作融合，利用 `FFMA` 形式减少 FP 指令数。

实现要点：

- 不再单独调用 `scale_S_accum()`
- 在 `exponentiate_tensor()` 中把 `S * scale - max_scaled` 合并
- `scale_l_O()` 中使用与新标度匹配的 `exp2f` 形式

同时把“快速指数近似”显式化：用 `exp2f()` 路径来表达原先 `expf()` 的快速实现逻辑。

### 关键代码改动

`softmax.cuh`（`scale_l_O`）

```cpp
template <int QO_fragments, int d_head_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void
scale_l_O(
    accum_t (&m_next)[QO_fragments],
    accum_t (&m_cur)[QO_fragments],
    accum_t (&l)[QO_fragments],
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t softmax_scale
) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t scale = exp2f((m_cur[q] - m_next[q]) * softmax_scale);
        m_cur[q] = m_next[q];
        l[q] *= scale;
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= scale;
        }
    }
}
```

`softmax.cuh`（`exponentiate_tensor`）

```cpp
template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void
exponentiate_tensor(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m)[QO_fragments],
    accum_t softmax_scale
) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t max_scaled = m[q] * softmax_scale;
        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] = exp2f(S_accum[q][k] * softmax_scale - max_scaled);
        }
    }
}
```

### Kernel 6 效果

- 吞吐：`67.11 -> 67.23` TFLOPS（约 `99.9%` reference）
- FP pipeline 压力下降，Tensor Core 利用率小幅回升

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter6/RTX3090_tflops_6_all.svg" alt="RTX3090_tflops_6_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：Kernel 6 性能结果
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter6/rtx3090_5_6_tensor_fma_pipe_util.svg" alt="rtx3090_5_6_tensor_fma_pipe_util" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：Kernel 5/6 管线利用率对比
  </figcaption>
</figure>

## Kernel 7：Auto-Tuning

到这里为止，我们已经堆叠了多个可配置优化开关。  
单一固定 block 配置不一定全局最优，因此需要系统化搜索配置空间。

### 配置空间

`flash_attention.cuh`

```cpp
struct FlashForwardKernelConfig {
    const int d_head; // [128]
    const int B_r;    // [64, 128]
    const int B_c;    // [32, 64]
    const int n_warps;// [4]

    const bool async_copy;
    const bool swizzled;
    const bool eager_load_blocks;

    const int Q_mma_load_K_fragments;
    const int K_mma_load_K_fragments;
    const int V_mma_load_K_fragments;

    const bool mma_double_buffer_loads;
    const bool optimized_softmax;
};
```

实际搜索会先过滤明显劣质组合（例如不启用 swizzling、spill 过重配置），再在可行子空间里比较吞吐。

### 最优配置结果

最佳配置为：

`(128, 64, 64, 4): async+eager+swizzled+load_0_2_0_fragments+buffer+opt_softmax`

该配置在 RTX 3090 上达到约 `101.5%` reference。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter6/RTX3090_tflops_7_all.svg" alt="RTX3090_tflops_7_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 4：Kernel 7（Auto-tuned）性能结果
  </figcaption>
</figure>

### Kernel 6 vs Kernel 7

Kernel 7 的关键变化之一：让 `Q` 在 mainloop 期间持续驻留 RF，减少每轮 `ldmatrix` 次数。  
对应的 SMEM 相关 stall 进一步下降：

| Stall | Kernel 6 | Kernel 7 | Delta |
| --- | --- | --- | --- |
| `barrier` | 4.82% | 2.72% | -2.11% |
| `mio_throttle` | 2.46% | 1.95% | -0.51% |
| `short_scoreboard` | 1.94% | 1.70% | -0.24% |

### Block size 简要对比（RTX 3090）

| 配置 | TFLOPS | 相对 Kernel 7 |
| --- | --- | --- |
| `(128, 64, 64, 4): load_0_2_0 + buffer + opt_softmax` | 68.31 | 100.0% |
| `(128, 64, 32, 4): load_2_2_0` | 67.39 | 98.64% |
| `(128, 128, 32, 4): load_2_2_0 + buffer + opt_softmax` | 67.36 | 98.61% |
| `(128, 128, 64, 4): load_2_2_0 + opt_softmax` | 54.26 | 79.42% |

Occupancy 角度（示意）：

| 配置 | Registers / Thread | SMEM / CTA | Warps / SM |
| --- | --- | --- | --- |
| `(128,64,64,4)` | 229 | 48KiB | 8 |
| `(128,64,32,4)` | 168 | 32KiB | 12 |
| `(128,128,32,4)` | 255（0B spill） | 48KiB | 8 |
| `(128,128,64,4)` | 255（272B spill） | 64KiB | 4 |

## A100 上的表现

同一套优化在 A100 上仅约 `80%` reference，说明已经进入架构相关瓶颈阶段：

| 配置 | TFLOPS | 相对 Reference |
| --- | --- | --- |
| `(128,64,64,4): load_0_2_0 + opt_softmax` | 149.71 | 80.31% |
| `(128,128,32,4): load_2_2_2 + buffer` | 142.82 | 76.62% |
| `(128,64,32,4): load_2_2_2 + opt_softmax` | 135.24 | 72.55% |
| `(128,128,64,4): load_2_2_2 + opt_softmax` | 130.14 | 69.81% |
| `Reference` | 186.41 | 100.00% |

这也是下一章（Part 7）的重点：专门针对 A100 画像与瓶颈拆解。

## 小结

Chapter6 完成了两个关键收官动作：

- **Kernel 6（FP 指令融合）**：降低 softmax 的 FP32 指令压力，性能稳定贴近 reference
- **Kernel 7（Auto-tuning）**：系统搜索后，在 RTX 3090 上达到并略超 reference（约 `101.5%`）

同时也得到一个重要结论：跨架构迁移并非“同参同效”，A100 仍存在显著性能差距，需要独立 profiling 与优化策略。
