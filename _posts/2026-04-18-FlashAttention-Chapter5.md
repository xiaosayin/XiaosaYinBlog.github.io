---
layout:     post
title:      Flash Attention 2 Chapter5
subtitle:   Cutlass GEMM 优化
date:       2026-04-18
author:     BY
header-img: img/post-bg-2015.jpg
catalog: true
auto_heading_numbering: true
tags:
    - Cuda, Flash Attention
---

## 简介

在 Chapter4 中，我们通过 swizzling 消除了主要的 SMEM bank conflicts，性能已经接近 reference。  
这一章进入 Cutlass 风格的 GEMM 优化阶段，重点是进一步提升 overlap 与可扩展性。

本章实现 3 组渐进优化：

1. **Kernel 3**：`GMEM -> SMEM` eager loading（double buffering）
2. **Kernel 4**：片上 LD/ST 与计算交错（fragment interleaving）
3. **Kernel 5**：`SMEM -> RF` double buffering

虽然在当前配置下，后两者并非都带来正向收益，但它们是后续 auto-tuning 的关键能力。

## Kernel 3：提前加载 K/V（GMEM->SMEM 双缓冲）

在 Kernel 2 里，`K/V` 往往“即用即取”，导致 warp 在等待 GMEM 传输时空转。  
profile 显示 `long_scoreboard` stall 仍较高，因此我们要把加载时机前移。

### 安全加载点

要提前加载，必须保证不覆盖仍在使用的 SMEM 数据。关键同步约束：

- `K/V` 是跨 warp 协作路径，`GMEM->SMEM` 与 `SMEM->RF` 之间需要 `__syncthreads()`
- 迭代间也需要 `__syncthreads()`，避免下一轮覆写上一轮未读完的数据

调整后执行逻辑：

- Prologue 先预取 `K[0]`
- 主循环开头先确保上一轮数据可见，再预取当前 `V`
- softmax 后再做一次同步，在安全点预取下一块 `K`

这种方式本质就是 **double buffering**：计算当前块时，后台搬运下一块。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/Flash-Double-Buffering-Before-Shorter.svg" alt="Flash-Double-Buffering-Before-Shorter" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：优化前（传输与计算重叠不足）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/Flash-Double-Buffering-After.svg" alt="Flash-Double-Buffering-After" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：优化后（GMEM 传输与计算重叠）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/Single-To-Double-Buffering.svg" alt="Single-To-Double-Buffering" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：单缓冲到双缓冲的通用转变
  </figcaption>
</figure>

### Kernel 3 效果

`long_scoreboard` 从约 `15.15%` 降到约 `0.95%`，GMEM 等待显著下降。  
整体性能从约 `98.3%` 提升到约 `99.4%`（相对 reference）。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/RTX3090_tflops_3_all.svg" alt="RTX3090_tflops_3_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 4：Kernel 3 性能结果
  </figcaption>
</figure>

## Kernel 4：片上 LD/ST 与计算交错（Fragment Interleaving）

即便 GMEM 端优化后，kernel 仍存在一个问题：  
`K/V` 往往整 tile 全部进 RF 后才开始 `mma`，这会拉长“首个 load 到首个 compute”的间隔，并推高寄存器压力。

### 关键思路

把整 tile 拆成沿 `k` 维的小 sub-tiles，循环模式改为：

1. 载入一小段 fragments
2. 立即执行对应 `mma`
3. 再载入下一段 fragments
4. 重复直到完成

为了最大化复用，采用 Cutlass 常用的 **沿内层 `k` 维切片** 策略，而不是 A 行/B 列分离加载。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/mma_mnk_iter1.svg" alt="mma_mnk_iter1" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 5：低复用策略（示意 1）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/mma_mnk_iter2.svg" alt="mma_mnk_iter2" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 6：低复用策略（示意 2）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/mma_kmn_iter1.svg" alt="mma_kmn_iter1" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 7：沿 k 维切片策略（示意 1）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/mma_kmn_iter2.svg" alt="mma_kmn_iter2" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 8：沿 k 维切片策略（示意 2）
  </figcaption>
</figure>

### 核心代码变化

原先 RF 缓冲常见写法：

```cpp
uint32_t input[rows/8][cols/8];
```

改为仅保留 sub-tile 宽度（例如 `2` 个 `k` fragments）：

```cpp
uint32_t input[rows/8][2];
```

并在 `matmul()` 中以 `k` 维步长加载 + 计算：

```cpp
for (int k = 0; k < GEMM::TotalKFragments; k += 2) {
    if constexpr (!A_t::load_entire_block_into_rf) {
        A.copy_SM2RF(k);
    }
    B.copy_SM2RF(k);
    warp_fragment_mma_f32_accum(A.data(), B.data(), C.data(), A_col_offset, B_col_offset);
}
```

### 寄存器压力收益

这个优化不仅改善 overlap，还显著降低 register pressure，尤其让部分大 block 配置从“必然 spill”变成“可用配置”。

| 配置（示意） | 变化前 -> 变化后 |
| --- | --- |
| `(128,64,64,4)` used registers | `242 -> 207` |
| `(128,128,32,4)` spills | `304/356 -> 0/0` |
| `(128,128,128,4)` spill loads | `2208 -> 1312` |

### Kernel 4 效果

在当前测试设定下，Kernel 4 达到约 `100.0%` reference。  
这是本系列首次达到（并接近持平）reference 的里程碑。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/RTX3090_tflops_4_all.svg" alt="RTX3090_tflops_4_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 9：Kernel 4 性能结果
  </figcaption>
</figure>

## Kernel 5：SMEM->RF 双缓冲

与 Kernel 3 类似，这一步把双缓冲思想继续推进到 `ldmatrix` 阶段：

- 在寄存器中为下一组 fragments 预留额外 stage
- 当前 stage 参与 `mma` 时，后台预取下一 stage
- 迭代时在两个 stage 间 toggle

由于这一层是 warp-synchronous 路径，一般不需要显式 CTA barrier。

### 存储与 matmul 结构变化

为支持 staged buffering，RF 存储增加 stage 维度，例如：

```cpp
uint32_t input[rows/8][2];
// -> 
uint32_t input[2][rows/8][2];
```

`matmul()` 采用 stage toggle：

```cpp
int A_stage = 0, B_stage = 0;
A.copy_SM2RF(A_stage, 0);
B.copy_SM2RF(B_stage, 0);

for (int k = 0; k < GEMM::TotalKFragments; k += 2) {
    int k_load = k + 2;
    if (k_load < GEMM::TotalKFragments) {
        A.copy_SM2RF(A_stage_toggle ^ A_stage, k_load);
        B.copy_SM2RF(B_stage_toggle ^ B_stage, k_load);
    }
    warp_fragment_mma_f32_accum(A.data(A_stage), B.data(B_stage), C.data(0), A_col_offset, B_col_offset);
    A_stage ^= A_stage_toggle;
    B_stage ^= B_stage_toggle;
}
```

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/mma-Double-Buffering-Shorter.svg" alt="mma-Double-Buffering-Shorter" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 10：SMEM->RF 双缓冲示意
  </figcaption>
</figure>

### Kernel 5 效果

在当前配置下出现轻微回退：约 `100.0% -> 99.6%` reference。  
但该优化在其他 block 配置中可带来最高约 `4%` 的收益，因此对后续 auto-tuning 仍然非常关键。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/RTX3090_tflops_5_all.svg" alt="RTX3090_tflops_5_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 11：Kernel 5 性能结果
  </figcaption>
</figure>

## 小结

本章完成了 3 个典型 Cutlass GEMM 优化模块：

- **GMEM->SMEM 双缓冲**：显著压低 `long_scoreboard`
- **fragment interleaving**：提升搬运-计算 overlap，并显著降低寄存器压力
- **SMEM->RF 双缓冲**：为多配置 auto-tuning 提供性能上限空间

从工程角度看，这些优化不只是追某一组配置的峰值，更是在构建“可调参、可迁移”的 kernel 结构。  
下一章将继续推进：通过 FP 指令融合与 auto-tuning，争取整体超过 reference。
