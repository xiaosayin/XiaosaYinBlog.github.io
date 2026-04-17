---
layout:     post
title:      Flash Attention 2 Chapter7
subtitle:   A100 Profiling 深度分析
date:       2026-04-18
author:     BY
header-img: img/post-bg-2015.jpg
catalog: true
auto_heading_numbering: true
tags:
    - Cuda, Flash Attention
---

## 简介

在 Chapter6 中，我们在 RTX 3090 上已经达到并略超 reference（约 `101.5%`）。  
但同一套 kernel 在 A100 上只有约 `80.3%`，出现了接近 `20%` 的明显性能落差。

这一章的目标是：通过 A100 profiling 把差距来源拆清楚，回答一个核心问题：

> 为什么同为 Ampere 架构，A100 与 RTX 3090 会对同一 kernel 表现出如此不同的性能敏感性？

## A100 上的 Kernel 7 画像

我们将 Kernel 7 与 reference 做并排比较，重点看三类信息：

1. pipeline utilization 与 cycles
2. scalar 指令数量与类型
3. throughput ratio（`mma` vs FP32）对开销放大的机制

（测试条件：`seq_len=4096, d_head=128`）

## 计算管线利用率对比

在 A100 上，问题非常直观：

- Tensor pipeline：`60.5%`（我们） vs `75.4%`（reference）
- FMA pipeline：`29.2%` vs `20.5%`
- ALU pipeline：`21.2%` vs `13.1%`
- 标量管线总活跃度（FMA+ALU）高约 `50%`

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter7/A100_pipeline_util_7_all.svg" alt="A100_pipeline_util_7_all" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：A100 上 pipeline utilization 对比
  </figcaption>
</figure>

cycle 维度同样揭示了问题：  
尽管两者 tensor 相关总周期接近，但我们的 scalar 周期显著偏高，吞掉了大量本可用于维持 Tensor Core 连续供给的窗口。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter7/A100_pipeline_cycles_7_all.svg" alt="A100_pipeline_cycles_7_all" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：A100 上 pipeline cycles 对比
  </figcaption>
</figure>

## RTX 3090：为什么没这么痛？

在 RTX 3090 上，Kernel 7 的 tensor 利用率反而略高于 reference（约 `48.0%` vs `47.2%`）。  
虽然 scalar 管线同样偏高，但影响被“掩盖”得多。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter7/RTX3090_pipeline_util_7_cur_only.svg" alt="RTX3090_pipeline_util_7_cur_only" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：RTX 3090 上 pipeline utilization（同类对比）
  </figcaption>
</figure>

关键原因在于吞吐比例差异（后文详细展开）：  
RTX 3090 的 tensor/scalar 比例更“温和”，scalar 开销更容易被掩蔽；A100 则会把同样开销放大成明显瓶颈。

## Scalar Instruction Overhead

从指令计数看，Kernel 7 的额外负担主要在以下类型：

- `IMAD`
- `LOP3.LUT`
- `MOV`
- `SHF`
- 以及 `CS2R`（reference 基本没有）

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter7/7_128_32_vs_reference_instructions.svg" alt="7_128_32_vs_reference_instructions" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 4：Kernel 7 与 reference 的指令计数对比
  </figcaption>
</figure>

### 只看 opcode 还不够：必须看 modifier

Nsight 默认统计 base opcode 容易误导。  
比如高 `IMAD` 计数，可能大头并非真正乘加，而是：

- `IMAD.MOV.U32`（寄存器拷贝）
- `IMAD.SHL.U32`（位移）

展开 modifier 后，问题更清晰：

- register copy 约 `11.6x`
- logic op 约 `216x`
- shift 约 `23.8x`
- 标量指令总量约 `2x`，总体指令约高 `38.7%`

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter7/7_128_32_vs_reference_instructions_mod.svg" alt="7_128_32_vs_reference_instructions_mod" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 5：带 modifier 的指令拆分
  </figcaption>
</figure>

## Data Dependency：真正卡住关键路径的地方

数量多只是表象，更致命的是这些额外 scalar 指令会制造 RAW 依赖，阻塞关键指令：

- `cp.async`
- `ldmatrix`
- `mma`

典型例子：

```cpp
SHF.L.U32 R176, R176, 0x1, RZ ;
...
LDSM.16.M88.4 R32, [R176+0x4000] ; // ldmatrix
```

`ldmatrix` 必须等 `SHF` 完成后才能发射，直接拉长关键路径。

量化后可见，Kernel 7 的 dependency-creating 指令约为优化后（K16）的：

- `6.8x`（一组配置）
- `11.3x`（另一组配置）

这就是 Tensor Core“吃不饱”的直接原因。

## 为什么 A100 更容易被拖垮：吞吐比例放大效应

核心对比：

| Device | `mma` TFLOPs/s（16-bit 输入, 32-bit accum） | FP32 TFLOPs/s | `mma / FP32` |
| --- | --- | --- | --- |
| A100 | 311.84 | 19.5 | `16x` |
| RTX 3090 | 71 | 35.6 | `2x` |

结论：

- **A100（16x）**：Tensor Core 太快，scalar 依赖稍微堵一下就会大量空转
- **RTX 3090（2x）**：tensor 与 scalar 更平衡，同样依赖更容易被隐藏

因此，A100 把“过量 scalar + 强依赖链”放大成了显著性能损失。

## Block Size 限制：第二个关键问题

除吞吐比例外，还有一个结构性问题：Kernel 7 对大 block 配置支持不佳，特别是寄存器压力导致 spill。

在 A100 上，reference 倾向更大更高效的 block；  
但 Kernel 7 在相关配置上会出现较重 spill（例如每线程 `272B`），导致本地内存访问成本显著上升，抵消了大块带来的潜在收益。

这也解释了为什么 Kernel 7 的“天花板”会被卡住：  
既有 scalar 关键路径问题，又无法稳定利用更优 block 配置。

## 小结

Chapter7 的结论可以压缩成两条主因：

1. **Scalar overhead + 依赖链过长**
   - 指令明显冗余，且大量指令直接阻塞 `cp.async/ldmatrix/mma`
2. **寄存器压力限制 block size**
   - 在 A100 最有价值的配置上出现 spill，无法把硬件潜力转成有效吞吐

为什么 RTX 3090 看起来“没问题”？

- tensor/scalar 吞吐比更平衡（`2x`），更能掩蔽这些代价
- 硬件资源边界（如 SMEM）也在一定程度上遮蔽了问题

而 A100 恰恰会把这些短板全部暴露出来。

## 下一步

下一章（Part 8）的方向就是“手术刀式”去掉这些多余 scalar 负担：  
通过 SASS 级分析削减 register copy / shift / logic 指令，降低依赖链长度与寄存器压力，把 A100 性能显著拉回。
