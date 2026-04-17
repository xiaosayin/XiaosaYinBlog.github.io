---
layout:     post
title:      Flash Attention Appendix B
subtitle:   Block Size 配置的性能权衡
date:       2026-04-18
author:     BY
header-img: img/post-bg-2015.jpg
catalog: true
auto_heading_numbering: true
tags:
    - Cuda, Flash Attention
---

## 简介

本附录讨论一个在实战中非常关键但常被“经验化”的问题：  
**block size 配置到底如何影响 instruction pattern 与最终性能？**

分析对象使用 kernel 16（最终优化版本），目的是尽量减少代码生成噪声，让结论更聚焦在本质 tradeoff 上。

关注四个维度：

- `Q` 是 persist 在 RF，还是每轮 reload
- `B_r`（query tile size）
- 每个 warp 处理的 query rows 数
- `B_c`（key/value tile size）

> 除特别说明外，文中 instruction 计数按 “per warp per K/V tile” 口径讨论。

原文参考：[Appendix B - Block Size Configuration](https://lubits.ch/flash/Appendix-B---Block-Size-Configuration)。

## 总体原则

更大的 block（更大 `B_r`、更多 query rows/warp、更大 `B_c`）通常能提升数据复用，减少冗余访存或冗余 softmax 开销。  
但代价是更高的 SMEM 与寄存器压力，可能触发 occupancy 下降或 register spill。

同时需要注意：  
`mma` 总工作量本身并不会因为 block size 改变而改变，变化的是“组织方式”和“外围开销”。

## Persist vs Reload：Q 的核心取舍

### Persist Q（驻留 RF）

- 优点：减少重复 `ldmatrix`（少搬运）
- 缺点：寄存器压力高，尤其在较大 `B_r` 场景容易 spill

### Reload Q（每轮重载）

- 优点：释放寄存器预算，允许更大 `B_r`
- 缺点：增加 `ldmatrix` 指令

附录给出的典型倍率（reload 相对 persist）显示：

| `B_r` \\ `B_c` | 32 | 64 |
| --- | --- | --- |
| 64 | 1.25x | 1.125x |
| 128 | 1.5x | 1.25x |

这说明 reload 的额外成本在 “大 `B_r` + 小 `B_c`” 时更明显。  
但在很多情况下，这个成本会被“更大 `B_r` 带来的全局冗余减少”反向抵消。

## `B_r` 与 Query Rows/Warp：算术强度提升机制

## `B_r` 对 L2 Arithmetic Intensity 的影响

`B_r` 增大时，CTA 数量减少。  
由于每个 CTA 都要搬运完整 K/V 序列，CTA 减少就意味着全局重复搬运减少，`cp.async` 总数下降，L2 侧算术强度上升。

附录数据中可见，`B_r: 64 -> 128` 时，L2 load 相关指标有明显下降（代表重复搬运减少）。

## Query Rows/Warp 对 SMEM Arithmetic Intensity 的影响

warp 处理 query rows 越多，总 warp 数越少，对 K/V 的重复 `SMEM -> RF` 拷贝也越少。  
因此，SMEM 侧算术强度提升。

在固定 `n_warps=4` 时，常见现象是：

- query rows/warp 从较小值提升到更大值后，`ldmatrix` 总数接近减半
- 即使配合 reload Q 产生额外加载，净效果仍可能是“总加载减少”

这也是“更大 rows/warp 往往更优”的基础原因之一（前提是寄存器压力可控）。

## `B_c`：Softmax Overhead 的关键旋钮

`B_c` 不仅影响 tile 数，还直接影响 blocked softmax 的“额外管理开销”占比。  
从附录给出的结论看，`B_c` 越小，softmax overhead 占比越高：

- `B_c=32`：约 `52.6%`
- `B_c=64`：约 `35.7%`
- `B_c=128`：约 `21.7%`

这也是为什么在很多配置中，`B_c` 提升会明显降低 non-`mma` 指令压力。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/appendix-b/B_c_scaling_overhead.svg" alt="B_c_scaling_overhead" style="width: 90%; max-width: 920px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：`B_c` 对 softmax overhead 比例的影响
  </figcaption>
</figure>

## 典型性能对比（附录结论）

相对最佳配置（100%）的示意对比：

| 配置（示意） | 相对性能 |
| --- | --- |
| `B_r=128, B_c=64` | 100% |
| `B_r=128, B_c=32` | 89.3% |
| `B_r=64, B_c=64` | 84.1% |
| `B_r=64, B_c=32` | 83.4% |

直观解释：

- 小 `B_r` 往往导致更多 CTA，K/V 重复搬运更重
- 小 `B_c` 往往带来更高 softmax overhead
- 两者叠加时，non-`mma` 开销会更容易压制 tensor path

## Hopper / Blackwell 的适用性变化

附录也讨论了新架构的影响（尤其数据中心版本）：

- warpgroup / 新型 `mma` 与相关缓存机制，降低了部分 Ampere 时代的冗余加载
- 更大的 on-chip memory（SMEM/TMEM/RMEM 组合）提升了大 block 可行性
- 某些关于 query rows/warp 的限制在新架构下被弱化

但对 `B_r` 与 L2 强度的核心关系，依然成立：  
更大 `B_r` 仍然减少重复全局搬运。

对于 GeForce 线路（非 DC），许多数据中心特性不可用，因此 Ampere 时代这套 block size tradeoff 依旧高度相关。

## 小结

Appendix B 的核心价值是把“block size 调参”从经验结论变成可量化分析：

- **`B_r`**：主要影响 CTA 数与 L2 侧重复搬运
- **Query rows/warp**：主要影响 K/V 在 SMEM->RF 的冗余拷贝
- **`B_c`**：强影响 blocked softmax 的非 `mma` 开销占比
- **Persist vs Reload Q**：是“指令冗余 vs 寄存器压力”的显式交换

工程上最重要的启发是：  
优化不应只看单一指标（如 `ldmatrix` 数或某个 stall），而应在 **复用收益、寄存器预算、occupancy、softmax 开销** 四者之间做整体平衡。
