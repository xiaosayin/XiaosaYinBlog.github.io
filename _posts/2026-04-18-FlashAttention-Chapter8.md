---
layout:     post
title:      Flash Attention 2 Chapter8
subtitle:   指令削减（Instruction Reduction）
date:       2026-04-18
author:     BY
header-img: img/post-bg-2015.jpg
catalog: true
auto_heading_numbering: true
tags:
    - Cuda, Flash Attention
---

## 简介

在 Chapter7 中，我们已经定位了 A100 上性能落后的核心问题：

1. scalar 指令过多，且大量指令制造依赖链，阻塞 `cp.async` / `ldmatrix` / `mma`
2. register pressure 过高，限制了大 block 配置
3. A100 的 `mma:FP32` 吞吐比更悬殊，使上述问题被放大

本章目标是系统化“减指令”，通过 Kernel 8~11 四步优化，显著缩小与 reference 的差距。

优化路线：

1. **Kernel 8**：strided swizzling，减少逻辑/位移指令并大幅降寄存器压力
2. **Kernel 9**：优化 fragment storage，去掉大量寄存器 copy 指令
3. **Kernel 10**：去掉 `CS2R` 并优化首次 softmax 迭代
4. **Kernel 11**：补齐 RF->SMEM 路径的 strided swizzling

## Kernel 8：减少逻辑与位移指令 + 降低寄存器压力

先看 kernel 7 的 SASS，可见 `LOP3.LUT`、`IMAD.SHL.U32`、`SHF.L.U32` 大量夹在 `LDGSTS/LDSM` 前后，说明地址计算开销非常重。

根因并不是“swizzling 无效”，而是“swizzling 写法让编译器难以复用偏移”，导致每轮反复重算地址。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/7_swizzling_vs_no_swizzling_instructions_mod.svg" alt="7_swizzling_vs_no_swizzling_instructions_mod" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：Kernel 7 中 swizzled 与 non-swizzled 指令量对比
  </figcaption>
</figure>

### Strided Swizzling 思路

把“每次重算 offset”的形式改成“显式 stride + base offset”：

- 在线程维度编码规律化 stride
- 在 swizzle region 内复用偏移
- region 间只做基址切换

这样可以把地址计算暴露给编译器，帮助其缓存/复用。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/swizzling_regions_single.svg" alt="swizzling_regions_single" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：单个 swizzle region
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/swizzling_regions_repeated.svg" alt="swizzling_regions_repeated" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：swizzle region 的重复结构
  </figcaption>
</figure>

### Kernel 8 的关键收益

地址寄存器复用显著改善：

- `LDGSTS`：SMEM 地址寄存器从多寄存器压到近似单基址模式
- `LDSM`：地址寄存器复用大幅提升

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/7_LDGSTS_regs.svg" alt="7_LDGSTS_regs" style="width: 88%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 4：Kernel 7 的 LDGSTS 地址寄存器分布
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/8_LDGSTS_regs.svg" alt="8_LDGSTS_regs" style="width: 88%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 5：Kernel 8 的 LDGSTS 地址寄存器分布
  </figcaption>
</figure>

寄存器 spill 大幅下降（附录给出的典型配置）：

| Kernel | stack_frame | spill_stores | spill_loads |
| --- | --- | --- | --- |
| 7 | 272 | 336 | 304 |
| 8 | 16 | 32 | 16 |
| Reduction | 94.1% | 90.5% | 94.7% |

性能上，最佳配置切换到更大块尺寸后，约从 `149.71` 提升到 `163.76` TFLOPS（+9.38%）。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/A100_tflops_8_all.svg" alt="A100_tflops_8_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 6：Kernel 8 性能结果
  </figcaption>
</figure>

## Kernel 9：减少寄存器 Copy 指令

Kernel 8 之后，`IMAD.MOV.U32` 与 `MOV` 仍偏高。  
深入看 `LDSM` 与 `HMMA` 的 PTX->SASS 映射后，问题集中在寄存器对齐/分组要求与 fragment 布局不匹配上，尤其是 `B` operand。

核心修正：

- 调整 `K` 的 fragment 存储/加载布局（row-major）以匹配 `HMMA` 对 `B` 的连续寄存器需求
- 保留 `(16,16)` 作为最佳 fragment 形状（兼顾兼容性与加载/交错效率）

构建期指令计数（附录）：

| Kernel | `IMAD.MOV.U32` | `MOV` |
| --- | --- | --- |
| 8 | 187 | 159 |
| 9 | 36 | 18 |
| Reduction | 81% | 89% |

性能继续抬升：`163.76 -> 177.68` TFLOPS（+8.51%）。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/A100_tflops_9_all.svg" alt="A100_tflops_9_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 7：Kernel 9 性能结果
  </figcaption>
</figure>

## Kernel 10：去掉 CS2R + 优化首次 Softmax 迭代

此步主要做两件事：

1. 去掉用于零初始化的 `CS2R` 路径
2. 将 softmax 首轮迭代单独处理，减少不必要初始化与缩放

理论上应提升，但实测出现小幅回退：`177.68 -> 175.10` TFLOPS（-1.46%）。

回退原因：编译器调度改变后，`LDSM -> HMMA` 间隔缩短，`short_scoreboard` 明显上升。

| Stall Type | Kernel 9 | Kernel 10 | Delta |
| --- | --- | --- | --- |
| `short_scoreboard` | 2.36% | 7.79% | +5.44% |
| `wait` | 36.99% | 38.48% | +1.49% |

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/A100_tflops_10_all.svg" alt="A100_tflops_10_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 8：Kernel 10 性能结果（小幅回退）
  </figcaption>
</figure>

## Kernel 11：补齐 RF->SMEM 的 Strided Swizzling

最后一块拼图是输出路径 `RF -> SMEM`。  
它只在尾部执行一次（写回最终 `O`），所以理论收益有限，但做完后可恢复并进一步拉高性能。

这一方向的特点：

- 每线程 store 为 4B（非向量化）
- 同行线程共享 stride，不同线程仅 offset 不同
- 可在行级共享部分 stride 计算

最终性能：`175.10 -> 177.40` TFLOPS（+1.31%）。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/swizzling-R2Smem.svg" alt="swizzling-R2Smem" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 9：RF -> SMEM 的 strided swizzling
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter8/A100_tflops_11_all.svg" alt="A100_tflops_11_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 10：Kernel 11 性能结果
  </figcaption>
</figure>

## 总结

Kernel 8~11 的净效果：

- 从 Kernel 7 的约 `149.7` TFLOPS
- 提升到 Kernel 11 的约 `177.4` TFLOPS
- 总提升约 **18.6%**

这一阶段的核心收获不是单点技巧，而是“编译器友好”的实现原则：

1. 把可重用地址模式显式化（stride/base），避免每轮重算
2. 让 fragment 布局契合 `LDSM/HMMA` 的 SASS 对齐约束
3. 避免引入破坏调度间隙的局部改动（即便名义上减少了指令）

这使 A100 上的差距显著收敛，为后续最终微调阶段打下基础。
