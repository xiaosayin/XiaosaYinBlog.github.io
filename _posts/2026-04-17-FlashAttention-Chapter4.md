---
layout:     post
title:      Flash Attention 2 Chapter4
subtitle:   Bank Conflicts 与 Swizzling
date:       2026-04-17
author:     BY
header-img: img/post-bg-2015.jpg
catalog: true
auto_heading_numbering: true
tags:
    - Cuda, Flash Attention
---

## 简介

在 Chapter3（Kernel 1）中，我们已经拿到了一个可运行的 baseline kernel，但性能仅约为参考实现的一半。  
这一章的核心目标是定位瓶颈并解决它：**bank conflicts**。

通过 Nsight Compute 分析可以看到，Kernel 1 在 SMEM 上存在严重冲突，导致大量访存串行化。  
本章将引入并实现 **swizzling**，最终把性能从 `33.28` TFLOPS 提升到 `66.12` TFLOPS，接近参考实现。

## Kernel 2：Swizzling

Kernel 1 的关键问题：

- SMEM 带宽利用率异常高（约 `93.64%`，远高于无冲突情况下的期望）
- `short_scoreboard` 与 `mio_throttle` stall 占比明显偏高
- 根因是多处 `8-way` bank conflicts

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/kernel1_profile.jpeg" alt="kernel1_profile" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：Kernel 1 的 Nsight Compute 画像（冲突显著）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/kernel1_profile_raw_metrics.jpeg" alt="kernel1_profile_raw_metrics" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：Kernel 1 原始指标（可见共享内存冲突相关信号）
  </figcaption>
</figure>

## 16B 向量化访存下的 Bank 模型

对于 `16B` 的 LD/ST 指令，warp 的执行是分 phase 的：

| Phase | Threads |
| --- | --- |
| 0 | 0-7 |
| 1 | 8-15 |
| 2 | 16-23 |
| 3 | 24-31 |

可将其理解为：每个 phase 由 8 个线程组成，每线程访问 16B；因此有效地形成 **8 个 16B banks**（而非传统 32 个 4B banks）。

冲突判定原则：

- **同一 phase 内**：若多个线程访问同一 bank 的不同地址，则产生 bank conflict
- **跨 phase**：可访问同一 bank 而不互相冲突

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/banks_16B_0-7.svg" alt="banks_16B_0-7" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：16B 访存 Phase 0（线程 0-7）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/banks_16B_8-15.svg" alt="banks_16B_8-15" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 4：16B 访存 Phase 1（线程 8-15）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/banks_16B_8-15_conflict.svg" alt="banks_16B_8-15_conflict" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 5：同 phase 同 bank 不同地址导致冲突
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/banks_16B_2_phase_no_conflict.svg" alt="banks_16B_2_phase_no_conflict" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 6：跨 phase 访问同 bank 不冲突
  </figcaption>
</figure>

## Kernel 1 中的冲突位置

在本 kernel 中，冲突主要出现在两处：

1. **SMEM -> RF（`ldmatrix`）**：同 phase 线程访问同一 bank，出现 `8-way` 冲突
2. **RF -> SMEM（4B store）**：线程映射导致每 4 间隔线程落同 bank，同样 `8-way` 冲突

而 **GMEM <-> SMEM** 路径是无冲突的。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/ldmatrix_banks.svg" alt="ldmatrix_banks" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 7：SMEM -> RF（ldmatrix）中的 bank 冲突
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/R2SMEM_banks.svg" alt="R2SMEM_banks" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 8：RF -> SMEM 中的 bank 冲突
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/G2SMEM_banks.svg" alt="G2SMEM_banks" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 9：GMEM <-> SMEM 路径无冲突
  </figcaption>
</figure>

## 冲突带来的性能后果

bank conflict 会把本应并行的访存序列化，导致 wavefront 数显著增加。  
在 Kernel 1 中，SMEM 路径几乎被冲突拖满。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/1_wavefronts.svg" alt="1_wavefronts" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 10：冲突导致的共享内存 wavefront 放大
  </figcaption>
</figure>

典型 stall 指标（Kernel 1）：

| Stall | % of All Stalls |
| --- | --- |
| `short_scoreboard` | 56.37% |
| `math_pipe_throttle` | 11.88% |
| `mio_throttle` | 11.66% |
| `long_scoreboard` | 6.31% |

## Swizzling 思想

swizzling 的目标是：保持逻辑索引不变语义下，让访问分散到不同物理 bank，避免同 phase 冲突。

在 toy example（4x4）中，我们把原访问：

`arr[row][col]`

改为：

`arr[row][row XOR col]`

并且注意：**写入 SMEM 与读取 SMEM 都必须使用同一 swizzled 映射**，否则数据会错位。

```cpp
// write: GMEM -> SMEM
int swizzled_col = row ^ col;
smem[row][swizzled_col] = gmem_in[row][col];

// read: SMEM -> GMEM
int swizzled_col = row ^ col;
gmem_out[col][row] = smem[row][swizzled_col];
```

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_4x4_row_0.svg" alt="swizzling_4x4_row_0" style="width: 80%; max-width: 720px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 11：4x4 swizzling（row 0）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_4x4_row_1.svg" alt="swizzling_4x4_row_1" style="width: 80%; max-width: 720px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 12：4x4 swizzling（row 1）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_4x4_row_2.svg" alt="swizzling_4x4_row_2" style="width: 80%; max-width: 720px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 13：4x4 swizzling（row 2）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_4x4_row_3.svg" alt="swizzling_4x4_row_3" style="width: 80%; max-width: 720px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 14：4x4 swizzling（row 3）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_4x4_col_by_col.svg" alt="swizzling_4x4_col_by_col" style="width: 90%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 15：按列访问时的无冲突效果
  </figcaption>
</figure>

### Sudoku-like 映射视角

从本质上说，XOR 不是唯一选择。只要映射满足：

- 每行元素唯一
- 每列元素唯一

就可形成无冲突映射（类似 Sudoku 条件）。  
选择 XOR 的原因是：计算简单、零额外查表开销。

### Vectorized 与 Non-Vectorized 两类场景

- 对 `GMEM <-> SMEM`、`SMEM -> RF` 这类 `16B` 访存，按 16B bank 粒度做 swizzling
- 对 `RF -> SMEM` 的 `4B` store，需要先 swizzle 共享基址，再叠加线程内 offset

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_8x8_col.svg" alt="swizzling_8x8_col" style="width: 90%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 16：8x8 向量化 swizzling 访问模式
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/ldmatrix_single_fragment.svg" alt="ldmatrix_single_fragment" style="width: 90%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 17：单 fragment 在线程中的寄存器分布（用于理解 RF->SMEM）
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling-R2Smem.svg" alt="swizzling-R2Smem" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 18：RF -> SMEM 的 swizzled 写回映射
  </figcaption>
</figure>

## 代码改动

### Swizzling 函数

`swizzling.cuh`

```cpp
#define BANKS_PER_VEC4_ACCESS 8
#define ELEMS_PER_BANK 8

__forceinline__ __device__ constexpr int get_swizzled_col(const int &row, const int &col) {
    const int region_row = row % BANKS_PER_VEC4_ACCESS;
    const int bank_col = col / ELEMS_PER_BANK;
    const int bank_offset = col % ELEMS_PER_BANK;
    return ((region_row ^ bank_col) * ELEMS_PER_BANK) + bank_offset;
}
```

### 三条路径统一接入 swizzling

1. `GMEM <-> SMEM`：列坐标先过 `get_swizzled_col`
2. `SMEM -> RF`（含 transpose 版本）：`ldmatrix` 地址采用 swizzled 列
3. `RF -> SMEM`：按线程偏移写回时同样使用 swizzled 列

一句话原则：**凡是读写 SMEM 的路径，都必须在地址层保持同一套 swizzled 映射**。

## 性能结果

swizzling 后性能从 `33.28` TFLOPS 提升到 `66.12` TFLOPS，约 `2x` 提升。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/RTX3090_tflops_2_all.svg" alt="RTX3090_tflops_2_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 19：Kernel 2（swizzling）性能提升
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/kernel2_profile_raw_metrics.jpeg" alt="kernel2_profile_raw_metrics" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 20：Kernel 2 原始指标
  </figcaption>
</figure>

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/kernel2_profile_smem_table.png" alt="kernel2_profile_smem_table" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 21：Kernel 2 共享内存相关指标（冲突显著下降）
  </figcaption>
</figure>

### Stall 对比（Kernel 1 -> Kernel 2）

| Stall | Kernel 1 | Kernel 2 | Delta (1->2) | Reference |
| --- | --- | --- | --- | --- |
| `short_scoreboard` | 56.37% | 1.49% | -54.88% | 0.52% |
| `mio_throttle` | 11.66% | 0.74% | -10.92% | 1.37% |
| `long_scoreboard` | 6.31% | 15.15% | +8.84% | 0.43% |

结论：

- SMEM 冲突基本被消除，短等待显著下降
- 但 `long_scoreboard`（等待 GMEM）占比上升，说明下一阶段优化重点将转向访存重叠与流水化

## 小结

Chapter4 的关键价值是：通过 swizzling 消除了 Kernel 1 的主要结构性瓶颈（SMEM bank conflicts），把性能直接拉升到接近参考实现。  
下一章将继续引入 CUTLASS 风格的调优策略，重点提升计算与搬运的 overlap，进一步压缩与参考实现的差距。
