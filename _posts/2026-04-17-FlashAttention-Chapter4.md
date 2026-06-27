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

在 chapter3 中，我们使用第 2 部分介绍的指令构建了第一个 kernel，并在 RTX 3090 上达到了接近官方实现一半的性能。

在本篇中，我们将对该 kernel 进行性能分析，以定位性能瓶颈。我们会发现，bank conflict 是主要原因；随后将通过实现 swizzling 来解决这一问题，并获得约 $2\times$ 的性能提升。

## Kernel 2：Swizzling

我们的 Kernel 1 已经达到了接近参考实现一半的性能，但剩下的另一半性能究竟损失在哪里？Nsight Compute 给出了一个线索：bank conflict。

```shell
ncu --kernel-name your_kernel_name --set full -o report ./your_cuda_program

scp user@ecs_ip:/path/to/report.ncu-rep .

# 本地 Nsight compute 打开从 ecs 上拿到的文件
File -> Open -> report.ncu-rep
```

> **Nsight Compute Rules**
>
> 这些信息弹窗由 *Nsight Compute rules* 生成。Rules 是一组 Python 脚本，用于分析 kernel profile，并基于一组 metrics 指出潜在的性能问题。
>
> Nsight Compute 自带一组默认 rules，但你也可以编写自己的 rules。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/kernel1_profile.jpeg" alt="kernel1_profile" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：Kernel 1 的 Nsight Compute 画像（冲突显著）
  </figcaption>
</figure>

Bank conflict 会将本应并行的内存访问串行化。因此，warp 会把大部分时间花在等待 SMEM 请求上，并使用约 $4\times$ 的内存带宽。对于相同的工作量，我们占用了 **$93.64\%$ 的 SMEM 带宽**，而理论上只应需要约 **$23.48\%$**。


## 可选：验证 Bank Conflict

> **Nsight Compute Bank Conflicts**
>
> 即使 Nsight Compute 报告了 bank conflict，这也并不总是准确的。由于没有专门用于 bank conflict 的硬件计数器，profiler rules 有时可能会产生误导。要了解真实情况，我们需要查看 `derived__memory_l1_wavefronts_shared_excessive` 这个 metric，它等于：
>
> $$
> \texttt{memory_l1_wavefronts_shared}
> -
> \texttt{memory_l1_wavefronts_shared_ideal}
> $$
>
> 这个 metric 表示由于 conflict 额外需要的 wavefront 数量。如果该值为 $0$，则说明没有 bank conflict。每一行 SASS 汇编都有自己对应的这个 metric 值。
>
> 查看我们 kernel 的 profile 可以发现，这次 bank conflict 检测并没有误导我们：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/kernel1_profile_raw_metrics.jpeg" alt="kernel1_profile_raw_metrics" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：Kernel 1 原始指标（可见共享内存冲突相关信号）
  </figcaption>
</figure>

> 高亮行是 Kernel 1，而上面一行是用于对比的参考实现。花括号中的值表示该 metric 捕获到的 SASS 汇编行数。

上面展示的 SMEM conflict rule 所引用的 metrics 分别是 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 和 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`。我认为这两个 metrics 聚合了同时访问 SMEM 和 L1 cache 时产生的 bank conflict；这种情况可能发生，是因为 SMEM 和 L1 cache 是统一的。虽然我们无法控制这些 L1/SMEM arbitration conflict，但可以通过精心设计的内存访问模式，例如 swizzling，来消除仅由 SMEM 访问产生的 conflict。

## 16B 向量化访存下的 Bank 模型
在 GPU 底层的硬件上, shared memory 实际上就是 32 个 4B bank, 只不过因为我们在做 GMEM->SMEM 搬运时, 一个线程搬运 8 个 BF16, 也就是一个 16B, 所以逻辑上可以视为 8 个 16B bank;

必须理解的是, SMEM 一次对齐的合并内存事务(Phase)就是按 128B 来处理的;

原因如下：当一个 warp 执行一条 16B 的 LD/ST 指令时，硬件会将其拆分为 4 个独立的 phase，每个 phase 由 8 个线程处理。由于每个线程访问 16 bytes，也就是跨越 4 个连续的 4-byte bank，这 4 个 bank 实际上会组成一个 16-byte 的访问单元。

| Phase | Threads |
|---:|---|
| 0 | 0-7 |
| 1 | 8-15 |
| 2 | 16-23 |
| 3 | 24-31 |

由于一个 phase 中的每个线程都会访问 16 bytes，因此每个 phase 最多可以访问 $128$ bytes。从整个 warp 的角度看，每条指令总共会访问 $512$ bytes。

在 16B 访问下，banking 结构会发生显著变化。每个线程不再只访问一个 4-byte bank，而是一个组中的 8 个线程分别同时访问 4 个 bank。由于地址必须按 16-byte 对齐，这实际上会把 4 个连续的 4-byte bank 组合成一个 16-byte 访问单元。也就是说，我们不再处理 32 个独立的 bank，而是等效地拥有 8 个 bank，每个 bank 宽度为 16 bytes。为了保持表述一致，后文我会将这些 16B 宽的单元也称为 “bank”。

第一个 phase 大致如下：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/banks_16B_0-7.svg" alt="banks_16B_0-7" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：16B 访存 Phase 0（线程 0-7）
  </figcaption>
</figure>

第 2 个 phase 也类似:
<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/banks_16B_8-15.svg" alt="banks_16B_8-15" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 4：16B 访存 Phase 1（线程 8-15）
  </figcaption>
</figure>

其余 phase 也会以同样的方式继续执行。现在，如果同一组中的两个线程访问了同一个 bank 中的不同地址，那么这会在全部 4 个组成 bank 上产生 bank conflict：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/banks_16B_8-15_conflict.svg" alt="banks_16B_8-15_conflict" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 5：同 phase 同 bank 不同地址导致冲突
  </figcaption>
</figure>

不过，由于每个 phase 都会作为一次独立的内存访问来执行，来自不同组的线程即使访问同一个 bank 中的不同地址，也不会产生 conflict：因为每个 phase 一次都只处理 128B 的数据;

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/banks_16B_2_phase_no_conflict.svg" alt="banks_16B_2_phase_no_conflict" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 6：跨 phase 访问同 bank 不冲突
  </figcaption>
</figure>

不同阶段可能会经历不同程度的 bank conflict。例如，某个阶段可能出现 2-way conflict，而另一个阶段可能出现 8-way conflict。有些阶段可能完全无冲突，而有些阶段则冲突严重。关键洞察在于：要在整个 warp 范围内实现无冲突的内存访问，我们只需要确保在处理同一阶段的每组 8 个线程内部，不发生 bank conflict。

虽然前文主要讨论的是 $16\text{B}$ 访问，但还是要牢记, shared memory banking 物理上还是 32 个 4B bank;真正的 bank conflict, 就是同一阶段, 不同线程访问了同一个 4B bank 中的不同地址;

所以才有 
1. 单线程 16B 指令, 32 x 16B / 128B = 4 阶段; 每阶段 128B / 16B = 8 个线程;
2. 单线程 8B 指令, 32 x 8B / 128B = 2 阶段; 每阶段 128B / 8B = 16 个线程;

> **16B Vectorized Banks**
>
> - **基于阶段的执行（Phase-based execution）**：$16\text{B}$ 访问被拆分为 4 个阶段，每个阶段包含 8 个线程
> - **有效 bank 结构（Effective bank structure）**：8 个大小为 $16\text{B}$ 的 bank（而不是 32 个大小为 $4\text{B}$ 的 bank）
> - **冲突条件（Conflict conditions）**：当同一阶段中的线程访问同一个 bank 内的不同地址时，就会发生 bank conflict
> - **跨阶段无冲突（Cross-phase freedom）**：不同阶段可以访问同一个 bank，而不会产生冲突

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

## bank conflict 带来的影响

Bank conflict 会显著吞噬内存带宽。每一次 8-way conflict 都会将原本应当并行完成的访问串行化，因此需要 $8\times$ 数量的 wavefront；可以将 wavefront 理解为完成这些访问所需的访问次数。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/1_wavefronts.svg" alt="1_wavefronts" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 10：冲突导致的共享内存 wavefront 放大
  </figcaption>
</figure>

因此，与等价的无 bank conflict kernel 相比，我们的 kernel 的 SMEM 带宽利用率达到了 $93.64\%$，而后者仅为 $23.48\%$。SMEM bandwidth utilization 高：表示共享内存单元很忙, 并不是有效的带宽利用; 前者带宽被浪费在额外 wavefront 上, 后者访问更高效，完成同样工作只需要更少 SMEM 事务，因此 SMEM 利用率反而低。

### warp 停顿
在 Ampere 架构上，warp 理论上可以在每个时钟周期被发射一条指令。然而，多种条件都可能阻止某个 warp 执行指令，从而产生一个 stall cycle。Profiler 会以每 $n$ 个时钟周期为间隔，周期性地采样这些 stall cause（其中 $n$ 是可配置的），并将数据记录在 `smsp__pcsamp_warps_issue_stalled_*` 指标中。

这些指标可以揭示不同的 stall 场景：

- 如果一个 warp 成功获得指令发射机会，则其 stall cause 会被记录为  
  `smsp__pcsamp_warps_issue_stalled_selected`
- 当一个 warp 本可以接收一条指令，但调度器选择了另一个 warp 时，它会被标记为 `..._not_selected`
- 在 CTA 中等待其他 warp 到达 `__syncthreads()` checkpoint 的 warp，会产生 `..._barrier` stalls

可以在 Nsight Compute 文档中找到更多 stall cause。分析这些 stall pattern 可以为定位 kernel 的性能瓶颈提供有价值的观察。

对于我们的 kernel，

| Stall | % of All Stalls |
| --- | --- |
| `short_scoreboard` | 56.37% |
| `math_pipe_throttle` | 11.88% |
| `mio_throttle` | 11.66% |
| `long_scoreboard` | 6.31% |

shuqian

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
