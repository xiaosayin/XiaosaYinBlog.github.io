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

和参考核相比：

| Stall | Kernel 1 | Reference | Absolute Delta |
| :--- | :--- | :--- | :--- |
| `short_scoreboard` | 56.37% | 0.52% | -55.85% |
| `mio_throttle` | 11.66% | 1.37% | -10.29% |
| `long_scoreboard` | 6.31% | 0.43% | -5.87% |

与参考实现相比，我们的核函数在 `short_scoreboard` 和 `mio_throttle` 上表现出明显更高的停滞率（Stall Rates）。我将在本系列博客的最后一部分对线程束停滞（Warp Stalls）进行更深入的探讨，在此先做一个简要的概述。

当指令等待共享内存 (SMEM) 的加载/存储操作结果，或者其他相对高延迟的指令（如指数运算）的结果时，就会触发 `short_scoreboard` 停滞。该指标不包含全局内存 (GMEM) 和局部内存 (LMEM) 操作，这两种操作由 `long_scoreboard` 进行单独追踪。同时，当 Ampere 架构中用于处理耗时较长操作的指令队列被打满，迫使线程束 (Warps) 必须等待直至队列释放出可用空间时，便会发生 `mio_throttle` 停滞。

这些指标充分表明，与经过优化的参考实现相比，我们在等待内存传输完成上花费了极其不成比例的时间。

既然我们已经明确存储体冲突 (Bank Conflicts) 是当前首要的性能瓶颈，并理解了其对整体性能的负面影响，接下来让我们探讨一种经典的解决方案：数据重排 (Swizzling)。


## Swizzling 思想

数据重排（Swizzling）是优化 CUDA 核函数时消除存储体冲突（Bank Conflicts）的标准技术。其核心思想是重新分配内存访问模式，使得访问同一行或同一列的线程能够命中不同的物理存储体，从而消除串行内存访问带来的延迟。

为了更直观地理解数据重排，我们来看一个贯穿始终的简单示例：假设我们在全局内存（GMEM）中有一个按行主序（Row-major Format）存储的 $4 \times 4$ 矩阵，我们的目标是对其进行转置。

* 我们希望按行主序将其加载到共享内存（SMEM）中，因此我们将逐行（Row by row）进行拷贝。
* 我们希望按列主序（Column-major Format）将其存回全局内存中，因此我们将逐列（Column by column）进行写回。

为了便于说明，我们假定当前的共享内存总共被划分为 4 个存储体（Banks）。

数据重排通过将行索引与列索引进行按位异或（XOR）运算，将同一逻辑列中的元素映射到唯一的物理存储体中。我们在访问每个元素时将使用：

> **📝 Note**
> 
> 当我们在*写入和读取*共享内存（SMEM）时，必须对列索引进行数据重排（Swizzle），即应用该公式。

如果不采用数据重排，我们的代码大致如下所示：

```cpp
__shared__ T smem[4][4];
 
// Copying row by row from GMEM to SMEM.
int col = threadIdx.x % 4;
for (int row = 0; row < 4; ++row) {
	smem[row][col] = gmem_in[row][col];
}
 
// Copying col by col from SMEM to GMEM.
int row = threadIdx.x % 4;
for (int col = 0; col < 4; ++col) {
	gmem_out[col][row] = smem[row][col];
}	
 
```

应用了 swizzling 之后，代码应该像：
```cpp
// Copying row by row from GMEM to SMEM.
int col = threadIdx.x % 4;
for (int row = 0; row < 4; ++row) {
	int swizzled_col = row ^ col;
	smem[row][swizzled_col] = gmem_in[row][col];
}
 
// Copying col by col from SMEM to GMEM.
int row = threadIdx.x % 4;
for (int col = 0; col < 4; ++col) {
	int swizzled_col = row ^ col;
	gmem_out[col][row] = smem[row][swizzled_col];
}	
 
```


### 进一步探索 Swizzling
为了深入理解数据重排（Swizzling），我们可以用递归的思维来考虑它。这将有助于我们洞察这些模式之所以生效的内在原理。让我们从最简单的情况开始，逐步深入：

**基本情况 ($1 \times 1$)：** 当只有一个元素时，数据重排不产生任何作用。此时只有一种排列方式。

**情况 ($2 \times 2$)：** 我们可以将其按行进行分解：

* **第一行：** 直接对每个元素应用基本情况。
* **第二行：** 在行内交换位置，然后再对每个元素应用基本情况。

**情况 ($4 \times 4$)：** 可以将其视为四个按正方形排列的 ($2 \times 2$) 子网格：

* **第一行子网格：** 独立地对每个子网格应用 ($2 \times 2$) 模式。
* **第二行子网格：** 交换子网格的位置，然后对每个子网格应用 ($2 \times 2$) 模式。

**直观视觉效果**

异或（XOR）模式在每一行中产生了不同的交换行为：

**第 0 行 (Row 0)：** 列索引保持不变。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_4x4_row_0.svg" alt="swizzling_4x4_row_0" style="width: 80%; max-width: 720px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 11：4x4 swizzling（row 0）
  </figcaption>
</figure>

**第 1 行 (Row 1)：** 我们交换相邻的元素对 ($0 \leftrightarrow 1, 2 \leftrightarrow 3$)。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_4x4_row_1.svg" alt="swizzling_4x4_row_1" style="width: 80%; max-width: 720px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 12：4x4 swizzling（row 1）
  </figcaption>
</figure>

**第 2-3 行 (Rows 2-3)：** 交换跨度（Swap Distance）跃升为 2，但在每个完成交换的元素对内部，我们依旧重复着第 0-1 行的排列模式。

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

**列访问模式**

当我们逐列访问数据时，请观察会发生什么：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_4x4_col_by_col.svg" alt="swizzling_4x4_col_by_col" style="width: 40%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 15：按列访问时的无冲突效果
  </figcaption>
</figure>

### Sudoku-like 映射视角

请注意，在访问任意一列时，4 个线程中没有任何两个线程访问相同的存储体 (bank).这正是我们的数据重排 (swizzling) 函数想要实现的目标。

### 类数独映射 (Sudoku-Like Mapping)

实际上，我们并不一定非要使用异或 (XOR) 操作。我们需要的是一种类似数独的映射关系，需满足以下条件：

* 每一行都包含唯一的元素
* 每一列都包含唯一的元素

在这个“数独网格”中，每个单元格内的数字代表逻辑列 (logical column)，而该列所处的位置代表物理列 (physical column)（即其实际存储的位置）。

**如何解读该映射：**

* **行访问 (Row access)：** 找到目标行，并按顺序读取列索引。
* **列访问 (Column access)：** 对于列 $c$，找出 $c$ 在每一行中出现的位置 —— 这些位置指示了其实际的物理存储方位。

| r\c | 0 | 1 | 2 | 3 |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 2 | 3 | 1 | **0** |
| **1** | **0** | 1 | 2 | 3 |
| **2** | 1 | **0** | 3 | 2 |
| **3** | 3 | 2 | **0** | 1 |

*$4 \times 4$ 非异或的“类数独”映射 (non-XOR "sudoku-like" mapping)*

这种映射机制之所以能够生效，是因为假使我们：

* **任选一列** —— 每个元素（代表物理存储体 Bank）均是唯一的，从而避免了存储体冲突 (Bank Conflicts)。
* **任选一行** —— 所有的元素互不相同，确保了没有任何数据会被错误覆盖。

理论上，任何“类数独”的映射方式都能满足需求，但我们为什么偏偏青睐于异或 (XOR) 运算呢？原因在于，XOR 的计算速度极快，并且完全不需要占用任何额外的内存开销。

| r\c | 0 | 1 | 2 | 3 |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **0** | 1 | 2 | 3 |
| **1** | 1 | **0** | 3 | 2 |
| **2** | 2 | 3 | **0** | 1 |
| **3** | 3 | 2 | 1 | **0** |

*$4 \times 4$ 异或 (XOR) 映射*

###  Vectorized Swizzling

与其单独考虑每一个 bf16/fp16 元素，不如将我们的“元素”视为与共享内存 (SMEM) 中的存储体 (Banks) 具有相同的大小，这样理解起来可能会更加直观。因此，我们不再加载 8 个 16 位 (16-bit) 的元素，而是直接加载单个 16 字节 (16-byte) 的元素。为此，Cutlass 专门定义了 `uint128_t` 数据类型以满足这一需求。

为了加载由 16 字节元素组成的 $8 \times 8$ 网格，我们需要在 8 个存储体和 8 行之间递归地应用相同的 $\text{row} \oplus \text{col}$（行索引与列索引按位异或）映射机制。

以下是第 1 列和第 5 列的访问模式：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling_8x8_col.svg" alt="swizzling_8x8_col" style="width: 90%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 16：8x8 向量化 swizzling 访问模式
  </figcaption>
</figure>

### Non Vectorized Swizzling
寄存器堆到共享内存（$\text{RF} \rightarrow \text{SMEM}$）的操作带来了一项独特的挑战，因为它们并未像其他内存传输那样进行向量化（Vectorized）。与每个线程移动 16 字节的向量化 $\text{GMEM} \leftrightarrow \text{SMEM}$ 以及 $\text{SMEM} \rightarrow \text{RF}$ 操作不同，$\text{RF} \rightarrow \text{SMEM}$ 的传输在每条指令中每个线程仅拷贝 4 字节。然而，为了后续的 $\text{SMEM} \rightarrow \text{GMEM}$ 操作，我们依然需要维持数据重排（Swizzled）后的内存布局。我们该如何应对这一问题呢？

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/ldmatrix_single_fragment.svg" alt="ldmatrix_single_fragment" style="width: 90%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 17：单 fragment 在线程中的寄存器分布（用于理解 RF->SMEM）
  </figcaption>
</figure>

回顾一下 chapter2 中的片段布局：

* 每组 4 个线程负责存储一个片段内的一行（总共 16 字节）。
* 在每一组内部，各个线程拥有唯一的列偏移量（0、2、4 或 6 字节），但它们*共享同一个 16 字节对齐的基地址 (16B-aligned base address)*。

我们需要做的是，对一行中所有线程的*共享基地址*应用数据重排 (Swizzling)，然后再对应加上每个线程自身的偏移量。这种处理方式能够确保 SMEM 中的数据维持后续操作所需的重排布局。

下图展示了第二个片段的这种映射关系：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/swizzling-R2Smem.svg" alt="swizzling-R2Smem" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 18：RF -> SMEM 的 swizzled 写回映射
  </figcaption>
</figure>

### 额外注释：Hopper 架构

Hopper 及后续架构在以下层级之间进行数据拷贝时，简化了数据重排（Swizzling）机制：

* **$\text{GMEM} \leftrightarrow \text{SMEM}$：** Hopper 在 `cp.async.bulk*` PTX 指令中加入了内置的数据重排功能。该功能在 Hopper 架构、Blackwell 高性能计算（HPC）以及消费级 GPU 上均受支持。
* **$\text{SMEM} \rightarrow \text{RF}$：** Hopper 和 Blackwell HPC 拥有专用的矩阵乘法指令——这些指令同样内置了数据重排功能——能够直接对 SMEM（Blackwell 架构中为 TMEM；Blackwell 里的 TMEM = Tensor Memory。
它不是普通的 shared memory，也不是 register file，而是 Blackwell 第五代 Tensor Core 专用的一块片上内存，主要给 tcgen05.* 这类新 Tensor Core 指令使用。）进行操作。这彻底免去了任何手动拷贝的步骤。

然而，目前并没有针对 $\text{RF} \rightarrow \text{SMEM}$ 传输的硬件级加速支持，因此在这种特定场景下，依然需要手动执行数据重排。

尽管这些硬件层面的演进极大地简化了开发流程，但为了正确调用并充分发挥这些功能的潜力，深入理解数据重排的工作原理仍然是不可或缺的。

## 代码改动

### Swizzling 函数

既然我们已经理解了数据重排（Swizzling）背后的理论基础，并通过简单的示例观察了它的运作机制，接下来让我们深入了解其实际的代码实现。我们的数据重排函数正是建立在早前探讨过的 `row XOR col` 模式之上，但针对我们特定的内存访问模式进行了一些必要的调整。

`swizzling.cuh`

```cpp
#define BANKS_PER_VEC4_ACCESS 8
#define ELEMS_PER_BANK 8
 
__forceinline__ __device__ constexpr int get_swizzled_col(const int &row, const int &col) {
    // Restrict the swizzled column to the
    // (8, 128) byte region it's in.
    // Not strictly necessary, but we'll need it in later kernels.
    // 行的逻辑不变，仍然是对应的行
    const int region_row = row % BANKS_PER_VEC4_ACCESS;
 
    // Convert column byte offset to 16B bank index since we have 8 banks of 16B each.
    // This transforms the column coordinate from element space to bank space
    // 32x4B 的物理 bank，变成 8x16B 的逻辑 bank; ELEMS_PER_BANK = 8;
    const int bank_col = col / ELEMS_PER_BANK;
    
    // Preserve the byte offset within each 16B bank for non-vectorized RF→SMEM stores
    // This ensures threads in the same 4-thread group maintain their relative positions
    // 在 16B 的物理 bank 中，找到真正的元素所在列
    const int bank_offset = col % ELEMS_PER_BANK;
 
    // Apply XOR swizzling to distribute consecutive row accesses across different banks
    // Then reconstruct the final column address by scaling back to element space
    return ((region_row ^ bank_col) * ELEMS_PER_BANK) + bank_offset;
}
```

一个至关重要的细节是：我们必须对**每一次**共享内存 (SMEM) 访问——无论是读取还是写入——都应用数据重排 (Swizzling)。混合使用重排和非重排的访问方式将会导致数据损坏。

既然数据重排函数已经就绪，接下来让我们将其应用到所有的内存操作中。高亮的代码行展示了具体的修改之处。

### GMEM 和 SMEM 之间传数据
`load_store.cuh`
<div class="code-highlight-line-17">
{% highlight cpp %}
template <typename op, /* either GM2SM_async or SM2GM */
          TensorLDSTConfig CFG, typename value_t, typename index_t = int64_t>
__forceinline__ __device__ constexpr void copy_block_GSM(
	value_t *gmem, value_t *smem,
    index_t gmem_seq_stride,
    const int lane_id
) {
		// ...
 
        #pragma unroll

        for (int c = 0; c < col_fragments_per_row;
             c += col_fragments_per_iter) {
            const int col_fragment = c + thread_col_fragment;
            // Apply swizzling to prevent bank conflicts during later column-wise access in `ldmatrix`.
            
            const int smem_col = get_swizzled_col(cur_row, col_fragment * COLS_PER_FRAGMENT);
 
            op()(&gmem[cur_row * gmem_seq_stride +
                       col_fragment * COLS_PER_FRAGMENT],
                 &smem[cur_row * CFG.smem_cols +
                       smem_col]);
        }
    }
}

{% endhighlight %}
</div>
<script>
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".code-highlight-line-17 code").forEach(function (code) {
    var lines = code.innerHTML.split("\n");
    code.innerHTML = lines.map(function (line) {
      if (line.indexOf("get_swizzled_col") !== -1 && line.indexOf("smem_col") !== -1) {
        return '<span class="code-line-highlight">' + line + "</span>";
      }
      return line;
    }).join("\n");
  });
});
</script>


### SMEM->RF
#### $Q , K^{(j)}$
`load_store.cuh`
```cpp
#define ROWS_PER_FRAGMENT 8
#define COLS_PER_FRAGMENT 8
#define ELEMS_PER_VEC4_ACCESS 8
 
template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int col_fragment_offset = 0
) {
	// ...
        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
			const int smem_col_fragment = thread_col_fragment + c + col_fragment_offset;
            // Use swizzled addresses to match the layout from GMEM→SMEM transfers
            const int smem_col = get_swizzled_col(cur_row, smem_col_fragment * ELEMS_PER_VEC4_ACCESS);
 
            ldmatrix_x4(&smem[cur_row * CFG.smem_cols +
                        smem_col],
                        regs[r][c], regs[r + 1][c], regs[r][c + 1],
                        regs[r + 1][c + 1]);
        }
    }
}
 
```

#### $V^{(j)}$
```cpp
template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id,
    const int row_fragment_offset = 0
) {
	// ...
        #pragma unroll
        for (int c = 0; c < CFG.RF.row_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = thread_col_fragment + c;
            const int smem_col = get_swizzled_col(cur_row, smem_col_fragment * ELEMS_PER_VEC4_ACCESS);
 
            ldmatrix_x4_transpose(
                &smem[cur_row * CFG.smem_cols +
                      smem_col],
                regs[c][r], regs[c][r + 1], regs[c + 1][r], regs[c + 1][r + 1]);
        }
    }
}
 
```

### RF->SMEM
```cpp
template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_RF2SM(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t *smem,
    const int lane_id
) {
    // ...
    #pragma unroll
    for (int r = 0; r < CFG.RF.row_fragments; ++r) {
        const int cur_row = thread_row + r * rows_per_iter;
        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = c;
            // Apply swizzling to maintain consistent layout for later SMEM→GMEM transfers
            const int smem_col = get_swizzled_col(cur_row, smem_col_fragment * ELEMS_PER_VEC4_ACCESS + thread_inner_col);
 
            reinterpret_cast<uint32_t *>(
                &smem[cur_row * CFG.smem_cols +
                      smem_col])[0] = regs[r][c];
        }
    }
}
```

把（Swizzling）应用在所有的内存操作中，让我们来看看它所带来的性能影响：

## 性能结果

数据重排（Swizzling）优化带来了很好的效果：**我们的性能实现了翻倍，从 33.28 提升至 66.12 TFLOPs**，这让我们极其逼近参考实现的性能水平！

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/RTX3090_tflops_2_all.svg" alt="RTX3090_tflops_2_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 19：Kernel 2（swizzling）性能提升
  </figcaption>
</figure>

### Profile 一下
已经消除了所有的 bank conflict;

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/kernel2_profile_raw_metrics.jpeg" alt="kernel2_profile_raw_metrics" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 20：Kernel 2 原始指标
  </figcaption>
</figure>

这一显著的性能飞跃，将访问同等容量共享内存 (SMEM) 所需的波前 (Wavefronts) 数量大幅削减了 8 倍。此前存在的 8 路存储体冲突 (8-way Bank Conflicts) 迫使内存操作被强制串行化，实质上等同于将有效内存带宽压缩至原来的 1/8。通过彻底消除这些冲突，我们成功恢复了内存访问的并行特性，这不仅使硬件得以充分释放其预期的吞吐能力，更为实际的计算任务腾出了大量宝贵的时钟周期 (Cycles)。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter4/kernel2_profile_smem_table.png" alt="kernel2_profile_smem_table" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 21：Kernel 2 共享内存相关指标（冲突显著下降）
  </figcaption>
</figure>

### 核函数 2 线程束停滞分析 (Kernel 2 Warp Stalls)

如果我们观察执行核函数的线程束（Warps）在停滞分布上的变化，可以发现情况得到了极大的改善！

| 停滞类型 (Stall) | 核函数 1 (Kernel 1) | 核函数 2 (Kernel 2) | 差值 (Delta 1->2) | 参考实现 (Reference) |
| :--- | :--- | :--- | :--- | :--- |
| `short_scoreboard` | 56.37% | 1.49% | -54.88% | 0.52% |
| `mio_throttle` | 11.66% | 0.74% | -10.92% | 1.37% |
| `long_scoreboard` | 6.31% | 15.15% | +8.84% | 0.43% |

然而，从相对比例来看，线程束在等待全局内存（GMEM）请求时发生的停滞仍然超出了我们的预期。高达 15.15% 的线程束停滞归因于 `long_scoreboard`，而参考实现中的这一比例仅为 0.43%。数据重排（Swizzling）极大概率并未直接导致这种停滞的增加，而是由于其他瓶颈被消除，让它变得更加凸显。进一步减少此类停滞，将有助于我们抹平最后残余的性能差距。

## 小结

Chapter4 的关键价值是：通过 swizzling 消除了 Kernel 1 的主要结构性瓶颈（SMEM bank conflicts），把性能直接拉升到接近参考实现。  
下一章将继续引入 CUTLASS 风格的调优策略，重点提升计算与搬运的 overlap，进一步压缩与参考实现的差距。
