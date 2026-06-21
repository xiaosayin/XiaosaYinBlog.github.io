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

在上一部分中，我们实现了 swizzling，并通过消除 bank conflict（存储体冲突）取得了显著的 $2\times$ 性能提升。我们的 kernel 现在在 RTX 3090 上已达到参考实现性能的 $98\%$。

在本部分中，我们将实现 CUTLASS 所采用的标准 GEMM 优化技术。CUTLASS 是 NVIDIA 面向高性能线性代数 kernel 的 C++ 模板库。由于我们已经获得了大部分可挖掘的性能收益，接下来的改进幅度可能看起来更为有限。不过，这些技术对于以下目标至关重要：

- **隐藏内存延迟（Memory latency hiding）**：将数据传输与计算重叠执行，以减少空闲时间。
- **降低寄存器压力（Register pressure reduction）**：支持更大的 block size，以及更多 kernel 配置。

在当前配置下，某些优化甚至可能带来轻微的性能回退；但它们能够使其他 block size 获得更好的性能，并在后续部分中提供可供自动调优的变量。

我们将逐步实现三项优化：

1. Kernel 3：双缓冲加载 K/V（Eager K/V Loading）

我们将在 GMEM $\rightarrow$ SMEM 层级实现双缓冲：在计算当前 $K^{(i)}$ 和 $V^{(i)}$ block 的同时，加载下一组 $K^{(j)}$ 和 $V^{(j)}$ block。

这项经典优化技术将 GMEM 数据传输与计算重叠执行，使 GMEM 停顿减少 $93\%$。

2. Kernel 4：Fragment 交错加载（Fragment Interleaving）

我们不会在计算前将完整的 tile 全部加载到 RF，而是将它们划分为 sub-tile，并将加载过程与矩阵运算交错执行。

这会缩短第一次 SMEM $\rightarrow$ RF 传输与第一次 GEMM 运算之间的时间间隔，同时显著降低所有 block 配置下的寄存器压力。

3. Kernel 5：SMEM $\rightarrow$ RF 双缓冲

我们将把双缓冲扩展到 fragment 加载，为隐藏 `ldmatrix` 延迟分配额外的寄存器空间。

尽管该优化在当前配置下会带来轻微的性能回退，但它能为其他对自动调优至关重要的 block size 带来显著的性能收益。

## Kernel 3：双缓冲加载 K/V（GMEM->SMEM 双缓冲）

在 Kernel 2 里，`K/V` 往往“即用即取”，导致 warp 在等待 GMEM 传输时空转。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/Flash-Double-Buffering-Before-Shorter.svg" alt="Flash-Double-Buffering-Before-Shorter" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：优化前（传输与计算重叠不足）
  </figcaption>
</figure>


这一低效问题在我们的性能分析中表现得非常明显：**$15.15\%$ 的 warp 停顿**来自等待 GMEM 传输（`long_scoreboard`），而参考 kernel 中该比例约为 $0.43\%$。这表明，通过更好的内存调度，我们仍有相当大的优化空间。

**解决方案：**在计算前一批数据时，更早地开始加载 block，使数据传输在真正需要这些数据之前完成。  

### 安全加载点

为安全地实现这项优化，我们必须遵守同步约束：

**内存依赖：**

- $K^{(j)}$ 和 $V^{(j)}$：在 GMEM $\rightarrow$ SMEM 与 SMEM $\rightarrow$ RF 之间需要执行 `__syncthreads()`（所有 warp 协作参与）。
- 迭代之间：需要执行 `__syncthreads()`，以确保覆盖 SMEM 中的数据之前，所有 warp 都已完成读取。

**问题：**在不引发竞态条件（race condition）的前提下，我们何时可以开始将 $K^{(j)}$ 和 $V^{(j)}$ 从 GMEM $\rightarrow$ SMEM 加载到共享内存中？

下面给出 kernel 的一个简化版 Python 伪代码，其中移除了 barrier 和 `cp.async.wait()`。

```python
1  # Prologue
2  cp.async_and_commit(Q_SM, Q_GM[offset])
3  
4  for blk in range(T_c):
5    cp.async_and_commit(K_SM, K_GM[blk])
6    load_SMEM2RF(K_RF, K_SM)
7    gemm(S_RF, Q_RF, K_RF^T)
8    
9    online_softmax(S_RF, m_RF, l_RF, O_RF)
10   convert_to_b16(P_RF, S_RF)  
11   cp.async_and_commit(V_SM, V_GM[blk])
12   load_SMEM2RF(V_RF, V_SM)
13   gemm(O_RF, P_RF, V_RF)
14 
15 # Epilogue  
16 #...  
```

加载任意数据块的最早时机，是在当前计算不再需要访问上一个数据块的共享内存 (SMEM) 之后，即 `load_SMEM2RF` 操作刚结束之时。然而，将其放置在 `gemm` 操作之后，能够为编译器提供更大的优化空间。

* 对于 $\mathbf{K}^{(j)}$，是在执行完 $\mathbf{Q}\mathbf{K}^\top$（第 7 行）之后。
* 对于 $\mathbf{V}^{(j)}$，是在执行完 $\tilde{\mathbf{P}}\mathbf{V}$（第 13 行）之后。

让我们将每次加载操作“上提”至对应的上述位置。针对 $\mathbf{K}^{(j)}$，我们需要为首次加载 (initial load) 增加一个特殊的处理逻辑。

```python
1  # Prologue       

2  cp.async_and_commit(Q_SM, Q_GM[offset])
3  cp.async_and_commit(K_SM, K_GM[0]) # 改动 0, 首个 K 块处理  
4  
5  for blk in range(T_c):
6      cp.async_and_commit(V_SM, V_GM[blk]) # 改动 1，提前加载 V,   GMEM->SMEM  

7      load_SMEM2RF(K_RF, K_SM)
8      gemm(S_RF, Q_RF, K_RF^T)
9      
10     online_softmax(S_RF, m_RF, l_RF, O_RF)
11 
12     if blk < T_c - 1:
13         cp.async_and_commit(K_SM, K_GM[blk+1]) # 改动2，提前加载 K, GMEM->SMEM

14     convert_to_b16(P_RF, S_RF)
15     load_SMEM2RF(V_RF, V_SM)
16     gemm(O_RF, P_RF, V_RF)
17  
18 # Epilogue    

19 # ...
```

现在，我们需要加入同步屏障。关键在于：为了防止竞态条件，我们需要设置两个屏障点：

| 屏障位置 | 目的 |
|---|---|
| 循环开始处 | 确保 $K^{(j)}$ 的传输在使用前完成。<br><br>防止仍在被读取的 $V^{(j-1)}$ 被覆盖。 |
| softmax 之后 | 确保 $V^{(j)}$ 的传输在使用前完成。<br><br>确保覆盖数据前，所有 warp 都已完成对 $K^{(j)}$ 的读取。 |

<pre><code class="language-python"># Prologue
cp.async_and_commit(Q_SM, Q_GM[offset])
cp.async_and_commit(K_SM, K_GM[0])
 
for blk in range(T_c):
<span style="display:block; background:rgba(255, 213, 79, 0.25);">    cp.wait()
    __syncthreads()</span>
    cp.async_and_commit(V_SM, V_GM[blk])
    load_SMEM2RF(K_RF, K_SM)
    gemm(S_RF, Q_RF, K_RF^T)
    
    online_softmax(S_RF, m_RF, l_RF, O_RF)
 
<span style="display:block; background:rgba(255, 213, 79, 0.25);">    cp.wait()
    __syncthreads()</span>
    if blk &lt; T_c - 1:
        cp.async_and_commit(K_SM, K_GM[blk+1])
    convert_to_b16(P_RF, S_RF)
    load_SMEM2RF(V_RF, V_SM)
    gemm(O_RF, P_RF, V_RF)
 
# Epilogue
# ...</code></pre>

这种调度安排在确保正确同步的前提下，实现了内存传输与计算操作之间重叠（Overlap）的最大化。

当前的执行流（Execution flow）如下所示：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/Flash-Double-Buffering-After.svg" alt="Flash-Double-Buffering-After" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：优化后（GMEM 传输与计算重叠）
  </figcaption>
</figure>


### 双缓冲
我们刚刚实现的是一种经典优化技术，称为**双缓冲（double buffering）**。其思想很简单：数据加载时，与其原地等待、无所事事，不如维护两个 buffer 并在它们之间交替使用。当我们正在一个 buffer 上进行计算时，同时将数据加载到另一个 buffer 中。

在这里，我们为 $K^{(j)}$ 和 $V^{(j)}$ block 都分配了额外的 SMEM 空间。当我们忙于计算当前 block 时，下一个 block 已经在后台加载。这样一来，warp 无需空闲等待内存传输完成。

值得一提的是，我们早在 Occupancy 一节中就已经为 $V^{(j)}$ 分配了这部分额外的 SMEM 切片——此前只是没有有效地利用它。

下面展示双缓冲在一般情况下如何改变这一执行模式：

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/Single-To-Double-Buffering.svg" alt="Single-To-Double-Buffering" style="width: 90%; max-width: 980px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：单缓冲到双缓冲的通用转变
  </figcaption>
</figure>

### Kernel 3 效果
这项优化如何提升了性能？性能分析显示，由 `long_scoreboard` 导致的 warp 停顿显著下降：从 $15.15\%$ 降至 $0.95\%$。

这使性能从参考实现的 $98.3\%$ 提升至 $99.4\%$，取得了不错的改进。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/RTX3090_tflops_3_all.svg" alt="RTX3090_tflops_3_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 4：Kernel 3 性能结果
  </figcaption>
</figure>

## Kernel 4：片上 LD/ST 与计算交错（Fragment Interleaving）

Kernel 3 已成功将 GMEM 停顿减少 $93\%$，我们因此解决了 GMEM $\rightarrow$ SMEM 传输的瓶颈。不过，kernel 仍会在开始任何计算之前，将完整的 $Q$、$K^{(j)}$ 和 $V^{(j)}$ tile 加载到 RF 中。Kernel 4 通过将内存传输与计算交错执行来解决这一低效问题，从而降低寄存器压力。

目前，我们的 kernel 会在执行任何 `mma` 运算之前，将完整的 $Q$、$K^{(j)}$ 和 $V^{(j)}$ tile 加载到 RF 中。

这带来了两个问题：

1. 我们应当能够在不等待整个 tile 完成复制的情况下，先开始执行部分矩阵乘法。

2. 尽管当前选择的 kernel 配置 $B_r = B_c = 64$ 尚不足以导致寄存器溢出，但某些更大的 block 维度会触发这一问题。若希望其他 block 配置也可行——因为当前选择最终未必是最优配置——我们就需要降低寄存器压力。

### 加载 fragments

与其一次性加载全部 fragment，我们将它们划分为若干子集，并称之为 *sub-tile*。随后，`mma` 循环会先加载一个 fragment sub-tile，对其中的 fragment 执行 `mma` 运算；再加载下一个 sub-tile，对其中的 fragment 执行 `mma`，如此循环。

为最大化性能，我们希望以尽可能提高数据复用率的方式加载 fragment。主要有两种策略：

**策略 1：按行加载 A，按列加载 B（低效）**

- B 的每一列都会被多次加载（A 的每一行各加载一次）。
- **Fragment 强度：**每次内存传输对应 $0.89$ 次计算。(AI 给的数据，不一定准确)
- 会产生冗余加载，从而给内存流水线带来压力。

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

**策略 2：沿 $K^{(i)}$ 维度加载 A 和 B（最优）**

- 沿内层 $K^{(i)}$ 维度，加载 A 与 B 的“切片（slice）”。
- 每个 fragment 恰好只使用一次。
- **Fragment 强度：**每次内存传输对应 $1.62$ 次计算——几乎是策略 1 的两倍。
- 这是 CUTLASS 所采用的方法。

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


我们将依据所使用的 `mma` 指令 `m16n8k16` 对张量进行分块。因此，会沿 $k$ 维度切分，每个 slice 的宽度为 $16$ 个元素，即 $2$ 个 fragment（如上图所示）。

有些 tile 已经完整地位于 RF 中，无需再从 SMEM 加载，例如 $\tilde{P}^{(j)}$；这些 tile 将保持不变。

### Tile 维度表

| 张量 | 存储格式 | 完整形状（Fragments） | Sub-tile 形状（Fragments） | Tile 数量（Fragments） | `mma` 矩阵变量 |
|---|---|---:|---:|---|
| $Q$ | Row major | $(2, 16)$ | $(2, 2)$ | $(1, 8)$ | A |
| $K^{(j)}$ | Row major | $(8, 16)$ | $(8, 2)$ | $(1, 8)$ | B |
| $V^{(j)}$ | Column major | $(16, 8)$ | $(16, 2)$ | $(1, 4)$ | B |

*Sub-tile 在 $k$ 维度上进行分块后的 RF 形状。*

按 K 维切割的意思，到底是什么？  
`mma` 指令 `m16n8k16`
$$
D = AB^\top + C
$$

其中：

- $A$ 的形状为 $(m, k)$  
- $B$ 的形状为 $(n, k)$  
- $C$ 与 $D$ 的形状均为 $(m, n)$，并且两者可以指向同一块内存地址 

其实就是按照 K 维循环的意思，每次按照 K 维，从 SMEM 读 2 个 fragments 到 RF 中；  

### 寄存器压力
正如前文所述，此举同样有助于大幅缓解寄存器压力（Register Pressure）。以下展示了在不同 Block Size（线程块大小）配置下，编译期产生的寄存器溢出（Spills）情况：

| $(d_{\text{head}}, B_r, B_c, n_{\text{warps}})$ | spilled | stack_frame (bytes) | spill_stores (bytes) | spill_loads (bytes) | registers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| (128, 64, 32, 4) | False | 0 | 0 | 0 | 212 |
| (128, 64, 64, 4) | False | 0 | 0 | 0 | 242 |
| (128, 128, 32, 4) | True | 304 | 356 | 324 | 255 |
| (128, 128, 64, 4) | True | 728 | 960 | 836 | 255 |
| (128, 128, 128, 4) | True | 1456 | 2148 | 1808 | 255 |

如果要让更大的 Block Size 在自动调优（Kernel 7）中切实可行，我们必须显著降低寄存器压力（Register pressure）。

某些配置在栈上溢出了将近 **1.5 KB** 的数据！仅 (128, 128, 64, 4) 这一种配置就产生了 2148 字节的溢出存储（Spill stores）和 1808 字节的溢出加载（Spill loads）。这种庞大的开销会造成内存流水线拥堵，导致这些配置在实际应用中变得不可行。

**什么会导致寄存器溢出？** 当寄存器文件（Register file）容量不足时，编译器会将寄存器中的值溢出到本地内存（LMEM，路径为 L1 -> L2 -> DRAM）。这会带来额外的开销，进而污染缓存、降低带宽并导致内存传输延迟。

**性能影响因场景而异：** 发生在 Prologue/Epilogue 阶段的溢出对性能的影响微乎其微，但紧凑循环（Tight loops）内的溢出会严重拖慢有效的计算工作。在算术强度（Arithmetic intensity）较高的情况下，少量的溢出有时可以通过最优的指令调度（Instruction scheduling）被隐藏起来。

你可以配置 `nvcc`，按 Kernel 粒度输出这些详细信息。详情请参阅 `nvcc Flags for Register Spilling`。

### 对应代码上的变化

我们将修改数组的大小，使其仅容纳我们将要存储在寄存器文件 (RF) 中的子切片 (sub-tiles) 数量。这仅会影响 $\mathbf{Q}$、$\mathbf{K}^{(j)}$ 和 $\mathbf{V}^{(j)}$。

```cpp
uint32_t input[rows/8][cols/8];
// becomes
uint32_t input[rows/8][2];
改为仅保留 sub-tile 宽度（例如 `2` 个 `k` fragments）：
```

为了适应这一调整，我们对 warp_fragment_mma_f32_accum() 进行了一些修改，并增加了一个封装函数 matmul()，用于在各个子切片上循环调用 warp_fragment_mma_f32_accum()。

```cpp
// It's possible for K_fragments_A != K_fragments_B because either tensor can be buffered over sub-tiles.
template <typename value_t, const int M_fragments, const int N_fragments,
          const int K_fragments_A, const int K_fragments_B,
          typename accum_t = float>
__forceinline__ __device__ constexpr void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments_A], // K 维方向上的 A
    uint32_t (&regs_B)[N_fragments][K_fragments_B], // K 维方向上的 B
    accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT],
    int A_col_fragment_offset = 0, int B_col_fragment_offset = 0) {
    constexpr int K_iters = constexpr_min(K_fragments_A, K_fragments_B); // k 维方向纬度
    #pragma unroll
    for (int k = 0; k < K_iters; k += MMA_K_FRAGMENTS_PER_ITER) {
        #pragma unroll
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            #pragma unroll
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1],
                    
                    // k 维累加
                    regs_A[m][k + A_col_fragment_offset],
                    regs_A[m + 1][k + A_col_fragment_offset],
                    regs_A[m][k + 1 + A_col_fragment_offset],
                    regs_A[m + 1][k + 1 + A_col_fragment_offset],
                    
                    regs_B[n][k + B_col_fragment_offset],
                    regs_B[n][k + 1 + B_col_fragment_offset],
                    // end k 维累加
                    
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}
```

以下是该封装函数 (Wrapper) 的实现代码。至于 `GEMM` 结构体，我们无需过多深究，它仅仅用于存放各个矩阵的配置信息以及全局 GEMM 运算的相关参数。

```cpp
template <typename GEMM>
__device__ constexpr void matmul(typename GEMM::A_t &A, typename GEMM::B_t &B,
                                 typename GEMM::C_t &C) {
    // If ::load_entire_block_into_rf is set for either A_t or B_t, then
    // we assume the block has already been loaded.
    using A_t = typename GEMM::A_t; // Q or P
    using B_t = typename GEMM::B_t; // K or V
 
    constexpr int fragments_per_iter = 2;
 
    // GEMM::TotalKFragments is
    // - d_head / 8 for QK^T
    // - B_c / 8    for PV
    #pragma unroll
    for (int k = 0; k < GEMM::TotalKFragments; k += fragments_per_iter) {
		// Load fragments along K dimension (2 at a time)
		// Q is pre-loaded, P is computed in RF - only load if needed
		if constexpr (!A_t::load_entire_block_into_rf) {
			A.copy_SM2RF(k);  // Load Q fragments from SMEM
		}
		// Always load K/V fragments from SMEM (2 fragments per iteration)
		B.copy_SM2RF(k);
 
		// Calculate column offsets for accessing the right fragment data
        int A_col_offset = A_t::load_entire_block_into_rf ? k : 0;
        int B_col_offset = B_t::load_entire_block_into_rf ? k : 0;
        
        // Perform outer product: each A row × each B column
        // This gives optimal fragment reuse compared to row-by-row approach
        warp_fragment_mma_f32_accum(A.data(), B.data(), C.data(),
                                    A_col_offset, B_col_offset);
    }
}
```

### 寄存器使用情况

在所有配置下，寄存器压力（Register pressure）的降低都非常显著：

**关键改进：**

* **当前配置 $(64,64)$：** 寄存器 $242 \rightarrow 207$
* **参考配置 $(128,32)$：** 溢出量 $304 \rightarrow 0$，彻底消除了所有寄存器溢出（Spills），从而使该配置切实可行
* **最大溢出配置 $(128,64)$：** 溢出存储（Spill stores） $2208 \rightarrow 1312$ 字节（减少了 $40.6\%$）

最大的收获在于使 **$(128, 32, 4)$ 配置变得可行**，这正是参考内核（Reference kernel）在 RTX 3090 上所采用的配置。然而，诸如 $(128, 64, 4)$ 这样的大型配置要想真正投入使用，仍需要进一步的优化工作。

| $(d_{\text{head}}, B_r, B_c, n_{\text{warps}})$ | spilled | stack_frame | spill_stores | spill_loads | used_registers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| (128, 64, 32, 4) | False | 0 | 0 | 0 | 212 $\rightarrow$ 168 |
| (128, 64, 64, 4) | False | 0 | 0 | 0 | 242 $\rightarrow$ 207 |
| (128, 128, 32, 4) | True $\rightarrow$<br>False | 304 $\rightarrow$ 0 | 356 $\rightarrow$ 0 | 324 $\rightarrow$ 0 | 255 |
| (128, 128, 64, 4) | True | 728 $\rightarrow$ 272 | 964 $\rightarrow$ 336 | 840 $\rightarrow$ 304 | 255 |
| (128, 128, 128, 4) | True | 1360 $\rightarrow$ 840 | 2208 $\rightarrow$ 1312 | 1868 $\rightarrow$<br>1120 | 255 |

### Kernel 4 效果

Fragment 交错（Fragment interleaving）达成了一项重要的里程碑：实现了 100.0% 的基准性能（Reference performance）。

通过将共享内存到寄存器（SMEM → RF）的加载操作与计算操作交错排布，并配合最优的 Fragment 数据复用策略（算术强度达到 1.6x，对比之前的 0.89x），我们在整个优化历程中首次跨越了基准性能的门槛。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter5/RTX3090_tflops_4_all.svg" alt="RTX3090_tflops_4_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 9：Kernel 4 性能结果
  </figcaption>
</figure>

书签
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
