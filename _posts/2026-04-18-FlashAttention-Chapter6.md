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

在 chapter5 中，我们实现了来自 CUTLASS GEMM 库的三项主要优化：eager block loading、带有 fragment interleaving 的 sub-tiling，以及 double buffering。这些技术帮助我们将内存传输与计算进行重叠，在 RTX 3090 上达到了参考实现 $99.6\%$ 的性能。

在这一部分中，我们将实现最后两项优化：

- **浮点指令融合**：利用官方实现中的一个巧妙优化（Kernel 6）
- **自动调优（auto-tuning）**：用于寻找最优的 kernel 配置（Kernel 7）

最终，我们的实现将略微超越官方实现，达到参考吞吐量的 $101.5\%$。


## Kernel 6：提升 FP32 吞吐

我们之前的优化主要聚焦于 tensor 运算；从 roofline 图中可以看到，我们现在已经达到了 RTX 3090 上 matmul 的峰值 FLOPs/s。若要进一步提升性能，就需要转而分析其他瓶颈。


### Roofline 视角

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter6/rtx3090_roofline_tensor_full.svg" alt="rtx3090_roofline_tensor_full" style="width: 90%; max-width: 1000px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：Roofline 视角下的瓶颈定位（Tensor core 接近饱和）
  </figcaption>
</figure>

roofline 图展示了三种不同的算术强度（arithmetic intensity），其中与我们的 kernel 最相关的是 **L2 强度**：

- **L1 强度看起来会被人为抬高：**
  - 在使用带有 `.cg` 选项的 `cp.async` 将数据从 GMEM 拷贝到 SMEM 时，我们会绕过 L1 cache
  - 因此，只有从 SMEM 回写到 GMEM 的 $O$（以及任何寄存器溢出）才会计入 L1 流量

- **DRAM 强度只包含：**
  - 从主存发起的初始数据加载
  - 溢出的 sector（即被挤出并需要重新加载的 cache line）
  - 所以这就对应于一种“最佳情况”的内存带宽场景

- **L2 强度对于我们的分析最为准确：**
  - 它捕获了全部内存传输，因为每个 CTA 都会从 L2 搬运相同数量的数据
  - 由于没有 L1 cache 的参与，L2 能看到我们全部的内存流量
  - 因此，它最能真实反映我们 kernel 的算术强度

这里其实就是说, L2 能反应现在 tensor core 的算术强度已经达到一个理论上的峰值了;(怎么制作和看这张图, 详见附录);

## 哪里还可以优化?

既然我们已经最大化了 Tensor Core 的利用率，接下来需要考察还有哪些操作消耗了大量计算周期。

观察注意力机制公式：

$$
\operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\mathsf{T}}}{\sqrt{d_{\text{head}}}}\right)\mathbf{V}
$$

我们可以将其中的操作分为两类：

- **矩阵乘法**：使用 Tensor Core 对 FP16/BF16 数值执行计算。
- **Softmax 操作**：使用标准浮点运算单元，以 FP32 精度执行计算。

在 Tensor Core 已经饱和的情况下，提升 FP32 Softmax 的性能便成为下一个优化目标。


## 在 Softmax 中融合浮点乘法与加法
这里是编译器经常做的指令融合, 来看看怎么做吧;  

我们可以使用融合乘加（FFMA）指令，将 `d = a * b + c` 合并为一条指令，从而减少 Softmax 计算中的指令数量。下面考虑以 warp 为执行粒度时，每个 KV tile 所需的指令数。

回顾一下，每个 warp 在一个由 \((8, 4)\) 个线程构成的网格中存储一个矩阵分块。令：

- $W_r = \frac{B_r}{8 \cdot n_{\text{warps}}} = B_r / 32$
- $W_c = B_c / 4$
- $W_d = d_{\text{head}} / 4$

当前的 Softmax 实现会显式地将 $S^{(j)}$ 缩放 $\frac{1}{\sqrt{d_{\text{head}}}}$，这需要 $W_rW_c$ 条独立指令（第 1 行）。下面分析其计算成本。

> 注意：右侧表达式给出了每一行所需的指令数量（包括加法、乘法、指数运算以及 max/比较操作）：

$$
\begin{aligned}
(0)\quad & \alpha = 1 / \sqrt{d_{\text{head}}},\quad S^{(j)} = \mathbf{Q}\mathbf{K}^{\mathsf{T}} \\
(1)\quad & S^{(j)} := \alpha \cdot S
&& W_rW_c \\
(2)\quad & m^{(j)} = \max\left(m^{(j-1)}, \operatorname{rowmax}(S^{(j)})\right)
&& W_rW_c + 2W_r\;(\text{WARP SHFL}) + 2W_r \\
(3)\quad & \tilde{\mathbf{P}}^{(j)} = \exp\left(S^{(j)} - m^{(j)}\right)
&& W_rW_c + W_rW_c\;(\text{EXP}) \\
(4)\quad & z^{(j)} = \exp\left(m^{(j-1)} - m^{(j)}\right)
&& W_r + W_r\;(\text{EXP}) \\
(5)\quad & \ell^{(j)} = z^{(j)} \odot \ell^{(j-1)} + \operatorname{rowsum}\left(\tilde{\mathbf{P}}^{(j)}\right)
&& W_r\;(\text{FFMA}) + W_r(W_c - 1) \\
(6)\quad & \tilde{\mathbf{P}}^{(j)} := \operatorname{convert}_{\text{fp32}\rightarrow\text{bf16}}\left(\tilde{\mathbf{P}}^{(j)}\right)
&& W_rW_c / 2 \\
(7)\quad & \tilde{\mathbf{O}}^{(j)} = z^{(j)} \odot \tilde{\mathbf{O}}^{(j-1)} + \cdots
&& W_rW_d
\end{aligned}
$$

上述操作合计后，每个 tile 共需要：

$$
W_r\left(5.5W_c + W_d + 6\right)
$$

条指令。

### 解读一下上面的计算, 直接看可能有点晕
核心是：把一个 warp 负责的矩阵分块，按线程二维布局均匀切分；$W_r$、$W_c$、$W_d$ 就是**每个线程在对应维度上持有的元素数量**。

设一个 KV tile 的尺寸为：

$$
B_r \times B_c
$$

因为 $Q (B_r, d_head), K(B_c, d_head)$, 所以 $P/S 尺寸是(B_r, B_c)$

其中 $B_r$ 是行方向大小，$B_c$ 是列方向大小。一个 warp 有 32 个线程，文中将其逻辑排布成：

$$
8 \times 4
$$

即 8 个“行线程组”与 4 个“列线程组”。

此外，$n_{\text{warps}}$ 个 warp 会共同覆盖 tile 的行方向。因此，行方向一共由：

$$
8 \cdot n_{\text{warps}}
$$

个线程分片共同处理。每个线程对应的行数为：

$$
W_r = \frac{B_r}{8 \cdot n_{\text{warps}}}
$$

我们这里的一般配置: $n_{\text{warps}}=4$，所以：

$$
W_r = \frac{B_r}{8 \cdot 4} = \frac{B_r}{32}
$$

列方向则由 warp 内的 4 个列线程组切分，因此每个线程负责：

$$
W_c = \frac{B_c}{4}
$$

同理，输出向量的 head 维度 $d_{\text{head}}$ 也沿 4 个列线程组切分：

$$
W_d = \frac{d_{\text{head}}}{4}
$$

换一种更直观的说法：

$$
\underbrace{B_r}_{\text{tile 的行数}}
=
\underbrace{n_{\text{warps}}}_{\text{warp 数}}
\cdot
\underbrace{8}_{\text{每 warp 的行线程组}}
\cdot
\underbrace{W_r}_{\text{每线程管多少行}}
$$

$$
\underbrace{B_c}_{\text{tile 的列数}}
=
\underbrace{4}_{\text{每 warp 的列线程组}}
\cdot
\underbrace{W_c}_{\text{每线程管多少列}}
$$

---

总指令数的计算，就是把每一行的指令成本相加。($W_r, W_c$ 这么算其实就是分摊到单线程,你要执行多少次对应的计算;)

其中，与 $W_rW_c$ 成正比的部分为：

$$
\begin{aligned}
\text{第 (1) 行：} &\quad W_rW_c \\
\text{第 (2) 行：} &\quad W_rW_c \\
\text{第 (3) 行：} &\quad W_rW_c + W_rW_c = 2W_rW_c \\
\text{第 (5) 行：} &\quad W_r + W_r(W_c-1) = W_rW_c \\
\text{第 (6) 行：} &\quad \frac{W_rW_c}{2}
\end{aligned}
$$

因此：

$$
W_rW_c + W_rW_c + 2W_rW_c + W_rW_c + \frac{W_rW_c}{2}
=
5.5W_rW_c
$$

与输出维度相关的是第 (7) 行：

$$
W_rW_d
$$

剩余的按行处理、shuffle、比较和指数操作开销合并为：

$$
6W_r
$$

所以总成本为：

$$
5.5W_rW_c + W_rW_d + 6W_r
$$

提取公因子 $W_r$：

$$
\boxed{
W_r\left(5.5W_c + W_d + 6\right)
}
$$

这也揭示了优化动机：第 (1) 行的显式缩放单独需要 $W_rW_c$ 次乘法；若把它融合进后续计算的 FFMA 中，就能消除这部分独立指令。

#### 第二步的计算
第 (2) 步是在计算每一行的运行时最大值（running maximum）：

$$
m^{(j)}
=
\max\left(
m^{(j-1)},
\operatorname{rowmax}\left(S^{(j)}\right)
\right)
$$

它是 online softmax 的一部分：第 $j$ 个 KV tile 到来时，既要得到该 tile 内每行的最大值，也要和此前所有 tile 的最大值 $m^{(j-1)}$ 合并。

其成本：

$$
W_rW_c + 2W_r\;(\text{WARP SHFL}) + 2W_r
$$

可以拆成三部分。

---

##### 1. 在线程本地求最大值：$W_r(W_c-1)$

每个线程持有一个大小为：

$$
W_r \times W_c
$$

的局部分块。对于其中每一行的 $W_c$ 个元素，求最大值需要 $W_c-1$ 次比较。

因此，$W_r$ 行的本地归约成本为：

$$
W_r(W_c-1)
$$

---

##### 2. 与历史最大值合并：$W_r$

得到当前 tile 的行最大值后，还要和此前累计的最大值比较：

$$
m^{(j)}
=
\max\left(m^{(j-1)}, \operatorname{rowmax}(S^{(j)})\right)
$$

每一行需要一次比较，因此成本为：

$$
W_r
$$

将前两部分合并：

$$
W_r(W_c-1) + W_r
=
W_rW_c
$$

这就是公式中的第一项。

---

##### 3. 在 4 个逻辑列线程组之间做归约：$2W_r$ 次 shuffle + $2W_r$ 次比较

每个线程只拥有该行的一部分列：

$$
W_c = \frac{B_c}{4}
$$

因此，某一行的完整 $B_c$ 个元素被分布在 4 个逻辑列线程组中。每个线程先求得自己的局部最大值后，还需要让这 4 个线程交换数据并完成最终归约。

4 个值求最大值需要：

$$
\log_2 4 = 2
$$

轮归约。每一轮都包含：

1. 一次 `WARP SHFL`：从配对线程取得其局部最大值；
2. 一次 `max`：将取得的值与自己的局部最大值比较。

对每一个线程持有的 $W_r$ 行，都要进行这两轮操作，因此：

$$
2W_r\;(\text{WARP SHFL})
+
2W_r\;(\text{max/compare})
$$

---

因此，第 (2) 行的完整成本是：

$$
\boxed{
W_rW_c
+
2W_r\;(\text{WARP SHFL})
+
2W_r
}
$$

其中：

- $W_rW_c$：线程本地的行最大值计算，以及与历史最大值 $m^{(j-1)}$ 的合并；
- $2W_r\;(\text{WARP SHFL})$：在 4 个逻辑列线程组之间交换局部最大值；
- $2W_r$：交换后的两轮 `max` 比较。

---

### 为什么 $d_{\text{head}}$ 沿列排列？

这里讨论的不再只是注意力分数矩阵 $S$，还包括输出矩阵的累积计算。

注意力计算的形状是：

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^{\mathsf T}
\in \mathbb{R}^{B_r \times B_c}
$$

随后用概率矩阵乘以 Value：

$$
\mathbf{O} = \mathbf{P}\mathbf{V}
$$

其中：

$$
\mathbf{P}\in\mathbb{R}^{B_r \times B_c},
\qquad
\mathbf{V}\in\mathbb{R}^{B_c \times d_{\text{head}}}
$$

所以输出 tile 的形状是：

$$
\mathbf{O}\in\mathbb{R}^{B_r \times d_{\text{head}}}
$$

这里的 $d_{\text{head}}$ 是 $\mathbf{O}$ 的**列维度**，也是 $\mathbf{V}$ 的第二维。因此，文中将 warp 的 4 个逻辑列线程组分配给它：

$$
W_d = \frac{d_{\text{head}}}{4}
$$

更明确地说，线程逻辑布局在不同阶段处理的是不同矩阵：

| 阶段 | 矩阵形状 | 逻辑列方向对应 |
|---|---:|---|
| Softmax | $\mathbf{S},\mathbf{P}\in\mathbb{R}^{B_r\times B_c}$ | $B_c$，故每线程处理 $W_c=B_c/4$ 个元素 |
| 输出累积 | $\mathbf{O}\in\mathbb{R}^{B_r\times d_{\text{head}}}$ | $d_{\text{head}}$，故每线程处理 $W_d=d_{\text{head}}/4$ 个元素 |

所以并不是说 $d_{\text{head}}$ 在注意力分数矩阵 $S$ 中“沿列排列”；它在 $S$ 中已经被归约掉了。它是在最终输出矩阵 $\mathbf{O}$ 的列方向上出现：

$$
\tilde{\mathbf O}^{(j)}
=
z^{(j)}\odot\tilde{\mathbf O}^{(j-1)}
+
\tilde{\mathbf P}^{(j)}\mathbf V^{(j)}
$$

第 (7) 行要更新大小为 $B_r\times d_{\text{head}}$ 的输出分块，因此其每线程工作量是：

$$
W_rW_d
$$

这种映射是该 kernel 的线程布局选择；理论上也可以采用转置或其他分块方案，但这里采用“8 个逻辑行组切 $B_r$，4 个逻辑列组切 $B_c$ 或 $d_{\text{head}}$”的方式，便于让同一套 warp 布局服务于 Softmax 与输出累积。

### 官方实现中的优化
 不再单独对 $S^{(j)}$ 进行缩放，而是将缩放与减法融合到指数函数的输入中，并通过一条 FFMA（融合乘加，fused multiply-add）指令完成。具体如下：

1. **缩放最大值（第 \(2'\) 行）**：计算

   $$
   \tilde{m}^{(j)} = \alpha \cdot m^{(j)}
   $$

   这会额外增加 $B_r$ 条指令。

2. **融合指数函数的输入（第 3 行）**：在执行指数运算之前，通过一条 FFMA 指令计算

   $$
   \alpha \cdot S^{(j)} - \tilde{m}^{(j)}
   $$

3. **更新缩放因子（第 4 行）**：相应调整 $\ell$ 与 $\tilde{\mathbf{O}}^{(j)}$ 计算中使用的缩放因子。

> `*` 表示该行经过了修改。

$$
\begin{aligned}
(0)\quad
& \alpha = \frac{1}{\sqrt{d_{\text{head}}}},
\qquad
S^{(j)} = \mathbf{Q}\mathbf{K}^{\mathsf{T}}
\\[4pt]
(1)^{*}\quad
& \text{移除：}\quad
S^{(j)} := \alpha \cdot S
&& \text{移除：}\quad W_rW_c
\\[4pt]
(2)\quad
& m^{(j)} =
\max\left(
m^{(j-1)},
\operatorname{rowmax}(S^{(j)})
\right)
&& W_rW_c + 2W_r\;(\text{WARP SHFL}) + 2W_r
\\[4pt]
(2')^{*}\quad
& \tilde{m}^{(j)} = \alpha \cdot m^{(j)}
&& +W_r
\\[4pt]
(3)^{*}\quad
& \tilde{\mathbf{P}}^{(j)}
=
\exp\left[
\alpha \cdot S^{(j)} - \tilde{m}^{(j)}
\right]
&& W_rW_c\;(\text{FFMA}) + W_rW_c\;(\text{EXP})
\\[4pt]
(4)^{*}\quad
& z^{(j)}
=
\exp\left[
\tilde{m}^{(j-1)} - \tilde{m}^{(j)}
\right]
&& W_r + W_r\;(\text{EXP})
\\[4pt]
(5)\quad
& \ell^{(j)}
=
z^{(j)} \odot \ell^{(j-1)}
+
\operatorname{rowsum}\left(\tilde{\mathbf{P}}^{(j)}\right)
&& W_r\;(\text{FFMA}) + W_r(W_c-1)
\\[4pt]
(6)\quad
& \tilde{\mathbf{P}}^{(j)}
:=
\operatorname{convert}_{\text{fp32}\rightarrow\text{bf16}}
\left(\tilde{\mathbf{P}}^{(j)}\right)
&& W_rW_c/2
\\[4pt]
(7)\quad
& \tilde{\mathbf{O}}^{(j)}
=
z^{(j)} \odot \tilde{\mathbf{O}}^{(j-1)}
+ \cdots
&& W_rW_d
\end{aligned}
$$

该优化方案每个 tile 仅需：

$$
W_r\left(4.5W_c + W_d + 7\right)
$$

条指令。


## 显式使用快速指数近似

当前已经使用了快速指数近似：其中，`expf()` 在内部通过 `exp2f()` 实现，并利用以下恒等式：

$$
e^x = 2^{x \cdot \log_2(e)}
$$

我们将在代码中显式采用这一近似，并将 $\log_2(e)$ 因子直接合并到缩放系数中：

$$
\begin{aligned}
(0)^{*}\quad
& \alpha = \frac{\log_2(e)}{\sqrt{d_{\text{head}}}},
\qquad
S^{(j)} = \mathbf{Q}\mathbf{K}^{\mathsf{T}}
\\[4pt]
(2)\quad
& m^{(j)}
=
\max\left(
m^{(j-1)},
\operatorname{rowmax}(S^{(j)})
\right)
&& W_rW_c + 2W_r\;(\text{WARP SHFL}) + 2W_r
\\[4pt]
(2')^{*}\quad
& \tilde{m}^{(j)} = \alpha \cdot m^{(j)}
&& +W_r
\\[4pt]
(3)^{*}\quad
& \tilde{\mathbf{P}}^{(j)}
=
2^{\left[
\alpha \cdot S^{(j)} - \tilde{m}^{(j)}
\right]}
&& W_rW_c\;(\text{FFMA}) + W_rW_c\;(\text{EXP})
\\[4pt]
(4)^{*}\quad
& z^{(j)}
=
2^{\left[
\tilde{m}^{(j-1)} - \tilde{m}^{(j)}
\right]}
&& W_r + W_r\;(\text{EXP})
\\[4pt]
(5)\quad
& \ell^{(j)}
=
z^{(j)} \odot \ell^{(j-1)}
+
\operatorname{rowsum}\left(\tilde{\mathbf{P}}^{(j)}\right)
&& W_r\;(\text{FFMA}) + W_r(W_c-1)
\\[4pt]
(6)\quad
& \tilde{\mathbf{P}}^{(j)}
:=
\operatorname{convert}_{\text{fp32}\rightarrow\text{bf16}}
\left(\tilde{\mathbf{P}}^{(j)}\right)
&& W_rW_c/2
\\[4pt]
(7)\quad
& \tilde{\mathbf{O}}^{(j)}
=
z^{(j)} \odot \tilde{\mathbf{O}}^{(j-1)}
+ \cdots
&& W_rW_d
\end{aligned}
$$

注意，指令数量保持不变；这里仅仅是将原本已经使用的快速指数近似显式地表达出来。

## 最终算法

参考实现采用了一个略有不同的变体：

$$
z^{(j)}
=
2^{\left[
\alpha\cdot\left(m^{(j-1)}-m^{(j)}\right)
\right]}
$$

与直接存储 $\tilde{m}^{(j)}$ 相比，该方法额外需要 $W_r$ 条指令，但性能略有提升。我们也采用这一方案，最终每个 tile 需要：

$$
W_r\left(4.5W_c + W_d + 8\right)
$$

条指令。

$$
\begin{aligned}
(0)^{*}\quad
& \alpha = \frac{\log_2(e)}{\sqrt{d_{\text{head}}}},
\qquad
S^{(j)} = \mathbf{Q}\mathbf{K}^{\mathsf{T}}
\\[4pt]
(2)\quad
& m^{(j)}
=
\max\left(
m^{(j-1)},
\operatorname{rowmax}(S^{(j)})
\right)
&& W_rW_c + 2W_r\;(\text{WARP SHFL}) + 2W_r
\\[4pt]
(2')^{*}\quad
& \tilde{m}^{(j)} = \alpha \cdot m^{(j)}
&& +W_r
\\[4pt]
(3)^{*}\quad
& \tilde{\mathbf{P}}^{(j)}
=
2^{\left[
\alpha\cdot S^{(j)}-\tilde{m}^{(j)}
\right]}
&& W_rW_c\;(\text{FFMA}) + W_rW_c\;(\text{EXP})
\\[4pt]
(4)^{*}\quad
& z^{(j)}
=
2^{\left[
\alpha\cdot\left(m^{(j-1)}-m^{(j)}\right)
\right]}
&& 2W_r + W_r\;(\text{EXP})
\\[4pt]
(5)\quad
& \ell^{(j)}
=
z^{(j)}\odot\ell^{(j-1)}
+
\operatorname{rowsum}\left(\tilde{\mathbf{P}}^{(j)}\right)
&& W_r\;(\text{FFMA}) + W_r(W_c-1)
\\[4pt]
(6)\quad
& \tilde{\mathbf{P}}^{(j)}
:=
\operatorname{convert}_{\text{fp32}\rightarrow\text{bf16}}
\left(\tilde{\mathbf{P}}^{(j)}\right)
&& W_rW_c/2
\\[4pt]
(7)\quad
& \tilde{\mathbf{O}}^{(j)}
=
z^{(j)}\odot\tilde{\mathbf{O}}^{(j-1)}
+ \cdots
&& W_rW_d
\end{aligned}
$$

官方这里多 $W_r$ 条指令, 但是实测性能略高的原因分析:  
这么算和我们推演的优化差异就在于, 是否需要保存缩放后的 $\tilde{m}^{(j-1)}$, 按照官方这么做,是不需要保存 $\tilde{m}^{(j-1)}$ 的, 这样可以减少寄存器压力,所以理论上可以得到性能上的提升;

<div style="margin: 24px 0; padding: 20px 28px; border: 1px solid #f3c98b; border-radius: 10px; background: #fffaf0;">

<h3 style="margin-top: 0; color: #df8a35;">⚠️ 数值精度</h3>

这些对 kernel 6 的修改会轻微改变其浮点误差特性：

- 通过 FMA 融合乘法与加法通常能够降低舍入误差。
- 通过将 logits 缩放 $\log_2(e)$，并使用 `exp2f()` 近似指数函数，会略微增大近似误差。

是否可以接受这一变化取决于具体应用场景。如果应用对数值误差尤其敏感，在采用这些修改之前，可能需要更深入地评估这些权衡及其影响。

</div>

## 总结

通过将缩放操作与指数计算融合，我们降低了每个 tile 的 Softmax 指令数量：

- **优化前：**$W_r\left(5.5W_c + W_d + 6\right)$ 条指令
- **优化后：**$W_r\left(4.5W_c + W_d + 8\right)$ 条指令

这意味着每个 warp tile 的指令数减少了：

$$
W_r\left(W_c - 2\right)
$$

对于当前的块大小：

$$
B_r = 64,\qquad B_c = 64
$$

总指令数减少为：

$$
\frac{28}{252} = 11.1\%
$$

## 代码

为实现这一点，我们将修改两个函数：

- `scale_l_O()`
- `exponentiate_tensor()`

由于相关逻辑已经被折叠进来，我们现在不再调用 `scale_S_accum()`。

关于缩放与 `exp2f`，这里补充说明：在代码中，`softmax_scale` 精确地等于

$$
\log_2(e) / \sqrt{d_{\text{head}}}
$$

为了保持可读性，我们在表述时仍然使用自然指数形式；但在实现中，实际通过 `exp2f` 来完成指数运算，所依据的恒等式为

$$
e^x = 2^{x \log_2(e)}
$$

因此，FFMA 可以直接生成最终的指数输入参数。

`softmax.cuh`

```c++
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
        accum_t scale = exp2f((m_cur[q] - m_next[q]) * softmax_scale); // 优化体现在这里

        m_cur[q] = m_next[q];
        l[q] *= scale;
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= scale;
        }
    }
}

```

`softmax.cuh`
```c++
template <int QO_fragments, int KV_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void
exponentiate_tensor(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m)[QO_fragments],
    accum_t softmax_scale
) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t max_scaled = m[q] * softmax_scale; // 改动 1
        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] = exp2f(S_accum[q][k] * softmax_scale - max_scaled); // 改动 2
        }
    }
}

```

## Kernel 6 效果

我们的浮点指令融合优化使吞吐量获得了小幅提升：从 $67.11$ TFLOPs 提高到 $67.23$ TFLOPs，相当于参考实现性能的 $99.9\%$。


<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter6/RTX3090_tflops_6_all.svg" alt="RTX3090_tflops_6_all" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 2：Kernel 6 性能结果
  </figcaption>
</figure>

### profile 一下
通过将 softmax 相关的浮点指令数量减少 $11.1\%$，我们也相应地将 FP pipeline 的利用率降低了 $12.41\%$。

这种 FP pipeline 压力的下降还带来了一个次级收益：tensor core 的利用率略微提升了 $0.17\%$，从 $47.11\%$ 增加到 $47.19\%$。


<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/chapter6/rtx3090_5_6_tensor_fma_pipe_util.svg" alt="rtx3090_5_6_tensor_fma_pipe_util" style="width: 90%; max-width: 960px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 3：Kernel 5/6 管线利用率对比
  </figcaption>
</figure>

## Kernel 7：Auto-Tuning (调参-手动 doge)

到目前为止，我们一直使用固定的 block 配置 \((B_r = B_c = 64)\) 来优化 kernel。然而，我们已经实现的每一项改进，本质上都对应着一个可独立开关的可配置组件。

自动调优（auto-tuning）是系统性探索这一配置空间、以寻找最优编译期参数的标准做法。

我们的配置空间包括：

`flash_attention.cuh`

```cpp
struct FlashForwardKernelConfig {
    const int d_head; // [128]
    const int B_r;    // [64, 128]
    const int B_c;    // [32, 64]
    const int n_warps;// [4]

    const bool async_copy; // always true. this was for testing purposes.
 
    // Kernel #2
    const bool swizzled;
 
    // Kernel #3
    const bool eager_load_blocks;

    // Kernel #4
	  // This can be either 0 or 2.
	  // If it is:
	  // - 0: load the entire tile into the RF at once before executing any matmuls
	  //   - additionally for Q, persist without reloading.
	  // - 2: load sub-tiles 2 fragments wide at a time
    const int Q_mma_load_K_fragments;
	  const int K_mma_load_K_fragments;
	  const int V_mma_load_K_fragments;

    // Kernel #5
	  const bool mma_double_buffer_loads;

    // Kernel #6: fusing FP multiplication and addition instructions
	  const bool optimized_softmax;

};
```

每引入一项新的优化，我们的配置空间都会呈指数级增长。为了使自动调优（auto-tuning）在实践中仍然可处理，我们会预先筛除那些已知表现次优的配置。这包括如下 kernel：

- 不使用 swizzling 的 kernel
- 存在过量寄存器溢出（register spilling）的 kernel

### 可选配置

为便于表述，我们将采用一种标准化格式来描述 kernel 配置：

```
({d_head}, {B_r}, {B_c}, {n_warps}):  
 {async}+{eager}+{swizzled}+load_{q_fragments}_{k_fragments}_{v_fragments}_fragments+{buffer}+{opt_softmax}
```

其中，block 配置与 fragment 数始终会给出，而其他选项仅在被启用时才会出现。

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

Kernel 6 与 Kernel 7 之间的主要差异在于：在 Kernel 7 中，$Q$ 在整个 mainloop 期间始终保留在寄存器文件（RF）中，因此每个 warp 只需要执行一次从 SMEM $\rightarrow$ RF 的 $Q$ 加载。这减少了每次迭代中执行的 `ldmatrix` 指令数量，并显著降低了与 SMEM 相关的 warp stall：

| Stall | Kernel 6 | Kernel 7 | Delta |
|---|---:|---:|---:|
| `barrier` | 4.82% | 2.72% | -2.11% |
| `mio_throttle` | 2.46% | 1.95% | -0.51% |
| `short_scoreboard` | 1.94% | 1.70% | -0.24% |

### 简要的 Block Size 分析

下面给出其他 block size 下的最佳配置，以及它们相对于最优配置的表现。  
**注意：** 从这里开始，所有 kernel 都默认采用 `async+eager+swizzled` 配置，因此为了便于阅读，后续将省略 `async+eager+swizzled+` 这一前缀。

| $(d_{\text{head}}, B_r, B_c, n_{\text{warps}})$ | TFLOPs (seqlen=4096) | 相对 Kernel 7 的性能 |
|---|---:|---:|
| $(128, 64, 64, 4)$: `load_0_2_0_fragments+buffer+opt_softmax` | 68.31 | 100.0% |
| $(128, 64, 32, 4)$: `load_2_2_0_fragments` | 67.39 | 98.64% |
| $(128, 128, 32, 4)$: `load_2_2_0_fragments+buffer+opt_softmax` | 67.36 | 98.61% |
| $(128, 128, 64, 4)$: `load_2_2_0_fragments+opt_softmax` | 54.26 | 79.42% |

occupancy 表如下：


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

ToDo: 根据 A100 架构特性去调优;  

## 小结

### 浮点指令融合（Kernel 6）

- 在 online softmax 中，将 attention logit 的缩放与最大值减法融合，使浮点指令数量减少了 $11.1\%$
- 在 RTX 3090 上达到了参考实现 $99.9\%$ 的性能

### 参数调优（Kernel 7）

- 系统性地探索配置空间，以寻找最优参数
- 最优配置为 $(128,\ 64,\ 64,\ 4)$，其中 $Q$ 在寄存器文件（RF）中持久驻留
- 在 RTX 3090 上超越了参考实现，达到 $101.5\%$



## 附录: Roofline 图的生成方式与解读

这张 Roofline 图通常不是直接由某个可视化工具“一键生成”的成品图，而是通过如下流程构建：

1. **使用性能分析工具采集指标**  
   通常使用 **Nsight Compute** 对 kernel 进行 profile，采集：
   - kernel 执行时间
   - Tensor Core / matmul 的运算量（或根据问题规模计算理论 FLOPs）
   - 各层存储层级的数据传输字节数，例如：
     - DRAM bytes
     - L2 bytes
     - L1 bytes

2. **计算性能（Performance）**  
   纵轴表示性能，定义为：
   $$
   \text{Performance} = \frac{\text{Operations}}{\text{Time}}
   $$
   这里的 Operations 通常取 tensor core matmul 的浮点操作数，因此单位是 $\text{OP/s}$ 或 $\text{FLOP/s}$。

3. **计算算术强度（Arithmetic Intensity）**  
   横轴表示算术强度，定义为：
   $$
   \text{Arithmetic Intensity} = \frac{\text{Operations}}{\text{Bytes transferred}}
   $$
   对不同的存储层级分别计算：
   - DRAM 强度：
     $$
     I_{\text{DRAM}} = \frac{\text{Operations}}{\text{Bytes}_{\text{DRAM}}}
     $$
   - L2 强度：
     $$
     I_{\text{L2}} = \frac{\text{Operations}}{\text{Bytes}_{\text{L2}}}
     $$
   - L1 强度：
     $$
     I_{\text{L1}} = \frac{\text{Operations}}{\text{Bytes}_{\text{L1}}}
     $$

4. **绘制不同层级的 Roofline**  
   对每一级存储带宽 $B$，Roofline 由下面的公式给出：
   $$
   P(I) = \min(P_{\text{peak}},\ I \cdot B)
   $$
   其中：
   - $P_{\text{peak}}$ 表示 GPU 的峰值计算吞吐（这里对应 Tensor Core matmul 峰值）
   - $I$ 表示算术强度
   - $B$ 表示某一级存储层级的带宽（例如 DRAM / L2 / L1）

   因此，每一条 roofline 都由两部分组成：
   - **斜线部分**：带宽受限区，性能随算术强度线性增长
   - **水平部分**：计算受限区，性能达到峰值后不再继续上升

5. **将 kernel 对应的点绘制到图上**  
   由于同一个 kernel 可以基于不同层级的字节数计算出不同的算术强度，因此图中会出现：
   - 一个对应 DRAM 的点
   - 一个对应 L2 的点
   - 一个对应 L1 的点

---

### 一般使用什么工具

这类图通常通过以下组合完成：

- **Nsight Compute**：采集底层性能计数器
- **Python + matplotlib**：根据采集到的数据自行绘制 Roofline 图
- 有时也可以参考 Nsight Compute 自带的 Roofline 页面，但如果需要像图中这样手动标注 L1 / L2 / DRAM 点、绘制自定义注释和说明，通常仍然会选择自己重画

因此，更准确地说，这张图通常是：

> **先用 Nsight Compute 采集性能指标，再用 Python 脚本按照 Roofline 模型绘制而成。**

---

### 这张图应该怎么看

阅读 Roofline 图时，可以按照下面的思路理解：

#### 1. 先看纵轴：性能是否已经接近峰值

图中的横向屋顶表示计算峰值 $P_{\text{peak}}$。  
如果 kernel 对应的点已经接近这条水平线，说明该 kernel 的性能已经接近硬件的峰值计算吞吐。

在这张图中，几个点基本都贴近水平屋顶，因此可以得出：

> 当前 kernel 的 Tensor Core matmul 吞吐已经接近 RTX 3090 的峰值。

这也意味着，进一步优化 Tensor Core 核心计算本身的空间已经很小。

---

#### 2. 再看横轴：不同层级下的算术强度差异

同一个 kernel 会对应三个不同的算术强度点：

- **L1 点**：最靠右
- **DRAM 点**：通常居中
- **L2 点**：更靠左

这说明：

$$
I_{\text{L1}} > I_{\text{DRAM}} > I_{\text{L2}}
$$

其原因是分母“传输字节数”在不同层级的统计口径不同：

- L1 层看到的字节数最少，因此算术强度看起来最大
- DRAM 只统计真正去主存的数据，因此通常比 L2 更“理想化”
- L2 会看到更多实际发生的片上缓存层级流量，因此更接近真实的数据搬运成本


### Roofline 图中几条线性斜线是如何画出来的

Roofline 图中的几条线性斜线，分别对应不同存储层级的**带宽上界**，例如：

- DRAM roof
- L2 roof
- L1 roof

这些线并不是“拟合”出来的，而是直接根据 Roofline 模型公式画出来的理论上界。

---

#### 1. 基本公式

对于任意一个存储层级，其 Roofline 都满足：

$$
P(I) = \min\left(P_{\text{peak}},\ I \cdot B\right)
$$

其中：

- $P(I)$：在算术强度为 $I$ 时可达到的理论性能上界
- $P_{\text{peak}}$：峰值计算性能
- $B$：该存储层级的峰值带宽
- $I$：算术强度（Arithmetic Intensity）

---

#### 2. 为什么会是一条斜线

当算术强度较小时，kernel 处于**带宽受限**区域，此时性能上界由

$$
P(I) = I \cdot B
$$

决定。

由于这是一个一次函数，所以在普通坐标系下它是一条过原点的直线；  
而在 Roofline 图常用的双对数坐标系下，它表现为一条**固定斜率的直线**。

因此：

- 给定 DRAM 带宽 $B_{\text{DRAM}}$，就有一条 DRAM 斜线
- 给定 L2 带宽 $B_{\text{L2}}$，就有一条 L2 斜线
- 给定 L1 带宽 $B_{\text{L1}}$，就有一条 L1 斜线

它们的区别只在于带宽 $B$ 不同。

---

#### 3. 为什么不同层级的斜线平行但位置不同

因为它们都满足：

$$
P(I) = I \cdot B
$$

在对数坐标下：

$$
\log P = \log I + \log B
$$

可以看到：

- 斜率都是 $1$
- 截距由 $\log B$ 决定

因此不同层级的 roofline 斜线通常是**互相平行**的，只是上下平移：

- 带宽越大，线越靠上
- 带宽越小，线越靠下

例如如果：

$$
B_{\text{L1}} > B_{\text{L2}} > B_{\text{DRAM}}
$$

那么对应的三条斜线位置关系就是：

- L1 斜线最高
- L2 斜线居中
- DRAM 斜线最低

---

#### 4. 它们如何与水平峰值线拼成 “roof”

Roofline 之所以叫 “roofline”，是因为它最终形成了一个“屋顶”形状。

对于每个层级，性能上界取：

$$
P(I) = \min\left(P_{\text{peak}},\ I \cdot B\right)
$$

这意味着：

- 当 $I$ 较小时，使用斜线部分：
  $$
  P(I) = I \cdot B
  $$
- 当 $I$ 足够大时，达到计算峰值上界：
  $$
  P(I) = P_{\text{peak}}
  $$

二者的交点满足：

$$
I \cdot B = P_{\text{peak}}
$$

所以交点的横坐标为：

$$
I^* = \frac{P_{\text{peak}}}{B}
$$

这就是每条斜线与水平峰值线相接的位置。

---

#### 5. 实际绘图时怎么做

画图时一般会先取一组横轴上的算术强度采样点，例如：

$$
I \in [10^{-1}, 10^{5}]
$$

对每个 $I$ 计算：

$$
P_{\text{roof}}(I) = \min\left(P_{\text{peak}},\ I \cdot B\right)
$$

然后把这些点连起来，就得到一条完整的 roofline。

如果只想单独画“斜线部分”，那就直接画：

$$
P(I) = I \cdot B
$$

---

#### 6. 一个具体例子

假设某 GPU 的：

- 峰值 Tensor Core 性能为
  $$
  P_{\text{peak}} = 5.8 \times 10^{13}\ \text{OP/s}
  $$
- DRAM 带宽为
  $$
  B_{\text{DRAM}} = 9 \times 10^{11}\ \text{byte/s}
  $$
- L2 带宽为
  $$
  B_{\text{L2}} = 2 \times 10^{12}\ \text{byte/s}
  $$
- L1 带宽为
  $$
  B_{\text{L1}} = 7 \times 10^{12}\ \text{byte/s}
  $$

那么三条斜线分别就是：

##### DRAM 斜线
$$
P_{\text{DRAM}}(I) = I \cdot B_{\text{DRAM}}
$$

##### L2 斜线
$$
P_{\text{L2}}(I) = I \cdot B_{\text{L2}}
$$

##### L1 斜线
$$
P_{\text{L1}}(I) = I \cdot B_{\text{L1}}
$$

完整 roofline 则是：

$$
P_{\text{DRAM roof}}(I) = \min\left(P_{\text{peak}},\ I \cdot B_{\text{DRAM}}\right)
$$

$$
P_{\text{L2 roof}}(I) = \min\left(P_{\text{peak}},\ I \cdot B_{\text{L2}}\right)
$$

$$
P_{\text{L1 roof}}(I) = \min\left(P_{\text{peak}},\ I \cdot B_{\text{L1}}\right)
$$

---

#### 7. 用 Python 画这些斜线的典型写法

```python
import numpy as np
import matplotlib.pyplot as plt

P_peak = 5.8e13
B_dram = 9.0e11
B_l2   = 2.0e12
B_l1   = 7.0e12

I = np.logspace(-1, 5, 500)

roof_dram = np.minimum(P_peak, I * B_dram)
roof_l2   = np.minimum(P_peak, I * B_l2)
roof_l1   = np.minimum(P_peak, I * B_l1)

plt.figure(figsize=(10, 6))
plt.loglog(I, roof_dram, '--', label='DRAM Roof')
plt.loglog(I, roof_l2, '--', label='L2 Roof')
plt.loglog(I, roof_l1, '--', label='L1 Roof')
plt.loglog(I, np.full_like(I, P_peak), '-', label='Compute Roof')

plt.xlabel('Arithmetic Intensity [OP/byte]')
plt.ylabel('Performance [OP/s]')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.show()

```

### 为什么能得出这三个结论

图中给出的三个结论分别是：

- **L1 intensity appears artificially high**
- **DRAM intensity only includes best-case main-memory traffic**
- **L2 intensity is most accurate for our analysis**

下面分别解释为什么可以从图和实现方式中得出这三个结论。

---

#### 结论一：L1 强度看起来被“人为抬高”了

之所以说 L1 intensity 被人为抬高，是因为这里的 kernel 在从 GMEM 拷贝到 SMEM 时，使用了带 `.cg` 选项的 `cp.async`。这种方式会绕过 L1 cache。

这意味着：

- GMEM $\to$ SMEM 的大量读取并不会计入 L1 流量
- 真正计入 L1 的，主要只剩下：
  - 从 SMEM 回写到 GMEM 的 $O$
  - 以及可能存在的寄存器溢出（register spill）

因此，L1 层统计到的字节数偏少，即：
$$
\text{Bytes}_{\text{L1}} \text{ 很小}
$$
从而导致：
$$
I_{\text{L1}} = \frac{\text{Operations}}{\text{Bytes}_{\text{L1}}}
$$
被显著放大。

这就是为什么图上的 **L1 点最靠右**，看起来算术强度特别高。  
但这种“高”并不代表 kernel 真的具有那么高的有效数据复用，而是因为 L1 没有看到大部分实际的数据传输。

因此可以得出：

> **L1 intensity 并不能真实反映该 kernel 的数据搬运代价，它看起来偏高，是统计口径造成的。**

---

#### 结论二：DRAM 强度只反映“最佳情况”的主存流量

DRAM intensity 的分母只统计真正访问主存（off-chip DRAM）的字节数。  
这通常包括：

- 首次从主存加载的数据
- cache line 被驱逐后重新加载的数据

也就是说，DRAM 只关心“有没有真的出片访问主存”。

如果某些数据已经驻留在片上缓存（例如 L2）中，那么这些访问不会继续计入 DRAM 流量。于是 DRAM 看到的字节数往往偏少，得到的算术强度就会偏高：

$$
I_{\text{DRAM}} = \frac{\text{Operations}}{\text{Bytes}_{\text{DRAM}}}
$$

这对应的是一种更理想、更乐观的情形：  
即很多数据重用已经由缓存体系“免费”帮你吸收了，因此 DRAM 压力不大。

所以 DRAM intensity 更像是在回答：

> **如果只看真正触达主存的流量，这个 kernel 的算术强度是多少？**

这也是为什么文中说它代表的是一种 **best-case memory bandwidth scenario**。

---

#### 结论三：L2 强度最适合用于分析这个 kernel

L2 intensity 被认为最准确，核心原因有两个：

##### （1）L2 能看到更完整的流量

由于该实现中绕过了 L1，因此很多实际发生的数据搬运都会直接体现在 L2 层上。换句话说：

- L1 看不到全部流量
- DRAM 又只看到最外层、最理想化的那部分流量
- 只有 L2 最接近“每个 CTA 实际搬了多少数据”

因此：
$$
I_{\text{L2}} = \frac{\text{Operations}}{\text{Bytes}_{\text{L2}}}
$$
更能反映真实的数据访问代价。

##### （2）L2 对每个 CTA 的数据搬运更稳定

对于这个 kernel 而言，每个 CTA 从 L2 获取的数据量更具有代表性；相比之下：

- L1 受到是否绕过 cache 的影响太大
- DRAM 受到 cache 命中与否、是否发生驱逐等因素影响更大

因此，L2 intensity 更适合作为分析 kernel 是否仍然受内存搬运限制的依据。

所以可以得出：

> **L2 intensity 最能反映该 kernel 的真实 arithmetic intensity，因此在分析这个 kernel 时最有参考价值。**

---

### 如何从图的点位关系直观看出这些结论

从图上可以做如下直观观察：

1. **三个点的纵坐标几乎相同**  
   说明它们对应的是同一个 kernel 的实际性能，只是以不同的字节统计口径计算出了不同的横坐标。

2. **L1 点最靠右**  
   说明按 L1 统计时，算术强度最大，即分母最小。这正对应了“L1 统计不到大部分流量，因此强度被抬高”。

3. **DRAM 点也比较靠右，但没有 L1 那么夸张**  
   说明 DRAM 字节数比 L1 大一些，但仍然偏少，因此它描述的是一种相对理想化的带宽情形。

4. **L2 点更靠左**  
   说明 L2 统计到的字节数更多，因此它给出的算术强度更保守，也更接近真实搬运成本。

5. **L2 点仍然接近水平屋顶**  
   说明即便采用更真实的 L2 统计口径，kernel 也已经接近计算峰值。  
   因此可以推断：
   - 当前实现已经不再明显受 matmul 核心计算不足的限制
   - 后续优化应更多关注其他瓶颈，而不是继续只盯着 tensor core 运算本身

---

### 小结

这张图的生成流程可以概括为：

> 使用 Nsight Compute 采集 kernel 的运行时间、运算量以及 DRAM / L2 / L1 的传输字节数，再利用 Roofline 公式
> $$
> P(I) = \min(P_{\text{peak}}, I \cdot B)
> $$
> 在 Python 中绘制得到。

而它所表达的关键信息是：

- kernel 的 Tensor Core matmul 性能已经接近峰值；
- L1 强度由于绕过 L1 cache 而被人为抬高；
- DRAM 强度只反映了较理想化的主存流量；
- L2 强度最接近 kernel 的真实算术强度，因此最适合用于性能分析。
