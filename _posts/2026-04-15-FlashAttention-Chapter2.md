---
layout:     post   				    # 使用的布局（不需要改）
title:      Flash Attention 2 Chapter2 				# 标题 
subtitle:   基本子块 #副标题
date:       2026-04-15 				# 时间
author:     BY 						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
auto_heading_numbering: true       # 启用标题自动编号（h2/h3/h4）
tags:								#标签
    - Cuda, Flash Attention
---

## 简介

在这一章, 我们将会去探索 CUDA 中组成 Falsh Attention 核的基础操作.Flash Attention 的性能取决于两点：其一，通过 Tensor Core 充分提升计算吞吐；其二，通过高效的数据搬运降低内存瓶颈。

为了说明这些因素为何关键，先来看一个典型 attention 切片的算术复杂度：设有 64 个 query 向量，关注 4096 个 key-value 向量，且 $d_{\text{head}}=128$。对应的 attention 计算为：

$$
\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}}\right)V
$$

在该设定下，计算量约为 $135.8\text{M}$ 次浮点运算，而从 GMEM 读取/写回的数据仅约 $2.1\text{M}$ 字节。由此可得算术强度（arithmetic intensity）约为 $64$，即每从内存加载 1 字节数据，平均可执行约 64 次数学运算。

<!-- 关于算术强度的计算代码，可参见附录（appendix）。 -->

这种较高的算术强度，使 Flash Attention 成为 Tensor Core 的理想负载。与常规浮点运算单元相比，这类专用计算单元能够提供明显更高的吞吐能力；但其峰值性能通常建立在“计算远超于访存”的前提之上。当前约 $64\!:\!1$ 的计算/访存比意味着：Tensor Core 可以持续执行矩阵运算，而数据搬运则在后台并行进行，从而最大化硬件利用率。

接下来我们将分三个阶段构建这套指令工具箱；在每个阶段中，都优先采用 Ampere 架构上性能最优的指令：

1. **高吞吐矩阵运算**：使用 Tensor Core 的 `mma`（matrix multiply-accumulate）指令。  
2. **高效内存操作**：使用能够持续维持高带宽利用率的数据搬运原语(汇编原子指令)。  
3. **配套支撑操作**：包括数据类型转换及其他关键实现细节。
