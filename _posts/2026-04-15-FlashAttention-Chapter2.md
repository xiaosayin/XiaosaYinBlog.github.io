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

这种较高的算术强度，使 Flash Attention 成为 Tensor Core 的理想负载。与常规浮点运算单元相比，这类专用计算单元能够提供明显更高的吞吐能力；但其峰值性能通常建立在“计算远超于访存”的前提之上。当前约 $64:1$ 的计算/访存比意味着：Tensor Core 可以持续执行矩阵运算，而数据搬运则在后台并行进行，从而最大化硬件利用率。

接下来我们将分三个阶段构建这套指令工具箱；在每个阶段中，都优先采用 Ampere 架构上性能最优的指令：

1. **高吞吐矩阵运算**：使用 Tensor Core 的 `mma`（matrix multiply-accumulate）指令。  
2. **高效内存操作**：使用能够持续维持高带宽利用率的数据搬运原语(汇编原子指令)。  
3. **配套支撑操作**：包括数据类型转换及其他关键实现细节。

## 矩阵乘法
Flash Attention 的性能本质上取决于两次矩阵乘法，因此我们首先讨论如何利用 Ampere 架构的 Tensor Core 来加速这两类计算。在内核实现中，输入与输出张量采用 `bf16/fp16` 精度，而 `softmax` 的计算则使用 `fp32`，以保证数值稳定性。

Ampere 架构上的 Tensor Core 以 **fragment** 为基本操作对象。可以将其理解为存放在线程寄存器中的矩阵 tile。

> **Fragment 定义**  
> 在本系列中，术语 **fragment** 特指跨一个 warp 存储于寄存器文件中的一个 $(8,8)$ tile。  
> 其中，warp 内每个线程仅持有 2 个元素，但整个 warp 协同完成乘法计算。  
> 这就是 Tensor Core 运算的基本单元。  
> 一个 warp 里面有 32 个线程, 而每个线程自己独占的寄存器里面存放 2 个元素, 总共就是 64个元素, 对应上述所说的$(8,8)$ tile.

### mma 指令
那么，如何在实现层面真正驱动 Tensor Core 呢？关键就在于 `mma`（matrix multiply-accumulate）这条 PTX 指令。

需要说明的是，编程 Tensor Core 还有另一类指令接口：`wmma`。但在本文中我们选择 `mma`，主要原因是其 fragment 布局对开发者更透明，便于精细控制数据映射与指令行为。若你想进一步了解这一取舍背后的思考，可参考 `wmma` API。

<div style="border:1px solid #d9eaf7;background:#f7fcff;border-radius:12px;padding:14px 16px;margin:14px 0;">
  <p style="margin:0 0 8px 0;font-weight:700;font-size:18px;">PTX / mma / wmma 关系速览</p>
  <p style="margin:0 0 8px 0;">
    <strong>PTX</strong> 是 CUDA 的中间层汇编（虚拟 ISA）。CUDA C++ 代码通常先编译为 PTX，再由驱动/JIT 转为目标 GPU 的机器码。
  </p>
  <p style="margin:0 0 8px 0;">
    <strong>mma</strong> 是 PTX 层的低层 Tensor Core 矩阵乘加指令；<strong>wmma</strong> 是 CUDA 提供的高层 Warp Matrix API，底层通常映射到 mma/相关 Tensor Core 指令。mma 和 wmma, 以我的理解,类似于 cpu 中 asm 和 intrinsic 的关系;
  </p>
  <p style="margin:0 0 6px 0;"><strong>三层关系：</strong></p>
  <ul style="margin:0 0 8px 18px;padding:0;">
    <li>CUDA C++（高层）</li>
    <li>wmma API（高层封装）</li>
    <li>PTX mma（低层指令）→ GPU 机器码（最终执行）</li>
  </ul>
  <p style="margin:0;"><strong>实践取舍：</strong>wmma 开发更快；mma 控制更细、调优空间更大，适合 FlashAttention 这类极致性能内核。</p>
</div>

`mma` 指令执行的运算形式为：

$$
D = AB^\top + C
$$

其中：

- $A$ 的形状为 $(m, k)$  
- $B$ 的形状为 $(n, k)$  
- $C$ 与 $D$ 的形状均为 $(m, n)$，并且两者可以指向同一块内存地址  

需要注意的是，尽管我们在内存中以行主序（row-major）存储 $B$（即同一行元素在内存中相邻），`mma` 实际参与乘法的是 $B^\top$。在将 attention 张量映射到这些操作数时，这一点尤为关键。

对于本文使用的具体 `mma` 指令，其操作数的形状与数据类型由指令规格决定，维度为 $(m=16,\ n=8,\ k=16)$。在 Ampere 架构上，针对 16-bit 输入、32-bit 累加的候选指令有两种：`m16n8k8` 与 `m16n8k16`。综合效率考虑，本文选择 `m16n8k16`。

**表 1：`m16n8k16` 指令下各操作数的形状与寄存器映射**

| Operand | DType    | Shape (Variables) | Shape (Elements) | Shape (Fragments) | Shape (Registers) |
|--------|----------|-------------------|------------------|-------------------|-------------------|
| A      | BF16/FP16| $(m, k)$          | $(16, 16)$       | $(2, 2)$          | $(2, 2)$          |
| B      | BF16/FP16| $(n, k)$          | $(8, 16)$        | $(1, 2)$          | $(1, 2)$          |
| C/D    | FP32     | $(m, n)$          | $(16, 8)$        | $(2, 1)$          | $(2, 2)$          |

解释一下这个表; Elements 和 Fragments 都比较好理解, Fragments 就是 Elements Shape 除以 8x8 得到的;   
Registers 则是按照每个线程视角来看, 每 8x8 tile, 一个线程私有寄存器里面有两个 elements, 对 BF16 这种 16bit 的数据而言, 寄存器单位是 uint32_t, 所以一个寄存器单位可以存两个元素, 对于 (16,16) 其实是 持有 8 个元素, 分别存在 (2,2) uint32_t 的寄存器槽位中, B 同理;  
对于 C 而言, 因为 `m16n8k16` 是一个 warp 级别指令, 计算完整个矩阵操作后, 得到 (16,8) 个元素的矩阵,分到每个线程头上就是 16x8/32 = 4, 每个线程持有 4个 32bit 的值,所以对应寄存器槽位是 (2,2);  

### Attention Tensors 如何映射到 mma 指令上呢?
在明确 `mma` 指令的操作数定义之后，下面讨论 Flash Attention 中各张量如何映射到这些操作数。该映射至关重要，因为它直接决定了整个 kernel 中数据的存储组织与访问方式。

Flash Attention 包含两次核心矩阵乘法：

1. $\mathbf{QK}^\top$：计算 query 与 key 之间的注意力分数  
2. $\tilde{\mathbf{P}}\mathbf{V}$：将注意力权重作用于 value  

由于第一步存在转置操作，同时两步对内存布局效率的要求不同，这两次乘法对应的“张量到操作数”映射也不相同。

下面给出相应的存储布局。

| Tensor | Operand | Storage Format in SMEM & GMEM | SMEM Tile Shape | Storage Format in RF | Effective Shape in RF (not actual storage)|
|---|---|---|---|---|---|
| $\mathbf{Q}$ | A | row-major | $(B_r, d_{\text{head}})$ | row-major | $(B_r, d_{\text{head}})$ |
| $\mathbf{K}^{(j)}$ | B | row-major | $(B_c, d_{\text{head}})$ | row-major | $(B_c, d_{\text{head}})$ |
| $\tilde{\mathbf{P}}$ | A | N/A | N/A | row-major | $(B_r, B_c)$ |
| $\mathbf{V}^{(j)}$ | B | row-major | $(B_c, d_{\text{head}})$ | col-major* | $(d_{\text{head}}, B_c)$ |

注: 首先就要理解,上述这张表其实是 block level(CTA level) 处理的数据;  
这张表刚看会令人很费解, 不知道这是在做什么,突然又冒出来这么多参数; 在这里我来简要解释一下 flash Attention 怎么将大矩阵的乘法,分散到每一个 warp 中去做; 

输入数据 shape(Q,K,V): (batch_size = 16, seq_len=4096, n_heads = 16, d_head = 128)

```python
def generate_qkv(cfg: QKVConfig):
    q = torch.randn(
        (cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.d_head),
        dtype=cfg.dtype,
        device=cfg.device,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    return q, k, v
```

然后固定 batch 和 n_heads 维度, d_head 是每个 head 的维度, 所以 (seq_len, d_head) 就需要被分成小矩阵块去做 attention 的过程, 然后就是 $B_r$ 和 $B_c$ 就是按照 seq_len 这个维度来切割, 以决定每个 Block 处理多大的 $Q$,$K$,$V$ 块, 为了更好理解核函数的实现,暂定 $B_r = 64, B_c=64$. 更细致的后面会讲到;  

<div style="border:1px solid #dbe7f3;background:#f7fbff;border-radius:12px;padding:14px 16px;margin:14px 0;">
  <p style="margin:0 0 10px 0;font-size:28px;line-height:1;"><strong>✎ 为什么 V 在 RF 中需要列主序存储</strong></p>

  <p style="margin:0 0 10px 0;">
    `mma` 指令执行的是 $AB^\top + C$，而我们目标计算的是 $\tilde{\mathbf{P}}\mathbf{V}$（不带显式转置）。为使两者一致，可采用如下映射策略：
  </p>

  <ul style="margin:0 0 10px 18px;padding:0;">
    <li>在 GMEM 与 SMEM 中，$\mathbf{V}$ 保持常规行主序（row-major）存储；</li>
    <li>加载到 RF 时，对其做一次逻辑转置，使其在寄存器视角下等效为 $\mathbf{V}^\top$；</li>
    <li>这样 `mma` 实际计算 $\tilde{\mathbf{P}}(\mathbf{V}^\top)^\top=\tilde{\mathbf{P}}\mathbf{V}$，结果与目标表达式一致。</li>
  </ul>

  <p style="margin:0 0 10px 0;">
    例如，若 $\mathbf{V}$ 在 GMEM/SMEM 中形状为 $(64,128)$，则在 RF 中按转置视角组织为等效 $(128,64)$，从而无需在计算路径中额外插入一次显式 transpose，即可得到正确输出。
  </p>

  <p style="margin:0;">
    该转置通常通过 `ldmatrix` 的 transpose 变体完成，后续章节将详细展开（<strong>ldmatrix Transpose</strong>）。
  </p>
</div>
