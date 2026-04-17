---
layout:     post
title:      Flash Attention Appendix
subtitle:   实验、配置与指令补充说明
date:       2026-04-18
author:     BY
header-img: img/post-bg-2015.jpg
catalog: true
auto_heading_numbering: true
tags:
    - Cuda, Flash Attention
---

## 说明

本文是 Flash Attention from Scratch 系列的附录整理版，主要聚焦：

- benchmarking / profiling 的实验设置
- arithmetic intensity 的计算脚本
- `nvcc` 与寄存器 spill 相关编译参数
- Ampere 设备关键规格与吞吐数据
- `mma`、`cp.async`、`wmma` 相关指令级补充

原文参考：[Appendix](https://lubits.ch/flash/Appendix)。

## AI 使用说明

原作者在附录中给出的 AI 使用范围：

- 主要用于 Python 脚本代码生成
- 用于博客文字编辑
- 少量用于 C++ 代码

## Benchmarking 与 Profiling 环境

实验环境（原文）：

- CUDA `12.8`（driver `570.133.20`）
- Python `3.12`
- PyTorch `torch==2.5.1+cu124`
- Flash Attention 2 fork（固定到 commit `b36ad4e`，做了若干工程性裁剪）

### Benchmark 设置

- sequence lengths：`512, 1024, 2048, 4096, 8192, 16384`
- 每个 kernel：warmup `32` 次，计时 `128` 次用于 TFLOPs 统计

固定频率如下：

| Device | SM Clock | DRAM Clock |
| --- | --- | --- |
| RTX 3090 | 1680 MHz | 9501 MHz |
| A100 PCIe 80GB | 1110 MHz | 1512 MHz |

> A100 仅支持固定 DRAM clock（1512 MHz）。

查看设备可用时钟：

```bash
nvidia-smi --query-supported-clocks=gr,mem --format=csv
```

### Nsight Compute Profiling 命令

```bash
ncu \
  --config-file off \
  --export /path/to/profile \
  --force-overwrite \
  --target-processes application-only \
  --kernel-name regex:device|flash \
  --warp-sampling-interval 1 \
  --warp-sampling-max-passes 1000 \
  --warp-sampling-buffer-size 536870912 \
  --set full \
  --apply-rules no \
  --import-source no \
  /path/to/python /path/to/repo/tools/benchmark/run_kernels.py 4096 128 --kernels $KERNEL_CONFIG_NAME \
  --n_runs 32
```

## Arithmetic Intensity 计算

附录中的 Python 代码（单 tile pair）：

```python
ELEM_SIZE = 2  # bytes

def tile_softmax_flop(B_r, B_c, d_head) -> int:
    # Kernel 6-16
    return B_r * (4 * B_c + d_head + 4)
    # Kernel 1-5
    return B_r * (5 * B_c + d_head + 3)

def kv_tile_flop(B_r, B_c, d_head) -> int:
    QK_flops = 2 * B_r * d_head * B_c
    PV_flops = 2 * B_r * B_c * d_head
    softmax_flops = tile_softmax_flop(B_r, B_c, d_head)
    return QK_flops + PV_flops + softmax_flops

def gmem_transfer_size(B_r, B_c, d_head) -> int:
    return d_head * 2 * (B_r + B_c) * ELEM_SIZE

def arithmetic_intensity(B_r, B_c, kv_seq_len, d_head) -> float:
    return (
        kv_tile_flop(B_r, B_c, d_head) * (kv_seq_len // B_c)
    ) / gmem_transfer_size(B_r, kv_seq_len, d_head)
```

## 编译参数：寄存器 Spill 排查

常用 `nvcc` 选项：

- `-Xptxas=-warn-spills`
- `-Xptxas=-warn-lmem-usage`
- `--resource-usage`

无 spill 时典型输出：

```text
ptxas info    : Function properties for ...
       0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 163 registers, used 1 barriers, 448 bytes cmem[0]
```

有 spill 时典型输出：

```text
ptxas info    : Function properties for ...
       456 bytes stack frame, 668 bytes spill stores, 580 bytes spill loads
ptxas info    : Used 255 registers, used 1 barriers, 456 bytes cumulative stack size, 448 bytes cmem[0]
```

## Compute Capability 与设备吞吐

### 关键 Compute Capability 规格

| Device | Compute Capability | Max Shared Memory (CTA/SM) | Max 32b Registers / Thread | # 32b Registers / SM |
| --- | --- | --- | --- | --- |
| A100 | 8.0 | 163KB / 164KB | 255 | 65536 |
| RTX 3090 | 8.6 | 99KB / 100KB | 255 | 65536 |

### Ampere 设备吞吐对比

| Device | GMEM Bandwidth | `mma` TFLOPs/s (16b input, 32b accum) | FP32 TFLOPs/s | `mma / FP32` | MUFU TFLOPs/s | Compute Capability |
| --- | --- | --- | --- | --- | --- | --- |
| A100 | 1.94 TB/s | 311.84 | 19.5 | 16x | 4.875 | 8.0 |
| RTX 3090 | 936.2 GB/s | 71 | 35.6 | 2x | 4.45 | 8.6 |

## Kernel 规格与属性

### Kernel Specification

- forward pass only
- non-causal attention
- `d_head = 128`
- no dropout / no KV caching
- Q/K/V sequence lengths 相同
- sequence lengths 可被 block sizes 整除
- 输入输出 `bf16/fp16`，softmax 在 `fp32`

### Kernel 7 指令比例（附录）

| Kernel | FP32+INT : `mma` ratio | Non-`mma` : `mma` ratio |
| --- | --- | --- |
| Kernel 7 (ours) | 3.0 - 5.2 | 4.2 - 6.5 |
| Optimized Kernel | 1.6 - 2.36 | 2.41 - 3.67 |

## 指令补充

### PTX 与 SASS 对应

| PTX | SASS |
| --- | --- |
| `ldmatrix` | `LDSM` |
| `cp.async` | `LDGSTS` |
| `mma`（16-bit input） | `HMMA` |

### `mma` 指令（m16n8k16）

本文核心 PTX 形式：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 d, a, b, c
```

要点：

- `mma`：matrix multiply-accumulate
- `.sync.aligned`：warp 内同步且对齐执行
- `m16n8k16`：形状规格
- `.row.col`：A row-major，B col-major
- `.f32.f16.f16.f32`：`D, A, B, C` 数据类型

## `cp.async` vs 传统加载

传统拷贝路径（cache miss 常见路径）：

`GMEM -> L2 -> L1 -> RF -> SMEM`

`cp.async`（合适对齐与配置下）可以走：

`GMEM -> L2 -> SMEM`

这能绕过 RF 中转，降低寄存器压力；配合 `.cg` 还可减少 L1 污染。

<figure style="text-align:center; margin: 16px auto;">
  <img src="{{ site.baseurl }}/img/flashAttention/appendix/a9fb6b92aef1065471e7974f1b719d99_MD5.jpeg" alt="appendix_cp_async_path" style="width: 90%; max-width: 900px; height: auto;">
  <figcaption style="margin-top: 8px; color: #666; font-size: 14px;">
    图 1：`cp.async` 与传统路径对比（附录原图）
  </figcaption>
</figure>

传统代码示例：

```cpp
for (int i = 0; i < N_COPIES; ++i) {
    smem[i] = gmem[i];
}
```

对应 SASS（示意）：

```text
LDG.E.128 ...
LDG.E.128 ...
STS.128 ...
STS.128 ...
```

`cp.async` 示例：

```cpp
for (int i = 0; i < 4; ++i) {
    cp_async<16>(smem + 16 * i, gmem);
}
cp_async_commit();
cp_async_wait<0>();
```

对应 SASS（示意）：

```text
LDGSTS.E.BYPASS.LTC128B.128 ...
LDGDEPBAR
DEPBAR.LE SB0, 0x0
```

### Commit / Wait 的价值

`cp.async` 支持显式分组与提交：

- 可把不同 tensor 的加载放进不同 group
- 可以只等待必要 group，其他传输继续 in-flight

这是传统加载路径较难实现的调度优势。

## `wmma` 与 `mma`

`wmma` 是高层 API，fragment 布局对开发者不透明；  
`mma` 是更低层的 PTX 指令，能够直接控制寄存器中的 fragment 组织。

对于本文这类高性能手写 kernel，`mma` 的可控性更有利于：

- 直接在 RF 上做运算
- 减少不必要 SMEM 往返
- 避免额外 MIO 压力

## 形状与映射速查

### `mma.m16n8k16` 操作数形状

| Operand | DType | Shape (Variables) | Shape (Elements) | Shape (Fragments) | Shape (Registers) |
| --- | --- | --- | --- | --- | --- |
| A | BF16/FP16 | `(m, k)` | `(16, 16)` | `(2, 2)` | `(2, 2)` |
| B | BF16/FP16 | `(n, k)` | `(8, 16)` | `(1, 2)` | `(1, 2)` |
| C + D | FP32 | `(m, n)` | `(16, 8)` | `(2, 1)` | `(2, 2)` |

### Warp 级线程到坐标映射（Kernels 1-8）

| Operation | Row | Column |
| --- | --- | --- |
| `mma` fragment / RF -> SMEM | `(tid % 32) / 4` | `(tid % 4) * 2` |
| SMEM -> RF (`ldmatrix`) | `tid % 16` | `((tid % 32) / 16) * 8` |
| GMEM <-> SMEM | `(tid % 32) / 8` | `tid % 8` |

## 小结

这份附录的价值在于提供“可复现实验 + 指令级解释 + 设备差异基线”三类信息。  
如果正文关注的是“优化策略”，那么附录更像是“验证与推导工具箱”：

- 如何稳定 benchmark / profile
- 如何读 spill 与资源占用
- 如何从 PTX/SASS 视角理解性能瓶颈
