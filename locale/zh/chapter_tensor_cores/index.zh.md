(chap_tensor_cores)=
# Tensor Core: `tcgen05`

:::{admonition} Overview
:class: overview

- `tcgen05` 是 Blackwell 的 Tensor Core instruction family。其 MMA instruction 以 cooperative 方式执行 tile matrix-multiply-accumulate work，instruction 由一个 elected thread 提交。
- accumulator 存放在 TMEM 而非 register 中。epilogue 后来用 `tcgen05.ld` 将其带回 register。
- `cta_group::1` 和 `cta_group::2` 控制一个 CTA 还是两个 CTA 合作在 MMA 上。那个选择也改变 M dimension 如何映射到 TMEM。
- block-scaled MMA mode，如 `mxfp8` 和 `nvfp4`，添加 scale-factor operand。data operand 存放在 SMEM，而 scale factor 通过 TMEM stage。
:::

稠密 linear algebra 是现代 GPU 花费大部分有用 work 的地方。一个普通 CUDA core matrix multiply 无法接近芯片的 advertised peak（{ref}`chap_background`）。快速 GEMM 和 attention kernel 通过向 Tensor Core 提供正确的 tile shape、layout 和 synchronization 来达到那个 peak。

基本 operation 自 Volta 以来在精神上没有改变。Tensor Core 消费 matrix tile、乘它们并累加结果。从 generation 到 generation 变化的是 operation 如何发出、operand 如何 layout 以及 accumulator 存放哪里。

Blackwell 对最后一部分做出重大改变。`tcgen05` 的 accumulator 不再作为 long-lived register fragment 保持。它被写入 Tensor Memory 或 TMEM（{ref}`chap_tmem`）。那一个改变影响了整个 kernel。MMA 写入 TMEM。completion 被异步跟踪。epilogue 后来从 TMEM 加载 accumulator 并将其恢复为转换和 store 所需的 register fragment。

本章聚焦 compute instruction 本身。TMA（{ref}`chap_tma`）负责将 operand 移入 SMEM。TMEM 负责持有 accumulator 和一些 scale-factor operand。`tcgen05.mma` 是介于这两个 memory movement 之间的 Tensor Core operation。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tcgen05_intro.html" title="tcgen05 and Tensor Memory" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: `tcgen05` accumulator behavior。切换 A 或 B 的 transpose，选择 output width `N`，并步进 `K` iteration 查看 partial sum 在 TMEM 中累加。*

## `tcgen05` MMA

`tcgen05` MMA 是 Blackwell Tensor Core matrix-multiply-accumulate instruction。它是一个 cooperative instruction。work 为 warpgroup 执行，在某些 mode 它可以涉及来自同一 cluster 的两个 CTA。instruction 不是由每个 thread 独立发出。一个 elected thread 代表参与 group 提交 operation。

将 MMA 分为三个 question 是有帮助的。

第一个 question 是谁合作。normal mode 使用一个 CTA，写为 `cta_group::1`。larger mode 使用 cluster 中的两个 CTA，写为 `cta_group::2`。两种情况下，instruction 代表 tile 上的一个 Tensor Core operation，而不是一个 thread 的 scalar operation。

第二个 question 是 operand 和 result 存放哪里。data operand 通常存放在 SMEM。某些 variant 还可以从 TMEM 读 A operand。accumulator 写入 TMEM。operand layout 必须匹配 Tensor Core 期望的，包括 data operand 使用的 swizzled shared memory layout（{ref}`chap_data_layout`）。

第三个 question 是如何观察 completion。`tcgen05.mma` 是 asynchronous。发出 MMA 并不意味着 multiply-accumulate 已完成。instruction 在 operation 提交后返回，而 Tensor Core 继续运行。kernel 使用 commit group 和 `mbarrier` 来了解 result 何时 ready（{ref}`chap_async_barriers`）。

那个 asynchronous behavior 正是使 overlap 可能的原因。快速 kernel 不发出 MMA 并立即 stall 直到完成。它可以发出 MMA、开始准备 later tile，并仅在 result 实际需要时才等待。代价是每个 handoff 必须显式。如果 epilogue 在 MMA completion barrier 触发之前读取 TMEM，它读取太早。

## Accumulator 存放在 TMEM

在 Ampere 和 Hopper 上，accumulator 作为 register 向 program 暴露。MMA 产生 per-lane register fragment，epilogue 直接消费该 fragment。这很简单，但它将 accumulator size 绑定到每个 thread 的 register budget。

Blackwell 打破了那个 link。`tcgen05.mma` 将其 accumulator 写入 TMEM，一个 scoped 到 CTA 的 Blackwell memory space。accumulator 可以通过 compute phase 保持在 TMEM 中，epilogue 后来使用 `tcgen05.ld` 将其加载回 register。

这改变了 kernel 的 shape。register fragment 在边缘仍然重要。epilogue 仍然需要 register，以便 convert、应用 elementwise work 并存储 result。但 long-lived accumulator state 不再是 register allocation problem。它是一个 TMEM allocation 和 layout problem（{ref}`chap_tmem`）。

这就是为什么 `tcgen05` 和 TMEM 必须一起理解的原因。MMA instruction 决定计算什么 tile。TMEM 决定 accumulator 落在哪里。epilogue 必须使用 matching load path 来恢复 accumulator，使其在预期的 register layout 中。

## `cta_group::1` 和 `cta_group::2`

`tcgen05` MMA 可以在 `cta_group::1` 或 `cta_group::2` mode 运行。

在 `cta_group::1` 中，一个 CTA 拥有 MMA。其 operand 在该 CTA 的 SMEM 中，其 accumulator 写入该 CTA 的 TMEM。

在 `cta_group::2` 中，cluster 中的两个 CTA 合作在一个 MMA tile 上。每个 CTA 有自己的 SMEM 和自己的 TMEM。accumulator 不存储在一个跨越两个 CTA 的物理 TMEM region 中。它在两个 CTA 之间分割，每个 CTA 持有自己的部分。even CTA 发出 instruction 并提交 pair 的 completion barrier。

那个 choice 很重要，因为它改变 logical accumulator tile `C(M, N)` 如何映射到 TMEM。TMEM 有 128 hardware Lane row 和最多 512 hardware Col column。在 TIRx layout notation 中，那些 axis 写为 `TLane` 和 `TCol`。MMA mode 决定 `C` 的 row 和 column 如何放置到那些 TMEM axis 上。

有四个有用的 case 需要记住。

下图遵循 demo color convention：紫色标记 SMEM operand，橙色标记 TMEM accumulator state，绿色标记 Tensor Core MMA path。CTA identity 通过 label 和 position 显示，而不是改变那些 hardware color。

### `cta_group::1`，`M = 128`

这是最简单的 case。一个 CTA 计算 128-row tile。TMEM 也有 128 Lane row。mapping 因此是直接的：accumulator 的 row `m` 映射到 Lane `m`，N dimension 映射到 TMEM column。

result 填充 128 Lane row × N Col column。这是 baseline picture。CTA 在 SMEM 中拥有 A 和 B，在其 TMEM 中拥有完整 accumulator tile。

![cta_group::1, M=128: row m 直接映射到 TMEM Lane m](../img/mma_cg1_m128.svg)

### `cta_group::1`，`M = 64`

对于 `M = 64`，accumulator 只有 64 row，但 TMEM 仍有 128 Lane row。hardware 不简单地将 row 0 到 63 pack 到 lane 0 到 63。相反，它将它们 spread 到 128 lane，以四个 16-row run。

row 0 到 15 去往 lane 0 到 15。row 16 到 31 去往 lane 32 到 47。row 32 到 47 去往 lane 64 到 79。row 48 到 63 去往 lane 96 到 111。

这在 lane 16 到 31、48 到 63、80 到 95 和 112 到 127 留下 gap。那些 gap 是有意的。使用不同的 lane alignment，另一个独立 `M = 64` MMA 可以占用 complementary lane。这使两个较小 M tile 共享 128-lane TMEM structure 而不互相干涉。

N dimension 仍映射到 TMEM column。unusual 部分仅是 M row 在 Lane 上的 placement。

![cta_group::1, M=64: four 16-row run 在 Lane stride 32，为另一个 aligned M=64 tile 留出空间](../img/mma_cg1_m64.svg)

### `cta_group::2`，`M = 256`

当 M dimension 大于一个 CTA 能自然持有时，MMA 可以使用 `cta_group::2`。对于 `M = 256`，split 是直接的。CTA 0 持有 row 0 到 127。CTA 1 持有 row 128 到 255。

每个 CTA 使用自己的 TMEM Lane row 0 到 127 和完整 N column。physically，这是两个单独的 128-row TMEM region，每个 CTA 一个。logically，它们形成一个 256 × N accumulator tile。

每个 CTA 还提供对应其 M row 的 A 部分。B 按 mode 需要向两个 CTA 可用。even CTA 负责发出 MMA 并提交 pair 的 completion barrier。

这是 {ref}`chap_gemm_advanced` 中 two-CTA cluster GEMM 使用的 mode。

![cta_group::2, M=256: M contiguously split 到两个 CTA，每 CTA 128 row](../img/mma_cg2_m256.svg)

### `cta_group::2`，`M = 128`

`cta_group::2`，`M = 128` mode 仍使用两个 CTA，但 M dimension 更短。由于总共只有 128 row，每个 CTA 接收 64 M row。

remaining lane capacity 用于 pack N dimension。在每个 CTA 内部，N 的一半占用 lane 0 到 63，另一半占用 lane 64 到 127。这使每个 CTA 使用所有 128 Lane row，即使它只拥有 64 M row。

因此 split 有两个部分。M 在 CTA pair 间分割，每 CTA 64 row。然后 N 在每个 CTA 内在 TMEM Lane row 的下半和上半间分割。

![cta_group::2, M=128: 每 CTA 64 M row，两半 N 在 lower 和 upper Lane half 上 stack](../img/mma_cg2_m128.svg)

在这些 mode 中，principle 相同。`tcgen05.mma` 计算一个 logical accumulator tile，但那个 tile 必须放置到物理 128 Lane × 最多 512 Col TMEM space。mode 和 M shape 决定那个 placement。kernel 的其余部分在后来读取 accumulator 时必须使用相同 mapping。

对于这里的 kernel，accumulator 通常在 TMEM 中为 f32。那是 common high-accuracy path。它不是唯一可能的 accumulator type。`.kind::f16` path 可以 accumulate 在 f16。

## Operand Placement

对于 dense MMA mode，A 和 B 在 MMA 运行前在 SMEM 中准备。TMA 负责将 global memory tile 移入 SMEM。kernel 将那些 SMEM tile 排列在 Tensor Core 期望的 layout 中，包括任何需要的 swizzle。

accumulator C 写入 TMEM。那是与早期 generation 的主要区别。epilogue 不直接从 MMA instruction output 接收 accumulator。它必须显式用 `tcgen05.ld` 从 TMEM 加载。

在 `cta_group::1` 中，一个 CTA 提供 operand 并拥有 accumulator。在 `cta_group::2` 中，每个 CTA 从 SMEM 提供自己的 operand 侧，每个 CTA 拥有 accumulator 自己的 TMEM portion。当 A 按 M split 时，每个 CTA 保留其 M slice 的 A row。B 按 mode share，因为两个 M slice 都乘以相同的 N × K tile。

这个 separation 在阅读 kernel 时很重要。SMEM placement 回答 Tensor Core 如何读 A 和 B。TMEM placement 回答 accumulator 去到哪里。两个 layout 由 MMA mode 关联，但它们不是相同 memory space 且不能视为 interchangeable。

## Block-Scaled MMA

dense mode 直接从 SMEM 读 data operand 并 accumulate 到 TMEM。block-scaled MMA 添加两个更多 operand：A 和 B 的 scale-factor tensor。

这用于极低精度 format，如 `mxfp8` 和 `nvfp4`。low-precision format 高效，但其 dynamic range 小。单个 global scale 通常太粗略。如果 scale 为 largest value 选择，smaller value 失去 precision。如果 scale 为 small value 选择，larger value 可能 clip。

block scaling 通过给 small K block 分配 scale factor 来解决这个问题。连续 K element 组共享一个 scale。MMA 概念上 dequantize 每个 block 用其 scale，然后在 accumulator type 中 accumulate product。

对于 A 和 B，这引入了两个 scale-factor tensor：

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中 `SFK = K / B`，`B` 是沿 K 的 block size。

确切 block size 取决于 format。重要点是 scale axis 在更粗粒度上跟随 K。每个 scale factor 描述 K value 的 block，而不是一个 individual element 也不是整个 matrix。

mathematical shape 是：

```text
acc += (Aq * scale_a) * (Bq * scale_b)
```

其中 `Aq` 和 `Bq` 是 quantized low-precision value，scale 在 accumulate 前恢复其 approximate magnitude。

scale dtype 也很重要。使用 `e8m0` scale，每个 scale 有效为 power of two。使用 `e4m3` scale（如 `nvfp4`），scale 是一个 small floating-point value 并表示 power of two 之间的 value。

## Scale Factor 存放哪里

block-scaled `tcgen05.mma` 与 dense MMA 在一个重要 placement rule 上不同：scale factor 从 TMEM 读取。

data operand A 和 B 仍在 SMEM 中 stage。scale factor SFA 和 SFB 通过 TMEM stage。由于 TMA load 到 SMEM，scale factor 通常需要一个 extra step。kernel 首先 load 它们到 SMEM，然后用 `tcgen05.cp` 从 SMEM 复制到 TMEM。只有 scale factor 在 TMEM 中后，block-scaled MMA 才能读取它们。

这给 scale factor 一个不同于 data operand 的 movement path：

```text
A, B:     global memory 到 SMEM，然后 MMA 读 SMEM
SFA, SFB: global memory 到 SMEM，然后 tcgen05.cp 复制 SMEM 到 TMEM，然后 MMA 读 TMEM
```

scale factor 的 TMEM layout 是 compact 的。一个 128-row scale vector 可以 pack 到 32 Lane row，使用基于 `r % 32` 的 lane position mapping 和沿列 `r / 32`。data 然后可以 broadcast 到读取完整 128 Lane space 的四个 warp（{ref}`chap_layout_generations`）。

这是一个为什么 TMEM layout 必须显式的好例子。accumulator layout 和 scale-factor layout 都在 TMEM 中，但它们不是相同 layout。accumulator 使用 MMA output mapping。scale factor 使用 block-scaled MMA 期望的 compact layout。

## Scale Factor 在 `cta_group::2` 中

在 two-CTA case，scale factor 跟随它们 scale 的 data。

SFA scale A。由于 A 按 M 在 CTA pair 间 split，SFA 也按 M split。每个 CTA 持有对应其 A row 的 SFA row。

SFB scale B。由于两个 CTA 都乘以相同 B tile，SFB 必须对两个 CTA 可见。实际上，这意味着 SFB 在 CTA pair 间 multicast。

这是 block-scaled cluster GEMM 中 common loading pattern 的来源。SFA per CTA load，使用 CTA 自己 M slice 的 mask。SFB broadcast 到 pair，因为两个 CTA 都需要相同 N-side scale factor。

![Block-scaled MMA placement: A 和 B pack 在 SMEM；SFA、SFB 和 C 在 TMEM，SFA 按 M 在 CTA 间 split 且 SFB 在 CTA pair 间 multicast](../img/mma_block_scaled.svg)

## Keeping the MMA Contract 匹配

一个 Blackwell GEMM tile 通过几个 specialized path 移动。

TMA 将 A 和 B 从 global memory 带入 SMEM。对于 block-scaled mode，它还带 scale factor 到 SMEM。`tcgen05.cp` 在需要时将 scale factor 移到 TMEM。`tcgen05.mma` 读其 operand、在 Tensor Core 上异步运行，并将 accumulator 累加到 TMEM。completion barrier 告诉 kernel accumulator 何时 ready。epilogue 然后用 `tcgen05.ld` 加载 accumulator 从 TMEM 回 register 并存储最终 output。

在这些 path 中，kernel 必须保持三个 contract 匹配：SMEM operand layout、TMEM accumulator 或 scale-factor layout，以及使 next consumer 安全运行的 asynchronous completion signal。
