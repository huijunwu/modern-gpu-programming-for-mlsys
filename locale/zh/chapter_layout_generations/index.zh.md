(chap_layout_generations)=
# Tensor Core Operand Layout 跨 GPU 代际

:::{admonition} Overview
:class: overview

- 跨 Ampere、Hopper 和 Blackwell，Tensor Core 仍执行相同 high-level operation：`D = A B + C`。
- 从一代到下一代变化的是 operand 如何到达 Tensor Core、支持哪些 tile shape 和 dtype，以及 accumulator 存放哪里。
- Ampere 使用 warp-level register fragment。shared memory tile 用 `ldmatrix` 加载到 fragment，accumulator 保持在 register 中。
- Hopper 使 `wgmma` 通过 matrix descriptor 直接从 shared memory 读 operand。descriptor 命名 Tensor Core 期望的 shared memory swizzle format。
- Blackwell 保持 shared memory operand 路径但将 accumulator 移到 TMEM。block-scaled MMA 也将其 scale factor stage 通过 TMEM。
- 两个 memory constraint 在所有代际中都仍然存在：global memory coalescing 和 shared memory bank conflict。
:::

从远处看，Tensor Core operation 看起来稳定。它乘 A 和 B 的 tile、加 accumulator C、产生 D。那个 form 自 Volta 以来相同。

围绕那个 operation 的 detail 没有保持固定。一个在一代上快的 kernel 可能在下一代上慢。一个使用错误 layout 的 kernel 也可能计算错误 answer，即使 logical math 仍说 `D = A B + C`。原因是 Tensor Core 不消费 abstract matrix。它消费非常 specific hardware layout 中的 operand。

本章跟踪那个 layout contract 跨三代代际。Ampere 通过 warp-level register fragment 暴露 Tensor Core。Hopper 将 input operand 移到 shared memory descriptor。Blackwell 保持 shared memory operand 但将 accumulator 移到 TMEM。operation 仍是 matrix-multiply-accumulate，但进入和离开 Tensor Core 的 path 每次都改变。

{ref}`Data Layout <chap_data_layout>` 章的 layout notation 是我们描述这些 contract 的语言。Blackwell TMEM detail 在 {ref}`chap_tmem` 中单独覆盖。

## 从未消失的两个 Constraint

在 Tensor Core 参与前，两个普通 memory constraint 已经塑造 GPU kernel 的 layout。

第一个是 global memory coalescing。当 warp 的 32 lane 发出 global memory load 时，memory system 希望 address 落入少量 contiguous、aligned memory segment。如果 address 分散，warp load 变成多个 memory transaction。相同 logical data movement 消耗更多 bandwidth 和更多 time。

第二个是 shared memory bank conflict。shared memory 分为 32 bank。如果 warp 中 lane 访问映射到相同 bank 的不同 address，那些 access 不能一次性全部 serve。hardware serialize 它们。一个作为 flat shared memory array 看起来无害的 layout 因此可能因为 bank pattern 而慢。

swizzling 是修复 shared memory 侧的通常方式。logical tile 保持相同，但 physical address mapping permute，使 access pattern spread across bank 而不是 stack 到一个 bank。

这两个 constraint 甚至适用于从不使用 Tensor Core 的 kernel。Tensor Core kernel 添加第三个 constraint：operand 必须排列在 Tensor Core instruction 本身期望的 layout 中。本章其余内容是关于那个第三个 constraint 如何跨 Ampere、Hopper 和 Blackwell 改变。

## Ampere: Warp Lane 上的 Register Fragment

在 Ampere-class GPU 上，main Tensor Core instruction 是 warp-level `mma.sync.aligned.m16n8k*` family。重要 fact 是 instruction 读写 data 的位置：register。

A、B 和 C 或 D accumulator 都是分布在 warp 32 lane 上的 per-thread register fragment。shared memory 仅是 staging area。在 MMA 能运行前，operand tile 必须从 shared memory 移到 instruction 期望的 exact register fragment layout。

data path 如下所示：

```text
SMEM 到 register 用 ldmatrix
register 到 register 用 mma.sync
register 回 SMEM 用普通 store
```

大多数 Ampere layout story 跟随这个 path。kernel 必须将 tile store 在 shared memory 中为可高效 load 的 form，然后用 `ldmatrix` 产生 `mma.sync` 所需的 register fragment。

## Ampere Tensor Core 期望什么

Ampere Tensor Core 读由 8 × 8 subtile unit 构建的 register fragment。这些是 `ldmatrix` load 且 MMA 消费的 unit。

取 `mma.m16n8k16`，fp16 或 bf16 input 且 fp32 accumulation 作为 concrete case。accumulator tile shape 为 `16 × 8`。它以固定 pattern 分布在 32 lane 上。

对于 C 或 D accumulator，lane `l` 持有 row：

```text
l / 4
l / 4 + 8
```

和 column：

```text
2 * (l % 4)
2 * (l % 4) + 1
```

因此每个 lane 拥有四个 fp32 accumulator value：来自两个 8-row half 的两 row，与两 adjacent column cross。四 consecutive lane 覆盖一个 row 的八 column。

A operand 使用相同 M-side row carve。K dimension spread 在 `l % 4` 和 lane 持有的 register 上。对于 fp16 或 bf16，每 32-bit register pack 两个 K value。

B operand 使用 matching K placement 并将 N side spread 在 lane group 和 register 上。

exact detail 随 instruction shape 和 dtype 变化，但 principle 固定。Tensor Core 期望特定 per-lane register fragment。如果 value 不在那些 register 中按那个 pattern，instruction 将乘错 element。

在 layout notation 中，m8n8 fragment 是这种用 named lane axis 写的 pattern，例如：

```text
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@m)]
```

两个 `laneid` 一起描述 row 和 column piece 如何在 lane 间 scatter，最终 `m` 成分描述 per-lane register slot。

## `ldmatrix`: Shared Memory 到 Register Fragment

`ldmatrix` 是 Ampere instruction 连接 shared memory 和 Tensor Core register fragment。它是 warp-collective load。一个 instruction 将一个或多个 8 × 8 16-bit matrix 从 shared memory 移到 `mma.sync` 期望的 distributed register layout。

instruction form 是：

```text
ldmatrix.sync.aligned.m8n8.x1.shared.b16
ldmatrix.sync.aligned.m8n8.x2.shared.b16
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

带可选 `.trans` qualifier。

`x1`、`x2`、`x4` 控制每 instruction 移动多少 8 × 8 matrix。`x1` 移动一个 8 × 8 matrix。`x2` 移动两个 8 × 8 matrix，使两个 8 × 8 或一个 8 × 16 matrix 可在一次 load 中 move。`x4` 移动四个 8 × 8 matrix。

`.trans` qualifier 控制是否 transpose load 的 8 × 8 matrix。这对 A 和 B operand 都重要，因为 MMA 期望特定 transpose state。

## Hopper: Shared Memory Descriptor 和 `wgmma`

Hopper 引入 `wgmma`，一个 warpgroup-level MMA instruction。它不消费 register fragment。相反，它从 shared memory 读 operand 通过 matrix descriptor。

descriptor 描述 shared memory 中的 operand layout。它记录 tensor shape、stride、element type、swizzle mode 和 swizzle type。instruction 然后使用 descriptor 从 shared memory load tile 并执行 MMA。

这意味着 kernel 不再需要 `ldmatrix`。operand 留在 shared memory 中，MMA 直接读它。shared memory 不再是 staging area。它是 operand 的 home。

这简化了 data path：

```text
SMEM 到 SMEM 通过 wgmma 读
SMEM 到 register 通过 wgmma 写
register 到 SMEM 通过普通 store
```

accumulator 仍在 register 中，但 operand 现在在 shared memory 中。

descriptor 的关键 role 是它命名 shared memory swizzle。kernel 必须将 operand tile store 在 shared memory 中为 descriptor 声明的 swizzle。如果 tile 以不同 swizzle store，MMA 读错 value。

```text
descriptor.swizzle == SW32_4X4  表示 32-byte swizzle
descriptor.swizzle == SW64_4X4  表示 64-byte swizzle
descriptor.swizzle == SW128_4X4 表示 128-byte swizzle
```

## Blackwell: TMEM Accumulator 和 Block-Scaled MMA

Blackwell 保持 shared memory operand 路径但将 accumulator 移到 TMEM。`tcgen05.mma` 从 shared memory 读 A 和 B 并将 accumulator 写入 TMEM。

这意味着 epilogue 必须从 TMEM load accumulator 到 register。`tcgen05.ld` 执行那个 load。

此外，Blackwell 引入 block-scaled MMA。这种 mode 允许 low-precision format，如 `mxfp8` 和 `nvfp4`，使用 block-level scale factor 来扩展 precision。scale factor 存放在 TMEM 中。

这意味着 kernel 必须管理两个 TMEM allocation：accumulator 和 scale factor。

```text
A, B:     global memory 到 SMEM 通过 TMA
SFA, SFB: global memory 到 SMEM 到 TMEM 通过 tcgen05.cp
C:        TMEM 通过 tcgen05.mma 写
```

accumulator layout 和 scale factor layout 在 TMEM 中不同。accumulator 使用 MMA output mapping。scale factor 使用 compact layout（32 Lane row，`warpx4` broadcast）。

## 两个 Constraint 仍然存在

尽管 layout 跨代际改变，两个 basic memory constraint 仍然存在。

第一个是 global memory coalescing。kernel 必须安排 global memory access 以 minimize transaction。TMA 帮助这个，但它仍要求 descriptor 正确描述 tensor layout。

第二个是 shared memory bank conflict。无论 operand 是加载到 register 还是直接由 Tensor Core 消费，shared memory 的 bank layout 影响 access speed。swizzle 仍然是修复 bank conflict 的主要工具。

## 总结

Tensor Core operation 在 high-level 上稳定：`D = A B + C`。path 进入和离开 Tensor Core 随代际改变。Ampere 使用 register fragment。Hopper 使用 shared memory descriptor。Blackwell 使用 shared memory operand 和 TMEM accumulator。

layout 是使每个 generation 工作的 contract。kernel 必须将 operand 排列在 hardware 期望的 layout 中。layout notation 是描述那个 contract 的语言。
