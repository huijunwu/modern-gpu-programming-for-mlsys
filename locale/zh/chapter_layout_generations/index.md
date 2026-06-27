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

`.x1`、`.x2`、`.x4` form load 一个、两个或四个 8 × 8 matrix。row base address 由 lane 提供。对于 matrix `m` 和 row `r`，base address 来自 lane `m * 8 + r`。这意味着 `.x1` 使用 lane 0 到 7 作为 row address，`.x2` 使用 lane 0 到 15，`.x4` 使用 lane 0 到 31。

result 直接落入 MMA fragment。对于基本 8 × 8 case，lane `l` 接收 Tensor Core 期望的 row 和 column pair。一个 plain loop 的 per-lane `ld.shared` instruction 必须手动重现那个 scatter。`ldmatrix` 作为一条 warp-collective instruction 执行 shared-memory-to-fragment rearrangement。

`.trans` form 在 load 时 transpose 每个 8 × 8 matrix。这在 operand 以与 MMA instruction 期望相反的 orientation store 时使用。

![ldmatrix load 一个 8×8 shared memory tile 到 warp register fragment；Ampere 上 reverse direction 使用普通 store，专用 stmatrix instruction 后来在 Hopper 上出现](../img/ldstmatrix.svg)

## 写回 Ampere Fragment

`mma.sync` 完成后，accumulator 仍是 register fragment。epilogue 必须将那个 fragment 移出。

在 Ampere 上，没有 `ldmatrix` 的专用 reverse。kernel 使用普通 per-thread store，有时在 store 前用 warp shuffle 或 local rearrangement，将 accumulator 以有用 layout 写入 shared memory 或 global memory。

这保持 Ampere model 简单但也暴露大量 layout work 给 kernel。input 侧使用 `ldmatrix` 创建 fragment。compute instruction 读和写 register fragment。output 侧由那些 fragment 的普通 store 处理。

## Ampere 上的 Swizzle

Ampere kernel 已经需要 shared memory swizzle。原因是 shared memory tile 通常以一个 access pattern write 并以另一个 read。

假设 tile 沿 row 从 global memory fill。row-major layout 使那个 write coalesced 且 bank friendly。但 `ldmatrix` 可能后来以有效沿 column 走或跨 8 × 8 subtile 的 pattern read tile。使用 plain row-major layout，那些 read 可能 stack 到相同 shared memory bank。

对于简单 `(8, 64)` float16 tile，一个 row 是：

```text
64 * 2 bytes = 128 bytes
```

正好是一个完整 shared memory bank line。沿固定 column 走每 row 前进 128 byte，因此 bank index 重复。八 row 可 collapse 到相同 bank，创建 8-way conflict。

改为 plain column-major layout 不解决整个问题。它通常将 conflict 移到另一个 access。row write 变得更差而 column-style read 变得更好。

XOR swizzle 通过使 physical column 依赖于 row 来修复这个问题。一个简单 version 是：

```text
physical_col = logical_col xor row
```

logical tile 不变。shared memory 中的 physical placement permute，使 row-style write 和 Tensor Core read pattern 都能避免 bank conflict。

在 Ampere 上，这个 swizzle 通常通过 hand-written shared memory index math 表达。后来代际使其成为 hardware engine 使用的 descriptor format 的一部分。

![在 plain row-major tile 上，row write spread across bank 而 column read 在一个 bank 上 collide；XOR swizzle 将 column read scatter across bank 而不放弃 coalesced row write](../img/swizzle_conflict.svg)

## Hopper: `wgmma`、Shared Memory Descriptor 和 Swizzle Format

Hopper 改变 Tensor Core path 的 input 侧。代替要求每个 operand 用 `ldmatrix` load 到 register，Hopper `wgmma` 可直接从 shared memory 读 operand。

B operand 从 shared memory matrix descriptor 读。A operand 可从 shared memory descriptor 或 register 读，给出 `.ss` 和 `.rs` form。

这移除了 SMEM-sourced operand 的显式 `ldmatrix` step。它不移除 layout requirement。Tensor Core 仍期望 operand 以精确 shared memory format store。不同之处是 format 现在通过 matrix descriptor 描述给 hardware。

## Hopper Tensor Core 期望什么

Hopper shared memory matrix descriptor 是 shared memory 中 matrix tile 的 compact description。它告诉 `wgmma` 如何将 logical operand coordinate 转为 shared memory address。

descriptor 包括如以下 field：

```text
start address
leading dimension offset
stride dimension offset
swizzle mode
base offset
```

exact interpretation 取决于 operand major mode。对于 K-major tile，一个 stride 沿 K 前进，另一个沿 M 前进。对于 MN-major tile，角色互换。

swizzle mode 是 shared memory descriptor format 之一，如：

```text
SWIZZLE_NONE
SWIZZLE_32B
SWIZZLE_64B
SWIZZLE_128B
```

swizzle mode 决定两件事。它决定 descriptor 使用的 atom shape，它决定在那个 atom 内应用的 XOR permutation。例如，128-byte swizzle mode 将 operand 视为 8-row × 128-byte atom 的 grid，swizzle 在每个 atom 内应用。

kernel 仍必须正确放置 byte。TMA 通常 fill shared memory tile，TMA descriptor 必须使用 `wgmma` descriptor 后来命名的相同 swizzle format。如果 TMA write 128-byte swizzled tile，`wgmma` descriptor 必须将其作为 128-byte swizzled tile read。如果 descriptor 和 data 不一致，Tensor Core 将读 scrambled operand。

这是从 Ampere 的主要 shift。swizzle 不再仅隐藏在 hand-written shared memory indexing 内。Hopper 使其成为 first-class descriptor format。write tile 的 TMA load 和 read tile 的 `wgmma` instruction 都可命名相同 format。

![Hopper shared memory matrix descriptor 将 operand coordinate 映射到 swizzled shared memory atom：descriptor stride 选择 atom，swizzle 选择 atom 内的 byte position](../img/smem_descriptor.svg)

## Hopper Output 仍使用 Register

Hopper 改变 input path，但 accumulator 仍在 register 中。

`wgmma` instruction 将 accumulator 写入 per-thread register fragment。exact fragment size 和 register count 取决于 instruction shape，如 `m64nNk16`，其中 N 改变 accumulator register 数量。但 basic idea 与 Ampere 相同：epilogue 消费 register fragment。

因此 Hopper 有 mixed layout model。input operand 可直接来自 shared memory descriptor，swizzle 由 hardware 描述。output accumulator 仍是 register layout 问题。

Blackwell 改变那个 output 侧。

## Blackwell: `tcgen05` 和 TMEM

Blackwell 对 data operand 保持 shared memory descriptor idea。A 和 B 仍在 shared memory 中以 Tensor Core 期望的 layout 准备。某些 mode 也可从 TMEM 读 A operand。

major change 是 accumulator。`tcgen05.mma` 将 accumulator 写入 Tensor Memory（TMEM），而不是将其保持为 long-lived register fragment。在 compute phase 期间，accumulator 留在 TMEM 中。epilogue 后来使用 `tcgen05.ld` 将其 load 回 register。

这将 output layout 问题从 register 移到 TMEM。kernel 必须 allocate TMEM、选择正确 TMEM layout、等待 MMA completion，然后用 matching `tcgen05.ld` path 恢复 accumulator fragment 供 epilogue 使用。

`cta_group::1` 和 `cta_group::2` 如何在单个或多个 CTA 间 split accumulator 的 detail 在 {ref}`chap_tensor_cores` 中覆盖。与之前代际最不同的 layout 是 block-scaled scale-factor layout。

## TMEM 中的 Scale Factor Layout

block-scaled MMA mode，如 `mxfp8` 和 `nvfp4`，添加 scale-factor operand。除 A 和 B 外，MMA 读：

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中 `SFK` 是 K scale block 数量。

data operand A 和 B 在 shared memory 中。scale factor 在 TMEM 中。这给它们不同的 movement path。

TMA 从 global memory load 到 shared memory。它不直接 load 到 TMEM。因此 scale factor 通常分两步移动：

```text
global memory 到 shared memory 用 TMA
shared memory 到 TMEM 用 tcgen05.cp
```

只有在那个 copy 之后，scale factor 才在 `tcgen05.mma` 期望读它们的 memory space 中。

TMEM scale-factor layout 使用 TMEM hardware coordinate Lane 和 Col。在 TIRx layout notation 中，那些 axis 写为 `TLane` 和 `TCol`。

128-row scale vector 被 compact 到 32-lane group 然后在 TMEM 的四个 32-lane window 间 replicate。在 layout notation 中，core pattern 是：

```text
S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
```

shard 放置 base 32-row group：

```text
TLane = r
TCol  = s
```

replica term 在 lane offset 0、32、64 和 96 添加 copy：

```text
TLane = r + 32 * q, 其中 q in {0, 1, 2, 3}
TCol  = s
```

这是 `warpx4` broadcast pattern。相同 compact scale-factor group 在完整 128-lane TMEM space 中可见。

32-bit `TCol` cell 内还有 byte packing。packing 取决于 `scale_vec` mode：

```text
1X: 一个 scale value broadcast 跨 32-bit cell
2X: 两个 scale value pack，每个 duplicate
4X: 四个 K-block scale value pack
```

![scale_vec byte packing：1X 将单个 scale broadcast 跨 4-byte cell；2X pack 两个 scale，每个 duplicate；4X pack 四个 K-block scale](../img/sf_scale_vec.svg)

这种 packing 在 Ampere 或 Hopper 上没有直接对应物，因为那些代际没有 TMEM scale-factor operand 用于 `tcgen05` block-scaled MMA。

在 `cta_group::2` 中，scale factor 跟随它们 scale 的 data。SFA scale A，因此它按 M 在两个 CTA 间 split，匹配每个 CTA 拥有的 A row。SFB scale B，B 由计算的两个 CTA half 共享，因此 SFB multicast 到两个 CTA（{ref}`chap_tensor_cores`）。

## 反复出现的 Fragment

尽管周围 memory path 改变，一个 structure 不断返回：m8n8-style register fragment。

在 Ampere 上，`ldmatrix` 构建那个 fragment 供 `mma.sync` read。

在 Hopper 上，`wgmma` 将 accumulator 写为 register fragment 供 epilogue 使用。

在 Blackwell 上，accumulator 在 compute 期间留在 TMEM 中，但 `tcgen05.ld` 在 epilogue 处理和 store 之前将其 load 回 register fragment（{ref}`chap_tmem`）。

因此 fragment 没有消失。它的 role 改变。早期代际在整个 compute phase 将 accumulator 保持在那里。Blackwell 主要在 TMEM 和 epilogue 的 boundary 使用它。

## 贯穿线

在 Ampere 上，kernel 显式构建 Tensor Core register fragment。shared memory swizzle 主要通过 index math 是 kernel 的 responsibility。

在 Hopper 上，Tensor Core 可通过 matrix descriptor 直接从 shared memory 读 operand。swizzle 成为 TMA 和 `wgmma` 共享的 named descriptor format。

在 Blackwell 上，input 侧仍使用 shared memory operand，但 accumulator 移到 TMEM。block-scaled MMA 还添加必须 stage 到 TMEM 的 scale-factor operand。

descriptor 不移除 layout work。它们使 contract explicit。kernel 仍必须确保 data movement path、memory layout 和 Tensor Core instruction 都一致。write swizzled SMEM tile 的 TMA descriptor、read 那个 tile 的 MMA descriptor 和附加到 buffer 的 layout 必须都描述相同 physical arrangement。

如果那些 pieces 中任何一个不一致，hardware 仍会运行。但它将读错 byte 或慢速读它们。这就是为什么 layout 不是 Tensor Core kernel 周围的 decoration。它是 instruction interface 的一部分。
