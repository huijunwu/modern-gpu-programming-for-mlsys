(chap_data_layout)=
# Data Layout 及其 Notation

:::{admonition} Overview
:class: overview

- 一个 *data layout* 将 tensor 的 logical index 映射到 physical location，它决定了 coalescing、bank conflict 以及 engine 能否读取 tile。
- 本书使用一种 notation 来写 layout：`S[(shape) : (strides)]`，带有 named axis（`@laneid`、`@TLane`、…）和一个 replication term `R[...]` 用于 broadcast 或 copied data。
- swizzle 是一种 address 的 XOR remapping，消除了 shared memory bank conflict。
:::

相同的数字，以不同的 physical arrangement 写入 memory，在同一个 GPU 上的运行速度可能相差一个数量级。

原因是 tensor 的 logical index 对它的 byte 实际存放在哪里没有说明。hardware 对那个 placement 高度敏感：它决定了 32 lane 的 load 是 coalesce 为一个 transaction 还是散成 32 个，它们的 address 是落在不同的 memory bank 还是碰撞并 serialize，甚至一个 tile 是否根本匹配 Tensor Core 能读取的 byte arrangement。

机器学习程序通常通过 logical shape 来描述 tensor。一个 **data layout** 补充了缺失的 physical 部分：它说明了 logical index 为 `(i, j, …)` 的 element 位于哪里，无论是在 memory 中、register 中，还是其他 hardware storage 中。

本章介绍了现代 GPU 编程中出现的主要 layout。为了保持讨论的可处理性，我们开发了一种紧凑的 **notation**，用于描述机器学习 system 遇到的各种情况。我们最后介绍 **swizzling**，它是使对一个 tile 的行访问和列访问同时高效的技术。

## Shape–Stride Model

在我们进入 GPU-specific layout 之前，值得从最简单的开始，因为本章的其余内容都建立在其之上。在核心，一个 layout 只有两件事：一个 **shape** 和匹配的一组 **stride**。我们将这对写为 `S[(shape) : (stride)]`，为了找到 logical index 的位置，我们计算该 index 与 stride 的点积。例如，一个 row-major 4×4 矩阵如下所示：

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

这不过是以紧凑方式写出的经典 shape/stride model（CuTe notation 的 row-major 简化版本），后面所有内容都由此构建。

事实上，你几乎肯定已经使用过这个 model。任何写过 PyTorch 或 NumPy 的人都用过，因为这些库中的 tensor 恰恰就是一个 shape 加上一个 stride，覆盖一个 flat storage buffer：

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← 恰好是 S[(3, 4) : (4, 1)]
```

一旦你以这种方式看待 tensor，就很清楚为什么那么多"reshape" operation 完全不会 touch data。它们只是重写 stride，并返回对同一个 storage 的 **view**，最清晰的例子是 transpose 或 permute：

```python
tt = t.permute(1, 0)               # 或 t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← stride 互换，没有 data 移动
tt.data_ptr() == t.data_ptr()      # True，相同的 byte
```

这里 `t.permute(1, 0)` 在 *相同* memory 上是 `S[(4, 3) : (1, 4)]`：transpose 纯粹是 stride 的变化，没有一个 byte 移动。对 contiguous tensor 的 `reshape` 或 `view` 的故事相同：对旧 storage 的新 shape 和新 stride。（NumPy 行为相同；唯一区别是其 `.strides` 以 byte 计数而非 element。）

这正是 layout 在 GPU 上工作的方式，本章的其余内容实际上是对一个 idea 的一系列变体：tile 的 mapping（无论是到 memory，还是通过我们即将介绍的 named axis 到 lane 和 register）是固定 buffer 上的 stride rule，因此重新排列 tile 通常是一种 *layout* 变化而非 copy。不过，我们应该小心这种推理的边界。zero-copy 故事在单个 linear address space 上的 logical view 上清晰成立；在 GPU 上，它仅在新 view 与现有 byte 和 ownership arrangement 兼容时适用。一旦你改变哪个 thread 或 register 拥有 element，或改变 SMEM swizzle，你通常需要 real data movement：load、store、shuffle、`ldmatrix`、transpose。

## Tile Layout

到目前为止，我们描述了整个 tensor 的 layout。然而，GPU kernel 很少一次对整个 matrix 操作；它们工作在更小的 tile 上，这些 tile 被 hardware 的不同部分加载、变换和计算。好消息是 tiling 不需要新概念。它仍然只是一个 layout，只不过现在用更多维度来写。将一个 8×8 matrix 切成 2×4 tile，我们得到一个 4-D layout，coordinate 为 `(tile_row, row_in_tile, tile_col, col_in_tile)`，stride 选择使每个 tile 保持 contiguous：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

logical `(i, j)` 首先变为 `(i//2, i%2, j//4, j%4)`，然后通过 stride。值得注意的是，该 notation 表达 tiling 时根本不需要任何特殊的"tile"概念：它与之前相同的 shape–stride model，只是 index 被分为 outer 和 inner coordinate。

下面的 interactive visualization 展示了 logical matrix index 如何分解为 tile coordinate 然后映射到 physical address。

```{raw} html
<iframe src="../../_extra/demo/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: 点击 cell 查看其 tiled index 和 address。*

## Named Axis

到目前为止，`S[...]` 中的每个 stride 都命名为 linear memory 中的 offset，我们将 address 视为那里的 location。不过在 GPU 上，data 可以存放在不止一个地方：除了 memory，tile 可能分布在 warp lane、thread register 或 TMEM lane 和 column 上。为了统一描述所有这些情况，我们用 **named axis** 扩展 notation。思路是让每个 stride coefficient 携带一个 axis tag，说明它通过哪个 space 移动：`@m` 用于普通 memory，`@laneid` 用于 warp lane，`@reg` 用于 register，`@warpid` 用于 warp，`@TLane` / `@TCol` 用于 TMEM coordinate。有了 tag，单个 layout 不仅可以描述 data 在 memory 中的位置，还可以描述它如何分布在对其操作的 hardware resource 上。

一旦 memory tag 显式化，memory 中的 row-major 8×16 tile 就是简单的

```text
S[(8, 16) : (16@m, 1@m)]
```

当 layout 描述 *分布在 thread 上* 而非放在 memory 中的 data 时，tag 开始发挥作用。例如 `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]`：它不再指向 linear memory，而是将 row 和 column 映射到 lane ID 和 per-lane register。这里 `laneid` 表示 warp 内的 warp lane index，即 `thread_index % warp_size`。这正是在 {ref}`chap_layout_generations` 中你将遇到的 tensor-core register fragment。

下面的 interactive visualization 展示了 layout 如何如何将 tensor element 分布在 warp lane 和 per-lane register 上，而不是将它们放在 linear memory 中。

```{raw} html
<iframe src="../../_extra/demo/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: 一个在 `@laneid` 和 `@reg` 上的 layout；点击 cell 查看哪个 lane/register 持有它。*

## Distributed Layout

使 named axis 如此有用的是，它们使我们能够在 system 的多个层级统一描述 placement，包括 *across whole device* 的 placement。我们刚刚在单个 GPU 内用于 lane 和 register，但完全相同的 idea 向外延伸：像 `@gpuid_x` 和 `@gpuid_y` 这样的 axis 可以说 data 在 GPU mesh 中的位置，有了它们，notation 就捕捉到了 distributed training 和 inference 中出现的 sharding pattern。axis 尚未捕捉的一件事是 *replication*——复制到多个地方的 data，所以我们添加了 notation `R[n : stride]`，其中 `R` 标记 replicated dimension。例如，`R[2 : 1@gpuid_x]` 描述沿 `@gpuid_x` axis 的 replication。将两者结合起来，单个表达式可以同时将 tensor sharding 到 2×2 GPU mesh 并沿一个 axis 复制：

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

下面的 demo 在一个小型 GPU mesh 上展示了这种 combined partition-and-replication pattern。点击任意 cell 查看哪个 device 持有它，并观察 `@gpuid_x` replication 如何将一个 identical copy 放置到配对的 device 上；按钮在 fully-sharded、shard + replica 和 shard + offset layout 之间切换。

```{raw} html
<iframe src="../../_extra/demo/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: 一个分布在 2×2 GPU mesh 上的 layout；点击 cell 查看哪个 device(s) 持有它。*

### Intra-Kernel Replication Pattern: TMEM 中的 Scale Factor

我们刚刚为 GPU mesh 引入的 replication dimension `R[...]` 不仅关于多个 device。相同的 construct 还描述了完全发生在单个 kernel 内部的事情：hardware *broadcast across lane* 的 data。Blackwell 的 block-scaled MMA（{ref}`chap_layout_generations`）就是一个好例子。其 scale factor 存放在 TMEM 中，一个 128-row 的 scale vector 只存储在 **32 TMEM lane** 中，logical row `r` 去往 TMEM lane `r % 32`，`r // 32` 沿 column 运行。这 32 个 stored TMEM lane 然后在 **replicated along the TMEM `TLane` axis**，从 32 到 128 TMEM lane，使得读取 warpgroup 中每个 warp 在其 32-lane TMEM window 中找到一份副本。这是一种 `warpx4` broadcast，我们用 replication dimension 来写。读取由这些 warp 的 thread 执行：

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

这给出了 stride 为 32 TMEM lane 的四个 replica：TMEM lane `l`、`l+32`、`l+64` 和 `l+96` 都持有相同的 scale。与之前一样，replication dimension 不携带新 data；它只是说"相同的 value，位于四个 TMEM lane position"，就像刚才 `@gpuid_x` 在 GPU mesh 上 broadcast row 一样。

下面的 interactive demo 同时展示了两个步骤：紧凑 pack 到 32 TMEM lane，然后 `warpx4` broadcast 到 128 个 reading lane。

```{raw} html
<iframe src="../../_extra/demo/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; min-width:1040px; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: 点击 scale factor `SFA[m, sf]`；它 pack 到 TMEM 中 lane `m mod 32`，column `(m // 32)·4 + sf`，然后沿 `TLane` axis broadcast `warpx4` 到四个 lane copy（`l`、`l+32`、`l+64`、`l+96`），每个 warp 一个 32-lane window。*

每 column 内的 byte packing（`scale_vec` 1X/2X/4X mode）和 `cta_group::2` split 在 {ref}`chap_layout_generations` 中覆盖。

已经了解 CuTe 的读者可以将本章的 notation 视为它的 row-major 变体，扩展了显式 hardware-named axis 和专门的 replication structure。

## Swizzle Layout

本章最后一个 layout 存在以解决一个 specific hardware problem。GPU 上的 shared memory 组织成 memory bank，当不同 lane 落到不同 bank 时，access 运行最快。当多个 lane 反而到达 *相同* bank 内的不同 address 时，hardware 别无选择只能 serialize 它们，我们付出 **bank conflict** 的代价。

在 tensor program 中这很难避免，因为 memory 不是纯 linear order access 的。处理 matrix 时，我们通常需要读取同一 tile 的 row slice 和 column slice，这就产生了真实的 tension：对 row-wise access 高效的 layout 倾向于对 column-wise access 产生 bank conflict，而偏好 column 的 layout 会伤害 row。**Swizzling** 是设计用来打破这种 tension 的技术。

swizzle 的 idea 是 permute address mapping，通常将 column index 与 row XOR，使 *row 和 column access* 最终都 spread across bank。它提供的 conflict-free guarantee 是 specific 的：它适用于匹配的 element width、swizzle mode 和 access pattern（engine descriptor 期望的那个），而不适用于任意 element width 或 alignment。

第一个下面的 interactive demo 使这一点具体化。点击一个 column index 并观察每个 element 落到哪个 bank：在左侧的 plain row-major tile 中，一个 column 将所有 8 个 element 汇集到单个 bank，因此 read serialize 为 8 个 cycle；在右侧的 XOR-swizzled layout 中，相同 column spread 到 8 个不同 bank 并 single cycle 读取。

```{raw} html
<iframe src="../../_extra/demo/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: 一个 8×8 tile，plain row-major 中 column 出现 bank conflict，XOR swizzle 后 conflict-free。*

这个小 8×8 example 捕捉了核心 idea，但 real GPU memory 有更多的 bank 比那个 toy picture 暗示的。为了使 swizzling 在全尺度工作，我们不将整个 tile 视为一个 monolithic object。相反，我们将 memory 切成小 segment，并在每个 segment 内应用 swizzle pattern。实践中最常见的情况是 `SWIZZLE_128B`，围绕 128-byte segment 组织，使相同的 row/column-remapping trick 自然地适应 32-bank memory system。

下面的 interactive demo 展示了那个 specific hardware swizzle `SWIZZLE_128B`，使 repeating segment-by-segment pattern 在我们在 format 间泛化之前可见。

```{raw} html
<iframe src="../../_extra/demo/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: 128-byte segment 内的 `SWIZZLE_128B` pattern；通过 read cycle 步进，查看 `physical_sector = logical_sector XOR row` 如何将每个 column spread 到不同 bank。*

相同的 idea 扩展到 128-byte 情况之外。为了简化 visualization，我们现在用单个 color block 来指代一个 segment，而不是绘制 individual bank。一般而言，hardware 定义一个 small repeating **atom** 在其上应用 permutation，不同 swizzle mode 选择不同 atom size。`SWIZZLE_128B` 使用 8 × 128 B atom，`SWIZZLE_64B` 使用 8 × 64 B atom，`SWIZZLE_32B` 使用 8 × 32 B atom；然后整个 tile 用 whichever atom in use 来 tile。

最后一个 interactive demo 允许你在这几个 format 之间切换（包括一个 16 B interleaved mode），选择 data type，hover 任意 cell 直接 inspect 一个 atom 内的 element arrangement，这是推理 load/store instruction 期望哪个 swizzle 的恰当 detail level。

```{raw} html
<iframe src="../../_extra/demo/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: 选择 swizzle format（和 data type）查看其 atom shape（8 × N B）；hover cell 查看其 element 如何 permute。*

你应该选择哪个 mode？rule of thumb 是偏好 tile 能 fill 的 *largest* atom。一个 N-byte atom 需要 tile 的 contiguous dimension 至少 N 字节，且是其倍数，因此 `SWIZZLE_128B` 仅当 row 跨至少 128 字节（或 64 个 `float16` element）时适用。当它 fits 时，它是 default choice，因为其 8 × 128 B atom 覆盖一个 full 128-byte bank line，因此一次将 column scatter 到所有 32 bank，在 fp16 中一次提供 8 row 和 8 column 的 conflict-free access。不过，当 problem 的 shape 迫使 contiguous dimension 变小时，tile 不能再 fill 一个 128 B atom，你降到 `SWIZZLE_64B` 或 `SWIZZLE_32B`——row 仍能覆盖的 largest atom。

你从不会手工计算这些 permuted address，值得精确说明 swizzle 与 `S[...]` notation 的关系：它 *不是* 那个 affine map 的一部分。它是一个单独的、non-affine layer，composited on top of it。`S[...]` layout 将 element 放置在一个 linear memory（`@m`）address，然后 swizzle permute 那个 address，在 TIRx layout API 中写为 `ComposeLayout(swizzle, tile)`（{ref}`chap_tirx_layout_api`）。你的工作只是在选择一个 consistent mode 跨每个 touch 该 tile 的 op，然后让 composed layout 完成其余工作。

相同的 composed layout 也是 hardware 填充的，这就是 swizzling 和 tiling 汇合的地方。TMA descriptor 是多维的，因此单个 three-dimensional box 可以同时描述 tile 的 atom tiling 和每个 atom 内的 swizzle；一个 TMA load 然后 atom by atom 排列 tile 并在写入 shared memory 时 swizzle 它（{ref}`chap_tma`），没有单独的 swizzling pass。*每个* engine 需求 *哪个* swizzle 是 generation-specific，那是下一章的主题。
