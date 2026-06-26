(chap_tmem)=
# Special Memory: TMEM

:::{admonition} Overview
:class: overview

- TMEM 是一个 Blackwell-only memory space，被 `tcgen05` 使用。它是每个 SM 上的 2D scratchpad，有 128 Lane row 和最多 512 Col column。
- `tcgen05.mma` 将其 accumulator 写入 TMEM。block-scaled MMA 也使用 TMEM 存放 scale factor。
- TMEM 由 Lane 和 Col 寻址。在 TIRx layout notation 中，这两个 hardware axis 写为 `TLane` 和 `TCol`。
- TMEM 不像 register 那样自动分配。kernel 必须显式 allocate 和 free 它，单位是 32 column。
- 普通 shared memory load 和 store 无法访问 TMEM。data 在 TMEM、register 和 shared memory 之间通过专用 asynchronous `tcgen05` instruction 移动。
:::

在 Hopper 及更早的 GPU 上，Tensor Core（{ref}`chap_tensor_cores`）accumulator 存放在 register 中。这种 model 容易推理。MMA instruction 产生一个 register fragment，kernel 在 compute phase 中保持该 fragment live，epilogue 后来读取它、转换它并存储结果。

问题是 register pressure。register 是固定 per-thread resource。随着 MMA tile 变大，accumulator fragment 也变大。在某一点，accumulator 开始 crowd out thread 需要保持的其他值。更大的 tile 对 Tensor Core throughput 有好处，但将整个 accumulator 保持在 register 中使那些更大的 tile 更难使用。

Blackwell 改变了这个 data path 部分。`tcgen05` 的 accumulator 不需要在整个 compute phase 留在 register 中。相反，`tcgen05.mma` 将 accumulator 写入 Tensor Memory 或 TMEM。TMEM 是早期 NVIDIA GPU 没有的 memory space。它是 SM 上的 2D scratchpad，形状为 128 Lane row × 最多 512 Col column，并且 scoped 到使用它的 CTA。

那个额外的 memory space 使 Blackwell 能够支持更大的 Tensor Core tile 而不迫使整个 accumulator 进入 per-thread register。但 TMEM 不像 register 那样是自动的。compiler 不会简单地将其作为普通 register storage 分发。kernel 必须 allocate TMEM、用正确的 layout 寻址它、用正确的 instruction 移入移出 data，并在 CTA 完成后 free 它。

## 2D Address Space

TMEM 不是一个 flat byte array。它是一个 2D address space。hardware 将它的两个 coordinate 命名为 Lane 和 Col。有 128 Lane row 和最多 512 Col column。每 Col 是一个 32-bit column。

那个 shape 很重要，因为 `tcgen05.mma` 使用这个 2D 结构将 accumulator 写入 TMEM。一个 TMEM location 由 Lane coordinate 和 Col coordinate 描述，而不是由单个 shared-memory-style byte offset 描述。

当 kernel 在 TIRx 中声明一个 TMEM buffer 时，它赋予该 buffer 在这两个 hardware coordinate 上的 layout。在 layout notation（{ref}`chap_data_layout`）中，我们将 TMEM Lane axis 写为 `TLane`，将 TMEM Col axis 写为 `TCol`。这些名称不是要取代 official hardware terminology。它们是 layout axis name，使 TMEM dimension 在 DSL 内显式化。

例如，一个 accumulator tile 可以写为：

```text
S[(128, N) : (1@TLane, 1@TCol)]
```

这表示 tile 沿 hardware Lane dimension 有 128 row，沿 hardware Col dimension 有 `N` column。在 layout notation 中，这两个 dimension 显示为 `TLane` 和 `TCol`。layout 是直接的：相邻 row 沿 `TLane` 移动，相邻 column 沿 `TCol` 移动。下图展示了那个 grid，hardware Lane 沿 128 row 向下，hardware Col 沿 column 横跨。

../../../../img/tmem_grid.png)

主要观点是 TMEM 是 tile layout story 的一部分。它不仅仅是 Tensor Core 的 hidden backing store。kernel 必须 name 那个 memory、从中分配 column，并使用与 `tcgen05` instruction 读写该 memory 的方式匹配的 layout。

## Allocation

在 kernel 可以使用 TMEM 之前，它必须 reserve 其中的空间。这与 register 不同。register 由 compiler 分配。TMEM 由 kernel 显式 allocate。

allocation 是 per CTA 完成的。CTA 中的一个 warp 请求一个 TMEM column range。请求以 32 column 为单位发出，requested column count 根据 hardware allocation rule 向上舍入。allocation 后，CTA 收到一个 base TMEM address。后来 `tcgen05` instruction 使用那个 base address 访问 reserved region。

将 TMEM 视为 budgeted CTA resource 是有益的，就像 shared memory 一样。CTA 拥有它已分配的 TMEM column。kernel 决定需要多少 column 用于 accumulator、scale factor 或 temporary staging。当 CTA 完成时，它必须 free 那个 allocation。

这使得 TMEM 成为 kernel resource planning 的一部分。更大的 accumulator tile 可能改善 Tensor Core throughput，但它消耗更多 TMEM column。block-scaled MMA 可能需要额外 TMEM space 用于 scale factor。kernel 必须使这些用途在 available TMEM budget 内，就像它必须在 SMEM budget 内 fit shared memory buffer 一样。

## Reading and Writing TMEM

普通 `ld.shared` 和 `st.shared` instruction 无法访问 TMEM。TMEM 是一个单独的 address space，因此 data 通过专用 `tcgen05` instruction 移动。

有三个主要路径。

第一条路径是 `tcgen05.ld`，它从 TMEM 加载 data 到 register。这是 epilogue 在 MMA phase 后使用的路径。accumulator 已在 TMEM 中产生，但 epilogue 通常需要 register fragment，以便它 cast、应用 elementwise operation 并存储最终结果。

在 DSL level，一个 TMEM load 分布在 warpgroup 上。它 lower 为四个 warp-level `tcgen05.ld` operation，每 warp 一个。每 warp 处理 128 TMEM Lane row 中的 32 个，因此四个 warp 一起覆盖完整 Lane dimension。在 layout notation 中，完整 dimension 是 `TLane` axis。

instruction 本身来自一个 load shape family，如 `.16x64b`、`.16x128b`、`.16x256b`、`.32x32b` 和 `.16x32bx2`，repeat factor 从 `.x1` 到 `.x128`。chosen shape 决定了读取多少 TMEM column 以及每个 thread 接收多少 register。

重要结果是 register fragment layout。对于 common epilogue path，lane `l` 从 TMEM row `l / 4` 和两个 column 接收值。这产生了与早期 generation 直接从 MMA 暴露的相同类型的 per-lane accumulator fragment（{ref}`chap_layout_generations`）。那种连续性很重要。它意味着 Blackwell epilogue 可以复用相同 register-level cast 和 store structure，这些结构已用于 Ampere `mma` 或 Hopper `wgmma`，即使 accumulator 在 compute phase 期间存放在 TMEM 中。

../../../../img/tcgen05_ldst.svg)

第二条路径是 `tcgen05.st`，它将 data 从 register 存回 TMEM。这是 `tcgen05.ld` 的反方向。当 thread 已持有 register fragment 并需要将其放入 TMEM 时使用它。例如，某些 operand 或 intermediate value 可能在写入 TMEM 供 later `tcgen05` operation 之前 stage through register。

第三条路径是 `tcgen05.cp`，它从 shared memory 复制 data 到 TMEM。这是一个 bulk copy path，常用于 block-scaled MMA 中的 scale factor。在这种情况下，TMA 或普通 thread code 首先在 shared memory 中准备 scale data，`tcgen05.cp` 将其移到 Tensor Core 期望的 TMEM layout。

所有三个路径都是 asynchronous。一个 `tcgen05.ld`、`tcgen05.st` 或 `tcgen05.cp` instruction 可以在 data movement 完成之前返回。因此，kernel 必须在消费结果或复用 storage 之前使用正确的 completion mechanism（{ref}`chap_async_barriers`）。

wait path 取决于 instruction。`tcgen05.ld` 通过 `tcgen05.wait::ld` 完成。`tcgen05.st` 通过 `tcgen05.wait::st` 完成。`tcgen05.cp` 像 `tcgen05.mma` 一样，通过 commit group 和 `mbarrier` 完成。如果 data 从一个 thread set 传送到另一个，kernel 可能还需要 fence，使 receiving thread 按 intended order 看到 completed write。

TMEM 位于 Blackwell Tensor Core data path 中间。TMA 将 operand stage 到 shared memory。`tcgen05.mma` 读取 operand 并将 accumulator 累加到 TMEM。对于 block-scaled MMA，scale factor 也可以 stage 到 TMEM。compute phase 之后，`tcgen05.ld` 将 accumulator 带回 register，epilogue 转换并存储最终 output。
