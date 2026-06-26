(chap_tirx_layout_api)=
# TIRx Layout API

:::{admonition} Overview
:class: overview

- TIRx layout API 将 {ref}`chap_data_layout` 的 layout notation 转为 compiler object。main object 是 `TileLayout`、`SwizzleLayout` 和 `ComposeLayout`。
- `TileLayout` 描述 over named hardware axis 的 affine placement。它从 shard spec `S[...]`、replica spec `R[...]` 和 optional offset 构建。
- layout 将一个 logical coordinate 映射到一个或多个 physical coordinate。`layout.apply()` 评估那个 mapping。
- `SwizzleLayout` 描述基于 XOR 的 shared memory swizzle，用于避免 bank conflict。`ComposeLayout` 将 swizzle stack 在 tile layout 上。
- ready-made constructor 如 `tmem_datapath_layout`、`tcgen05_atom_layout` 和 `wg_local_layout` 覆盖 kernel 中反复出现的 hardware layout。
:::

{ref}`chap_data_layout` 介绍了全书使用的 notation：tile shape、named axis 上的一组 stride，和 optional replication term 用于 copied 而非 partitioned 的值。本章将那个 notation 转为 compiler 使用的 API。

goal 是页面上的 notation 和 kernel 中的 code 看起来几乎相同。当你写一个 layout 如：

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

你不仅写 explanation。你在构建一个 `TileLayout` object，可 attach 到 buffer。之后，每 touch 那个 buffer 的 tile operation 都可以从 layout 读取其 placement。placement 写一次、check 一次，并由 compiler 复用。

layout 在 pool allocate 时或 buffer declare 时 attach：

```python
pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

从那时起，buffer 携带其 physical placement。tile operation 不需要重复每 element 存放哪里。

layout object 位于一个 module：

```python
from tvm.tirx.layout import (
    TileLayout,
    SwizzleLayout,
    ComposeLayout,
    S,
    R,
    laneid,
    warpid,
    tid_in_wg,
    TLane,
    TCol,
    m,
    tcgen05_atom_layout,
    tmem_datapath_layout,
)
```

API 背后有一个 central idea。layout 不必须将 logical index 映射到单个 physical address。它将 logical index 映射到 named axis 上的一组 physical coordinate。通常情况那 set 有一个 element。当 replication present 时，相同 logical element 有几个 physical placement。

这就是为什么 layout model 有三个 piece：shard、replica 和 offset。shard place element。replica copy 它到 additional coordinate。offset shift 整个 placement。

## Layout 示例

下面 example 展示 API 的 basic shape。

TMEM 中的 accumulator 可写为 TMEM axis 上的 direct placement：

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])
```

这里 logical row 映射到 `TLane`，logical column 映射到 `TCol`。在 {ref}`chap_tmem` 中，hardware coordinate 称为 Lane 和 Col。在 TIRx layout notation 中，那些 hardware axis 写为 `TLane` 和 `TCol`。

block-scaled MMA scale-factor layout 使用 replication：

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)
```

shard 在 TMEM 中 place 一个 32-row group。replica 重复那个 group 四次，stride 为 32 lane，因此 32-row group 在完整 128-lane TMEM space 中 visible。

tensor-core register fragment 可 distribute 在 lane 和 warp 上：

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

相同 physical axis 可出现多于一。这个 example 中，两个不同 iter 都 contribute 到 `laneid`。没有 explicit axis 的 stride 使用 default memory axis `m`。

在 real kernel，常见 hardware layout 通常来自 constructor：

```python
acc = tmem_datapath_layout("D", 128, 256)

ld = tcgen05_atom_layout("32x32b", (128, 64), "float32")
```

这些 constructor 返回普通 `TileLayout` object。它们是 convenience，不是 separate mechanism。你可 inspect 返回 layout、与其他 layout compose，或在 shape unusual 时手写底层 `S[...]` 和 `R[...]` form。

## Interactive Demo

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../tirx-layout-demo/index.html" title="TIRx layout API interactive demo" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: 选择 layout 并 hover element 查看其 physical placement。*

## How Layout Applies

layout 将 logical coordinate 映射到 physical coordinate。`layout.apply()` 做那个 work。

`apply()` 接受 logical coordinate 并返回 physical coordinate。对于 shard-only layout，它返回单个 coordinate。对于有 replication 的 layout，它返回 coordinate list。

在 code，`apply()` 是：

```python
result = layout.apply((i, j, ...))
```

那 result 可用于 compute address 或 verify placement。

## Swizzle Layout

swizzle 是 shared memory address 的 XOR remapping。它使 row 和 column access 都 conflict-free。

在 TIRx，swizzle 是 `SwizzleLayout`。`SwizzleLayout` 描述 swizzle 的 XOR transform。

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

那些 parameter 控制 XOR mask 如何 compute：

```text
per_element:  log2 of element size in bytes (e.g. 3 for fp16, 2 for fp32)
swizzle_len:  log2 of swizzle width in elements (e.g. 3 for 8 elements)
atom_len:     log2 of atom height in elements (e.g. 3 for 8 elements)
```

swizzle 是 non-affine transform。它不是 shape-stride map。它是 XOR permutation。

## Compose Layout

`ComposeLayout` 将 swizzle stack 在 tile layout 上。result 是 composite layout：

```python
composed = ComposeLayout(swizzle, tile)
```

那 layout 可用于 buffer 或 pool allocation。当 buffer 以 compose layout alloc，compiler 将 swizzle 应用于 tile placement。

那 result 是 element 在 shared memory 中 swizzle。swizzle 由 TMA 或 thread code 应用，取决于 dispatch path。

## Ready-Made Constructor

大多数 kernel 使用 common hardware layout。TIRx 提供 constructor 用于那些 layout：

```python
tcgen05_atom_layout(shape_kind, shape, dtype)
tmem_datapath_layout(operand_kind, M, N)
wg_local_layout(shape, dtype, ...)
```

那些 constructor 返回 `TileLayout` object。它们是 convenience。你可总是手写 `S[...]` 和 `R[...]]`。

### `tcgen05_atom_layout`

`tcgen05_atom_layout` 构造 `tcgen05` MMA 的 operand layout。它返回 A 或 B operand 的 layout，以匹配 MMA instruction 期望的 layout。

```python
A_layout = tcgen05_atom_layout("m16n8k16", (128, 64), "float16")
```

那个 layout 描述 operand 在 shared memory 中的 placement，以匹配 MMA 期望的 layout。

### `tmem_datapath_layout`

`tmem_datapath_layout` 构造 TMEM accumulator 的 layout。它返回 accumulator 在 TMEM 中的 placement。

```python
acc = tmem_datapath_layout("D", 128, 256)
```

那 layout 匹配 `tcgen05.mma` 写入 TMEM 的 accumulator layout。

### `wg_local_layout`

`wg_local_layout` 构造 warp-level register fragment 的 layout。它描述 fragment 在 warp lane 和 register 间的 placement。

```python
frag = wg_local_layout((8, 4), "float32", ...)
```

那 layout 用于 `tcgen05.ld`，它将 accumulator 从 TMEM 加载到 register。

## Swizzle 公式

swizzle 对 shared memory address 应用 XOR permutation。formula 是：

```text
x    = m >> per_element
addr = ((x ^ ((x >> swizzle_len) & ((1 << atom_len) - 1))) << per_element) | (m & ((1 << per_element) - 1))
```

这里 `m` 是 linear address（以 element 计数）。swizzle 将 `m` 映射到 swizzled address。

对于 `per_element=3`、`swizzle_len=3`、`atom_len=3`，formula 简化为：

```text
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)
```

那 XOR pattern 使 row 和 column access 都 conflict-free。

### 为什么 XOR？

XOR permutation 是 swizzle 的关键 insight。row-major layout 使 row access 是 conflict-free，但 column access 在相同 bank 上碰撞。XOR permutation 使 column access spread across bank，同时保持 row access conflict-free。

关键 observation 是 XOR 是 self-inverse。`x ^ c ^ c = x`。这使 inverse permutation 简单：apply same XOR 将 swizzled address 映射回 original address。

### Swizzle 的 Bank Pattern

shared memory 分为 32 bank。每 bank 服务一个 4-byte（对于 fp16）或 8-byte（对于 fp32）access。

swizzle formula 使 element 在 bank 间 spread。对于 `(8, 64)` float16 tile：

```text
bank = floor(addr / 2) mod 32
```

row access（`i` 变化，`j` 固定）仍 conflict-free。column access（`j` 变化，`i` 固定）现在也 conflict-free，因为 XOR spread column 在 bank 间。

## Worked Example: 128B Swizzle 在 `(8, 64)` float16 Tile

回到 row-major float16 tile：

```text
m = 64 * i + j
```

使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

transform 变为：

```text
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)
```

由于：

```text
m = 64 * i + j
```

我们可以写：

```text
q = floor(j / 8)
r = j mod 8
```

swizzled address 是：

```text
addr = 64 * i + 8 * (q xor i) + r
```

现在看 column `j = 0`。那 `q = 0` 和 `r = 0`，所以：

```text
addr = 72 * i
```

对于 float16，bank 是：

```text
bank = floor(addr / 2) mod 32
```

八 row 映射到：

```text
i = 0: bank 0
i = 1: bank 4
i = 2: bank 8
i = 3: bank 12
i = 4: bank 16
i = 5: bank 20
i = 6: bank 24
i = 7: bank 28
```

column 现在 touch 八 distinct bank。conflict 消失。

没有 swizzling，相同 column 有 address：

```text
m = 64 * i
```

因此：

```text
bank = floor(64 * i / 2) mod 32 = 0
```

每 row 落地到 bank 0，access serialize。swizzle 只 change 物理 placement，但那足以使 column access 变为 conflict-free。

那 guarantee 取决于以设计方式使用 swizzle。dtype、swizzle width 和 access shape 必须匹配 TMA 和 MMA descriptor mode。128-byte float16 swizzle 设计围绕 relevant 16-byte row chunk 和 Tensor Core access pattern。它不是 promise 任意 shared memory access 变为 conflict-free。本章顶部 demo 使这 visible：选择 dtype 和 swizzle mode，watch column 在没有 swizzle 时 collapse 到一个 bank，然后 scatter 到 bank view 一旦匹配 swizzle 应用。

## Design Rationale

layout API 遵循三个 design choice。

首先，它支持 general shape。hardware tile 不总是 power of two。global tensor、shared memory stage、TMEM accumulator 和 scale-factor buffer 常有来自 capacity limit 或 algorithm choice 的 shape。layout model 将那些 shape 视为 normal。

其次，mapping 从 logical coordinate 到 physical coordinate。那个 direction 重要，因为 replication common。一个 logical element 可能在几个 physical place。logical-to-physical map 直接表示那为 coordinate set。

第三，hardware axis 是 explicit。layout 不使用 anonymous dimension 并依赖 context 后来解释它们。`tx`、`tid_in_wg`、`laneid`、`warpid`、`TLane` 和 `TCol` 的差异写入 layout 本身。

legality 和 feasibility check 不是 layout object 单独的工作。layout 可以说 data 存放哪里。higher-level tile primitive 决定给定 operation 能否 legally and efficiently 使用那 placement。那 separation 保持 layout API 小，同时仍给 compiler 足够 information 来 dispatch real hardware operation。
