(chap_tirx_layout_api)=
# TIRx Layout API

:::{admonition} Overview
:class: overview

- TIRx layout API 将 {ref}`chap_data_layout` 中的 layout notation 转为 compiler object。main object 是 `TileLayout`、`SwizzleLayout` 和 `ComposeLayout`。
- `TileLayout` 描述在 named hardware axis 上的 affine placement。它由 shard spec `S[...]`、replica spec `R[...]` 和可选 offset 构建。
- layout 将一个 logical coordinate 映射到一个或多个 physical coordinate。`layout.apply()` 评估那个 mapping。
- `SwizzleLayout` 描述用于避免 bank conflict 的基于 XOR 的 shared memory swizzle。`ComposeLayout` 在 tile layout 上 stack 一个 swizzle。
- `tmem_datapath_layout`、`tcgen05_atom_layout` 和 `wg_local_layout` 等 ready-made constructor 覆盖 kernel 中反复出现的 hardware layout。
:::

{ref}`chap_data_layout` 介绍了本书使用的 notation：tile shape、在 named axis 上的一组 stride、以及用于被 copy 而非 partition 的 value 的可选 replication term。本章将那个 notation 转为 compiler 使用的 API。

goal 是页面上的 notation 和 kernel 中的 code 看起来几乎相同。当你写 layout 如：

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

你不仅在写 explanation。你在构造一个可 attach 到 buffer 的 `TileLayout` object。之后，每个 touch 那个 buffer 的 tile operation 都可以从 layout 读它的 placement。placement 写一次、check 一次、被 compiler reuse。

layout 在从 pool allocate 时或在 declare buffer 时 attach：

```python
pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

从那时起，buffer 携带它的 physical placement。tile operation 不需要重复每个 element 存放在哪里。

layout object 在一个 module 中：

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

API 背后有一个 central idea。layout 不必将 logical index 映射到单个 physical address。它将 logical index 映射到 named axis 上的一组 physical coordinate。在通常 case 那组有一个 element。当 replication present 时，相同 logical element 有几个 physical placement。

这就是为什么 layout model 有三部分：shard、replica 和 offset。shard place element。replica 将它 copy 到额外 coordinate。offset shift 整个 placement。

## 通过 Example 看 Layout

下面 example 展示 API 的 basic shape。

TMEM 中的 accumulator 可以写为在 TMEM axis 上的 direct placement：

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])
```

这里 logical row 映射到 `TLane`，logical column 映射到 `TCol`。在 {ref}`chap_tmem` 中，hardware coordinate 叫 Lane 和 Col。在 TIRx layout notation 中，那些 hardware axis 写为 `TLane` 和 `TCol`。

block-scaled MMA scale-factor layout 使用 replication：

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)
```

shard 在 TMEM 中 place 一个 32-row group。replica 以 32 lane stride repeat 那个 group 四次，因此 32-row group 在完整 128-lane TMEM space 中 visible。

tensor-core register fragment 可 distribute 在 lane 和 warp 上：

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

相同 physical axis 可出现多次。在这个 example 中，两个不同 iter 都 contribute 到 `laneid`。没有显式 axis 的 stride 使用 default memory axis `m`。

在 real kernel 中，common hardware layout 通常来自 constructor：

```python
acc = tmem_datapath_layout("D", 128, 256)

ld = tcgen05_atom_layout("32x32b", (128, 64), "float32")
```

这些 constructor 返回普通 `TileLayout` object。它们是 convenience，不是 separate mechanism。你可以 inspect 返回的 layout、将它与其他 layout compose、或者当 shape unusual 时手写 underlying `S[...]` 和 `R[...]` form。

## Interactive Demo

在 mechanics 之前，有 something concrete 可以 poke 是有帮助的。下面 demo 让你选择 preset layout、编辑 logical shape 和 `S` 或 `R` term、选择 dtype 和 swizzle mode、并点击 element 看哪个 physical coordinate 或 coordinate 拥有它。

```{raw} html
<p>
  <a class="reference external" href="../_static/tirx-layout-demo/index.html"
     target="_blank" rel="noopener"
     style="display:inline-block; padding:10px 18px; background:#3b82f6;
     color:#fff !important; font-weight:700; border-radius:8px;
     text-decoration:none;">▶ Open the demo full screen ↗</a>
</p>
<iframe id="tirx-layout-demo-frame" src="../_static/tirx-layout-demo/index.html?notitle"
        style="width:100%; height:1040px; border:1px solid #dfe1e6;
        border-radius:10px; margin:10px 0 6px; display:block;"
        title="TIRx interactive layout demo" loading="lazy"></iframe>
<script>
// The demo (viz-base.js) posts its content height; size the iframe to fit so
// there is no inner scrollbar. This demo is responsive (fills the width), so
// only the height follows content.
(function () {
  var f = document.getElementById('tirx-layout-demo-frame');
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    if (f && e.source === f.contentWindow) f.style.height = d.height + 'px';
  });
})();
</script>
```

demo 有用是因为 API 的大部分只是 demo 展示内容的 precise version。logical element 进入 layout。layout flatten 它、将它 split 到它的 iter 上、在 named axis 上 accumulate coordinate、然后在需要时 apply replication。

## TileLayout

`TileLayout` 是 main affine layout object。它通常用与文本中相同的 notation 写：

```python
TileLayout(S[shape : strides])
```

`S` term 是 shard spec。你可以读它：取这个 shape 的 logical tile 并用这些 stride 在 named axis 上 place 它。

当 value 需要在多个地方出现时，shard spec 用 replica spec 扩展：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])
```

可选 offset 也可添加：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)
```

在 surface 下，这些 piece 由 iter 表示。iter 是一个 triple：

```text
(extent, stride, axis)
```

它描述在一个 named axis 上的 strided walk。extent 告诉 iter 有多少 position。stride 告诉每个 step 移动多远。axis 告诉哪个 hardware coordinate 正在被 change。

layout 有三个部分。

### Shard

shard 或 `D` 是由 `S[...]` 构建的部分。它将 logical index partition 到一个或多个 iter 上并产生 base physical coordinate。

例如：

```python
S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
```

有四个 shard iter。它们的 extent 是 `8`、`2`、`4` 和 `2`。它们的 stride 将 data place 在 `laneid`、`warpid`、再次 `laneid` 和 default memory axis `m` 上。

这 generalise 了普通 shape-and-stride rule。不同之处在于 stride attach 到 named hardware axis 而不是单个 flat address。

### Replica

replica 或 `R` 描述相同 logical element 的额外 physical copy。replica iter 独立于 logical index。它们 enumerate hardware space 中的 extra offset。

例如：

```python
R[2 : 4@warpid]
```

创建两个在 `warpid` axis 上由四个 warp 分隔的 copy。

replication 不是 convenience trick。它描述 real hardware behavior。一些 data broadcast 在 warp、lane 或 memory region 上。logical-to-physical mapping 自然地 support 那个，因为一个 logical element 可以映射到一组 physical coordinate。

### Offset

offset 或 `O` 是添加到每个 result 的 fixed coordinate。

例如：

```python
5@warpid
```

在 `warpid` axis 上将整个 placement shift 五。

offset 用于将 tile place 在 chosen base coordinate、reserve region 供 exclusive use、或描述在同一 resource 中在另一个 tile 之后开始的 tile。

### 将 Piece 放在一起

layout 按顺序 apply 这三个部分。

首先，shard compute base coordinate。然后 replica 将那个 coordinate fan out 到零个或多个额外 copy。最后，offset shift 每个 coordinate。

对于 logical coordinate `x`，result 是：

```text
L(x) = { D(x) + r + O | r in R }
```

如果没有 replica，`R` 只包含零 offset，因此 result 是 singleton set。如果有 replica，result 包含每个 replica position 的一个 coordinate。

在 TIRx syntax 中，完整 layout 看起来像这样：

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从左到右读，shard place logical tile，replica 在四个 warp ID 之外创建第二个 copy，offset 将整个 placement shift 到从 `warpid = 5` 开始。

如果 iter 已经作为 object 构建，相同 layout 可以直接构造：

```python
TileLayout.from_iters(shard, replica, offset)
```

大多数 user code 使用 `S[...]` 和 `R[...]` notation 因为它更接近 mathematical form。

## Named Axis

layout 中的 axis 不是 anonymous dimension。每个 axis name 一个 real hardware coordinate 或 compiler-level placement coordinate。

example 包括：

```text
bx, by, bz
cbx, cby, cbz
tx
warpid
laneid
wgid
tid_in_wg
wid_in_wg
m
P, F
Bank
TLane, TCol
```

如 `bx`、`by` 和 `bz` 的 grid axis 在 CTA 间 place work。如 `cbx`、`cby` 和 `cbz` 的 cluster axis 在 CTA cluster 内 place work。如 `tx`、`warpid`、`laneid`、`tid_in_wg` 和 `wid_in_wg` 的 thread axis 描述 CTA 或 warpgroup 内的 ownership。axis `m` 是 default linear memory axis。`P` 和 `F` 用于 two-dimensional scratchpad-style placement。`Bank` name shared memory bank。`TLane` 和 `TCol` 是 TMEM Lane 和 Col coordinate 的 TIRx layout name。

axis name 是 layout 的一部分。这很重要因为具有相同 integer value 的两个 coordinate 可以 mean 不同 hardware thing。`1@tx` 不同于 `1@tid_in_wg`。`1@laneid` 不同于 `1@TLane`。layout 保持那些 meaning explicit。

## Forward Mapping

evaluate layout mean 取 logical coordinate 并 compute 它 physical 上 land 在哪里。API method 是：

```python
layout.apply(*coord)
```

对于没有 replication 的 layout，result 是一个 coordinate dict。有 replication 时，result 是一组 coordinate dict。coordinate dict 将 axis name 映射到 integer position，如：

```python
{"laneid": 7, "warpid": 2, "m": 1}
```

evaluation rule 有四个 step。

首先，以 row-major order flatten logical coordinate。对于 logical coordinate：

```text
x = (x0, x1, ..., xr-1)
```

在 logical shape 内：

```text
(S0, S1, ..., Sr-1)
```

flat index 是：

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

第二，将那个 flat index split 到 shard extent 上。如果 shard extent 是：

```text
(e0, e1, ..., en-1)
```

那么 split 产生 component：

```text
c0, c1, ..., cn-1
```

使用 shard extent 上相同 row-major order。

第三，使用 stride 将每个 component accumulate 到它的 axis 上。如果 shard iter `k` 有 extent `ek`、stride `sk` 和 axis `ak`，那么 component `ck` contribute：

```text
ck * sk @ ak
```

对相同 axis 的所有 contribution 加在一起。然后添加 offset。

第四，apply replica iter。每个 replica iter contribute 独立于 logical coordinate 的额外 offset。如果有几个 replica iter，layout enumerate 所有 combination。

这个 rule 的一个 useful consequence 是 layout 不需要 hard-code input shape。它需要的是 logical tile 有与 shard extent 乘积相同 total element 数量。一旦那个成立，flattening 和 splitting define mapping。

## Case Study: Tensor Core Register Tile

考虑 distribute 在两个 32 lane warp 上的 logical `(8, 16)` tile。每个 lane 拥有一个小 register fragment。register slot 由 default memory axis `m` 表示。

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

取来自 `(8, 16)` tile 的 logical element `(i, j)`。

row-major flat index 是：

```text
flat = 16 * i + j
```

按 shard extent `(8, 2, 4, 2)` split 给出：

```text
c0 = i
c1 = floor(j / 8)
c2 = floor(j / 2) mod 4
c3 = j mod 2
```

shard contribution 是：

```text
laneid = 4 * c0 + c2
warpid = c1
m      = c3
```

添加 offset `5@warpid` 后，变为：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5
m      = j mod 2
```

replica term：

```python
R[2 : 4@warpid]
```

向 `warpid` 添加 `0` 或 `4`。因此完整 mapping 是：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5 + 4 * r, 其中 r in {0, 1}
m      = j mod 2
```

shard 将 tile place 在 warp 5 和 6 上。replica 然后将它 copy 到 warp 9 和 10。因此相同 logical element 出现在两个 warp position。

这个 example 展示为什么 model 使用一组 physical coordinate。replication 不能自然地由从 physical coordinate 到 logical coordinate 的 function 表示。它可以自然地由从一个 logical coordinate 到几个 physical coordinate 的 function 表示。

## Case Study: Blackwell Tensor Memory

相同 layout model 适用于 memory placement。axis 不必是 thread axis。它们可以是 memory axis。

TMEM 由 hardware Lane 和 Col coordinate 寻址。在 TIRx layout notation 中，那些 axis 写为 `TLane` 和 `TCol`。

考虑这个 layout：

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

如果 logical tile shape 是 `(2, 128, 112)`，split component 只是 logical coordinate 本身。对于 element `(a, l, c)`，mapping 是：

```text
TLane = l
TCol  = 112 * a + c
```

extent-128 iter 带 stride `1@TLane` fill 128 TMEM Lane row。extent-2 iter 带 stride `112@TCol` 和 extent-112 iter 带 stride `1@TCol` 一起 cover 224 column：

```text
TCol in [0, 224)
```

224-column span 是 intentional。TMEM layout 不必是 2 的幂。block-scaled FP8 GEMM 可能选择 224-column accumulator，因为完整 256-column tile 不会留下足够 TMEM capacity 供两个 accumulator stage 加上 scale factor。layout API 可以直接 express 那个 shape。

## Scale Factor Layout

上面 accumulator layout 是 pure placement。每个 logical accumulator element 映射到一个 TMEM coordinate。block-scaled MMA 的 scale factor 不同，因为相同 physical group 可能需要跨几个 warp window visible。这是 replication 变得有用的地方。

compact scale-factor layout 可以写为：

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

shard 在 TMEM 中 place 一个 32-row scale-factor group：

```text
TLane = r
TCol  = s
```

对于 logical scale coordinate `(r, s)`。

replica term 创建四个由 32 lane 分隔的 copy：

```text
TLane = r + 32 * q, 其中 q in {0, 1, 2, 3}
TCol  = s
```

因此 32-row group 在 TMEM lane 0 到 31、32 到 63、64 到 95 和 96 到 127 处 visible。这是 `warpx4` broadcast pattern（{ref}`chap_layout_generations`）。四个 warp-sized TMEM lane window 中的每个看到相同 scale-factor group。

在完整 block-scaled MMA layout 中，这个 atom 与 M row 和 K scale-factor group 的 outer iter 组合。几个 scale factor 也可能 pack 到一个 32-bit `TCol` cell，取决于 scale-factor dtype。例如，fp8 scale factor 可以将四个 value pack 到一个 32-bit column cell。可选 stride-zero reuse 和 pipeline-depth iter 然后可以 describe scale reuse 跨多个 MMA 和 double buffering。

important part 是相同 `TileLayout` model describe 两个 case。accumulator 是 TMEM 中的 single placement。scale factor 是相同 TMEM address space 中的 replicated placement。

## Ready-Made Layout

大多数 kernel 不手写每个 hardware layout。TIRx 为经常出现的 layout 提供 constructor。

```python
tmem_datapath_layout(datapath, rows, cols)
```

返回 `tcgen05.mma` 写入的 TMEM accumulator layout。`datapath` argument 选择 row placement pattern。例如，`"D"` 对应 `M = 128` identity-style placement，而 `"F"` 对应 `M = 64` scattered placement。

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

返回 `tcgen05.ld` 或 `tcgen05.st` atom 移动的 register tile layout。instruction shape example 包括 `.32x32b`、`.16x64b`、`.16x128b` 和相关 form。在 DSL level 这是 warpgroup-distributed tile。在 lowering 期间它变为四个 warp-collective `tcgen05.ld` 或 `tcgen05.st` instruction，每个 warp 一个，每个 warp 处理它自己 32 TMEM lane。

```python
wg_local_layout(cols, rows=128)
```

返回 warpgroup-local register tile，通常在 `tid_in_wg` 上每个 thread 一个 row。

这些 helper 在那里以避免手写 common hardware mapping。它们不 hide model。每个 helper 返回由上面描述的相同 `S` 和 `R` piece 构建的普通 `TileLayout`。

## SwizzleLayout 和 ComposeLayout

`TileLayout` 是 affine。它可以 express stride、replication 和 named axis 上的 offset。那对许多 placement 足够，包括 thread fragment、TMEM tile 和 compact scale-factor layout。

shared memory swizzle 需要其他东西。用于避免 bank conflict 的 swizzle 不是 affine stride pattern。它是 linear shared memory address 的基于 XOR 的 permutation。

TIRx 因此将 swizzling 保持为 separate layout object：

```python
SwizzleLayout(...)
```

并与 tile layout compose 它：

```python
ComposeLayout(swizzle, tile)
```

tile layout 首先产生 linear memory address。swizzle 然后 permute 那个 address。保持这两个 layer separate 比强制 XOR permutation 进入 affine layout model 更 clean。

## 为什么 Swizzle

shared memory 分为 32 bank，每个 bank word 持有 4 byte。当一个 access 的 lane touch 相同 bank 中不同 address 时，access 被 bank conflict serialize。

plain row-major tile 可以 structurally 创建这个 conflict。考虑 row-major layout 的 `(8, 64)` float16 tile：

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

logical element `(i, j)` 有 linear element address：

```text
m = 64 * i + j
```

每个 row 是 64 float16 value，或 128 byte。那正好是一个完整 shared memory bank line。如果 warp 以固定 `j` 沿 column read，每个 row step 前进一个完整 128-byte line。bank index repeat，因此 column read 跨 row collapse 到相同 bank。

swizzle 通过使 low address bit 依赖于 higher row bit 来 change 这个。否则会反复 land 在相同 bank 的 column scatter 到不同 bank。

## Swizzle Transform

`SwizzleLayout` 由三个 integer parameter 控制：

```text
per_element = M
swizzle_len = B
atom_len    = S
```

input 是 linear element address `m`。

`m` 的 low `M` bit 保持不变。这 preserve 一个小 contiguous group of element。higher bit shift down 到 temporary value：

```text
x = m >> M
```

然后 `x` 在 position `[S, S + B)` 的 bit group XOR 到 `x` 的 bit group `[0, B)`。swizzled address 然后通过放回 unchanged low `M` bit 形成。

等价地：

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low
```

对于 layout well formed，`S` 必须至少 `B`。

transform 的 point 不是 change tile 中哪些 logical element。它 change 那些 element 在 shared memory 中 land 在哪里。MMA 仍然 read 相同 logical tile。swizzle 使 physical bank pattern 更好。

## 选择 Swizzle Parameter

在 normal use 中，swizzle parameter 从 dtype 和 shared memory swizzle mode 选择。common mode 是 32-byte、64-byte 和 128-byte swizzle。

`per_element` parameter 选择使小 vector-sized group 保持 contiguous。对于 float16，16-byte vector 包含 8 element，因此：

```text
M = log2(8) = 3
```

与 128-byte swizzle，layout 使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

这保持 16-byte vector group intact，同时仍然 permute larger shared memory address pattern 足够打破 column bank conflict。

大多数 code 不应该手写 derive 这些 parameter。dtype 和 descriptor mode 通常 determine 它们。对 programmer important 的是 TIRx layout 中的 swizzle、TMA descriptor 和 MMA expectation 都 match。

因此 swizzled shared memory allocation 看起来像：

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)
```

composed layout 是 attach 到 shared memory buffer 的那个。

## Element 的 Bank 和 Line

看 swizzle 是否帮助，将 swizzled element address 转回 shared memory bank。

让 `addr` 是 swizzled element address，`b` 是 element size 以 byte 计。byte address 是：

```text
byte = addr * b
```

bank 是：

```text
bank = floor(byte / 4) mod 32
```

128-byte bank line 是：

```text
line = floor(byte / 128)
```

对于 float16，`b = 2`，因此 bank formula 变为：

```text
bank = floor(addr / 2) mod 32
```

这是下面 worked example 使用的 formula。

## Worked Example: `(8, 64)` float16 Tile 上的 128B Swizzle

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

因为：

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

现在看 column `j = 0`。那么 `q = 0` 且 `r = 0`，因此：

```text
addr = 72 * i
```

对于 float16，bank 是：

```text
bank = floor(addr / 2) mod 32
```

因此八个 row 映射到：

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

column 现在 touch 八个 distinct bank。conflict 消失了。

没有 swizzling，相同 column 有 address：

```text
m = 64 * i
```

因此：

```text
bank = floor(64 * i / 2) mod 32 = 0
```

每个 row land 在 bank 0，因此 access 被 serialize。swizzle 只 change physical placement，但那足够将 column access 转为 conflict-free 一个。

这个 guarantee 取决于以 design 的方式使用 swizzle。dtype、swizzle width 和 access shape 必须 match TMA 和 MMA descriptor mode。128-byte float16 swizzle 围绕 relevant 16-byte row chunk 和 Tensor Core access pattern design。它不是 promise 任意 shared memory access 变得 conflict free。本章顶部的 demo 使这个 visible：选择 dtype 和 swizzle mode，看 column 在没有 swizzle 时 collapse 到一个 bank，然后在 apply matching swizzle 后 scatter 到 bank view。

## Design Rationale

layout API 跟随三个 design choice。

首先，它 support general shape。hardware tile 不总是 2 的幂。global tensor、shared memory stage、TMEM accumulator 和 scale-factor buffer 经常有来自 capacity limit 或 algorithm choice 的 shape。layout model 将那些 shape 视为 normal。

第二，mapping 从 logical coordinate 到 physical coordinate。这个 direction important 因为 replication common。一个 logical element 可能在几个 physical place。logical-to-physical map 直接将其表示为一组 coordinate。

第三，hardware axis explicit。layout 不使用 anonymous dimension 并依赖 context 后来 explain 它们。`tx`、`tid_in_wg`、`laneid`、`warpid`、`TLane` 和 `TCol` 之间的 difference write 到 layout 本身。

legality 和 feasibility check 不是 layout object 单独的工作。layout 可以说 data place 在哪里。higher-level tile primitive decide 给定 operation 是否可以 legally 和 efficiently 使用那个 placement。这个 separation 保持 layout API small，同时仍然给 compiler 足够 information 来 dispatch real hardware operation。
