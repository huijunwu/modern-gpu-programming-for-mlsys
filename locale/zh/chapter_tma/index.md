(chap_tma)=
# Async Data Movement: TMA

:::{admonition} Overview
:class: overview

- TMA 是一个 hardware engine，用于 global memory 和 shared memory 之间的 asynchronous tile copy。一个 thread 发出 copy，engine 移动 byte。
- 一个 TMA copy 由 tensor-map descriptor 描述。descriptor 告诉 engine global tensor shape、stride、tile coordinate 和 shared memory swizzle mode。
- 在 load path，TMA 在写入 shared memory 时可以 swizzle tile，使 tile 直接落到 Tensor Core 期望的 layout。
- TMA load 通过带 byte-count tracking 的 `mbarrier` 完成。TMA store 使用 commit group 和 wait group。
:::

只有当 Tensor Core 有 data 可消费时，它才有用。在 GEMM 或 attention kernel 中，一旦 pipeline 饱满，math 可能是 compute-bound（{ref}`chap_performance`），但 pipeline 只有在下一 operand tile 及时到达时才能保持饱满。

移动 tile 的旧方法是让 thread 自己复制它。每个 thread 计算 address、从 global memory 发出 load、将值 store 到 shared memory。这有效，但它花费 warp instruction 在 address arithmetic 和 copy bookkeeping 上，而不是 compute。它还使 copy path 在原本应该 feed Tensor Core 的相同 warp 的 instruction stream 中可见。

Tensor Memory Accelerator（或 TMA）将这项工作移入 hardware copy engine。一个 thread 发出 tile copy。copy engine 然后在 global memory 和 shared memory 之间异步移动一个 rectangular tile。当 engine 移动 byte 时，CTA 的其余部分可以继续其他 work。

TMA 还处理 part 的 layout problem。Tensor Core 不仅需要 shared memory 中有正确的 value。它需要它们以正确的 shared memory layout 放置。在 load path，TMA 在写入 tile 时可以应用 shared memory swizzle。这使得 tile 直接落到后续 MMA 期望的 layout。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../../../../_extra/demo/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: TMA 将 tile 从 global memory 复制到 shared memory。切换 swizzle mode 并 hover source cell 查看它在 shared memory 中的落地位置。*

## One Thread Issues, Hardware Moves the Tile

一个 TMA copy 始于一个 issuing thread。那个 thread 不 loop 遍历 tile 中的所有 element。它给 hardware 一个 copy 的 description，然后 TMA engine 执行 transfer。

main input 是一个 tensor-map descriptor。descriptor 描述 global tensor 以及 tile 应如何从中读取。它记录 information 如 tensor shape、stride、element size、tile shape 和 swizzle mode。issuing thread 还提供 tile 应落到的 shared memory address。

instruction 发出后，copy 异步运行。issuing thread 可以继续。CTA 中其他 thread 也可以继续。transfer 现在是 TMA engine 的职责，而不是普通 load 和 store instruction 的 loop。

这给 kernel 两种表达相同 logical operation "copy this tile" 的不同方式。

一种 path 是 thread copy。thread 合作从 global memory load 并 store 到 shared memory。这使得 kernel 对每个 access 有直接 control，但它消耗 thread instruction 和 register 用于 address calculation。

另一种 path 是 TMA copy。一个 thread 发出 transfer，hardware copy engine 执行 rectangular copy。这是大型 regular tile 的 natural path，尤其是 Tensor Core kernel 使用的 operand tile。

这两种 path 有不同的 synchronization rule 和不同 performance behavior。在它们之间选择是一个 dispatch decision。layout 告诉 kernel 它想要什么 memory arrangement。scope 告诉它哪些 thread 或 CTA 参与。dispatch 决定 copy 是由普通 thread code 还是 TMA 实现。

## Swizzled Layout

移动 tile 还不够。tile 还必须以 Tensor Core 能高效读取的 layout 放置到 shared memory 中。

这就是使用 TMA swizzling 的地方。当 TMA 将 tile 写入 shared memory 时，它可以 permute shared memory address pattern。global memory tile 仍然是 logical rectangle，但 destination shared memory layout 可以 swizzle。

swizzle mode 是 TMA descriptor 的一部分。一旦 descriptor 设置好，issuing thread 不需要手动应用 swizzle。engine 在 byte 落到 shared memory 时应用它。

重要的 requirement 是 agreement。TMA descriptor、shared memory tile layout 和后续 MMA instruction 必须都描述相同的 layout（{ref}`chap_data_layout`）。如果 TMA 以 swizzle 一种方式写入 tile，但 MMA 读它时假设是另一种 swizzle，hardware 仍将精确执行所要求的事情。byte 只是被安排得与计算不匹配。

这是 layout notation 不仅仅是一个 bookkeeping device 的点。DSL 使用的 layout 必须匹配 TMA descriptor 和 Tensor Core instruction 使用的 hardware layout。例如，如果 kernel 说一个 operand tile 以 128-byte swizzled layout 存储，TMA descriptor 必须使用匹配 swizzle mode，MMA dispatch 必须期望相同 shared memory arrangement。上面的 demo 允许你在 no swizzle 和 128-byte swizzle 之间切换；hover source element 查看 swizzle 应用后它落到的位置。

理解 swizzle 的一个有用方式是：TMA 不改变 logical tile。它改变 logical element 物理上落到 shared memory 中的位置。后续 MMA 仍然消费相同 logical A 或 B tile。swizzle 只决定 tile 在 shared memory bank 上如何排列。

## 3D TMA for Tiling 和 Swizzling

一个 plain TMA copy 移动 flat 2D tile，但 Tensor Core 想要的 shared memory layout 通常 *tiled into swizzle atom*（来自 {ref}`chap_data_layout` 的 8 x 128-byte atom）。TMA 通过一个额外 descriptor dimension 处理它。**3D TMA** 将 shared memory box 描述为 `(group, row, col)`，其中 group dimension 跨 atom 遍历，inner two address 在一个 atom 内。一个 single 3D copy 然后将 tile 排列 atom by atom（tiling）并在每个 atom 内应用 swizzle，所以 data 到达时已经是 MMA 期望的 layout，没有单独的 tiling 或 swizzling pass。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../../../../_extra/demo/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: 一个 3D TMA copy，addressed 为 (group, row, col)，tiling 到 swizzled shared memory。*

选择 swizzle *format* 与这个 tiling 相关。更宽 swizzle 将 column scatter 到更多 bank，所以 128-byte swizzle 在 fits 时是 default，但一个 N-byte atom 需要 tile 的 contiguous dimension 填满它。一个因 shape constraint 而小的 tile 因此不能使用 128-byte swizzle 并必须降到 64-byte 或 32-byte：rule of thumb 是选择 tile 能 fill 的 largest swizzle（{ref}`chap_data_layout`）。下面的 demo 直接显示 constraint：一个 16 x 16 tile 上的 128-byte swizzle 只有当 tile 分成匹配 atom 的 16 x 8 group 时才 conflict-free。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../../../../_extra/demo/tiling_constraint.html" title="Swizzle imposes a tiling constraint" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
<script>
(function () {
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    document.querySelectorAll('iframe.demo-tma3d').forEach(function (f) {
      if (e.source === f.contentWindow) f.style.height = d.height + 'px';
    });
  });
})();
</script>
```
*Interactive: 一个 16 x 16 tile 上的 128-byte swizzle，分成 16 x 8 group 后 conflict-free。*

## Completion: Load

copy 是 asynchronous 的，所以发出它还不够。consumer 不能仅因 TMA instruction 已发出就读取 shared memory tile。tile 只有在 engine 完成写入 byte 后才安全可读。

对于 TMA load，completion signal 是一个 `mbarrier`（{ref}`chap_async_barriers`）。

通常 sequence 是：

1. 初始化或重用 pipeline stage 的 `mbarrier`；
2. 告诉 barrier TMA transfer 预期写入多少 byte；
3. 发出 TMA load；
4. 让 TMA engine 在 byte 到达时更新 barrier；
5. 使 consumer 在读取 shared memory tile 前等待 barrier phase。

byte count 通过类似 operation 设置：

```text
mbarrier.arrive.expect_tx(bytes)
```

这做两件事。它记录 expected transfer size，它还执行 issuing thread 的 barrier arrival。barrier 仅因为这个调用发生并不 complete。它仍等待 TMA engine 报告 expected byte 已到达。

随着 transfer 进行，engine 对 barrier 执行 complete-tx update。barrier phase 仅在两个 condition 都满足时才翻转：arrival count 满足，且 pending byte count 归零。

consumer 然后等待那个 barrier。一旦 wait 为 expected phase 完成，shared memory tile 就 ready。此时 MMA path 可以安全读取它。

../../../../img/tma_sync_flow.png)

这是与其他 asynchronous producer-consumer handoff 相同的 barrier model。producer 是 TMA engine。consumer 是 MMA path 或任何读取 shared memory tile 的其他 code。barrier 是它们之间的 explicit handoff。

## Completion: Store

TMA store 将 data 反方向移动，从 shared memory 到 global memory。它们也是 asynchronous 的，但 completion mechanism 不同。

一个 TMA load 通常 feed 同一 kernel 内 consumer。MMA path 需要知道 shared memory tile 何时 ready。这就是 load path 使用 `mbarrier` 的原因。

一个 TMA store 通常将 final data 写入 global memory。通常没有 immediate in-kernel consumer 等待 stored result。kernel 主要需要知道何时安全重用 shared memory buffer 或完成 store sequence。

为此，TMA store 使用 commit group 和 wait group。kernel 发出一个或多个 store、commit group，然后等待 group drain。wait 完成后，该 group 中的 store 从 kernel 角度看已完成，store 使用的 shared memory region 可以安全重用。

所以 rule 很简单：

```text
TMA load:  通过带 byte-count tracking 的 mbarrier 等待
TMA store: 通过 commit group 和 wait group 等待
```

两种 mechanism 在不同 handoff point 服务相同 purpose。load 需要使 shared memory tile 对后续 consumer 可见。store 需要确保 outgoing transfer 在 kernel 重用 source storage 或依赖 store 已 drain 前完成。

## 为什么 TMA 对 Pipelining 重要

当 TMA 是 pipeline 的一部分时，它最有用。kernel 可以在 Tensor Core 计算 current tile 时发出 future tile 的 load。load 在 background 运行。compute 在 foreground 运行。当 future tile 成为 current tile 时，barrier 连接两者。

一个典型 GEMM loop 重复使用这种 structure。shared memory 的一个 stage 保持当前被 MMA 消费的 tile。另一个 stage 正由 TMA 填充。随着 loop 推进，role 轮转。MMA 读取 stage 前，它等待该 stage 的 load barrier。TMA 覆盖 stage 前，kernel 确保 previous consumer 已完成它。

这就是为什么 TMA 和 `mbarrier` 通常一起出现在 Blackwell- 和 Hopper-style kernel 中。TMA 给 kernel 一个 asynchronous copy engine。barrier 给 kernel 一种精确方式，知道 copied byte 何时 ready。
