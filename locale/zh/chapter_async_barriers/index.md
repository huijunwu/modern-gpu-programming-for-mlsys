(chap_async_barriers)=
# Async Coordination: mbarrier

:::{admonition} Overview
:class: overview

- TMA 和 Tensor Core 是 asynchronous，因此发出 work 不等于完成它，consumer 需要 explicit completion signal。
- mbarrier 是那个 signal：producer arrive，consumer wait，它跟踪 arrival count 和（对于 TMA）byte count。
- 每 barrier 携带一个 *phase*，每轮 flip；wait 在正确 phase 是 safe gate consumer。
:::

TMA（{ref}`chap_tma`）和 Tensor Core（{ref}`chap_tensor_cores`）operation 是 asynchronous。当 kernel 发出 TMA load 或 `tcgen05` MMA，issuing thread 不等待 operation 完成。instruction 仅提交到 hardware engine；actual data movement 或 matrix operation 与 program 的其余部分 parallel 继续。

这有用，因为它使 memory movement 和 compute overlap。这也意味着 program order 不足以证明 data ready。later instruction 可能在 earlier asynchronous operation 完成前运行。如果 TMA 仍在写入 shared memory tile 当 MMA 开始读取它，MMA 读 incomplete data。如果 epilogue 在 Tensor Core 完成写入 accumulator 前读取 TMEM，它读错 value。如果 kernel wait 在错误 condition，它可能从不 progress。

因此 kernel 在每 asynchronous handoff 需要 explicit completion signal。`mbarrier` 是那个 signal。producer 在其 work complete 时在 barrier arrive，consumer 在使用 produced data 前在 barrier wait。相同 mechanism 用于 TMA 到 MMA handoff、MMA 到 epilogue handoff，以及 pipeline stage 间 buffer reuse。

barrier 不是一 shot flag。它携带 phase bit，每 barrier 完成一轮 arrival 后 phase bit 改变。phase 使一个 barrier 可以在多 loop iteration 中 reuse 而不混淆一轮 completion 与另一轮 completion。

## mbarrier

`mbarrier`（memory barrier 缩写）是 stored in shared memory 的 hardware synchronization object。conceptually，它包含两个 piece 的 state：arrival counter 和 phase bit。counter 告诉 barrier 当前 round 仍缺多少 arrival。phase bit 告诉 kernel barrier 当前在哪个 round。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:100%; min-width:1320px; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: `mbarrier` state view 显示 arrival counter、phase bit 和 `init`、`arrive`、`wait` operation；click field 聚焦它。*

barrier 以 initialization 开始。在 `init` 期间，kernel 设置 barrier 应该 expect 多少 arrival。barrier 从 phase 0 开始，counter load 到那个 expected arrival count。从那时起，barrier 等待所有 required producer 或 resource user 报告它们完成。

arrival 减少 barrier 仍等待的 work amount。kernel 不同部分可以以不同方式在 barrier arrive，那个 distinction 很重要。

对于 TMA load，usual arrival path 是 tx-count arrival。像 `mbarrier.arrive.expect_tx(bytes)` 这样的 operation 做两件事。首先，它算作 issuing thread 在 barrier 的 arrival。其次，它记录 TMA engine expected 转移多少 byte。barrier 不 completion 仅因 issuing thread 已 arrive。它还等待 TMA engine 在 transfer 完成时 drain byte count。phase 仅在两个 condition 都满足后 flip：normal arrival count 已归零，且 pending tx byte count 已归零。

这就是为什么 `expect_tx` 不应读为"一个更多 ordinary arrival"。它为 asynchronous copy 设置 byte budget。hardware 后来通过 complete-tx update 计算 actual copy completion。barrier 仅在 arrival 和 byte transfer 都完成时 complete。

对于 Tensor Core work，arrival path 不同。`tcgen05` MMA 不因 MMA 发出就自动 advance barrier。kernel 必须 explicit attach barrier arrival 到 commit path，例如用 `tcgen05.commit.mbarrier::arrive` operation。当那个 committed group complete，Tensor Core side 执行 barrier arrival。如果 kernel 忘记那个 commit arrival，在 barrier wait 的 consumer 将永远 wait。

normal thread 也可以直接在 barrier arrive。当普通 thread code 是 producer，或当一组 thread 宣布它已完成使用 resource 时使用。例如，在 consumer 完成读取 shared memory buffer 后，它可以到达 barrier 告诉 producer buffer 已 free 以 reuse。

waiting 是相同 protocol 的 consumer side。consumer wait 直到 barrier 完成当前 iteration expected 的 phase。只有那时才 safe 读 data 或 reuse barrier 保护的 resource。

重要点是 asynchronous hardware 不仅 run ahead 于 program；它还通过 barrier report completion 回。TMA 可以 signal shared memory tile ready。Tensor Core work 可以 signal TMEM result ready。ordinary thread 可以 signal buffer 不再 in use。barrier 给所有这些 case 相同 producer-consumer shape：producer arrive，consumer wait。

## Phase Tracking

barrier 通常不 allocate 用于 single use。pipelined K-loop 可能执行相同 handoff 数百次，为每 iteration 分配新 shared memory barrier 不 practical。相反，kernel 保持少量 fixed barrier set 并在 loop advance 时 reuse 它们。

phase bit 是使那个 reuse safe 的原因。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: 一个 reused barrier 跨多 pipeline iteration，显示 phase bit 在每 completed round 后 flip。*

每 barrier 完成当前 round 的所有 arrival，它 flip phase：phase 0 变为 phase 1，phase 1 变为 phase 0，以此类推。wait operation check consumer expected 的 phase。那个 expected phase 由 kernel 保持在 register 中。在一 stage 成功 wait 一轮后，kernel 在 barrier 用于下一轮前 toggle 其 local phase value。

这防止 kernel 将 old completion 误认为 new one。假设 barrier 用于一个 TMA load 并已 complete。如果 next loop iteration reuse 相同 barrier 而不 tracking phase，consumer 可能 observe 之前的 completion 并错误假设新 load ready。phase bit 分离那两个 round。iteration 0 wait 一个 phase，iteration 1 wait 相反 phase，iteration 2 再次 wait 第一个 phase，pattern 继续。

在 real pipeline，bookkeeping 通常 per stage。kernel 有 fixed number shared memory stage、matching fixed number barrier，和少量 phase value 在 register 中。随着 loop advance，每 logical iteration map 到一 physical stage，phase value 告诉 wait operation 它 wait 那个 physical barrier 的哪个 round。

这就是为什么 later GEMM code 不需要每 K tile 一个 barrier（{ref}`chap_gemm_async`）。它需要每 reusable stage 一个 barrier，加上 phase tracking。stage index 选择 shared memory buffer 和 barrier。phase value 区分那个 stage 的 current use 和 previous one。

**Try with your agent**: 给它一个 two-stage pipeline 并要求它 trace 四 iteration。对于每 iteration，列出 stage index、local phase value、barrier flip 时间，以及如果 phase 未在 stage reuse 前 toggle 会发生什么错误。

## Synchronization Rules

一旦 barrier 和 phase mechanism 清晰，tensor-core kernel 中 synchronization pattern 相当 mechanical。每 time 一 path produce data 或 release resource 另一 path 将 consume，handoff 必须 explicit。

有三种 common case。

第一种 case 是 thread code produce data 给 asynchronous engine。如果 thread write shared memory 且 later TMA store 或 MMA instruction 读那个 shared memory，kernel 必须在 engine 读前使 thread write visible。这需要 appropriate thread-level synchronization 或 fence。exact instruction 取决于 handoff scope，但原因始终相同：engine 不应在 producing thread 完成 write 前 observe shared memory buffer。

第二种 case 是 TMA produce data 给 MMA。TMA load 异步 fill shared memory tile。MMA path 不能仅因 TMA instruction 发出就推断 tile ready。TMA operation 必须关联 `mbarrier`，MMA path 必须在读 tile 前在 barrier wait。

第三种 case 是 MMA produce data 给 epilogue。`tcgen05` MMA 异步 write result 到 TMEM。epilogue 不能在 Tensor Core complete relevant work 前安全读取 accumulator。因此 MMA commit path 在 completion barrier arrive，epilogue 在 read TMEM 前在 barrier wait。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_tma_timeline.html" title="mbarrier signalling TMA completion" loading="lazy"
        style="width:100%; min-width:1320px; height:700px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: TMA load 通过 `mbarrier` signal completion。MMA path wait barrier 前读取 shared memory tile。Tensor Core 到 epilogue handoff 遵循相同 shape，除了 Tensor Core commit path 执行 arrival 而非 TMA。*

相同 idea 也适用于 resource reuse。barrier 不仅是 data-ready signal。它也可以是"resource free"signal。shared memory stage 不能 overwrite 直到所有 old tile consumer 完成。TMEM region 不能 reuse 直到 previous user 完成读取或写入它。那些 case，arrival 意味"我完成此 resource"，wait 意味"现在 reuse 此 resource 给 next stage 是 safe"。

这是阅读 pipelined GEMM kernel 中 synchronization 的正确方式。wait 和 arrive 不作为 defensive programming 散落。每个 mark 一个 concrete ownership transfer：tile become ready，accumulator become readable，或 buffer become reusable。一旦识别那些 handoff，control flow 变得 much easier 跟随。
