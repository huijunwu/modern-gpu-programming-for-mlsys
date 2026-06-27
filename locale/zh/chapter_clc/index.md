(chap_clc)=
# 高级: Cluster Launch Control

:::{admonition} Overview
:class: overview

- persistent kernel 保持固定 CTA 或 CTA cluster set resident（通常 size 为大约每 SM 一个 active work owner，但不依赖 guaranteed 1:1 mapping），并 loop 多个 output tile 而非 per tile 启动一个 CTA。
- Cluster Launch Control 是 Blackwell hardware mechanism，使 resident cluster 可以在 runtime 要求另一个 tile。它是围绕两个 PTX instruction 构建的 hardware work-stealing path：一个 instruction 请求 work，另一个读取请求是否成功。
- 主要 benefit 是更好的 tail behavior。当 tile cost 不均或 tile 数不能均匀分配在 available SM 上，early finish 的 CTA 可以 pull 更多 work 而非 idle。
:::

persistent GEMM 不将 CUDA grid 视为 fixed one-CTA-per-output-tile launch。相反，它启动较小 set 的 long-lived CTA 或 CTA cluster。每个计算一个 tile、advance 到另一个 tile、再计算，并继续直到 output space 完成。这是 {ref}`chap_gemm_advanced` 中构建的 execution pattern。

一旦 kernel 是 persistent，main scheduling question 变得简单：CTA 或 cluster 完成 current tile 后，next tile 来自哪里？

最简单 answer 是 static formula。例如，kernel 可以从 CTA id 计算 tile coordinate，然后 advance 一个 grid stride。这 easy to implement，当所有 tile cost 大致相同且 tile count 均匀分布在 GPU 上时 work well。但 schedule 在 work 实际运行前决定。如果少数 tile 花费 longer，或最后 few tile 分配 uneven，一些 SM 完成其 share early 而其他仍 work through the tail。

Cluster Launch Control 或 CLC，改变那个 scheduling model。而不是 upfront decide whole assignment，persistent cluster 可以向 hardware grid scheduler 要求另一个 not-yet-launched cluster work。如果请求成功，current cluster take over 那个 cluster coordinate 并计算对应 tile。如果请求失败，没有更多 work 可 steal，loop exit。

这与 thread block cluster 本身不同。thread block cluster（一起 launch 的 CTA，带 cluster-level synchronization 和 distributed shared memory access）在 Hopper 中引入（{ref}`chap_background`）。CLC 是 Blackwell 添加，使在那些 cluster coordinate 上的 scheduling dynamic。cluster 已是 launch unit；CLC 使 already-running cluster cancel pending launch 并 inherit 其 coordinate。

## 两个 Instruction

Cluster Launch Control 通过两个 PTX instruction 暴露。第一个 instruction 发送 asynchronous request 到 grid scheduler。第二个 instruction 读取 response。

request instruction 是 `clusterlaunchcontrol.try_cancel.async`。

`try_cancel` 要求 scheduler cancel pending cluster launch 并返回那个 cluster coordinate 给 caller。response 作为 16-byte record 写入 shared memory。由于 request 是 asynchronous，instruction 不等待 response 到达。相反，completion 通过 `mbarrier` 报告，使用与 TMA 相同的 barrier-and-phase model。

这是 important detail，因为它意味着 CLC 不引入新 waiting model。kernel issue 请求、associate 与 barrier，later wait barrier 前 read response。response arrival 通过 barrier signal 带 byte-count completion，与 other asynchronous hardware operation 相同 general style（见 {ref}`chap_async_barriers`）。

一旦 barrier fire，kernel use query instruction。

第一个 query 是 `clusterlaunchcontrol.query_cancel.is_canceled`。它返回 predicate 告诉 kernel cancellation 是否成功。true predicate 意味 scheduler 找到 pending cluster launch、cancel 它并返回其 coordinate。false predicate 意味没有 pending work 留下。

仅当 `is_canceled` 为 true 时 kernel 应 read coordinate。它用 `clusterlaunchcontrol.query_cancel.get_first_ctaid` 做那，extract canceled cluster 的第一个 CTA id。那个 CTA id 是 coordinate vector，通常读为 `(x, y, z)`，kernel 将其 decode 为它应计算 next 的 output tile。

这个 protocol 中没有 numeric sentinel tile id。kernel branch on predicate。如果 predicate 为 true，coordinate valid。如果 predicate 为 false，work-stealing loop done。

under the hood，这个 shape follow directly from what CLC doing。hardware 不是从 software queue 分配 abstract task。它 cancel 还未发生的 cluster launch。successful response 因此包含 real cluster coordinate。failed response 简单意味 launch queue exhausted。

## Work-Stealing Loop

用那两个 instruction，persistent scheduler 成为 short loop。

在 loop 中任何 point，cluster 有一个它负责计算的 tile。在开始那 tile 前，它发送 `try_cancel` 请求下一个。request run asynchronously。scheduler work on that request 时，cluster compute current tile。

current tile finished 后，cluster wait `mbarrier` 关联 `try_cancel` response。response ready 后，它 call `query_cancel.is_canceled`。如果 predicate 为 true，它 call `query_cancel.get_first_ctaid`，decode returned coordinate，并用它作为 next tile。如果 predicate 为 false，没有更多 work 留下，cluster exit。

在 code shape，loop 是：

1. issue `try_cancel` 可能 next tile；
2. request in flight 时 compute current tile；
3. wait response barrier；
4. query cancellation 是否成功；
5. 要么 continue 与 returned coordinate 或 exit。

request 的 placement 使 loop 有用。cluster 不等到完成 current tile 才 ask 更多 work。它先 ask，再 compute。那 overlap scheduler request 与 useful work。到 current tile done 时，next tile 的 answer 通常已 available。

这是 persistent kernel 在其他地方 use asynchronous copy 和 tensor-core barrier 的相同 basic reason。kernel 避免将 long-latency operation 直接放在 critical path。CLC apply 相同 idea 到 tile scheduling：early ask 下一个 work unit，compute current unit，然后在需要时 consume scheduling result。

## 与 Persistent GEMM 的关系

{ref}`chap_gemm_advanced` 中的 persistent GEMM 对 main walkthrough 使用 static scheduler。static scheduler easier to explain 因为 next tile 可以直接从 loop state 计算。例如，像 `ClusterPersistentScheduler2D` 这样的 scheduler 可以使用 output tile space 上的 grid-stride pattern assign tile。

CLC 是那个 static assignment 的 dynamic replacement。outer loop 保持相同：每个 resident cluster 重复 compute 一个 output tile 然后 advance 到另一个。变化是 next tile 来自哪里。用 static scheduler，next tile 由 formula 计算。用 CLC，next tile 由 hardware work stealing 返回。

那个 difference 在 launch 的 tail near 最重要。在 static schedule 中，remaining work 可能不 even distribute。一些 SM 可能 run out  assigned tile 而其他仍有 several。用 CLC，early finish 的 cluster ask 另一个 pending cluster coordinate。只要 launch queue 有 work，early finisher 保持 pull 更多 tile。

当 tile cost 不 uniform 时也 matter。一些 GEMM tile 可能因 boundary、masking、sparsity、grouped scheduling 或 around main matrix multiply 的 fused work 取不同 path。static schedule 假设 tile assignment 在任何 those cost 被 observe 前 sufficiently good。CLC 不需要那 assumption。它仅在 cluster 可用后 assign 更多 work。

在 TIRx，CLC 因此可以 expose 为 dynamic tile scheduler。programming model 不需要 change tile 的 computation。tile body 与 static scheduler 使用的相同 persistent GEMM body。scheduler 从"formula 计算 my next tile coordinate"变为"ask hardware next available cluster coordinate"。result 是相同 persistent loop，但带 hardware-driven work distribution 而非 fixed launch-time schedule。
