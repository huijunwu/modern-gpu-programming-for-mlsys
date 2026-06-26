(chap_background)=
# GPU Execution Model

:::{admonition} Overview
:class: overview

- 一个 kernel 运行在线性的 thread hierarchy（thread → warp → warpgroup → CTA → cluster → grid）之上，跨越多个不同的 memory space（registers、SMEM、GMEM、TMEM）。
- 计算分为 CUDA cores 和 Tensor Cores；TMA 等专用引擎负责搬运供给它们的数据。
- 一个 kernel 是一个 pipeline，将数据在这些 memory space 中分段传递，并在独立的 compute engine 和 data-movement engine 之间交接工作；反复出现的核心目标是在同一时间让这些引擎保持忙碌。
:::

要编写快速的 GPU 程序，理解硬件本身以及代码如何在硬件上运行至关重要。本章概述 GPU execution model：执行工作的 thread hierarchy、存储和搬运数据的 memory space，以及完成繁重任务的 compute engine 和 data-movement engine。我们先逐一介绍这些组成部分，然后将它们组合在一个 GEMM pipeline 中，以清楚地展示数据和执行如何在硬件中流动。本书后面几乎每一个优化，本质上都是以不同的方式在这些相同的部件之间分配工作。

现代 GPU 还包含大量专用硬件单元。为了先做一个初步展示，下面的 interactive demo 在聚焦每个部分之前，先展示了 Blackwell streaming multiprocessor 内的主要元素。你可以点击每个部分查看其细节。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../../../../_extra/demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: Blackwell SM，展示其 warps/warpgroups、shared memory、Tensor Memory 以及 Tensor Core 和 TMA engine。*

## The Execution Hierarchy

我们从负责执行工作的 thread 开始。GPU 不会将它那成千上万个 thread 呈现为一个扁平的 pool。相反，它将它们分组到一个嵌套的 hierarchy 中，其原因是合作发生在多个不同的规模上，而每个层级的存在都是为了让其中一个规模上的合作变得廉价。下图展示了 Blackwell 上的 hierarchy；你可以点击每一层来高亮显示它。

```{raw} html
<iframe src="../../../../_extra/demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: 点击一个层级：thread → warp → warpgroup → CTA → cluster → grid。*

- **Thread**：scalar 的执行单元。每个 thread 拥有自己的 program counter 和 register，并通过其在 warp 内的 lane ID 来标识。
- **Warp**：32 个 thread 以 SIMT（*single instruction, multiple threads*）方式执行。一个 warp 的 lane 一起发出同一条指令，但每个 lane 保留自己的 register 并且可以单独被 mask off，这正是使单个 warp 的各个 lane 能够走不同分支的原因。
- **Warpgroup**：四个连续的 warp，共 128 个 thread。Hopper 引入了 warpgroup 作为发出 warpgroup-level MMA（`wgmma`）的单元；在 Blackwell 上，它承担了第二个角色：它是 Tensor Memory access 的 cooperation 单元，128 个 thread 共同将一个 TMEM tile 移入或移出 register。
- **CTA**（*Cooperative Thread Array*，CUDA 中也称为 thread block）：硬件调度的基本单元。一个 CTA 运行在单个 SM 上，并拥有其中私有的 shared memory 分配。多个 CTA 可以同时驻留在同一个 SM 上，当这种情况出现时，它们会在彼此之间瓜分该 SM 的 shared memory 容量。
- **Cluster**：一组可以互相合作的 CTA，它们可能位于不同的 SM 上。cluster 中的 CTA 可以彼此同步，并且可以读写彼此的 shared memory，这种能力称为 distributed shared memory。

这些层级值得仔细推敲，因为与之前的架构不同，Blackwell 的关键操作 **并非全部由同一组 thread 发出**。一个 TMA copy 由单个 thread 发起，然后由硬件完成执行。一个 TMEM 到 register 的 load 是 warpgroup-distributed 的：四个 warp 合作，每个 warp 搬运它自己对应的那部分 TMEM tile。一个 `tcgen05` MMA 由一个被选出的 thread 提交，而一个 clustered MMA 则同时跨越两个 CTA。因此，每个操作都有其自身的自然粒度，执行它的那些 thread 集合就是本书反复回归的第一个 recurring design element——scope（scope、layout 和 dispatch）。

## Memory Space

这个 hierarchy 中的 thread 的速度取决于数据到达它们的速度，因此我们接下来要看这些数据存放在哪里。没有一种单一的 memory 既能做到很大又能做到很快；物理规律要求在 capacity 和 speed 之间做权衡。因此，GPU 提供多种 memory 而不是一种，每一个 memory 在 capacity-speed 权衡上处于不同的点，而一个 kernel 的工作方式就是在这些 memory 之间移动数据。每个 space 有它自己的 capacity、自己的 latency，以及关于谁可以访问它自己的规则。

| Memory | Ownership | Role | Notes |
|--------|-----------|------|-------|
| **Global (GMEM)** | Device-wide | Persistent tensor storage | 大容量 HBM，所有 SM 共享 |
| **Shared (SMEM)** | Per-CTA（one SM） | Tile staging | 低 latency scratchpad；B200 上最多 228 KB/SM |
| **Tensor Memory (TMEM)** | Per-CTA | MMA accumulator storage | Blackwell 上新增；被 `tcgen05` 使用 |
| **Register File (RF)** | Per-thread | Scalars 和 per-thread tile fragment | 速度快；存放 epilogue/temp 值 |

按顺序读，这些 space 描述了一条路径。本书中几乎所有 kernel 的 data path 都是 **GMEM → SMEM → (compute) → register → SMEM → GMEM**，而对于 tensor-core kernel，TMEM 位于这条路径的中间，在数学运算运行时存放 accumulator。

在这四个之中，**Tensor Memory (TMEM)** 是唯一一个在 pre-Blackwell 硬件上没有对应物的，它的完整细节要等到 {ref}`chap_tensor_cores` 才会详细介绍。不过现在值得理解它的动机。之前的 GPU 将大型 MMA accumulator 保存在 register 中，在那里它们会争夺稀缺的资源。Blackwell 则把 `tcgen05` accumulator 的 output 写入 TMEM——这是一个 CTA-scoped 的 2D scratchpad，每 CTA 128 lane × 最多 512 个 32-bit 列（该数组在物理上位于 SM 上）。然后 kernel 必须在 epilogue 之前显式地将 TMEM 读回 register。这个额外的步骤不是免费的，由此产生的两个后果会在书中反复出现。第一个是 TMEM read 是 **explicit 且 warpgroup-distributed** 的，由 warpgroup 中的四个 warp 合作完成。第二个是，与 register 不同，TMEM 必须 **explicitly allocated and freed**。

### Distributed Shared Memory Across a Cluster

cluster 是 hierarchy 中唯一一个其成员可以跨越多个 SM 的层级，而这种跨度带来了一种其他层级所不具备的 memory 能力。一个 CTA 运行在一个 SM 上，使用该 SM 的 shared memory 工作，但单个 CTA 的 SMEM budget 是有限的，大型 tile 往往需要超出一个 block 单独能够提供的 operand storage 或更多 reuse。Hopper 的答案是 **thread block cluster**：一组比独立 block 之间更紧密合作的 CTA，因为它们可以一起同步，并且可以读写彼此的 shared memory——这种能力称为 **distributed shared memory (DSMEM)**。Blackwell 保留了 cluster 并在此基础上扩展，增加了 dynamic scheduling（{ref}`chap_clc`）和 2-CTA cooperative MMA。

DSMEM 允许一个 CTA 直接寻址和访问 peer CTA 的 shared memory。一个 thread 可以指定 peer 的 SMEM 中的一个位置，并批量将 tile 直接从自己的 SMEM 复制到 peer 的 SMEM，当字节落地后触发一个 completion barrier（{ref}`chap_async_barriers`）。Part III 中的 2-CTA cluster GEMM 正是基于这种机制构建的，使用它来在两个 CTA 之间共享 operand tile，而无需将数据经由 global memory 重新路由。

下图展示了 CTA cluster 使得可能的额外 DSMEM 跳转；点击某个部分可以看到每个 CTA 拥有什么以及 cross-CTA read 在哪里发生。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../../../../_extra/demo/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: 一个 2-CTA cluster，每个 CTA 拥有一半的 A 和一半的 B，通过 cluster 读对方的 B（DSMEM），这一对 CTA 产生一个 256×256 output tile。*

## Compute: CUDA Cores 和 Tensor Cores

thread 和它们移动的数据必须在一个算术单元处相遇，而一个 SM 提供两种不同的数学引擎而不是一种。这两种引擎之间的分工决定了几乎所有 kernel 的编写方式，并且它们起到互补的作用。

- **CUDA core** 是通用 SIMT ALU。它们运行 scalar 和 vector 指令，处理 index arithmetic、elementwise math、reduction 和 control flow——即围绕繁重矩阵工作的 glue logic。
- **Tensor Core** 是 fixed-function 单元，在 *tile* 粒度执行稠密的 matrix multiply-accumulate，用单条指令计算 $D = AB + C$。

这种分工之所以重要，是因为 Tensor Core 提供的算术 throughput 远远超过 CUDA core——FLOP/s 高出约 10 倍或更多——因此稠密线性代数（GEMM、convolution 和 attention）只有在运行在 Tensor Core 上时才能达到峰值性能。因此，获得性能在很大程度上是一个让那些 Tensor Core 保持有数据可计算的问题。从一个 GPU 代际到下一个代际变化的是 *如何* 编程 Tensor Core 以及它们的结果 *最终* 存放在哪里。Hopper 引入了 asynchronous warpgroup MMA（`wgmma.mma_async`）；Blackwell 的第五代 Tensor Core——`tcgen05`——将 accumulator 放在 Tensor Memory 而不是 register 中，我们在 {ref}`chap_tensor_cores` 中专门讨论它。

cluster 以在 GEMM 章节中反复出现的两种方式来扩展这些 engine。**2-CTA cooperative MMA** 允许两个 CTA 各自贡献它们的 SMEM operand 到一个单一的、更大的 Tensor Core MMA tile 中。**TMA multicast** 让 data-movement engine 的一次 load 将同一个 GMEM tile 同时传送到多个 CTA，消除了单独 load 原本会造成的冗余 global traffic。这两种机制都建立在前面介绍的 distributed shared memory 之上。

## GEMM Data Pipeline

到目前为止，我们已经逐个介绍了这些硬件单元。为了看它们如何协同工作，我们可以用一个典型的 general-purpose matrix multiplication（GEMM）pipeline 作为例子。下面的 interactive demo 展示了三阶段 GEMM tile pipeline 中涉及的各个单元；点击一个操作（如 `tma load`）来高亮显示它跨硬件单元的 data path。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../../../../_extra/demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: Blackwell 上的 load → MMA → epilogue pipeline；点击一个操作来追踪它跨硬件单元的 data path。*

一个 GEMM tile 流经三个阶段。

1. **Load。** 一个 TMA copy（{ref}`chap_tma`）将一个 A 或 B operand tile 从 GMEM 流式传输到 SMEM。一个 thread 发出这个 copy，预先记录预期到达的字节数。当字节落地时，TMA engine 报告它们的进度，一个 completion barrier 仅在所有预期字节都已送达时翻转。
2. **Compute。** 一个 `tcgen05` MMA（{ref}`chap_tensor_cores`）从 SMEM 读出 operand tile，并将乘积累加到 TMEM tile 中。一个被选出的 thread 发出这个指令，当数学运算完成时它发出一个 barrier 信号。
3. **Epilogue。** warpgroup 将 TMEM accumulator 读回 register，将结果 cast 为 output dtype，并将其存储到 GMEM——经常是通过 staging 到 SMEM 并发出一个 TMA store 来实现。

这样写出来，这三个阶段看起来是完全顺序执行的，但慢 kernel 和快 kernel 之间的全部区别在于 **overlap**。一个 naive kernel 确实按顺序执行这些步骤（load，wait，compute，wait，store），因此每次都在等待前一个 engine 时让其他 engine 空转。一个快 kernel 则将它们 pipeline 化：当 Tensor Core 在计算 tile `k` 时，TMA engine 已经在获取 tile `k+1`，而 epilogue 正忙着排出 tile `k-1`，因此所有三个引擎在同一时间都保持忙碌。让三个 asynchronous engine 安全地将工作交接给彼此，正是 barrier 和 phase model（{ref}`chap_async_barriers`）的职责，而 Part III 的 GEMM ladder 就建立在其之上。

## What to Read Next

既然我们已经看到了 high-level 的全貌，接下来可以进入更深入了解主要机制的章节：

- {ref}`chap_tensor_cores` 详细解释 `tcgen05` compute 和 Tensor Memory。
- {ref}`chap_tma` 介绍基于 TMA 的 asynchronous data movement。
- {ref}`chap_async_barriers` 介绍协调这些 engine 的 mbarrier 和 phase model。
