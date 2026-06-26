(chap_performance)=
# 什么让 kernel 变快

:::{admonition} Overview
:class: overview

- roofline model 为 kernel 设定了一个性能上限。该上限由 memory bandwidth 或 compute throughput 决定。
- arithmetic intensity 决定哪个上限适用。它表示每移动一个 byte 所完成的有用 arithmetic 运算量。
- 低 arithmetic intensity 意味着 kernel 是 memory-bound。主要的出路是移动更少的 byte、更多地复用数据、fuse operation 或使用更小的 dtype。
- 高 arithmetic intensity 意味着 kernel 可以是 compute-bound。此时主要任务是保持 Tensor Core 处于忙碌状态。
- 在现代 GPU kernel 中，主要的杠杆是 overlap。只要 dependency graph 允许，TMA、Tensor Core、epilogue 和 store 应该在同一时间运行。
:::

一个 kernel 是否快，只能相对于一个上限来说。像 330 TFLOP/s 这样的数字本身可能看起来很大，但在一个可以持续约 2 PFLOP/s 稠密 fp16 或 bf16 Tensor Core 运算的 GPU 上，它意味着完全不同的事情。没有上限，很难判断一个 kernel 是接近硬件极限，还是仍然让大部分芯片处于空闲状态。

roofline model 给出了那个上限。它将 kernel 分为两类基本活动：移动 byte 和执行算术运算。如果 kernel 不能足够快地移动数据，memory bandwidth 就设定了限制。如果 kernel 有足够的数据复用和足够的算术运算量，compute throughput 就设定了限制。

本章中的数字以 NVIDIA B200 作为运行示例。遵循 {ref}`chap_background` 中的惯例，我们在推理时使用约数：大约 2 PFLOP/s 的稠密 fp16 或 bf16 Tensor Core throughput，以及大约 8 TB/s 的 HBM3e bandwidth。确切值取决于具体的设备、clock、power limit 和测量设置，因此应将它们视为数量级限制，而不是 datasheet 常数。

## Roofline Model

每个 kernel 都移动数据并执行算术运算。roofline model 由这两条路径中较慢的那条来约束 kernel。

compute ceiling 是硬件的最大算术 throughput。对于 B200 上的 Tensor Core GEMM，相关的 ceiling 是 Tensor Core throughput。对于 scalar 或 elementwise kernel，相关的 ceiling 可能是 CUDA core throughput 或其他功能单元。

memory ceiling 是 bandwidth 乘以 arithmetic intensity。如果 kernel 每移动一个 byte 只执行少量算术运算，memory bandwidth 就限制了性能。如果它每 byte 执行大量运算，memory 就不太可能是限制因素。

基本的 roofline bound 是：

```text
可实现的 FLOP/s <= min(峰值 FLOP/s, memory bandwidth * arithmetic intensity)
```

arithmetic intensity 是：

```text
arithmetic intensity = 有用 FLOPs / 移动的 byte 数
```

必须指定 memory level。对于 HBM roofline，byte 是 HBM byte。对于 L2 roofline，它们是 L2 byte。对于 SMEM roofline，它们是 shared memory byte。本章中，默认 roofline 是 HBM roofline。

在 roofline plot 上，x 轴是 arithmetic intensity，单位是 FLOP/byte。y 轴是可实现的 performance。memory roof 是一条斜线：

```text
performance = bandwidth * arithmetic intensity
```

compute roof 是一条水平线：

```text
performance = 峰值 FLOP/s
```

两条线在 ridge point 处相交：

```text
ridge point = 峰值 FLOP/s / bandwidth
```

对于本处使用的 B200 约数：

```text
ridge point ≈ 2000 TFLOP/s / 8 TB/s
            ≈ 250 FLOP/byte
```

arithmetic intensity 低于该值的 kernel 在 HBM roofline 下是 memory-bound。它无法达到峰值 Tensor Core throughput，因为它不能提供足够的每秒 byte 数来供给那么多算术运算。

arithmetic intensity 高于该值的 kernel 可以是 compute-bound。在那一点，memory traffic 不再是首要限制。剩下的工作是将 compute unit 驱动得足够好，以接近平坦的 roof。

roofline model 有用的部分不是 plot 本身。有用的部分在于它告诉 programmer 哪个资源是 binding 的。memory-bound kernel 不会因为其 math instruction 略微改善而变快。compute-bound kernel 不会因为节省了几个无关紧要的 byte 而变快。第一步是知道 kernel 在 ridge 的哪一侧。

![一个 B200 roofline，展示 example workload，包括 memory roof、compute roof 和 ridge point](../../../img/roofline.png)

## Common Workload 的 Arithmetic Intensity

arithmetic intensity 在成为实现细节之前，往往首先是 algorithm 属性。在编写 kernel 之前通常就可以做出粗略估计。

### Elementwise 和 Reduction

elementwise kernel（如 GELU）和 reduction 风格的 kernel（如 RMSNorm）读取和写入大型 tensor，但每 element 只执行少量 FLOP。

它们的 arithmetic intensity 很低。它们位于 ridge point 的左侧很远。这类 kernel 的最佳版本通常试图接近 memory bandwidth roof，而不是 Tensor Core compute roof。

对于这些 kernel，重要的问题是机械性的：

```text
load 和 store 是否 coalesced？
byte 是否只移动了一次？
该 operation 是否可以与 producer 或 consumer fuse？
dtype 是否可以更小？
TMA 或 vectorized access 是否有帮助？
```

如果没有 reuse 且没有 fusion 机会，memory roof 就是真正的 ceiling。

### GEMM

GEMM 是相反的情况。它的 arithmetic intensity 随着问题规模的增长而增长，因为每个加载的 tile 可以被复用于大量 multiply-accumulate 运算。

对于 `M = N = K` 的方形 fp16 matmul，理想的 arithmetic intensity 约为：

```text
AI ≈ 2N^3 / (3 * 2N^2)
   = N / 3 FLOP/byte
```

这个估计假设 A 和 B 各读一次，C 写一次，beta 为零，片上 reuse 是完美的，且没有额外的 metadata、padding 或冗余 traffic。实际 kernel 移动的数据比这个理想模型更多。但这个估计仍然有用。

在 `N = 4096` 时：

```text
AI ≈ 4096 / 3
   ≈ 1365 FLOP/byte
```

这远远在 B200 约 250 FLOP/byte 的 ridge point 的右侧。因此，大型 GEMM 在 HBM roofline 下是 compute-bound。目标不仅仅是减少 HBM traffic。目标是要使用 Tensor Core、保持它们有数据可计算，并将 data movement 与 compute overlap，从而使 compute roof 变为可达。

这就是为什么 naive GEMM 即使 GEMM 具有高 arithmetic intensity，仍然可能运行缓慢。algorithm 允许高性能，但 implementation 可能会让 Tensor Core 处于空闲状态。

### Attention

attention 介于这两个极端之间。它的 arithmetic intensity 取决于 sequence length、head dimension、tiling、masking，以及是否将 intermediate tensor materialize。

standard attention 中的关键问题是 score matrix。如果 kernel 将 score matrix 写入 HBM 后来又读回，它会将一个大型 intermediate 通过 memory 搬运。Flash Attention（{ref}`chap_flash_attention`）通过将相关 tile 保持在片上并避免那个 HBM round trip，从而提高了 arithmetic intensity。

因此，attention 优化部分是一个 roofline 问题，部分是一个 scheduling 问题。algorithm 被调整，使更少的 byte 进入 HBM。然后 kernel 被 scheduling，使剩余的 movement 和 compute overlap。

## Arithmetic Intensity 较低时

如果一个 kernel 在 ridge 的左侧，它就是 memory-bound。Tensor Core 或 CUDA core 可能处于空闲状态，因为瓶颈是 byte，而不是 arithmetic instruction。

有两种应对措施。

第一种应对措施是提高 arithmetic intensity。这是 leverage 更高的路径，因为它可以将 kernel 推向 compute-bound 区域。

最重要的技术是 fusion。低 arithmetic intensity 的一个常见来源是将 intermediate tensor 写入 HBM，然后在下一个 operation 中立即读回。fuse producer 和 consumer 可以将那个 intermediate 保留在 register、SMEM 或 TMEM 中。HBM round trip 消失了。

例子包括：

```text
GEMM 加上 elementwise epilogue
normalization 折叠到相邻 op 中
attention 计算不 materialize 完整的 score matrix
```

第二种技术是 blocking 以实现 reuse。如果一个 tile 只被加载一次，在驱逐前被多次使用，每个 byte 就能支持更多的 arithmetic work。GEMM 的高 arithmetic intensity 正是来自这种 reuse。其他 workload 在它们对某个 tile 有重复使用时，也可以运用同样的思路。

第三种技术是减少每值的 byte 数。从 fp32 变为 fp16、fp8 或 fp4 减少了 traffic 并增加了每 byte 的 FLOP。当格式需要 metadata、scale factor 或额外 conversion work 时，实际收益比原始 dtype 比值要小。block-scaled fp8 和 fp4 就是这类例子。即便如此，更小的 dtype 往往仍然是将 kernel 在 roofline 上向右推动的最直接方法之一。

第二种应对措施是接受 memory roof 并尝试达到它。有些 kernel 没有足够的工作可以 fuse，也没有足够的 reuse 可以利用。纯 copy、简单 elementwise operation，或对大型 tensor 的单 pass reduction，可能从根本上就是 memory-bound 的。

在这种情况下，目标不是超越 roof。目标是饱和它。

这意味着：

```text
每个 byte 只移动一次
避免冗余读
使用 coalesced 或 vectorized access
对于规则的 bulk tile 使用 TMA
保持足够的 memory request 在 flight
当 algorithm 允许时使用更小的 storage dtype
```

一旦 memory-bound kernel 达到了 memory roof，进一步的 compute 优化就没有帮助了。加快的唯一方法是改变 algorithm，使它移动更少的 byte。

## Optimization Ladder

roofline 说明了什么是不可能的。但它并没有说明达到那个限制有多容易。

一个大型 fp16 GEMM 在理论上可能是 compute-bound。这只意味着 HBM roof 不是主要限制。它并不意味着任何 implementation 都会达到 Tensor Core roof。缩小差距需要正确的 instruction、layout、staging、synchronization 和 scheduling。

Part III 中的 GEMM kernel 在 B200 上以一系列步骤展示（{ref}`chap_gemm_advanced`）。每一步保持相同的基本 algorithm，但改变 tile 的计算或调度方式。

GEMM ladder 中第一个可测量的大幅跃升，是从 thread-copy tiled 路径转向 TMA-backed 路径。TMA 将规则的 GMEM → SMEM tile movement 从 CTA thread 上卸载，使 kernel 通过 hardware-managed bulk copy 来给 Tensor Core 喂数据。

在那第一个跃升之后，主要改进来自 overlap 和 scheduling。TMA 将未来的 tile 带入 shared memory。`tcgen05.mma` 异步运行。epilogue 排出之前的结果。software pipelining 和 warp specialization 将这些组件组织起来，使 hardware engine 在同一时间处于活跃状态。

也没有规则要求每个 intermediate step 必须各自更快。像 warp specialization 这样的步骤可能会暂时将资源花费在一个不立即改善数字的结构上。如果它启用的是更简单结构无法表达的后续 overlap，它仍然可以是正确的步骤。

![B200 上的 GEMM 优化旅程：从同步 tiled baseline 到 TMA、warp specialization、CTA cluster 和 multi-consumer execution 的测量点](../../../img/gemm_perf.png)

## Overlap 是主要的杠杆

一旦 GEMM 是 compute-bound 且已经使用 Tensor Core，剩余的差距通常来自 idle time。

一个简单 kernel 可能会这样做：

```text
load tile k
compute tile k
store tile k
load tile k + 1
compute tile k + 1
store tile k + 1
```

这种 schedule 使硬件空闲。当 load 运行时，Tensor Core 等待。当 Tensor Core 运行时，copy engine 可能空闲。当 store 排出时，两者都可能等待。

一个 pipelined kernel 则尝试将独立 stage 一起运行：

```text
load tile k + 1
compute tile k
store tile k - 1
```

这是本书后面使用的 Blackwell kernel 结构背后的核心思想。TMA 处理 asynchronous data movement。`tcgen05.mma` 处理 asynchronous Tensor Core work。epilogue 和 store 处理 output side。`mbarrier` 对象连接各个 stage，使每个 consumer 仅在确实需要数据时才等待。

关键不是消除 dependency。关键是在 dependency 周围做 scheduling。tile `k` 的 MMA 不能在 tile `k` 被 load 之前开始。tile `k` 的 epilogue 不能在 tile `k` 的 MMA 完成之前读取 accumulator。但 tile `k + 1` 的 load 通常可以在 tile `k` 的 MMA in flight 时运行，而 tile `k - 1` 的 store 通常可以同时排出。

这就是为什么后面的许多章节聚焦于 asynchronous mechanism：

```text
TMA 用于 global memory 到 shared memory 的 movement
mbarrier 用于 load completion 和 resource handoff
tcgen05 用于 asynchronous Tensor Core compute
TMEM 用于 long-lived accumulator
warp specialization 用于分离 producer 和 consumer role
cluster 用于更大的 cooperative tile 和 multicast
```

它们是不同的 mechanism，但服务于同一个 scheduling 目标：在同一时间在多于一条 hardware path 上运行有用 work。

## Occupancy 和 Resource Pressure

overlap 不是唯一的 latency-hiding mechanism。更古老且更通用的机制是 occupancy。

occupancy 是驻留在 SM 上的 work 量。如果一个 warp stall，scheduler 可以运行另一个准备好的 warp。它通过保持一个独立 warp pool 可用性来隐藏 latency。

occupancy 受 per-SM resource 限制。主要限制是 register、shared memory、warp slot 和 CTA slot。一个每 thread 使用大量 register 或每 CTA 使用大量 shared memory 的 kernel 可能具有低 occupancy，因为只有少量 CTA 或 warp 能装入 SM。

许多现代 Tensor Core kernel 故意以降低 occupancy 的方式消耗资源。多 stage shared memory pipeline 消耗 SMEM。大型 register fragment 消耗 register。TMEM allocation 消耗 Tensor Memory 容量。warp specialization 可能为 producer 或 consumer role 预留整个 warp。

这种 trade 是有意为之。与其通过让大量无关 warp 驻留来隐藏 latency，这些 kernel 通过在更少量驻留 CTA 内的 explicit overlap 来隐藏 latency。一个 low-occupancy kernel 如果其 pipeline 使 TMA、Tensor Core 和 store 保持忙碌，仍然可以很快。

两种方法都没有普遍更好的。有些 kernel 需要高 occupancy，因为它们有不规则的 memory access 或有限的 explicit overlap。其他 kernel 需要 deep staging 和 specialization，因为这是有效 feed Tensor Core 的唯一方式。正确的问题不是 occupancy 是否高。正确的问题是 active hardware unit 是否保持忙碌。

## This Buys Later

本书的其余部分不断回归到同一个诊断：

```text
这个 kernel 在哪个 roof 之下？
什么 resource 是 binding 的？
什么 change 使 kernel 更接近那个 roof？
```

对于 memory-bound kernel，答案通常是更少的 byte 和更好的 bandwidth 使用。这意味着 fusion、coalescing、vectorized access、适用的地方使用 TMA，以及更小的 dtype。

对于 compute-bound GEMM，答案是先 Tensor Core，然后 overlap。kernel 必须 stage operand、发出 asynchronous MMA work、保持 pipeline 饱满，并在不 stall compute path 的情况下排出结果。

对于 Flash Attention，第一步是通过将 score 和 probability tile 保持在片上来提高 arithmetic intensity。之后，它使用与 GEMM 相同的 overlap 工具：tiled data movement、shared memory staging、asynchronous compute 和 careful resource handoff。

这提供了一个实用的 optimization workflow。估计 arithmetic intensity。定位 roof。决定 kernel 是 memory-bound 还是 compute-bound。然后优化实际设定 ceiling 的 resource。

没有那一步，kernel optimization 就变成了 guesswork。有了它，每个 change 都有一个理由：要么它提高了 arithmetic intensity，要么它使 memory path 更接近 bandwidth peak，要么它减少了 compute roof 下的 idle time。
