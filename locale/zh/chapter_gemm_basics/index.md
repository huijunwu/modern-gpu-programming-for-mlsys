(chap_gemm_basics)=
# 构建 Tiled GEMM

:::{admonition} Overview
:class: overview

- 从 TIRx tile primitive 构建正确 tiled GEMM，从单个 output tile 开始。
- Step 1 是 single-tile GEMM，Step 2 添加 K-loop accumulation，Step 3 跨 CTA spatial tile。
- correctness 优先；performance 是接下来两章的工作。
:::

GEMM 是这本书整个围绕的 workload。它在 linear layer、attention projection 和大多数 convolution implementation 下面，因此正确 GEMM 和快速 GEMM 之间的差异是在让大部分芯片 idle 和 saturate 它之间的差异。

那个 gap 太大，不能一次跨越。一个 saturating kernel 使你在同一时间 debug memory movement、accumulation、tiling 和 Tensor Core scheduling，没有可信赖的东西可比较。更 safe path 是从产生正确 answer 的最 small kernel 开始，然后一次一个 decision 增长它。

本章写那个 first correct tiled GEMM。之前章节介绍了 TIRx scope / layout / dispatch model 抽象；这里我们将其应用到 real kernel。我们从 128 × 128 output tile 开始并增长它为处理 full-size matrix 的 kernel，添加 K-dimension accumulation 然后跨多 CTA spatial tiling。

这是三个 walkthrough GEMM optimization path 章节中的第一个。在这个我们构建正确 tiled kernel 并停在那里。下一章（{ref}`chap_gemm_async`）用 TMA 替换 thread copy 并通过 pipelining overlap data movement 与 compute，{ref}`chap_gemm_advanced` 进一步用 warp specialization 和 CTA cluster。每章建立在之前一章上，因此 kernel 积累 feature 而非重新开始。

将每步读为编辑 single contract 带三项 terms 是有帮助的：哪个 **scope** 运行 operation、operand tile 使用哪个 **layout**、哪个 **dispatch** path 执行它。大多数步有一 primary change，因此我们打开它们用小 card 命名那个 change 并 call out 使 reuse safe 需要的任何 synchronization detail。Step 1 建立 baseline 其余 path 编辑。

## GEMM

GEMM 是 sit under linear layer、attention projection 和许多 convolution implementation 的 dense matrix multiply，因此快速 GEMM kernel 几乎在你查看的任何地方都有 payoff。本 tutorial 的 example 使用 $D = A B^{\top}$：

- $A$ 有 shape $M \times K$。
- $B$ 有 shape $N \times K$。
- $D$ 有 shape $M \times N$。
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$。

transpose 不是我们选择执行的 extra operation；它来自 data 如何存储。example 保持 $B$ 为 $N$ 行 length $K$，那是 linear-layer weight 通常来的 layout，因此沿 $K$ contracting 自然地读 $B^{\top}$ 无需任何 rearrangement。

贯穿 tutorial，我们按 TFLOPS 衡量 kernel throughput，计算每 multiply-add 的两个 floating-point operation 对 wall-clock time：

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM Data Path

本 tutorial 的每 optimization 归结为 data 存放哪里和如何移动，因此在我们写任何 code 前将那个 map out 是 worth。在核心，Blackwell GEMM kernel 围绕两个 activity 组织：在 memory 间移动 tile 和在它们上 compute。下图追踪 tile 从 input 到 output 接触每 memory：

../../../../img/memory_dataflow.png)

上图显示 baseline path 每 later optimization 编辑但从不替换。
从左到右读：operand tile 首先从 GMEM 移到 SMEM；`tcgen05.mma` 然后
消费 SMEM operand 并将 accumulator 写入 TMEM；最终 epilogue 将 TMEM
读回 register 前将 result 存储到 GMEM。记住那个 chain，因为每
step
下面改变 *how* 那跳之一发生；它从不改变跳本身。

## Optimization Path

上面 plain data path 足以得到正确 answer，但它使大部分 hardware idle。tutorial 其余部分通过一次一个添加 Blackwell feature 关闭那个 gap，每 feature 表达通过 TIRx tile primitive。我们将遵循的 path 依次访问那些 feature：

- **TMA async movement** 通过 Blackwell hardware copy path 移动 GMEM <-> SMEM tile，barrier 跟踪 completion。
- **Software pipelining** 使用多 SMEM stage 使 next K tile 的 data movement 可与当前 Tile 的 Tensor Core compute overlap。
- **Persistent scheduling** 保持固定 CTA pool，每 CTA 通过 tile scheduler 处理多 output tile，而非 per tile 启动一个 CTA。
- **Warp specialization** 将 producer、MMA consumer 和 writeback role 分到单独 warpgroup。
- **CTA cluster** 使两个 CTA 合作在单个、更大 Blackwell MMA tile 上。
- **Multi-consumer execution** 使用多 consumer warpgroup 同时计算 tile 不同 part，提高 compute density。

---

(chap_single_tile)=
## Step 1: Sequential Single-Tile GEMM

仍然 exercise full hardware path 的最 simple GEMM 是计算单个 output tile 的那个。因此那是我们开始的地方。Step 1 计算一个 128 × 128 output tile 带 K = 64，足够小使 nothing 必须 loop，data path 每 piece 出现 exactly once。没有重复，我们可以在线循环前隔离看到每跳。

> **这个 step 建立什么：baseline**
> - Scope: 单个 warpgroup 的 128 thread 顺序走整个 path，一 stage 后另一。
> - Layout: A 和 B tile 存放在 SMEM，accumulator 在 TMEM，result 通过 register stage out。
> - Dispatch: sync `Tx.copy` 携带 load，`tcgen05` 运行 MMA。

### Single-Tile Dataflow

baseline contract fixed，下一件事 pin down 是一个 tile 通过它 travel 的顺序。这个 first kernel 走 core GEMM data path exactly once，相同 GMEM → SMEM → TMEM → register → GMEM chain 来自 data-flow figure，没有 loop 包裹它。它 alloc 其 working memory、load operand、compute product、write result back、并清理它自己：

1. **Alloc**: SMEM（pool allocator）、TMEM（`tcgen05.alloc`）、mbarrier
2. **Load**: 所有 128 thread 合作复制 A 和 B tile 从 GMEM 到 SMEM（sync `Tx.copy`）
3. **Compute**: 单个 elected thread 发出 `Tx.gemm_async` + `tcgen05.commit`；所有 thread 在 mbarrier wait
4. **Writeback**: warpgroup 读 TMEM → register；每 thread cast fp32→fp16 并 write 到 GMEM
5. **Dealloc**: TMEM dealloc

### First Kernel 的四部分

full kernel 只有几 dozen line，但它在 parts 中 easier to digest。我们将在四个 parts 中读它（memory allocation、sync load、MMA dispatch 和 writeback）并在之后才将它们组装为一个 kernel。沿途出现的 API name 是 Part II 介绍的 TIRx tile-primitive vocabulary（{ref}`chap_tirx_primer`、{ref}`chap_tirx_layout_api`）。

**Memory allocation。** kernel 从为 operand 切出 shared memory 开始，加上 TMEM address slot 和 mbarrier：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

这里两个 detail 值得 pause。`pool.move_base_to(1024)` 将 Asmem 和 Bsmem 推到 offset 1024，为上面小 piece metadata 保留低 address，使 bulky operand tile 落在 clean boundary。而 `layout=A_layout` 要求 `tma_shared_layout` 为 swizzled SMEM placement 使 TMA 和 `tcgen05.mma` 都可直接读，恰好 Part II 描述的 layout-as-contract obligation。

**Sync load。** buffer 在 place 后，operand 仍必须 reach SMEM。这个 first version 我们让 CTA 自己 thread 做 copy：

```python
T.copy(A, Asmem)
T.copy(B, Bsmem)
```

那是 sync `Tx.copy`。它使所有 128 thread 合作将 A tile 从 GMEM 复制。每个 thread 计算它应 load 的 element 的 global address、load 那个 value、然后 store 它在 SMEM 匹配 layout。

copy 是 sync 因为 thread 在 `T.copy` 返回前不 move on。那使 safe：在 MMA 运行前 operand 保证在 SMEM。它不是 performance-optimized——thread 在 copy 时没有 execute other work——但它是 correct baseline。

**MMA dispatch。** operand 在 SMEM 后，kernel 发出 MMA。

```python
tcgen05.mma(
    acc=acc,
    A=Asmem,
    B=Bsmem,
    kind="wgmma.mma_async.z.vf8.m128n128k16.f16.f32",
)
tcgen05.commit()
T.mbarrier_wait(mma_bar, phase=1)
```

`tcgen05.mma` 发出 MMA。它是 asynchronous——instruction 提交 MMA 到 Tensor Core 并 return 而不 wait 完成。那使 kernel 在 Tensor Core 计算时开始 other work。

`tcgen05.commit` 提交 MMA。它是 explicit：在 MMA complete 前 barrier 不 receive arrival。

`T.mbarrier_wait` 在 MMA complete 前 wait。barrier 在 `tcgen05.commit` 提交 work 后 receive MMA arrival。当 barrier phase flip，MMA result 在 TMEM 中 ready。

**Writeback。** MMA complete 后，epilogue read accumulator 从 TMEM、convert dtype、并 store result 到 GMEM。

```python
acc = T.tmem_load(acc)
acc = T.cast(acc, "float16")
T.copy(acc, C)
T.tmem_free(acc)
```

`T.tmem_load` 将 accumulator 从 TMEM 加载到 register。它是 warpgroup-distributed：四个 warp 合作 read 完整 128 Lane TMEM。

`T.cast` 将 fp32 accumulator value 转换为 fp16。

`T.copy` 将 result 从 register 存到 GMEM。

`T.tmem_free` dealloc TMEM 分配。

### Full Kernel

组装 parts：

```python
@tirx.script
def gemm_kernel(A, B, C):
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    a_type = A.dtype
    b_type = B.dtype

    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    mma_bar = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
    pool.commit()

    T.copy(A, Asmem)
    T.copy(B, Bsmem)

    acc = tcgen05.alloc(tmem_addr, shape=(BLK_M, BLK_N), dtype="float32")
    tcgen05.mma(
        acc=acc,
        A=Asmem,
        B=Bsmem,
        kind="wgmma.mma_async.z.vf8.m128n128k16.f16.f32",
    )
    tcgen05.commit()
    T.mbarrier_wait(mma_bar, phase=1)

    acc = T.tmem_load(acc)
    acc = T.cast(acc, "float16")
    T.copy(acc, C)
    T.tmem_free(acc)
```

那是 correct single-tile GEMM。它 exercise full hardware path：GMEM → SMEM（通过 thread copy）、SMEM → TMEM（通过 `tcgen05.mma`）、TMEM → register（通过 `T.tmem_load`）、register → GMEM（通过 `T.copy`）。

它不是 fast——它没有 pipeline、没有 TMA、没有 warp specialization——但它是 correct baseline。那 baseline 是 rest of the optimization path 建立的基础。
