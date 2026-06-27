(chap_gemm_async)=
# 用 TMA Pipeline GEMM

:::{admonition} Overview
:class: overview

- 基本 GEMM 浪费时间轮流执行（copy tile、compute、copy 下一个），而两者可以同时运行。
- Step 4 切换到 TMA async load，Step 5 double-buffer SMEM 并 prefetch（PIPE_DEPTH=2）；完整的 load/compute overlap 在 Step 7 与 warp specialization 一起实现，Step 6 用 tile scheduler 使 kernel persistent。
- 目标是在 Tensor Core 处理当前 tile 的同时加载下一个 tile。
:::

Tensor Core 是芯片上最昂贵的 unit，上一章中正确的 tiled GEMM 使它们在大部分时钟周期内 idle。kernel 轮流执行：thread copy tile 到 shared memory，Tensor Core 处理它，thread copy 下一个 tile，Tensor Core wait。每个 stage stall 在前一个 stage 上，尽管加载下一个 tile 和计算当前 tile 使用完全不同的 hardware 且可以同时运行。缩小这个 gap 不需要新的 data path；tile、layout 和 math 已经正确。必须改变的是*何时*执行 work 和*由谁*来 schedule。本章保持 tile data path 不变，直接消除 idleness。

我们分三个渐进式 step 达到目标，在开始之前了解目标是有帮助的。在 Step 4 我们将 bulk GMEM ↔ SMEM transfer 交给 TMA，使 dedicated copy hardware 移动 tile 而不是 thread。在 Step 5 我们添加两 stage software pipeline，给下一个 K tile 一个可以落地的地方，而当前 tile 仍在被 multiply。在 Step 6 我们将 launch reshape 为 persistent kernel，由 tile scheduler 驱动，它 amortize per-tile setup 并让我们选择 tile order 保持 operand hot。贯穿始终，SMEM、TMEM 和 register layout 与上一章完全相同。唯一真正新的 idea 是 hardware unit 间的 asynchronous handoff：让一个 engine 领先于另一个运行，而不是 lockstep 地行进。

(chap_tma_async)=
## Step 4: TMA Async Load

我们的第一步是将 copy 本身从 critical path 移开。想想 CTA 在 Step 1-3 中做了什么：它的每个 thread 计算 address 并 issue load instruction，唯一原因是将 tile 搬运到 SMEM。这是将 instruction bandwidth 花在 plumbing 上而不是 math 上。Step 4 将 synchronous `Tx.copy` 替换为 TMA，一个 thread issue 一个 command，TMA engine 自己完成整个 tile transfer。从这里开始，example 在完整的 M=N=K=4096 size 下运行，而不是 Step 1-3 的小 size；它们的 end-to-end timing 出现在 {ref}`chap_gemm_advanced` 末尾的 *End-to-End Result* 表中。

> **这个 step 改变了什么：Dispatch**
> - Scope：不变，一个 warpgroup。
> - Layout：不变，相同的 SMEM/TMEM/register tile。
> - Dispatch：GMEM → SMEM load 从 sync `Tx.copy` 移到 TMA engine。

### TMA Issue Pattern

Step 4 的唯一变化是将 synchronous tile copy 替换为 TMA load，因此仔细观察 load 如何 issue 是值得的。对 source 的 edit 只有几行，但这些行背后的 execution model 在本质上不同。synchronous `Tx.copy` 是 CTA thread 用自己的 instruction 完成的 work；TMA copy 是一个 thread issue 的 command，之后 TMA hardware 完成所有搬运。将两者并排看是值得的。

**之前（Step 3）**：全部 128 个 thread 参与 copy，然后 `cta_sync` 使 shared memory write 可见：
```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])   # 全部 128 个 thread
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

**之后（Step 4）**：一个 thread issue TMA load，mbarrier 跟踪 hardware transfer 何时完成：
```python
tid = warp_id * 32 + lane_id                 # warpgroup 内 0..127
if tid == 0:  # 恰好一个 thread 启动 TMA
    Tx.copy_async(Asmem, A[...], dispatch="tma")
    Tx.copy_async(Bsmem, B[...], dispatch="tma")
    T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)  # TMA 期望的 byte
T.ptx.mbarrier.try_wait(tma_bar, phase)                  # MMA 读取 SMEM 前 wait
```

注意 load 的 gate 是 `tid == 0`，不是 `elect_sync()`，这个区别比看起来重要。`elect.sync` 每 warp 选举一个 active lane，warpgroup 有四个 warp，因此 `elect_sync()` 实际上会让四个 thread 进入 load protocol。问题是 protocol 向 mbarrier 宣布期望的 byte count，且必须恰好宣布一次；四次宣布将 corrupt count，wait 永远不会正确 release。通过 warpgroup-wide id 精确选择一个 thread 是避免这个问题的干净方法。

诚实地说明 speedup 来自哪里很重要。Step 4 仍然在每个 TMA load 后 wait，所以我们还没有 overlap load 与 compute；那是 Step 5 的工作。这里的 win 纯粹来自 data-movement path 的改变：

- `Tx.copy` 使用 CTA thread 计算 address 并 issue load/store instruction。
- TMA 使用一个 issued command 启动 hardware tile transfer。address generation、coalescing 和 swizzling 由 TMA descriptor 描述并由 TMA engine 执行。

因此尽管 Step 4 仍然在每个 load 上 block，它仍然更快。TMA 吸收 bulk transfer，使 CTA thread 不必花费 instruction bandwidth 搬运 tile，仅这个 saving 就足以改变性能。

### TMA Load 和 Store Synchronization

我们已经看到 TMA copy 如何 issue；故事的另 half 是知道它何时完成。切换到 TMA 同时改变两件事：谁启动 copy，以及 code 如何知道它何时完成。第一点从 code 中显而易见；第二点容易忽略，出错会给你 silent correctness bug 而不是 crash。使用 `Tx.cta.copy` 时，CTA thread 一起完成 copy，后续的 `cta_sync()` 足以知道它完成。使用 TMA 时，一个 selected thread issue `Tx.copy_async(..., dispatch="tma")`，engine 按自己的 schedule 执行 transfer，并通过 mbarrier 信号 completion。

这正是 `cta_sync()` 不再足够的原因。`cta_sync()` 只 wait CTA 自己的 thread 且只 order 它们的 shared memory write；它不知道 in-flight TMA transfer，因此它会 happily return 而 tile 仍在 arriving。fix 是使 completion explicit：对于 TMA load，selected thread 首先告诉 mbarrier 期望多少 byte，然后 CTA 在任何 MMA 触碰 SMEM tile 之前 wait *那个* mbarrier。下面的图 end-to-end 追踪那个 handshake。

![TMA Async Load: Synchronization Flow](../../img/tma_sync_flow.png)

上图隔离了 load-side handshake：一个 selected thread 启动 TMA，mbarrier 计算期望的 byte，MMA 在读取 SMEM 之前 wait release。图中"Elected Thread"指的是启动 TMA 的 selected thread，在我们的 code 中是 `tid == 0` thread，不是 `elect_sync()` lane。

总结 load path：selected thread issue 两个 `copy_async` call，然后跟 `arrive.expect_tx(total_bytes)`，byte count 精确是 mbarrier 应该等待的 data 量。一旦 engine 移动了那么多 byte，匹配的 `mbarrier.try_wait(phase)` release，只有那时 SMEM tile 才可以安全地 feed 给 MMA。

store side 经过相同的 hardware 但以不同方式 wait，因此在头脑中清楚区分两个 protocol 是值得的：load 用 mbarrier 和 byte count 跟踪 completion，store 用 commit group 和 wait group 跟踪它。thread 将 fp16 result 写入 `Dsmem` 并 synchronize 后，一个 selected thread 启动 `Tx.copy_async(D[...], Dsmem, dispatch="tma")`，然后 `cp_async.bulk.commit_group()` 后跟 `cp_async.bulk.wait_group(0)` block 直到 store drain。那个 wait 不是 optional：在 previous store 完成之前 `Dsmem` 不能 reuse 给下一个 tile。

**与你的 agent 一起尝试**：追踪 Step 4 一个 K tile 的 load 和 store synchronization。识别哪个 thread 启动每个 TMA command、哪个 mbarrier 或 commit group 跟踪 completion、哪个 wait 保护 MMA 读取 `Asmem` 和 `Bsmem`、哪个 wait 保护 `Dsmem` 的 reuse。为什么 `elect_sync()` 是 TMA load protocol 的错误 thread selection？

### Complete Kernel

complete kernel 将 TMA load 和 store 折叠到 Step 3 structure 中，保持其余结构不变。import 与之前相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

它包装在 `hgemm_v4(M, N, K)` 中，这是我们贯穿始终遵循的 pattern：wrapper 将 shape-dependent constant 和 layout 紧靠使用它们的 kernel。

```python
def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    F16_SIZE = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation（现在包括用于 TMA store 的 Dsmem）---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        # --- Inline helper ---
        @T.inline
        def tma_load(k_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([0])
            })
            Tx.copy_async(Asmem[:, :],
                          A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[:, :],
                          B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([0]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
            )

        @T.inline
        def mma(accum):
            Tx.gemm_async(
                tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                accum=accum, dispatch="tcgen05", cta_group=1
            )
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # --- K-loop with TMA async ---
        tid = T.meta_var(warp_id * 32 + lane_id)
        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)

            # 单个 thread issue TMA load
            if tid == 0:
                tma_load(k_st)

            # Wait TMA 完成；mbarrier release 携带 SMEM
            # visibility 到后续 MMA，因此不需要额外 fence。
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)

            # 单个 thread issue MMA
            if tid == 0:
                mma(accum=k != 0)

            # Wait MMA 完成
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_tma ^= 1
            phase_mma ^= 1

        # --- TMA Store Writeback ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        # 读 TMEM -> register（async；wait.ld 然后 cta_sync 确保 read 完成）
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        # Cast fp32 -> fp16
        Tx.cast(Dreg_f16[:], Dreg[:])
        # 写 register -> Dsmem，flush，然后 sync
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        # TMA store: Dsmem -> GMEM。一个 selected thread 启动 store 并在
        # Dsmem 被 reuse 之前 drain store group。
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

### Kernel 中的 TMA Configuration

那个 kernel 中几乎 everything 都是从 Step 3 带来的。只有五个 configuration point 实际携带 TMA 语义，值得逐个了解：

- **TMA config**：`{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` 告诉 `Tx.copy_async` 使用 TMA 并通过 `tma_bar` 报告 load completion。

- **Byte count**：`(BLK_M * BLK_K + BLK_N * BLK_K) * 2` 是两个 fp16 operand tile 加载的 byte 数。`arrive.expect_tx(...)` 将这个 count 给 mbarrier。

- **mbarrier initialization**：`init(tma_bar.ptr_to([0]), 1)` 创建 TMA load 使用的 completion barrier。

- **`@T.inline`**：`tma_load(...)` 和 `mma(...)` 是 helper function。它们在 compile time 展开到 kernel body 中，可以使用周围 kernel 的 variable。

- **TMA store synchronization**：epilogue 首先将 fp16 row 写入 `Dsmem`。`fence.proxy_async` 和 `warpgroup_sync` 使那些 thread-written SMEM value 准备好供 TMA store path 使用。store 然后使用 `commit_group()` 和 `wait_group(0)` wait SMEM-to-GMEM transfer 完成。

此时我们有正确的组件但错误的节奏。Step 4 仍然在启动匹配的 MMA 之前完成每个 load，因此 load 和 multiply 实际上从未同时运行；我们费了那么大劲分开的两个 engine 仍然轮流执行。下一步保持 TMA load 和 store path 不变，而是 rearrange schedule，使加载一个 K tile 可以在另一个 tile 上 compute 时进行。

(chap_software_pipeline)=
## Step 5: Software Pipeline（PIPE_DEPTH=2）

为什么 Step 4 不能 overlap load 与 compute，当两个 engine 明显独立时？障碍原来是 storage。只有一个 SMEM tile pair 时，下一个 load 无处可去：在当前 MMA 读完那个 pair 之前它不能开始，因为过早开始会覆盖仍在使用的 data。Step 5 通过 double-buffering shared memory 消除那个 storage conflict。single-warpgroup loop 仍然在每个 MMA 之后 wait 再启动下一个 TMA load，但它现在有 distinct stage 可以 prefetch 和 reuse。我们仍然在完整的 M=N=K=4096 size。

> **这个 step 改变了什么：Layout**
> - Scope：不变，一个 warpgroup。
> - Layout：单个 SMEM tile pair 变为 `PIPE_DEPTH`-stage ring buffer。
> - Dispatch：不变，TMA load 和 `tcgen05` MMA；这个 step 添加 prefetch 和 stage reuse，而完整的 load/compute overlap 在 Step 7 实现。

### Pipeline Walkthrough

使用 `PIPE_DEPTH=2` 时，kernel 分配两个 SMEM stage，给 load path 和 MMA path 单独的 slot 来工作。

将下面的图读作两 stage buffer 旨在启用的 pipeline structure，而不是这个 single-warpgroup kernel 的精确 execution trace。Step 5 构建 ring buffer 并 prefetch 后续 stage，但 main loop 仍然在 issue 下一个 TMA load 之前 wait 当前 MMA。完整的 load/compute overlap 在 Step 7 实现，warp specialization 给 TMA 和 MMA 单独的 role。

![*Pipeline PIPE_DEPTH=2，目标 schedule；这个 single-warpgroup step 只 prefetch，完整 overlap 在 Step 7 与 warp specialization 一起实现*](../../img/pipe_depth2.png)

一旦 primed，loop 在两个 stage 间交替。两个 TMA load 预先填充两个 stage；之后，loop wait 当前 stage，在其上运行 MMA，wait 那个 MMA 读完 stage，然后为 `k + PIPE_DEPTH` 启动 load 到刚刚变为 reusable 的 stage。这还不是 concurrent TMA/MMA schedule，但它建立了 ring-buffer structure，Step 7 将把它拆分到 producer 和 consumer role 中。

具体而言，code 在四个方面与 Step 4 不同：

1. `Asmem` 和 `Bsmem` 获得前导 `PIPE_DEPTH` dimension，使每个 stage 有自己的 SMEM storage。
2. `tma_bar` 变为数组，每个 stage 一个 mbarrier。
3. 在 main K loop 之前，kernel prefetch 前两个 stage。
4. K loop 使用 `stage = k % PIPE_DEPTH`：wait 当前 stage，在其上运行 MMA，然后为 `k + PIPE_DEPTH` reuse 那个 stage。

### Pipeline Mechanics

**1. Prefetch**：在 main loop 运行之前，我们加载前 `PIPE_DEPTH` 个 stage，使 loop 在第一次迭代时总是找到等待的 data：
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
    tma_load(s, s * BLK_K)
```

**2. Main loop**：对于每个 K tile 我们 wait 它的 stage 就绪，在其上运行 MMA，然后通过为 `PIPE_DEPTH` 之前的 tile 启动 load 立即将现在空闲的 stage 重新投入工作：
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

**3. Phase management**：这是容易出错的地方，但 rule 比初看简单。每个 barrier 的 phase-flip rule 直接来自那个 barrier 有多少 slot，这正是两个 barrier 在不同节奏上 flip 的原因。MMA accumulator 生活在单个 TMEM slot 中，因此 `mma_bar` 是单个 barrier（`mma_bar.ptr_to([0])`），每次迭代都 revisit，每次迭代 revisit 的 barrier 必须每次迭代 flip 它的 phase。TMA barrier 讲述不同的故事：它们形成 `PIPE_DEPTH`-element 数组，每个 stage 一个 barrier，任何给定 stage 的 barrier 每轮 ring 只回来一次。因此 `phase_tma` 只在 stage index 回绕到 0 时 flip：
```python
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

**与你的 agent 一起尝试**：使用 `PIPE_DEPTH=2` 和 `K_TILES=5`，要求它追踪 main loop。对于每个 `k`，列出 `stage`、传递给 wait 的 `phase_tma` 和 `phase_mma` value，以及是否 issue 新的 prefetch。`phase_tma` 精确在哪里 flip，为什么最后两次迭代没有 prefetch？

### Complete Kernel

complete kernel 逐字保留 Step 4 的 TMA load 和 store path，然后用我们刚描述的 staged buffer 和 phase logic 包装它。import 不变：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

它包装在 `hgemm_v5(M, N, K)` 中。`PIPE_DEPTH=2` constant 设置 pipeline stage 数（这里两个，恰好是 double buffering）：

```python
PIPE_DEPTH = 2

def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    # Double-buffered layout：第一 dimension 是 pipeline stage
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        # Double-buffered TMA barrier（每个 stage 一个），单个 MMA barrier
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # 初始化 barrier：TMA 用 PIPE_DEPTH，MMA 用 1
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                for s in range(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        @T.inline
        def tma_load(stage, k_offset):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        tid = T.meta_var(warp_id * 32 + lane_id)

        # === Prefetch：加载前 PIPE_DEPTH 个 stage ===
        if tid == 0:
            for s in range(min(PIPE_DEPTH, K_TILES)):
                tma_load(s, s * BLK_K)

        # === Main loop ===
        for k in range(K_TILES):
            stage = k % PIPE_DEPTH

            # Wait TMA 完成加载这个 stage
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

            # 在这个 stage 的 data 上运行 MMA
            if tid == 0:
                mma(stage, accum=(k != 0))

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

            # Issue 下一个 prefetch load（k + PIPE_DEPTH）
            next_k = k + PIPE_DEPTH
            if next_k < K_TILES:
                if tid == 0:
                    tma_load(stage, next_k * BLK_K)

            # TMA phase 在 stage 回绕时 flip
            if stage == PIPE_DEPTH - 1:
                phase_tma ^= 1

        # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

(chap_persistent_kernel)=
## Step 6: Persistent Kernel + Tile Scheduler

到目前为止 everything 都优化了单个 tile 内的 work。Step 6 改变了问题的 scale，跨 tile 优化。

Step 5 每 128 x 128 output tile 启动一个 CTA。对于 4096 x 4096 output，这意味着 1024 个 separate CTA，每个支付自己的 setup cost，然后在 tile 完成的那一刻消失。

Step 6 启动 fixed pool 的 CTA，然后要求每个 CTA 依次处理多个 tile。这给我们两件事：setup work 在几个 tile 间 amortize，tile assignment 移到 kernel 内部，scheduler 可以选择 reuse operand 的 order。我们仍然在完整的 M=N=K=4096 size。

> **这个 step 改变了什么：Scope**
> - Scope：fixed pool 的 persistent CTA，每个通过 scheduler 在多个 output tile 上 loop。
> - Layout：不变，相同的 per-tile SMEM/TMEM/register path。
> - Dispatch：不变。

### Persistent Scheduling

persistent kernel 的 defining idea 是它按 hardware 而不是 problem 来 size 它的 grid。它启动 `SM_COUNT` 个 CTA，大约每个 SM 一个，不管有多少 output tile，目的是保持每个 SM 持续占用。我们故意说"大约"：精确的 1:1 residency 不保证，因为它取决于 occupancy 和 hardware 如何选择 schedule CTA。

在我们 targeting 的 B200 上，`SM_COUNT=148`。那 148 个 CTA 中的每个通过 `ClusterPersistentScheduler2D` 在分配的 tile 上 loop。

第一个 payoff 是 amortization。TMEM allocation、barrier initialization 和 scheduler state 现在每个 CTA 发生一次，并在 CTA 处理的约 7 个 tile 间 reuse，而不是在 1024 个 throwaway CTA 间重复。

第二个 payoff 来自 scheduler 选择的 order。设置 `l2_group_size=8` 将 nearby tile 分组，因此共享 row band 的 tile reuse 相同的 A row-tile，共享 column band 的 tile reuse 相同的 B tile。连续运行那些 tile 保持 operand 在 L2 中 hot 而不是从 HBM 重新 fetch。这正是 Step 3 未利用的 reuse。

```python
bx = T.cta_id([SM_COUNT])  # 1D grid，每个 SM 一个 CTA

tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // BLK_M,
    num_n_tiles=N // BLK_N,
    l2_group_size=8,       # 将 8 个 nearby tile 分组
    num_clusters=SM_COUNT
)
tile_scheduler.init(bx)
```

在 tile 上 loop 带来一个容易忽略的 correctness consequence。每个 tile 运行自己的 fresh K-loop，这意味着它的 barrier phase 必须从已知 state 开始。在 Step 5 中一个 CTA 处理恰好一个 tile，因此一次初始化 `phase_tma` 和 `phase_mma` 完全没问题。在 Step 6 中那些 initializer 必须移到 `while tile_scheduler.valid()` loop *内部*，使每个 tile 开始于与其自己的 TMA 和 MMA work 匹配的 phase state，而不是继承前一个 tile 留下的任何东西：

```python
while tile_scheduler.valid():
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0
    ...
```

### Complete Kernel

结构上，kernel 不过是 Step 5 的 pipeline 包装在 tile-level outer loop 中。唯一新的 dependency 是 scheduler 本身，我们与其他一起 import 它：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
```

grid dimension 现在简单是 `SM_COUNT` 而不是 `(M//BLK_M, N//BLK_N)`，`ClusterPersistentScheduler2D` 接管给每个 CTA 分配 tile 的工作：

```python
SM_COUNT = 148  # NVIDIA B200 GPU 上的 SM 数量
PIPE_DEPTH = 2

def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 1D grid：每个 SM 一个 CTA（不再是 2D grid！）
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation（与 Step 5 相同）---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init（与 Step 5 相同）---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        # Tile scheduler：以 L2-friendly order 给 CTA 分配 tile
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT
        )
        tile_scheduler.init(bx)

        tid = T.meta_var(warp_id * 32 + lane_id)

        @T.inline
        def tma_load(stage, k_offset, m_st, n_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # === Outer loop：在 tile 上迭代 ===
        while tile_scheduler.valid():
            # 从 scheduler 获取当前 tile position
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # === Inner loop：与 Step 5 相同的 pipeline ===
            phase_tma: T.int32 = 0
            phase_mma: T.int32 = 0

            # Prefetch 前 PIPE_DEPTH 个 stage
            if tid == 0:
                for s in range(min(PIPE_DEPTH, K_TILES)):
                    tma_load(s, s * BLK_K, m_st, n_st)

            # Main K-loop
            for k in range(K_TILES):
                stage = k % PIPE_DEPTH
                T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)
                if tid == 0:
                    mma(stage, accum=(k != 0))
                T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1
                next_k = k + PIPE_DEPTH
                if next_k < K_TILES:
                    if tid == 0:
                        tma_load(stage, next_k * BLK_K, m_st, n_st)
                if stage == PIPE_DEPTH - 1:
                    phase_tma ^= 1

            # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
            Dreg = T.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = T.alloc_local((BLK_N,), d_type)
            Dreg_wg = Dreg.view(128, BLK_N,
                                layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if tid == 0:
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                              Dsmem[:, :], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)

            T.cuda.cta_sync()
            tile_scheduler.next_tile()  # 移动到下一个 tile

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## Exercises

1. 在 Step 4 中，`arrive.expect_tx` 使用 `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` byte。如果这个 byte count 太小或太大，mbarrier 会 wait 什么？
2. 在 Step 5 中，为什么每个 SMEM stage 需要自己的 TMA barrier 而不是两个 stage 共享一个 `tma_bar`？
3. 在 Step 6 中，`BLK_M=BLK_N=128` 时 4096 x 4096 output 有多少 output tile？使用 `SM_COUNT=148` 时，每个 persistent CTA 平均处理多少 tile？
