(chap_gemm_advanced)=
# 用 Warp Specialization 和 Cluster 扩展 GEMM

:::{admonition} Overview
:class: overview

- 带 pipeline 的 GEMM 仍然让一个 warpgroup 顺序执行 load、MMA 和 writeback，本章消除这一瓶颈。
- Step 7 将 warp 专化为不同角色，Step 8 添加 2-CTA cluster，Step 9 添加多个 consumer。
- 每一步消除一个串行瓶颈，最终接近 state-of-the-art 的 throughput。
:::

上一章中带 pipeline 的 GEMM ({ref}`chap_gemm_async`) 已经很快，但它仍然要求一个 warpgroup 包揽所有事情：发起 load、执行 MMA、然后把结果写回。即使有了 software pipeline，这一组线程仍然是三个 engine 汇聚的地方，成为瓶颈。

症状很容易看到：Tensor Core 运行时 TMA unit 空闲，结果写回内存时 Tensor Core 空闲，每个 engine 通过同一组线程等待其他 engine。突破方法是停止让一个团队做所有事情。

我们通过三步逐步扩大协作范围来实现这一想法。Step 7 ({ref}`chap_warp_specialization`) 将 warp 专化为 producer、consumer 和 writeback 角色。Step 8 ({ref}`chap_cta_cluster`) 将两个 CTA 组成 cluster，共享它们 shared memory 中的 operand。Step 9 ({ref}`chap_multi_consumer`) 添加第二个 MMA consumer，使一个预存的 tile 提供两倍的计算量。

把这三步看作同一模式在不同尺度上的展开是有帮助的。Step 7 将整个 pipeline 保留在一个 CTA 内：TMA 和 MMA 共享一个 warpgroup，writeback 在另一个 warpgroup 中运行。Step 8 将协作范围扩大到 CTA 之间，产生一个跨越两个 CTA 的 256×256 tile。Step 9 进一步推高计算密度：cluster 输出增长到 512×256，每个预存的 B tile 被两个 consumer 复用，我们到达了教程中最密集的变体。

贯穿所有这些步骤，有一件事保持不变：SMEM、TMEM 和 register layout 仍然遵守我们在前两章建立的 contract；改变的是*谁在协作*，而不是数据如何布局。Step 8 是协作范围首次超出单个 CTA，因此它的 operand tile 分布在两个 CTA 的 shared memory 中，一个 layout 沿 `cbx` cluster 轴跨越两个 CTA。


(chap_warp_specialization)=
## Step 7: Warp Specialization + Pipeline

单 warpgroup kernel 浪费了性能，原因很简单：每个线程走同一条路径——load、然后 compute、然后 write——因此当它 load 时 Tensor Core 无事可做，当它 compute 时 TMA engine 无事可做。解决办法是*warp specialization*：与其要求一个线程团队依次做每项工作，不如将每项工作交给专门的 warp，让这些 warp 同时运行，由 software pipeline 将它们缝合在一起。这是 GEMM 路径中最大的架构变化，本章的其余内容都建立在其之上。这里的 benchmark 使用 M=N=K=4096。

> **这个 step 改变了什么：Scope**
> - Scope：一个 warpgroup 按顺序执行 load → MMA → writeback，变为三个并发角色（TMA producer、MMA consumer、writeback），由 full/empty barrier 连接。
> - Layout：不变，与 Step 6 相同的 SMEM stage 和 TMEM accumulator。
> - Dispatch：不变，TMA load、`tcgen05` MMA。

**主题。**

- Warp specialization：将不同的 warp/warpgroup 专用于不同的任务

- 高级 barrier 抽象：`TMABar`、`TCGen05Bar`、`MBarrier`

- `PipelineState` 用于自动 stage/phase 管理

- `warpgroup_sync` barrier ID 用于 per-warpgroup 同步

（多 stage SMEM pipeline 和 persistent `ClusterPersistentScheduler2D` 与 Step 5-6 完全复用；这里只有 scope 拆分是新的。）

### 从 Sequential 到 Concurrent

在引入角色和 barrier 之前，先把 warp specialization 消除的调度瓶颈隔离出来是有帮助的。下面的图用 Step-4 风格的 sequential timeline 作为紧凑的参考，对应 Step 4-6 中 pre-specialization kernel 的情况，然后将其放在 Step 7 warp-specialized schedule 上方，使 engine 利用率的差异一目了然。

![Warp Specialization Timeline](../../img/warp_specialization_timeline.png)

上方是 pre-specialization 的单 warpgroup 模式：同一个未专化的线程组同时拥有 load path 和 MMA path，因此一个 engine 很容易在另一个 engine 活跃时空闲。Step 5 和 Step 6 通过 double buffering 和 persistent scheduling 改进了这一基线，但尚未将 loading 和 compute 拆分为独立的 producer 和 consumer 角色。在下方，specialization 打破了这种轮流机制：TMA producer 在 MMA consumer 忙于计算时预取下一个 tile，writeback 自行推进。producer warp 3 在 consumer warp 0 仍在处理当前 MMA 时发起下一个 load，因此两个 engine 都不必等待对方。load/MMA 交接使用两个 barrier：

- **`tma2mma`**（TMA → MMA）：信号 loaded SMEM data 已准备好供 MMA 消费。
- **`mma2tma`**（MMA → TMA）：信号 MMA 已读完一个 buffer，TMA 可复用它进行下一次 load。

图中有一个细节初看可能像是个错误：`mma2tma` 箭头跳过了一个 stage。原因是 ring buffer。当 `PIPE_DEPTH=2` 时有两个 SMEM buffer，stage 0 和 stage 1；TMA Load k=0 填充 buffer 0，TMA Load k=1 填充 buffer 1。当 MMA Compute k=0 读完 buffer 0 时，它发出 `mma2tma` 信号表示 buffer 空闲，但真正想取回 buffer 0 的 load 是 TMA Load k=2 而不是 k=1（它正在使用 buffer 1）。这就是为什么从 MMA Compute k=0 出发的 `mma2tma` 箭头一直延伸到 TMA Load k=2。release 跳过一个 stage 仅仅是因为 ring 有两个 slot。

### Warp 角色

timeline 展示了*为什么*要拆分工作；下一个问题是*谁*做每部分。specialization 将三项工作（load、compute、writeback）分配给特定的 warp，使它们可以同时运行。使用 `WG_NUMBER=2` 时，kernel 使用两个 warpgroup（在角色表中缩写为 WG）：

| 参与者 | 位置 | 工作 |
|-------|------|-----|
| **TMA Producer** | Warpgroup 1, warp 3 | 通过 TMA 持续加载 A 和 B tile |
| **MMA Consumer** | Warpgroup 1, warp 0 | 数据就绪后立即运行 MMA |
| **Writeback** | Warpgroup 0（所有 warp）| 读 TMEM 结果，写 GMEM |

### 4 个 Barrier

三个并发 actor 需要四个 barrier，这四个 barrier 自然地归为两个相反方向。forward path（TMA → MMA → Writeback）信号数据*就绪*，它的消息是"你在等待的 tile 已到达"。backward path（Writeback → MMA → TMA）信号 buffer*释放*："你想要的 slot 又空闲了"。一旦了解命名约定，名字就可以自读了：每个 barrier 都是 `source2destination`，所以 `tma2mma` 就是 TMA 向 MMA 发出信号的 barrier。

| Barrier | 类型 | 方向 | 含义 |
|---------|------|------|------|
| **tma2mma** | `TMABar` | TMA -> MMA | "SMEM data 已就绪" |
| **mma2tma** | `TCGen05Bar` | MMA -> TMA | "SMEM buffer 可复用" |
| **mma2ld** | `TCGen05Bar` | MMA -> Writeback | "TMEM result 已就绪" |
| **ld2mma** | `MBarrier` | Writeback -> MMA | "TMEM 空闲，可用于下一个 tile" |

为什么每个 barrier 有*特定类型*？类型取决于 producer 如何宣布完成。**TMA Load** 使用 `TMABar`，即带 byte 计数的 mbarrier：TMA hardware 本身在 transfer 的 bytes 落地后到达 barrier，因此 consumer 无需任何 thread polling 即可得知 data 就绪。**TMA Store** 不能使用这种方式（store 没有人可通知），因此回退到 `cp_async.bulk.commit_group()` + `wait_group(0)`，发起 thread 等待自己的 write 完成。**MMA 操作** 使用 `TCGen05Bar`，`tcgen05.commit()` 指令在 MMA 完成时信号 barrier。

这里的一个小细节将在 Step 8 中产生回报。`arrive` 调用传递 `cta_mask=0`，因为在单 CTA kernel 中没有其他 CTA 需要信号。当 Step 8 形成 cluster 时，这个参数变为非零，成为唤醒合作 CTA 的机制。

### PipelineState

四个 barrier 告诉角色 buffer 何时*就绪*；但 pipeline 循环时，仍需要跟踪每个角色当前在*哪个* buffer 上。`PipelineState` 管理的就是这些 bookkeeping。一个 ring buffer 同时携带两件 bookkeeping：当前在哪个 slot 上，以及正在等待该 slot barrier 的哪个"phase"。在 pipelined loop 中手动跟踪两者正是容易引发 off-by-one 错误的地方，而这里的 off-by-one 会使整个 kernel deadlock。`PipelineState` 的存在就是为了将两者绑定在一起，使你不必手动处理：

```python
tma_ps = PipelineState(PIPE_DEPTH, phase=1)   # Producer 开始时就绪（phase=1）
# tma_ps.stage = 当前 stage index
# tma_ps.phase = 当前 phase（0 或 1）
tma_ps.advance()                          # 推进到下一个 stage
```

初始 `phase` 决定了角色的第一个 `wait` 是放行还是阻塞，pipeline 两端的正确答案相反，这是容易出错的地方：
- `phase=1`（producer）-> 第一个 `wait(phase=1)` 看到 barrier 仍在 phase 0，因为 0 != 1 所以**立即通过**。这正是我们想要的，因为 buffer 开始时是空的，producer 应当可以自由开始填充它们。

- `phase=0`（consumer）-> 第一个 `wait(phase=0)` 看到 barrier 在 phase 0，因为 0 == 0 所以**阻塞**。同样是我们想要的，因为还没有 data，在 producer 到达之前 consumer 无数据可读。

给两端相同的起始 phase 会导致 deadlock 或更糟的 silent corruption，因此这个选择值得做对。

### `warpgroup_sync` Barrier ID

specialization 引入了一个容易踏入的同步 hazard。一旦每个 warpgroup 运行不同的 code path，熟悉的 `cta_sync()` 就会 deadlock：它使用 hardware barrier #0 且要求*所有* CTA thread 到达，但在 warpgroup 分支内只有部分 thread 在场。我们真正需要的是作用域限定在单个 warpgroup 的 barrier。GPU 给我们 16 个 named barrier（ID 0-15），因此 kernel 使用 `warpgroup_sync(10)`，它只同步一个 warpgroup 内的 thread。当多个 warpgroup 各自需要独立同步时（如 multi-consumer Step 9 中发生的情况），它们通过 `warpgroup_sync(wg_id + 10)` 使用不同的 ID，这样它们永远不会在同一个 hardware barrier 上冲突。

**实现。**

这里我们使用 `PIPE_DEPTH=2`，这是仍然能让 load 和 compute 重叠的最小深度。增加深度可以隐藏更多 memory latency，上限是 SMEM budget；下面 *Step 7 行为异常时* 的讨论详细分析了这一 trade-off。现在所有组件已就绪（角色、四个 barrier、`PipelineState` 和 warpgroup-scope sync），我们可以组装完整 kernel：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.pipeline import TMABar, TCGen05Bar, MBarrier, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

SM_COUNT = 148  # NVIDIA B200 GPU 上的 SM 数量
F16_SIZE = 2

def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)   # Warpgroup 0 全部 128 个 thread 到达
        pool.commit()

        # --- TMEM alloc + fence ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)
        tile_scheduler.init(bx)
        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

        # =============================================
        # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                # === TMA Producer ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            tma2mma.arrive(tma_ps.stage,
                                           (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                # === MMA Consumer ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        # 等待 TMEM 从上一 tile 的 writeback 释放
                        ld2mma.wait(ld_ps.stage, ld_ps.phase)
                        ld_ps.advance()

                        for k in range(K_TILES):
                            tma2mma.wait(mma_ps.stage, mma_ps.phase)
                            Tx.gemm_async(
                                tmem[:, :BLK_N],
                                Asmem[mma_ps.stage, :, :],
                                Bsmem[mma_ps.stage, :, :],
                                accum=(k != 0), dispatch="tcgen05", cta_group=1)
                            mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
                            mma_ps.advance()

                        # 信号 result 已就绪，供 writeback 使用
                        mma2ld.arrive(0, cta_group=1, cta_mask=0)
                        tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: Writeback
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((BLK_N,), d_type)

            while tile_scheduler.valid():
                # 等待 MMA result
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()

                # 读 TMEM -> register（warpgroup scope）
                reg = T.alloc_local((BLK_N,), acc_type)
                reg_wg = reg.view(128, BLK_N,
                    layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
                Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
                T.ptx.tcgen05.wait.ld()

                # 信号 TMEM 空闲（全部 128 个 thread 到达）
                ld2mma.arrive(0, cta_id=0, pred=True)

                # Cast fp32 -> fp16
                Tx.cast(reg_f16[:], reg[:])

                # 写 Dsmem + TMA store
                Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    if lane_id == 0:
                        Tx.copy_async(D[m_st:m_st+BLK_M, n_st:n_st+BLK_N],
                                      Dsmem[:, :], dispatch="tma")
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(10)

                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

要运行这些 kernel 中的任意一个，复用我们在 Step 1 中展示过的同一个 compile / run / check harness（{ref}`chap_gemm_basics`）：将 `hgemm_v1` 替换为 `hgemm_v7`、`hgemm_v8` 或 `hgemm_v9`，选择一个 problem size 例如 `M=N=K=4096`。注意 clustered step 需要 `M` 和 `N` 是 cluster tile 的倍数（Step 8 为 `256×256`，Step 9 为 `512×256`），因此很小的 `128×128` size 不会产生任何 tile。每个 step 在独立的 Python session 中编译，在切换 step 之前重启 kernel，因为 kernel 复用内部名称且 compiler 保持 per-session 状态。每个 step 的 timing 收集在下面的 *End-to-End Result* 中。

### Epilogue（Writeback）细节

Step 7 可以接受一个令人愉快的简单 epilogue。仅有 `BLK_N=128` 列，writeback warpgroup 在一次传递中将整个 TMEM tile 读入 register，然后发起一次 TMA store。Step 8 和 Step 9 将没有这种奢侈，这正是它们引入后面添加的 chunking 的原因，但现在的顺序是：

1. 等待 MMA：`mma2ld.wait(phase)`。本教程中的 Step 8 和 Step 9 在这里添加一个 `fence.after_thread_sync()` 作为保守的额外措施；MMA-completion mbarrier 已经覆盖了 ordering，大多数 kernel（包括 CUTLASS）省略了它，因此 Step 7 也省略了。
2. 读 TMEM -> register（每 thread 128 个 fp32，warpgroup scope，通过 `Tx.copy_async(reg_wg, tmem[:, :BLK_N])` 后跟 `T.ptx.tcgen05.wait.ld()`）。
3. 信号 MMA：`ld2mma.arrive(0, cta_id=0, pred=True)`（全部 128 个 thread 到达）；TMEM 现在对下一个 tile 空闲。两个 `arrive` kwargs 在 cluster step 中反复出现：`cta_id` 标识发出信号的 CTA，`pred=True` 意味着所有 thread 都到达。

### Step 7 行为异常时

`PIPE_DEPTH=2` 是能让 load 和 compute 重叠的最小值，但如果你遇到 deadlock 或 wrong result，问题通常出在 pipeline bookkeeping 上。以下是诊断 checklist：

1. **phase 反转**：如果 producer 和 consumer 使用相同的初始 phase，第一个 `wait` 要么立即释放（导致 consumer 读取未初始化 data）要么永远阻塞（deadlock）。检查 producer 的 `phase=1` 和 consumer 的 `phase=0`。

2. **barrier 计数错误**：`tma2mma.init(1)` 意味着每个 stage 只需 1 个 arrive。如果 TMA producer 在 `tid == 0` 检查内发起，那是对的。如果多个 thread arrive，barrier 将永远等待不存在的额外 arrive。

3. **ring buffer 溢出**：当 `PIPE_DEPTH=2` 时，stage 在 0 和 1 之间循环。如果 producer 在 consumer 调用 `mma2tma.arrive` 之前推进了两步，它将在 consumer 仍在读取时覆盖 SMEM data。

4. **SMEM budget**：每个 stage 需要 `(BLK_M + BLK_N) * BLK_K * 2` 字节的 SMEM。对于 `BLK_M=128, BLK_N=128, BLK_K=64`，每个 stage 约 32 KB，`PIPE_DEPTH=2` 需要约 64 KB。增加 `PIPE_DEPTH` 直到 SMEM 耗尽。

---

(chap_cta_cluster)=
## Step 8: 2-CTA Cluster

Step 7 让 engine 重叠了，但每个 CTA 仍然孤立地计算自己的 128×128 tile，重新加载邻居无法借用的 operand。Step 8 打破了这种隔离。两个 CTA 加入 cluster 并获得互相访问对方 shared memory 的能力，因此单个 cooperative `tcgen05` MMA 产生一个跨越两个 CTA 的 256×256 tile，一次 B 的 load 现在提供两倍的 MMA work。如前所述，M=N=K=4096。

> **这个 step 改变了什么：Scope + Layout + Dispatch**
> - Scope：协作范围现在跨越 cluster 中的两个 CTA，而不是一个。
> - Layout：operand tile 分布在两个 CTA 的 SMEM 中；CTA 0 拥有共享的 completion barrier（`remote_view`）。
> - Dispatch：MMA 获得 `cta_group` / `cta_mask` 使 `tcgen05` 作为 2-CTA cooperative op 运行。

**主题。**

- CTA cluster：多个 CTA 合作完成更大的 tile

- 通过 `map_shared_rank` 跨 CTA 访问 SMEM

- `cta_group=2` 用于 256x256 cluster tile 上的 cooperative MMA

- 使用 `cta_mask` 跨 CTA barrier signal


### Cluster Tile Shape

整个优化基于一个 hardware capability：使用 `cta_group=2`，MMA 被允许读取 *两个* CTA 预存的 operand tile，而不仅仅是它所在的那个。每个 CTA 加载 stored B 的一个 128-row slice，转置后变为 128 个 logical output column，cooperative MMA 将两个 slice 拼接回一个 operand。下图追踪了两个 CTA 的 A 和 B slice 如何组合成单个 256×256 cluster tile：

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster: cooperative MMA via cross-CTA SMEM read" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive：每个 CTA 拥有 A 的一个 row slice 和 stored-B 的一个 row slice，然后通过 cluster（DSMEM）读取另一个 CTA 的 stored-B slice。`B.T` 之后，两个 stored-B slice 覆盖完整的 output-column span，因此这对 CTA 产生一个 256×256 output tile。*

**为什么 A 和 B 在 cluster 中分割**：要看到 256×256 tile 如何被分区，回顾本教程将 GEMM 存储为 `D = A @ B.T`，其中 stored B 的形状为 `N x K`。cluster 中有两个 CTA 时，分割很清晰：

- **A 垂直分割**：CTA-0 持有 A0（row 0-127），CTA-1 持有 A1（row 128-255）。堆叠：`[A0; A1]`（256 row）。
- **Stored B 按 row 分割**：CTA-0 加载 B row 0-127，CTA-1 加载 B row 128-255。因为 math 使用 `B.T`，这两个 stored row slice 变为 logical right-hand operand 的两个 128-column slice。
- 使用 `cta_group=2`，MMA hardware 通过跨 CTA shared memory access 从**两个** CTA 的 SMEM 读取 B，因此它看到完整的 logical output-column span。
- 结果：两个 CTA 合作完成一个 256x256 output tile。每个 CTA 写入该 tile 的 128x256 row stripe。

值得停下来看看为什么这是一个真正的 win 而不仅仅是 work 的 reshuffle。每个 CTA 仍然只加载 128×K 的 A 和 128×K 的 B，因此 cluster 整体预存约 2× 单个 CTA 的 operand，但它产生一个 256×256 tile，携带约 4× 128×128 tile 的 output FLOP。因此 MMA 每 staged-operand byte 做大约两倍 work，因为每个 CTA 的 B slice 通过 cooperative MMA 被另一个 CTA 的 A slice 复用。换句话说，arithmetic intensity 大约翻倍，这正是仍然 memory-leaning 的 kernel 需要的杠杆：End-to-End 表中的 ~2.2× speedup 来自于将相同的 byte 提供给更多 math。

### Tile Address Calculation

现在 cluster 是 unit of work，tile scheduler 也必须按 cluster tile 计数。它返回的每个 `(m_idx, n_idx)` 命名一个完整的 256×256 region，cluster 内的两个 CTA 在该 region 之间分割。将 cluster coordinate 翻译为每个 CTA 实际加载的 per-CTA slice 如下：

```python
m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

两个 CTA 工作在*同一个* 256×256 cluster tile 上，单个 coordinate `cbx`（CTA 在 cluster 内的位置，0 或 1）选择出这个 CTA 在两个轴上的贡献。`m_st` 选择这个 CTA 拥有的 output row stripe，`n_st` 选择它馈入 cooperative MMA 的 stored-B slice，writeback 稍后发射 256-column output span 的两个 128-column half。还要注意 `num_m_tiles = M // 256` 和 `num_n_tiles = N // 256` 计数的是 cluster tile 而不是单个 CTA tile。

乍一看 `cbx` 出现在 `m_st` 和 `n_st` 中，好像 row offset 以某种方式泄漏到了 column，但两个用法都是正确的，值得理清原因。在 writeback path 上，`cbx` 只属于 M 轴：每个 CTA 拥有独立的 128-row stripe（`m_st = (m_idx * CTA_GROUP + cbx) * BLK_M`，因此 CTA-0 写入 row `m_idx*256 .. +128`，CTA-1 写入接下来的 128），然而两个 CTA 都写入 cluster tile 的*完整* 256 output column。这正是 store 的 column 从 cluster 的 `n_idx` 派生（`n_st_epi = n_idx * 256 + no * 128`，没有 `cbx`）而不是从 per-CTA `n_st` 派生的原因。`n_st` 携带 `cbx` 的原因是每个 CTA 加载不同的 stored-B row slice 到 MMA：在那里，`cbx` 是 *load* offset，而不是 CTA 的 output-column offset。

### Code Changes from Step 7

与 Step 7 的 diff 有六个编辑，每个编码我们刚刚描述的 cluster contract 的一个部分：

```python
# 1. Cluster launch
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])   # cbx = CTA index within cluster (0 or 1)

# 2. Cooperative MMA（曾是 cta_group=1）
Tx.gemm_async(..., cta_group=2)

# 3. Cross-CTA shared memory access
B_remote = T.ptx.map_shared_rank(Bsmem, cta_id=1)

# 4. Cross-CTA barrier
tma2mma_cta0 = T.decl_buffer(
    [CTA_GROUP], "uint64",
    data=T.ptx.map_shared_rank(tma2mma.ptr_to([0]), 0),
    scope="shared"
)

# 5. mma2tma / mma2ld arrive 从 cta_mask=0（单 CTA，Step 7）
#    变为 cta_mask=3（signal cluster 中的两个 CTA）
mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)

# 6. Cluster sync 在末尾替换 cta_sync
T.cuda.cluster_sync()
```


### Cluster-Scope Changes

这六个编辑都源于同一个转变：协作范围现在是 cluster 而不是单个 CTA。下面的要点说明了这种扩大在实践中意味着什么：每个 CTA 如何找到它的位置、cluster 在谁的 barrier 上协调、以及哪个 CTA 实际发射 cooperative MMA。

- **Cluster CTA ID**：`cbx` 告诉每个 CTA 它在 cluster 中的位置（0 或 1）。CTA-0 处理 A row 0-127，CTA-1 处理 row 128-255。

- **Remote barrier view**：在 cluster 中，每个 CTA 有自己的 SMEM 和自己的 barrier，这引出一个明显的问题：如果 CTA-1 需要等待 CTA-0 产生的东西，它实际接触谁的 barrier？答案是提名 CTA-0 的 barrier 为唯一的 coordination point，让 cluster 中的任何 CTA 都能到达它们。`map_shared_rank(tma2mma.ptr_to([0]), 0)` 返回指向 CTA-0 barrier 的 cluster-wide pointer，使用 TIRx wrapper `tma2mma.remote_view(0)`，从那时起每个 arrive 和 wait 都针对 CTA-0 的 copy。

- **MMA 仅从 CTA-0 dispatch**：很容易将 `cta_group=2` 读作并行发射两个 engine，但事实并非如此。CTA-0 精确地发射一个 `tcgen05.mma`，hardware 然后驱动*单个 cooperative* MMA 跨越两个 CTA，从两个 SM 的 SMEM 读取 operand 并将 accumulator 写入两个 SM 的 TMEM。CTA-1 不发射任何 MMA。（每个 SM 只有一个 `tcgen05` engine，因此 `cta_group=2` 是一个 cross-SM MMA，而不是两个 engine 并排运行。）这就是代码用 `if cbx == 0:` 保护 MMA 的原因。

- **Multicast arrive**：`tcgen05.commit(..., cta_group=2, cta_mask=3)` 仅由 CTA-0 发出但 signal 两个 CTA 的 barrier。`cta_mask=3`（二进制 `11`）意味着 CTA-0 和 CTA-1 都是 target。

- **ld2mma init count**：`init(128 * CTA_GROUP)` --- 两个 CTA 的 writeback warpgroup（各 128 thread）arrive。


**实现。**

```python
def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, 128))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, 128), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128 * CTA_GROUP)  # 两个 CTA 的 writeback thread
        pool.commit()

        # --- TMEM alloc（cooperative）---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler（cluster tile）---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        # --- Cross-CTA barrier view ---
        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 1: TMA Producer（warp 3）+ MMA Consumer（warp 0）
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(ld_ps.stage, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, :MMA_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: Writeback（256 column 分为 2 x 128-column chunk）
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((128,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                for no in T.unroll(2):  # 2 chunk 各 128 column = 256 总计
                    reg = T.alloc_local((128,), acc_type)
                    reg_wg = reg.view(128, 128,
                        layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
                    Tx.wg.copy_async(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if lane_id == 0:
                            n_st_epi = T.meta_var(n_idx * 256 + no * 128)
                            Tx.copy_async(D[m_st:m_st+BLK_M, n_st_epi:n_st_epi+128],
                                          Dsmem[:, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

                ld2mma.arrive(0, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**2 CTA 的改变。**

- `CTA_GROUP = 2`，`MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` --- 两个 CTA 的 writeback WG arrive

- TMA arrive byte count 包括两个 CTA：`CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` 和 `tcgen05.dealloc` 必须使用 `cta_group=2`

- Writeback 将 256 output column 分为两个 128-column chunk --- 一次读取全部 256 TMEM column 超出 register capacity。Step 9 将 chunk 进一步缩小到 `EPI_N=64`

- `cluster_sync()` 在末尾替换 `cta_sync()`（确保所有 CTA 在 TMEM dealloc 之前完成）

所有额外的 arithmetic intensity 直接反映在 wall clock 上：Step 8 在 4096³ 时达到 **0.104 ms**，比同 size 下 Step 1 算法的 70 ms 快约 676 倍（见 End-to-End 表）。kernel 现在倾向于 compute-bound，这正是 Step 9 的设置，我们在 Step 9 中添加第二个 MMA consumer 以保持更多 Tensor Core work in flight。

如果 Step 8 比 Step 7 *更慢*，罪魁祸首几乎总是新的 cluster contract 输入有误。首先检查三件事：TMA arrive byte count 是否为 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`；scheduler dimension 是否为 `num_m_tiles=M//256, num_n_tiles=N//256`（对应 256×256 cluster tile）；writeback 是否发射两次 TMA store，每个 128-column chunk 一次，每次在 Dsmem 复用之前排空。

---

Cluster 提高了跨 CTA 的 reuse。最后一步转向内部，提高每个 CTA 内的计算密度，方法是为 producer 提供第二个 MMA consumer 以保持 fed。


(chap_multi_consumer)=
## Step 9: Multi-Consumer Warp Specialization

到 Step 8 时 MMA 确实很忙，但单个 consumer warp 处理一个预存 B tile 的速度有限，而那个 B tile 整个时间在 SMEM 中闲置，可供任何愿意读取的人使用。最后的优化利用了这一点：它添加第二个 MMA consumer，用*不同*的 A block 乘以*同一个* B tile。每个 CTA 的计算密度翻倍，cluster 输出从 256×256 增长到 512×256。如前所述，M=N=K=4096。

> **这个 step 改变了什么：Scope + Layout**
> - Scope：一个 MMA consumer 变为两个，由 `warp_id` 选择。
> - Layout：一个预存 B tile 被两个 consumer 复用；A 增加 consumer 轴。
> - Dispatch：不变。

**主题。**

- 多个 MMA warp（consumer）以获得更高 throughput

- 多个 writeback warpgroup，使用独立的 barrier slot

- 本教程中最优化 GEMM 变体使用的结构


### Multi-Consumer 结构

添加第二个 consumer 意味着 kernel 现在有更多的角色需要布局：两个 MMA warp 而不是一个，以及匹配的第二个 writeback warpgroup 来排空额外的 accumulator。使用 `NUM_CONSUMER=2` 和 `WG_NUMBER=3` 时，kernel 现在跨越三个 warpgroup（在角色表中缩写为 WG）：

| Warpgroup | Warp | 角色 |
|-----------|------|------|
| **WG 2** | warp 0 | MMA consumer 0：`Asmem[..., 0] x B` -> TMEM 列 `[0:256]` |
| **WG 2** | warp 1 | MMA consumer 1：`Asmem[..., 1] x B` -> TMEM 列 `[256:512]` |
| **WG 2** | warp 3 | TMA producer：每个 stage 加载 2x A block + 1x B block |
| **WG 0** | 全部 | Consumer 0 的 writeback：读 TMEM `[0:256]` |
| **WG 1** | 全部 | Consumer 1 的 writeback：读 TMEM `[256:512]` |

整个安排 hinges on 一个不对称性。每个 consumer 用自己的 A block 乘以*同一个*预存 B tile，因此单个 B load 现在提供 2× 的 MMA work，B 的 load cost per useful FLOP 有效减半。我们共享 B 而不是 A 的原因是两个 consumer 覆盖不同的 M 行条带：它们的 A block 是真正不同的 data，而 B 对两者相同。Exercise 3 要求你说服自己这是唯一有效的共享方式。

### 与 Step 8 的变化

具体而言，支持第二个 consumer 在 kernel 的几个地方做了修改，每个变化都追溯到一个事实：现在有两个 A block 和两个 TMEM range 需要每个 stage 填充和排空，而 B 保持共享。下面的编辑为额外的 A block staging，给每个 consumer 自己的 barrier slot，并为更高的 512×256 cluster tile 调整 tile addressing。

- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` --- 每个 stage 2 个 A block，每个 consumer 一个

- TMA 加载 `Asmem[stage, 0]` 和 `Asmem[stage, 1]`，TMA arrive bytes 现在为 `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`（额外 A block）

- MMA warp `warp_id` 选择哪个 A block 和 TMEM range

- `mma2tma.init(NUM_CONSUMER)` --- 两个 consumer 每个 stage 信号 TMA

- `mma2ld` 和 `ld2mma` 的 `depth=NUM_CONSUMER` --- 每个 consumer 使用自己的 barrier slot（MMA 侧为 `warp_id`，writeback 侧为 `wg_id`）

- Tile address：`m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M` --- M 方向有额外的 `NUM_CONSUMER` 因子，因为每个 cluster tile 现在在 M 方向跨越 `NUM_CONSUMER` 个 consumer。Tile scheduler 使用 `num_m_tiles = M // 256 // NUM_CONSUMER`（cluster tile 为 512x256）

- Writeback 使用 chunked `EPI_N`，使每次迭代在 register 中保持更少的 TMEM-readback value


**实现。**

```python
def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP   # 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    WG_NUMBER = 3
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (NUM_CONSUMER, BLK_M, EPI_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, NUM_CONSUMER)   # depth=2, 每个 consumer 一个 slot
        ld2mma  = MBarrier(pool, NUM_CONSUMER)     # depth=2, 每个 consumer 一个 slot
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(NUM_CONSUMER)  # 每个 stage 期望 2 个 arrive
        mma2ld.init(1)              # 每个 slot 得到 1 个 arrive
        ld2mma.init(128 * CTA_GROUP)  # 两个 CTA 的 writeback thread
        pool.commit()

        # --- TMEM alloc (cooperative) ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler (512x256 cluster tile) ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256 // NUM_CONSUMER, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 2: TMA Producer (warp 3) + 2 MMA Consumer (warp 0, 1)
        # =============================================
        if wg_id == 2:
            if warp_id == 3:
                # === TMA Producer: 每个 stage 加载 2 个 A block + 1 个 B block ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    m_st_c1 = T.meta_var(m_st + CTA_GROUP * BLK_M)
                    Tx.copy_async(Asmem[tma_ps.stage, 0, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Asmem[tma_ps.stage, 1, :, :],
                                  A[m_st_c1:m_st_c1+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id < NUM_CONSUMER:
                # === MMA Consumer: warp_id 选择 A block 和 TMEM range ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(warp_id, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, warp_id * MMA_N:warp_id * MMA_N + MMA_N],
                                    Asmem[mma_ps.stage, warp_id, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0/1: Writeback（每个读自己 consumer 的 TMEM range）
        # =============================================
        elif wg_id < NUM_CONSUMER:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((EPI_N,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wg_id, wb_ps.phase)  # 等待 THIS consumer
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                # 以 EPI_N=64 列 chunk 读取 TMEM（256 列共 4 次迭代）
                for i in T.unroll(MMA_N // EPI_N):
                    reg = T.alloc_local((EPI_N,), acc_type)
                    reg_wg = reg.view(128, EPI_N,
                        layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
                    col_st = T.meta_var(wg_id * MMA_N + i * EPI_N)
                    col_end = T.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                    Tx.wg.copy_async(reg_wg[:], tmem[:, col_st:col_end])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if lane_id == 0:
                            m_st_epi = T.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = T.meta_var(n_idx * MMA_N + i * EPI_N)
                            Tx.copy_async(
                                D[m_st_epi:m_st_epi+BLK_M, n_st_epi:n_st_epi+EPI_N],
                                Dsmem[wg_id, :, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(wg_id + 10)

                ld2mma.arrive(wg_id, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**实现 notes。**

- 在 Step 9 设计中，`mma2ld` 和 `ld2mma` 各是一个 `depth=NUM_CONSUMER` 的共享对象，而不是每个 consumer 独立的对象。Slot 0 连接 MMA warp 0 到 Warpgroup 0，slot 1 连接 MMA warp 1 到 Warpgroup 1；MMA 侧用 `warp_id` 索引，writeback 侧用 `wg_id` 索引。

## End-to-End Result

下表报告了从 naive baseline 到 warp-specialized cluster kernel 的测量里程碑，以及 cuBLAS reference。NVIDIA B200 上的 reference 数字，M=N=K=4096，fp16，锁定时钟，1000 次迭代 timed benchmark：

| Step | 技术 | 时间 | 加速比 |
|------|------|------|--------|
| 1 | Sync load + MMA | 70 ms | 1× |
| 2 | K-loop accumulation | --- | 处理大于一个 tile 的 K |
| 3 | Spatial tiling | 53.6 ms | ~1.3× |
| 4 | TMA async load | 0.49 ms | ~142× |
| 5 | Software pipeline | --- | 重叠 load + compute |
| 6 | Persistent kernel | --- | L2 cache locality |
| 7 | Warp specialization | 0.23 ms | ~309× |
| 8 | 2-CTA cluster | 0.104 ms | ~676× |
| 9 | Multi-consumer | 0.094 ms | ~744× |
| --- | cuBLAS（reference）| 0.094 ms | ~744× |

表中每个时间（包括 70 ms Step 1 baseline）都在相同的 M=N=K=4096 size 下测量，这正是使 speedup chain 可以端到端比较的原因。值得精确说明那 70 ms 到底是什么，因为很容易误读。它*不是* {ref}`chap_gemm_basics` 中的单 tile Step-1 kernel 在 4096³ 下运行；那个 kernel 只计算一个 128×128 tile 且只在小 size 下运行。70 ms 是一个 naive 的 full-size baseline，采用相同的 sequential、single-tile 方法并扩展到完整的 4096³ problem。Step 1-3 在 {ref}`chap_gemm_basics` 中以小 size（128×128 和 256³）引入，以保持最初的 walkthrough 简单；这里的 Step 1 和 Step 3 行是它们的 full-size benchmark 对应物。其余的 dash（Step 2、5、6）标记展示了结构但未单独计时的 step。

将这些数字读作受控条件下的一次 B200 reference run，而不是 leaderboard entry。每个 step 中嵌入的 `{.python .input}` benchmark cell 是 smoke benchmark：它们适合观察 trend，不适合声称 peak performance。

四项技术贡献了几乎全部增益：

1. **TMA Async Data Movement**：hardware copy engine 替换 software copy（从 Step 1 → Step 4 约 142×）。正确理解这个 142× 很重要：它反映了从单个 128×128-tile kernel（grid 1×1）一直到完整的 tiled-and-parallel kernel（带 K-loop、spatial tiling 和多个 CTA）*加上* TMA 的全过程；它不是 TMA 的孤立贡献。隔离 TMA 意味着比较两个 full-size kernel，它们仅在 copy 机制上不同。
2. **Software Pipelining + Warp Specialization**：通过给 load 和 compute 各自专用的角色来重叠两者（从 Step 4 → Step 7 约 2.2×）。
3. **CTA Cluster**：2-SM cooperative MMA 提高跨 CTA 的 B-tile reuse（在此 benchmark 中从 Step 7 → Step 8 约 2.2×）。
4. **Multi-Consumer**：两个 MMA warp 获得更高计算密度（从 Step 8 → Step 9 约 10%）。

在测量的里程碑处绘制，同样的四项贡献描绘了从 synchronous tiled kernel 下降到 cuBLAS reference 的轨迹。下图显示了选定的测量点：

![GEMM Optimization Journey](../../img/gemm_perf.png)

注意增益随着列表下降而缩小，这有结构上的原因，而不是努力减弱。早期 step 针对的是*memory* 瓶颈（TMA 替换 software copy，cluster 提高 arithmetic intensity），而 70 ms 的大部分时间实际花在那里，因此这些 step 回报最多。到 Step 8 时 kernel 已经在 cuBLAS 的 ~10% 以内（0.104 vs 0.094 ms），接近 *compute-bound*，这意味着几乎没有剩余的 memory stall 可以隐藏；Step 9 的 multi-consumer overlap 恢复了剩下的大部分。在 compute ceiling 附近大约 10% 的最终增益正是预期结果：这是一个几乎解决的问题的 diminishing return，而不是弱优化的信号。

本章构建的所有东西（TMA load、`tcgen05` MMA、TMEM readback 和 warp-specialized barrier）直接带入下一章。Flash Attention 复用了所有这些东西，然后通过在两个 MMA phase 之间插入 online-softmax step 来提高难度，而不是简单地重复一个。


## Exercises

1. 如果在 Step 7 中将 TMA 和 MMA 的 `PipelineState` 的初始 `phase` 都设为 `0` 会发生什么？画出 deadlock 场景。
2. 在 Step 8 中使用 `cta_group=2` 时，TMA arrive byte count 为 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`。为什么每个 CTA 加载自己的 data 还要乘以 `CTA_GROUP`？
3. 在 Step 9 中，每个 consumer 处理不同的 M 行但相同的 B tile。为什么共享 B（而不是 A）是正确的选择？

**与你的 agent 一起尝试**：粘贴 Step 7 kernel 并要求它追踪一个 K-tile 通过四个 barrier（`tma2mma`、`mma2tma`、`mma2ld`、`ld2mma`）。对于每个 barrier，询问谁 wait、谁 arrive、哪个 tile 变为可读、以及之后哪个 buffer 变为可复用。
