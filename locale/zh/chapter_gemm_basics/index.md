(chap_gemm_basics)=
# 构建 Tiled GEMM

:::{admonition} Overview
:class: overview

- 从 TIRx tile primitive 构建正确 tiled GEMM，从单个 output tile 开始。
- Step 1 是 single-tile GEMM，Step 2 添加 K-loop accumulation，Step 3 跨 CTA spatial tile 以处理完整 matrix。
- correctness 第一；performance 是后面两章的工作。
:::

GEMM 是整个本书围绕的 workload。它位于 linear layer、attention projection 和 convolution 之下，这些 dominate GPU 的时间，因此正确 GEMM 与 fast GEMM 之间的 difference 是让 chip 大部分 idle 与 saturate 它的 difference。

那个 gap 太大不能一次跨越。saturating kernel 让你同时 debug memory movement、accumulation、tiling 和 Tensor Core scheduling，没有 trustworthy thing 可以 compare。safer path 是从产生 correct answer 的最小 kernel 开始，然后一次一个 decision 增长它。

本章写那个 first correct tiled GEMM。前面章节在抽象层面介绍了 TIRx scope / layout / dispatch model；这里我们将其应用到 real kernel。我们从一个 128 × 128 output tile 开始并将其增长为处理 full-size matrix 的 kernel，添加 K-dimension accumulation 然后跨许多 CTA spatial tiling。

这是三个 walkthrough 单个 GEMM optimization path 的章节的第一个。在这个中我们构建 correct tiled kernel 并停在那里。下一章（{ref}`chap_gemm_async`）用 TMA 替换 thread copy 并通过 pipelining overlap data movement 与 compute，{ref}`chap_gemm_advanced` 进一步用 warp specialization 和 CTA cluster 推进。每个 chapter 建立在前面一个上，因此 kernel accumulate feature 而不是重新开始。

将每个 step 读为对具有三个 term 的单个 contract 的 edit 是有帮助的：哪个 **scope** 运行 operation、operand tile 使用哪个 **layout**、以及哪个 **dispatch** path 执行它。大多数 step 有一个 primary change，因此我们用一个小 card 打开它们，name 那个 change 并 call out 使 reuse safe 需要的任何 synchronization detail。Step 1 建立 rest of the path edit 的 baseline。

## GEMM

GEMM 是位于 linear layer、attention projection 和许多 convolution implementation 之下的 dense matrix multiply，这就是为什么 fast GEMM kernel 几乎在你看的每个地方都 payoff。本 tutorial 的 example 使用 $D = A B^{\top}$：

- $A$ 有 shape $M \times K$。
- $B$ 有 shape $N \times K$。
- $D$ 有 shape $M \times N$。
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$。

transpose 不是我们选择执行的 extra operation；它来自 data 如何 store。example 保持 $B$ 为 $N$ 行长度 $K$，这是 linear-layer weight 通常来的 layout，因此沿 $K$ contract 自然地 read $B^{\top}$ 而无需任何 rearrangement。

贯穿 tutorial 我们用 TFLOPS throughput 测量 kernel，对 wall-clock time 计算每个 multiply-add 的两个 floating-point operation：

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM Data Path

本 tutorial 的每个 optimization 归结为 data 存放在哪里和如何 move，因此在我们写任何 code 之前 map 那个是值得的。本质上，Blackwell GEMM kernel 围绕两个 activity 组织：在 memory 间 move tile 和在它们上 compute。下面 figure trace tile 从 input 到 output 的路径上 touch 的每个 memory：

![*Memory Data Flow*](../img/memory_dataflow.png)

上面 figure 展示每个 later optimization edit 但从不 replace 的 baseline path。
从左到右 read：operand tile 首先从 GMEM move 到 SMEM；`tcgen05.mma` 然后
consume SMEM operand 并将 accumulator 写入 TMEM；最后 epilogue 在将 result store 到 GMEM 之前
将 TMEM read 回 register。保持这个 chain 在 mind，因为每个 step
下面 change *如何* 这些 hop 中的一个发生；它从不 change hop 本身。

## Optimization Path

上面 plain data path 足够得到 correct answer，但它让大多数 hardware idle。tutorial 的 rest 通过一次一个添加 Blackwell feature 来 close 那个 gap，每个通过 TIRx tile primitive express。我们将 follow 的 path 依次 visit 这些 feature：

- **TMA async movement** 通过 Blackwell hardware copy path move GMEM ↔ SMEM tile，barrier track completion。
- **Software pipelining** 使用多个 SMEM stage 使下一个 K tile 的 data movement 可以 overlap 当前 tile 上的 Tensor Core compute。
- **Persistent scheduling** 保持固定 pool of CTA，每个通过 tile scheduler 处理许多 output tile，而不是每 tile 一个 CTA。
- **Warp specialization** 将 producer、MMA consumer 和 writeback role 拆分到 separate warpgroup。
- **CTA cluster** 让两个 CTA 在单个 larger Blackwell MMA tile 上 cooperate。
- **Multi-consumer execution** 使用多个 consumer warpgroup 同时 compute tile 的不同 part，raise compute density。

---

(chap_single_tile)=
## Step 1: Sequential Single-Tile GEMM

仍然 exercise full hardware path 的最简单 GEMM 是计算单个 output tile 的那个。所以那是我们开始的地方。Step 1 计算 K = 64 的一个 128 × 128 output tile，足够小以至于 nothing 必须 loop，data path 的每个 piece 恰好出现一次。没有 thing repeat，我们可以在必须 reason 关于 loop 之前孤立地看每个 hop。

> **这个 step 建立什么：baseline**
> - Scope：128 thread 的单个 warpgroup 按顺序 walk 整个 path，一个 stage 接一个。
> - Layout：A 和 B tile 在 SMEM 中，accumulator 在 TMEM 中，result 通过 register stage out。
> - Dispatch：synchronous `Tx.copy` 携带 load，`tcgen05` 运行 MMA。

### Single-Tile Dataflow

baseline contract fixed 后，next thing 要 pin down 是一个 tile 通过它的 order。这个 first kernel 恰好一次 walk core GEMM data path，与 data-flow figure 中相同 GMEM → SMEM → TMEM → register → GMEM chain，没有 loop wrap 在它周围。它 allocate 它的工作 memory、load operand、compute product、write result back、并清理它自己：

1. **Allocate**：SMEM（pool allocator）、TMEM（`tcgen05.alloc`）、mbarrier
2. **Load**：所有 128 thread cooperative copy A 和 B tile 从 GMEM 到 SMEM（sync `Tx.copy`）
3. **Compute**：single elected thread 发出 `Tx.gemm_async` + `tcgen05.commit`；所有 thread 在 mbarrier 上 wait
4. **Writeback**：warpgroup read TMEM → register；每个 thread cast fp32→fp16 并 write 到 GMEM
5. **Deallocate**：TMEM deallocation

### First Kernel 的四个 Piece

完整 kernel 只有几十行，但分部分消化更容易。我们将在四个 piece（memory allocation、synchronous load、MMA dispatch 和 writeback）中读它，并在之后将它们组装成一个 kernel。沿途出现的 API name 是 Part II 介绍的 TIRx tile-primitive vocabulary（{ref}`chap_tirx_primer`、{ref}`chap_tirx_layout_api`）。

**Memory allocation。** kernel 从为 operand carve out shared memory 开始，加上 TMEM address 和 mbarrier 的 slot：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 byte)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 byte)
pool.move_base_to(1024)                           # Skip 到 offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

这里两个 detail 值得 pause。`pool.move_base_to(1024)` 将 Asmem 和 Bsmem push 到 offset 1024，为上面的小 piece of metadata reserve low address，因此 bulky operand tile land 在 clean boundary。`layout=A_layout` 向 `tma_shared_layout` 请求 swizzled SMEM placement，TMA 和 `tcgen05.mma` 都可以直接 read，正是 Part II 描述的 layout-as-contract obligation 类型。

**Synchronous load。** buffer in place 后，operand 仍然必须 reach SMEM。在这个 first version 我们让 CTA 自己的 thread 做 copy：

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

因为这里只有一个 tile（M=N=128, K=64），copy 整个 A 和 B 是整个 load。`Tx.cta.copy(...)` 使 CTA cooperate 在那个 copy 上，每个 thread 负责它自己 slice of data。跟随的 `T.cuda.cta_sync()` 做 double duty：它 wait 每个 thread finish 并 publish 它们的 shared memory write，因此当 MMA later read `Asmem` 和 `Bsmem` 时它看到 complete tile 而不是 half-filled buffer。这个 thread-driven copy 也是我们即将 replace 的 very first thing；下一章（{ref}`chap_gemm_async`）将它 swap out 为 TMA。

**MMA dispatch。** operand 现在 sit 在 SMEM 中，我们可以 issue MMA，我们从 single elected thread 做那个：

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

两个 nested guard 在两个 step 中 narrow issuer。outer `if warp_id == 0` 只保持 warpgroup 的 warp 0，inner `if T.ptx.elect_sync():` 然后在那个 warp 内 elect single active lane。一起它们留下恰好一个 thread 运行 `Tx.gemm_async` 和 `tcgen05.commit`。

明确那个 single thread 做和不做什么是值得的，因为 natural reading 是 misleading。single issuing thread *不* imply single-threaded multiply。computation 仍然是 full tile-level MMA：hardware 为 SMEM operand layout 和 TMEM accumulator layout 描述的 tile perform cooperative multiply。关键是 `Tx.gemm_async` 是一个 *tile operation*，不是一个 hardware instruction。K = 64 tile 比 hardware MMA K-atom（`MMA_K = 16`）宽，因此这个 tile op lower 为沿 K  stepped 的短 sequence of raw `tcgen05.mma` instruction，warpgroup cooperative drive 每个。只有一个 thread issue tile op 的原因是每个 underlying `tcgen05.mma` 本身是 *single-instruction* cooperative op：一个 launch drive tile MMA 的那个 K-atom。如果所有 128 thread issue sequence，相同 work 将 simply launch 128 次 over。最后，`accum=False` flag 告诉 MMA overwrite TMEM destination 而不是 add into 它，这是这里我们想要的，因为没有 prior partial sum 要 extend。

**Writeback。** product 现在 sit 在 TMEM 中，但 caller 想要它 back 在 GMEM 中作为 fp16。因此 epilogue 必须将 result bring down 通过 register 并沿路 cast 它：

```python
Dreg = T.alloc_local((BLK_N,), acc_type)        # per-thread fp32 register row
Dreg_f16 = T.alloc_local((BLK_N,), d_type)      # 相同 row，cast 为 fp16
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

MMA 在 TMEM 中留下 128 × 128 fp32 accumulator tile。fp32 是 deliberate：GEMM 沿 K sum 许多 product，保持 running sum 在 higher precision 压下否则 accumulate 的 rounding error。但 `D` 是 fp16，因此 value 不能 straight out。它们首先 land 在 register、在那里 narrow 为 fp16、然后 reach GMEM。

两个 register buffer 玩 distinct role。`Dreg` 是 `BLK_N` element 的 per-thread buffer，而 `Dreg_wg` 是那些相同 register 在 chosen layout 下的 warpgroup-wide *view*：

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

这个 layout 将 tile 的 first dimension 映射到 warpgroup 的 thread：thread 0 own row 0，thread 1 own row 1，依此类推到 row 127。second dimension 保持在每个 thread 自己 register buffer 内，因此 single thread 持有它一个 row 的所有 column。warpgroup 有 128 thread 且 tile 有 128 row，128 × 128 output  neatly divide 为每 thread 一个 row。

在那个 view 下 read accumulator out 正是 `Tx.wg.copy_async(Dreg_wg, tmem)` 做的，它 lower 为 Blackwell TMEM load path `tcgen05.ld`。因为那个 load 是 asynchronous，`T.ptx.tcgen05.wait.ld()` 必须 complete 在任何 thread touch `Dreg` 之前；否则 thread 将 read load 尚未 fill 的 register。

一旦 wait return，每个 thread 的 private `Dreg[:]` 持有它一个 logical output row 的 fp32 value。thread 将那些 narrow 为 `Dreg_f16` 中的 fp16，work out 它负责哪个 global row，

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

并 write `D[m_thr, n_st:n_st + BLK_N]`。row 在四个 warp 间 cleanly partition：warp 0 write row 0-31，warp 1 write row 32-63，warp 2 write row 64-95，warp 3 write row 96-127。

### Complete Kernel

现在我们将四个 piece stitch back 到一个 runnable kernel（M=N=128, K=64）。import 先来：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

kernel 包装在与 later step 使用的相同 `hgemm_vX(M, N, K)` style 中。Step 1 以 `M=N=128, K=64` 运行，因此 launch 恰好包含一个 output tile：

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K 记录 underlying hardware MMA tile；它们不
    # 传递给 gemm_async（它从 operand 和
    # accumulator tile 推导 MMA shape），因此 later step 省略它们。
    MMA_M, MMA_N, MMA_K = 128, 128, 16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # Step 1 是 single-tile kernel：M = BLK_M 且 N = BLK_N，因此 grid
        # 为 1×1。从 1×1 grid 开始使 per-CTA tile offset
        # (m_st, n_st) 平凡为零；Step 3+ 将此 generalise 到更大 M / N。
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # single warpgroup，因此 wg_id 始终为 0（下面 unused）
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
    
        # --- Barrier + TMEM init (仅 warp 0) ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
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
        phase_mma: T.int32 = 0
    
        # --- Load: 所有 thread copy global → shared (synchronous)。
        # M=BLK_M 且 N=BLK_N 时下面 slice 覆盖完整 matrix；
        # 保持 slice form 使到 Step 3（multi-tile）的 diff 最小。
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- Compute: single elected thread 发出 MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    
        # --- Writeback: TMEM → RF → GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

follow 的每个 GEMM step 以相同方式 compile、run 和 check 它自己，因此我们在这里完整 spell out 那个 scaffolding 一次，从那时起只显示 kernel。要 run later step，将它的 `hgemm_vX` 和 matching problem size drop in 替换下面的那些。一个 caveat 值得 remember：在 fresh Python session 中 compile single step 并在尝试另一个之前 restart，因为 example reuse inner name 且 compiler hold per-session state。

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device('cuda')  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

# ex.mod(...) 直接接受 torch tensor，与每章使用的相同 call form。
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
# Relative tolerance，与 warp-specialization 和 Flash Attention cell 相同：
# output magnitude 随 K 增长，因此 fixed absolute bound 将在 larger K 失败。
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

# Optional timing 对于 larger kernel。
ITERS = 10
for _ in range(3):
    ex.mod(A_tensor, B_tensor, D_tensor)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    ex.mod(A_tensor, B_tensor, D_tensor)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

Step 1 到 3 在 deliberately small size 运行（这里 128×128，Step 3 中 256³）以保持这些 first walkthrough simple to follow。{ref}`chap_gemm_advanced` 末尾的 cross-step *End-to-End Result* table 采取 opposite approach：它在单个 M=N=K=4096 size 测量每个 step，包括这个 Step 1 algorithm，因此它的 speedup ratio 可以直接 compare。

### Single-Tile Kernel 的 Limit

这个 kernel correct，那是 Step 1 的 whole point，但它只在 very narrow setting 中 correct。四个 limitation on purpose bake in，optimization path 的 rest 一次一个 lift 它们：

- 它只处理 single K tile，因此不能 contract over large K。
- 它只处理 single output tile，因此 M 和 N pin 到 128。
- 它使用 synchronous GMEM → SMEM copy 而不是 TMA。
- 它不 overlap data movement 与 compute，因此两个从不同时 run。

---

(chap_k_loop)=
## Step 2: K-Loop Accumulation

要 remove 的 first limit 是 smallest 一个。Step 1 只处理 single 64-wide K tile，但 real matrix contract over 远多于那个。在 Step 2 我们保持 single output tile 但让 K span 许多 64-wide chunk。

idea straightforward：每 chunk 一次 repeat load → MMA → wait sequence，并让每个 MMA accumulate 到相同 TMEM slot。real work，事实证明，在 synchronization 中。reuse 一个 mbarrier 跨 iteration 引入本章 first genuine correctness hazard。如果 code track wrong phase，wait 可以在它的 MMA 实际 finish 之前 return，silently corrupt result。下面 mechanics 精确展示那个如何 go wrong，以及如何 avoid 它。

> **这个 step 改变什么：Layout reuse**
> - Scope：unchanged，仍然 single warpgroup。
> - Layout/reuse：相同 SMEM tile pair 和 TMEM accumulator slot 在 K-loop 间 reuse。不 allocate 新 storage；operand tile 通过一对 fixed buffer stream，accumulator state 保持在一个 TMEM slot。
> - Synchronization：reuse MMA barrier 必须在每个 K chunk 上 advance 通过 right phase，否则 later wait 可以 observe earlier completion。
> - Dispatch：unchanged。

### K-Loop Mechanics

Step 1 contract over single 64-wide K tile；这里我们保持它的 single output tile 但让 K run 只要 matrix demand。要 cover 大于 64 的 K，我们以 `BLK_K=64` chunk walk K。每个 iteration load 下一个 A 和 B K-slice 到 SMEM 并 issue `Tx.gemm_async`。`accum` flag 是将这些 chunk stitch 到一个 dot product 的东西：在 first chunk，`accum=False` initialize TMEM accumulator，在每个 later chunk，`accum=True` 将那个 chunk 的 product add 到已经 sit 在 TMEM 中的 running sum。

synchronization 是 need care 的地方。我们 reuse single mbarrier 用于每个 MMA completion，safely reuse 它归结为 track 我们在 wait 哪个 barrier phase。mbarrier 携带 1-bit phase，0 或 1，它每次 expected arrival land 时 flip 到 other value。subtle part 是 wait condition 本身：`try_wait(bar, phase)` block 直到 barrier 的 internal phase 与 `phase` argument *differ*。所以我们 pass 的 argument 必须 name 我们 expect leave behind 的 phase，而不是我们 wait reach 的那个：

| K iteration | wait 前 Local `phase_mma` | `try_wait` wait 什么 | wait 后 Local update |
|---|---:|---|---:|
| 0 | 0 | barrier flip 到 1 | `phase_mma = 1` |
| 1 | 1 | barrier flip 到 0 | `phase_mma = 0` |
| 2 | 0 | barrier flip 到 1 | `phase_mma = 1` |

single line `phase_mma ^= 1` 是保持那个 table honest 的东西。drop 它，second iteration 仍然 call `try_wait(bar, 0)`，但 barrier 在 first MMA 后已经 flip 到 phase 1，因此 wait 看到 mismatch 并 immediately return，在 second MMA finish 之前。kernel 然后 read half-computed accumulator 并 report wrong answer 且没有 error。这是一个 compile 和 run perfect 的 bug，这正是为什么 phase flip 值得这么多 attention。

### Complete Kernel

下面完整 kernel 只是 Step 1 带有 fold in 的 K-loop 和 phase flip。import 与之前相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

它包装在 `hgemm_v2(M, N, K)` 中。grid 仍然 `[1, 1]`，因为我们仍然 compute single output tile；所有 grow 的是它的 K extent：

```python
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])  # 仍然一个 output tile (M=N=128)
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # === K-loop: 以 BLK_K chunk 迭代 K ===
        for i in T.serial(K_TILES):   # serial device loop (保持 full-K A/B parameter 正确 shape)
            # Load 第 i 个 K chunk
            Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            # MMA: first tile accum=False，rest True
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # Wait MMA，然后 flip phase
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # === Writeback (与 Step 1 相同) ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

---

(chap_spatial_tiling)=
## Step 3: Spatial Tiling (Multi-CTA)

K-loop 处理了 contraction dimension，但 M 和 N 仍然 pin 到 single 128 × 128 tile。real output 远大于一个 tile，因此 basic kernel 的 last piece 是用许多 tile 同时 cover M 和 N。Step 3 launch 2D grid of CTA，每 output tile 一个，并让 GPU 并行 compute 所有 tile。example 使用 M=N=K=256，给出 2×2 grid of tile，刚好足够使 indexing non-trivial 而不 bury 它。

> **这个 step 改变什么：Scope**
> - Scope：CTA 的 2D grid，每个 CTA own 一个 128 × 128 output tile。
> - Layout：unchanged；在 CTA 内，这是与 Step 2 相同 SMEM/TMEM/register path。
> - Dispatch：unchanged。

### Grid Mapping

grid shape 直接 follow tiling：每 128 × 128 output tile 一个 CTA，我们总共需要 `[M // BLK_M, N // BLK_N]` CTA。与 Step 2 相比 genuinely new work 只有教每个 CTA 哪个 slice of matrix 是 *它的* slice 要 compute。

CTA `(bx, by)` own 这个 output region：

```text
D[bx * BLK_M : (bx + 1) * BLK_M,
  by * BLK_N : (by + 1) * BLK_N]
```

要 produce 它，CTA 的 K-loop repeatedly load 它自己 A row band 和 B column band 的 matching K-slice：

```text
A[bx * BLK_M : (bx + 1) * BLK_M, k : k + BLK_K]
B[by * BLK_N : (by + 1) * BLK_N, k : k + BLK_K]
```

indexing 直接 follow `D = A @ B.T` convention：`bx` select A 和 D 的 row，而 `by` select B 的 row，那些在 apply transpose 后成为 D 的 column。

每 CTA 一个 tile 是最 simple mapping 工作，但它也 wasteful。row 中每个 CTA reload 相同 A tile 从 GMEM，column 中每个 CTA reload 相同 B tile，因此 nothing reuse neighboring CTA 已经 pull in 的 data。我们将 leave 那个 waste in place 现在；persistent scheduling（{ref}`chap_gemm_async` 中 Step 6）come back 到它并保持那些 shared operand hot 在 L2。

**与你的 agent 一起尝试**：以 `M=N=K=256`、`BLK_M=BLK_N=128` 和 `BLK_K=64`，要求它 trace CTA `(1, 0)` 和 CTA `(0, 1)`。对每个 CTA，list `m_st`、`n_st`、每个 K iteration 加载的 A 和 B slice、以及写入的 D region。哪些 B row 成为 D column 因为 kernel compute `D = A @ B.T`？

### Complete Kernel

kernel 再次是 Step 2，这次只有两个 change：grid shape 和 per-CTA offset。inner K-loop 和 writeback untouched。import 相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

grid 变为 `[M // BLK_M, N // BLK_N]` 而不是 `[1, 1]`，load 和 store 现在由 CTA 自己的 `m_st` 和 `n_st` offset：

```python
def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 2D grid: 每 128×128 output tile 一个 CTA
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0

        # Per-CTA tile offset
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # 带 offset A 和 B slice 的 K-loop
        for i in T.serial(K_TILES):   # serial device loop (保持 full-K A/B parameter 正确 shape)
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # Writeback 到 correct output tile
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## Exercise

1. 在 Step 1-3 中，`Tx.copy` 在 MMA 前将 A 和 B tile move 到 SMEM。为什么 kernel need `T.cuda.cta_sync()` 在 `Tx.gemm_async` read 那些 SMEM tile 之前？
2. 在 Step 2 中，如果从 K-loop remove `phase_mma ^= 1` 会发生什么？kernel wait 每个 MMA，还是 later wait 可以 too early pass？
3. 对于 M=N=4096 且 BLK_M=BLK_N=128，Step 3 launch 多少 CTA？哪些 operand tile 在 neighboring CTA 间 logically reuse，Step 3 是否 exploit 那个 reuse？
