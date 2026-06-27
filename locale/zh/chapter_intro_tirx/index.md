(chap_tirx_primer)=
# TIRx 入门

:::{admonition} Overview
:class: overview

- TIRx 是一个用于在 IR 层级编写 GPU kernel 的 Python DSL：你直接命名 hardware，但通过结构化 IR。
- 每个 tile operation 由三个 design element 控制：*scope*（哪些 thread）、*layout*（tile 存放在哪里）和 *dispatch*（哪条 hardware path）。
- 一个可运行的 single-MMA GEMM 展示了这三者；本书其余内容是这些 design element 的规模化应用。
:::

:::{admonition} 运行示例
:class: note

这些示例需要 Blackwell GPU（`sm_100a`，如 B200）。TIRx compiler 作为 Apache TVM wheel 的
`tvm.tirx` module 分发；与 CUDA 版 PyTorch 一并安装：

```bash
pip install apache-tvm
```

通过 `python -c "import tvm, tvm.tirx; print(tvm.__version__)"` 确认 import 成功。相同 setup
运行本书每个可运行示例。
:::

Part I 解释了 hardware 是什么。要让它计算任何东西，我们需要一种编程方式。

我们可以编写 raw CUDA 或 PTX，许多 fast kernel 正是那样编写的。问题是，真正决定 kernel behavior 的 decision 在那里很难看到：哪些 thread 运行 operation、每个 data tile 存放在哪里、以及哪条 hardware path 执行它。那些 choice 隐藏在内建函数参数、address arithmetic 和 convention 中。

TIRx（Tensor IR neXt）是一个 Python DSL，将这三个 decision 提升到明处：**scope**（哪些 thread 运行 operation）、**layout**（operand tile 存放在哪里）和 **dispatch**（哪条 hardware path 执行它）。它仍然直接命名 hardware concept，包括 thread、shared memory 和 tensor memory、barrier 以及 `tcgen05` MMA。不同之处在于，那些 choice 现在是 compiler 可以 lower、check 和 schedule 的结构化 IR。

与其在抽象层面引入这些 idea，我们将从一个完整 kernel 入手：一个最小 single-MMA GEMM。我们先让它运行，然后逐行读回，看 scope、layout 和 dispatch 如何塑造它，以及 kernel 如何被编译。kernel 依赖的 tensor layout model 在 {ref}`chap_tirx_layout_api` 中有独立展开，完整 language-feature set 在 {ref}`chap_language_reference`；这里我们将 focus 放在一个 kernel 和三个 design element 上。

## 第一个 Kernel: Single-MMA GEMM

我们承诺的 kernel 是一个最小 GEMM，精简到仍 exercise Tensor Core 的最小版本。它计算 `D = A B^T` 的单个 128 × 128 output tile，K = 64。整个 computation 表达为一个 `Tx.gemm_async` tile operation，从端到端。（那个 tile operation 不映射到单条 hardware instruction：因为 hardware MMA K-atom 是 16，K=64 tile lower 为短序列 `tcgen05.mma` instruction，沿 K 步进。DSL 的 point 正是我们写 tile，而不是写 sequence。）围绕那个 operation，kernel 做通常的 chore：它 allocate shared memory（SMEM）和 tensor memory（TMEM）、将 A 和 B 从 global memory copy 到 shared memory、发出 tile MMA 到 TMEM accumulator、通过 register 读回那个 accumulator、并 store result。尽管很小，这个 kernel 是 {ref}`chap_gemm_basics` 中我们攀爬的 GEMM ladder 的 Step 1，在那里它以完整 walkthrough 回归。

每个 TIRx kernel 从相同 handful import 开始，因此值得 upfront 看一次：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

我们将 kernel 包装在一个小 builder `hgemm_v1(M, N, K)` 中，它接受 problem shape 并返回 `PrimFunc`。对于我们选择的 shape `M=N=128, K=64`，launch 恰好包含一个 output tile，这正是保持这个 first version 足够简单、可一次读完的原因：

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
    
        # --- Load: 所有 thread copy global -> shared (synchronous)。
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
    
        # --- Writeback: TMEM -> RF -> GMEM ---
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

在我们读 kernel 之前，让我们确认它工作。我们编译它并将 output 与 torch reference 对比。我们不必拼出 exact architecture：arch（如 `sm_100a`）从 device auto-detect，因此 target `"cuda"` 足够，`tir_pipeline="tirx"` 是选择 TIRx lowering pipeline 的选项。一旦编译，`ex.mod(...)` 直接接受 torch tensor，中间无需 manual conversion。

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
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

## Scope、Layout、Dispatch

现在 kernel 运行了，我们可以读回它并问那些 line 实际决定了什么。从这个角度看，整个 kernel 是沿三个 design element 的 choice 集合。其中每个 operation 回答相同三个 question：*谁*运行它、它的 data *存放在哪里*、*如何*执行，而那三个 answer 正是 scope、layout 和 dispatch。本节其余部分一次取一个 design element；下面 interactive demo 让你看到每个 design element 控制哪些 line。

```{raw} html
<iframe src="../demo/tirx_dispatch.html" title="TIRx: scope, layout, dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: 点击 Scope / Layout / Dispatch 以 spotlight 每个 design element 控制的 kernel line。*

使用 demo 时，注意三个 question：

- **Scope: 谁运行 operation？** `Tx.cta.copy(...)` 是 CTA-scoped，因此所有 128 thread 帮助 GMEM → SMEM copy。`Tx.gemm_async(...)` 由 elected thread 发出一次，因为每个 lowered `tcgen05.mma` instruction 已经是 cooperative MMA launch。`Tx.wg.copy_async(...)` 是 warpgroup-scoped，因此 warpgroup 的 128 thread 按 row 分割 TMEM readback。
- **Layout: 每个 tile 存放在哪里？** A 和 B 使用 `tcgen05.mma` 期望的 swizzled SMEM layout。accumulator 在 `TLane`/`TCol` layout 下存放在 TMEM 中。register readback view 将 row 映射到 `tid_in_wg`，因此每个 warpgroup thread 拥有一个 row fragment。
- **Dispatch: 哪条 hardware path 执行它？** `Tx.gemm_async(..., dispatch="tcgen05", ...)` 选择 Blackwell Tensor Core path。copy operation 也有 dispatch choice：这个 first kernel 使用普通 thread copy，later GEMM step 将那些 copy 替换为 TMA，而不改变 surrounding scope 或 layout。

**与你的 agent 一起尝试**：从 first kernel 选三行：一个 copy、一个 MMA、一个 TMEM readback。要求它按 scope、layout 和 dispatch 标注每行，然后检查 answer 是否与 code 中的 guard、buffer layout 和 `dispatch=` argument 匹配。

## Compilation 如何工作

我们上面已经编译了 kernel 以测试它；现在我们更近距离地看那个 step 做什么。recipe 简短：将 `PrimFunc` 包装在 `IRModule` 中并交给 `tvm.compile(mod, target=..., tir_pipeline="tirx")`。这运行 TIRx lowering pipeline 并返回一个你直接调用的 `Executable`。

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

至少从 outline 层面知道 `tir_pipeline="tirx"` 启动了什么，这是值得的。pipeline 的 central pass `LowerTIRx` 将每个 tile primitive 对其 scope / layout / dispatch contract 进行 resolve：这正是我们刚讨论的三个 design element 实际兑现为 instruction 的地方。之后，通常的 host/device split 和 finalize step 产生可 launch module。如果你偏好，你也可以在 `with target:` block 内编译，它让 kernel pick up surrounding target context。

这个 flow 的一个 nice property 是 nothing 对你隐藏：result 可在两个 level 检查。你可以用 `.show()` 或 `.script()` 读 IR 本身，你可以从编译 module 直接读 compiler 最终发出的 CUDA C。

```python
kernel.show()                          # pretty-print TIRx (TVMScript)
print(kernel.script())                 # ... 相同内容，作为 string

# 从编译 Executable 生成的 CUDA C source：
print(ex.mod.imports[0].inspect_source())
```

这只是 sketch。对于完整 lowering story，覆盖所有 pass、tile-primitive dispatch 如何 resolve、以及 host/device split 如何做，见 {ref}`chap_arch`。

## 下一步去哪里

一个 kernel 足以认识 scope、layout 和 dispatch，并看到它们被编译和运行。三个 design element 中的每个，以及 kernel 本身，都通向一个将它推进的 chapter：

- {ref}`chap_tirx_layout_api`：上面 operand 和 accumulator placement 构建自的 tensor layout model（`TileLayout`、named axis、swizzle）。如果 layout design element 感觉是三个中最神秘的，从这里开始。
- {ref}`chap_language_reference`：完整 language-feature set，覆盖 parser utility、data type、buffer 和 memory、control flow 以及 thread synchronization，适用于你想要 complete vocabulary 而不是 tour 的时候。
- {ref}`chap_gemm_basics`：这个 kernel 作为 GEMM optimization path 的 Step 1，通过 K-loop accumulation、spatial tiling、TMA 和 warp specialization 构建。如果你想看相同三个 design element 扩展到 real kernel，这是 natural next stop。
