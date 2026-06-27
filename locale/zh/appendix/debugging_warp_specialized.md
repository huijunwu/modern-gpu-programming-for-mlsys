(chap_warp_spec_debug)=
# Debugging Warp-Specialized Kernel

{ref}`chap_gemm_advanced` 中的 GEMM Step 7-9 重叠 TMA load、`tcgen05` MMA 和 TMEM/SMEM writeback。相同的 debugging method 适用于 Flash Attention handoff：识别 role、识别每个 role 拥有的 storage，然后验证生成的 CUDA 是否符合那个 model。

不要从重写 kernel 开始。首先确保 run 是 valid 的，然后 inspect 生成的 CUDA。排除 environment 和 compile-time issue 后，这些 kernel 中的 runtime failure 通常归结为 broken handoff：未初始化的 barrier、错误的 arrival count、role guard 内隐藏的 collective、stale barrier phase，或在 producer 使 write visible 之前 reuse storage。

## Debugging Kernel 之前

首先排除 runtime context：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

这些 kernel target Blackwell（`sm_100a`）。如果 Python import 了过时的 TVM checkout，或 GPU 不是 Blackwell-class，在修改 kernel 之前先修复那个问题。然后运行 kernel 的最小 correctness check（例如 `run_correctness()`），再看 performance。

## Debugging Workflow

1. 在仍然失败的最小 shape 上复现 failure。如果 failure 是 illegal memory access，在下次 run 之前 restart Python。
2. 如果 compilation 失败，在读取 runtime synchronization code 之前检查安装的 API、target、`dispatch=` 和 buffer scope。
3. 保存 `inspect_source("cuda")` output。在重新读取 Python 之前，搜索其中的 role guard、`mbarrier_init`、`tcgen05`、`cp.async.bulk.tensor` 和 `cta_sync()`。
4. 为失败的 kernel path 编写 roles / storage / handoff / lifetime table。
5. 将生成的 CUDA 与那个 table 检查：barrier init 在 role branch 之前、期望的 TMA producer、MMA issuer、writeback group，且 warpgroup-only branch 内无 CTA-wide collective。
6. 将 run 分类为 deadlock、crash、wrong result 或 correct-but-slow，然后使用下面匹配的 section。
7. 一次改变一个 handoff：init count、arrive/wait phase、role guard、fence、TMA store drain、TMEM alloc/dealloc 或 tile-scheduler advance。
8. 在测量 performance 之前重新运行 correctness。

## What Transfers

对于任何 asynchronous kernel，在修改 code 之前做一个小 worksheet：

| 项目 | 写下什么 |
|---|---|
| Role | 发起每个 async operation 的精确 thread、warp、warpgroup 或 CTA。 |
| Storage | 每个 step 中每个 tile 的 live location：GMEM、SMEM、TMEM 或 register。 |
| Handoff | producer、consumer、signal object、arrival count、phase 以及使 data visible 的 fence 或 drain。 |
| Lifetime | 每个 storage slot 可以 reuse、read back 或 free 的最早点。 |

然后验证生成的 CUDA 是否符合 worksheet：

- Role guard 匹配 roles table。
- Barrier init 出现在 guarded role branch 之前。
- Collective operation 没有被 lane、warp 或 warpgroup guard 意外缩小。
- Arrive/wait phase 匹配 handoff table。
- TMA store drain、TMEM dealloc 和 SMEM reuse 只在 lifetime table 说它们合法之后发生。

对 TMA->MMA->writeback GEMM pipeline 和 Flash Attention 中的 score/softmax/value/correction handoff 使用同一个 worksheet。

## 如果 Compilation 失败

在 debugging runtime synchronization 之前修复 compile-time failure：

| Symptom | 可能区域 | 首先检查 |
|---|---|---|
| 未知 TIRx API 或 attribute error | 安装的 wheel 与 tutorial code 不匹配 | 打印 `tvm.__file__` 和 `tvm.__version__`；将 API name 与 {ref}`chap_language_reference` 比较。 |
| 不支持的 `dispatch=` | 选择的 target 或 primitive 不支持那个 path | 检查 `dispatch` argument 和 target capability；本教程中的 `tcgen05` path 需要 Blackwell。 |
| Buffer scope mismatch | buffer 通过错误的 hardware path 使用 | 检查 worksheet 的 storage row：TMEM 必须通过 `tcgen05` 访问，TMA operand 必须使用兼容的 GMEM/SMEM layout。 |
| Compile 成功但生成的 CUDA 缺少期望的 path | Dispatch 没有按期望的方式 lower | 在修改 algorithm 之前 inspect 生成的 CUDA 中的 `tcgen05` 和 `cp.async.bulk.tensor`。 |

## Inspecting Generated Code

对于任何 compiled kernel，保存 CUDA 以便搜索和 diff：

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)
```

生成的 code 将 TIRx construct 映射到 CUDA 如下：

| TIRx | 生成的 CUDA |
|------|------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` internal guard | `((int)threadIdx.x) < 1`（仅 CTA thread 0） |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

在读取完整 kernel 之前扫描这些 string：

| 生成的 CUDA | 检查 |
|---|---|
| `if (threadIdx.x < 1)` | 单个 CTA-thread guard，通常是 barrier initialization |
| `mbarrier_init` | Barrier initialization 存在且出现在 role branch 之前 |
| `tcgen05` | Tensor Core path 已生成 |
| `cp.async.bulk.tensor` | Copy lowered 到 TMA |
| `cta_sync();` | CTA-wide barrier；它必须在 `wg_id` branch 内部 |

## Step 7 Reference Skeleton

正确编译的 Step 7 kernel 有这个 top-level shape。下面的 guard 用 role name 编写以提高可读性；在生成的 CUDA 中搜索上表中对应的 expression。

```c
// (1) Barrier init：top level，仅 CTA thread 0
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // 全部 128 个 WG0 thread arrive
}

// (2) TMEM alloc：WG0 warp 0，issuing warp 的全部 lane
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) Fence + cta_sync，然后 phase init：producer=1, consumer=0

// (4) Warp-specialized loop
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) Cleanup：issuing warp，无 lane guard
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }
```

在修改 algorithm 之前检查这些：

- Barrier init 在 top level，不在 `wg_id` guard 内部。
- `tcgen05_alloc` 和 `tcgen05_dealloc` 有 warp guard 但无 lane guard；issuing warp 的全部 lane 参与。
- TMA 和 MMA loop 都迭代 `K_TILES` 次。
- Phase init 是 producer=`1`，consumer=`0`。

## Symptom Map

从 symptom 开始，但把它当作 clue 而不是最终 diagnosis：

| Clue | 可能区域 | 首先检查 |
|---|---|---|
| Kernel hang，然后 runtime 报告 unspecified launch failure | Deadlock | Barrier init placement、arrival count、`cta_sync()` placement 和 `next_tile()` participation |
| Illegal memory access、XID 或后续不相关的 CUDA call 也失败 | Crash / poisoned context | Restart Python，然后检查 pointer range、storage lifetime 和 collective participation |
| 错误 row 以 128-row 或 tile-sized stripe 出现 | Sync race 或 tile-index mismatch | Producer/consumer phase、scheduler advance 和哪个 warpgroup 拥有每个 row stripe |
| `NaN` 或明显无效的 value | Descriptor、operand setup 或 uninit accumulation | SMEM/TMEM descriptor setup、swizzle/layout 和 accumulator initialization |
| 有限但有 pattern 的错误 value | Stale 或部分 visible data | 缺少 fence、缺少 TMA store drain 或在 lifetime table 允许之前 reuse storage |
| Correct output 但没有期望的 speedup | Dispatch 或 resource issue | Generated CUDA path、pipeline depth、occupancy 和 register spill |

## 何时 Restart Python

CUDA error 不总是自动清理。在 illegal memory access、XID 或"CUDA context poisoned" error 之后，后续不相关的 call（如 `torch.randn`）可能继续失败。在测试下一个 fix 之前 restart Python process，否则你可能在 debugging 之前的 crash 而不是当前 code。

## Deadlock

按顺序检查这些：

- **Arrival count 与 init count 不匹配。** 常见 case：`MBarrier.init(128)` 但 `arrive` 被 `if warp_id == 0: if lane_id == 0:` guard，因此只有 1 个 thread arrive，wait 永远不返回。

  | Barrier | init(count) | 谁 arrive | Arrival |
  |---|---|---|---|
  | `TMABar`（tma->mma） | 1 | TMA engine 通过 `arrive(stage, bytes)` | 1 |
  | `TCGen05Bar`（mma->tma, mma->ld） | 1 | MMA warp 通过 `tcgen05.commit` | 1 |
  | `MBarrier`（ld->mma） | 128 | 全部 WG0 thread 通过 `arrive` | 128 |

- **Barrier init 嵌套在 `wg_id` guard 内部。** `.init()` lower 为 `if threadIdx.x < 1:`，意思是 CTA thread 0。CTA thread 0 在 WG0 中，因此 `if wg_id == 1:` 阻止每个 thread 运行 init。Init 必须在 top level；在 `inspect_source()` 中 `grep mbarrier_init` 来验证。

- **`cta_sync()` 在 warpgroup branch 内部。** `cta_sync` 是 `__syncthreads()`，需要全部 CTA thread。在 `if wg_id == 0:` 内部，WG1 永远达不到它。使用 `T.cuda.warpgroup_sync(10)` 作为 single-warpgroup barrier。

- **`tile_scheduler.next_tile()` 被一些 consumer-warpgroup thread 跳过。** Scheduler 跟踪 per-thread state；跳过它的 thread 可以永远 loop。

- **TMA 和 MMA 对 K-tile count 不一致。** 如果 MMA 做 `K_TILES - 1` 而不是 `K_TILES`，barrier phase 漂移，第二个 outer tile deadlock。

- **`PipelineState` 初始 phase 错误。** Producer 从 `phase=1` 开始使第一个 wait pass；consumer 从 `phase=0` 开始使第一个 wait block。如果两者从相同 phase 开始，第一个 handoff 可以立即 deadlock。

## Crash 和 Context Poisoning

常见原因：

- **`pool.commit()` 之后 `pool.alloc`。** Barrier wrapper 内部调用 `alloc`。正确顺序：`tmem_addr -> barrier wrapper -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`。
- **`tcgen05.alloc` 或 `tcgen05.dealloc` 带 lane guard。** Issuing warp 必须用全部 lane 参与。`if lane_id == 0:` 运行一个 thread，那是 undefined behavior。
- **`tcgen05.dealloc` 之前缺少 `cta_sync()`。** writeback 仍在读取时 TMEM 被 free。
- **OutOfRange GMEM 或 SMEM access。** 缩小到一个 tile，检查 scheduler 的 `m_idx` / `n_idx`，检查当前 shape 是 kernel tile 或 cluster tile 的倍数。

## Wrong Result

在猜测之前按 pattern 分类 wrong output。整个 row stripe 通常指向 producer/consumer phase、tile-index 或 role-ownership mismatch。`NaN` output 通常指向 descriptor setup、operand setup 或 uninit accumulation。有限但有 pattern 的 wrong value 通常意味着 consumer 读了旧 tile、部分写入的 tile 或 store 尚未 drain 的 data。

- **`tcgen05.commit` 在 `elect_sync` 外部。** 全部 32 个 thread 创建 commit group；31 个空 group 立即 signal mbarrier。TMA 可能在 MMA 读取之前 overwrite SMEM。
- **TMA store 之前缺少 `fence.proxy_async("shared::cta")`。** TMA engine 可能看不到 thread 的 SMEM write。
- **TMA store 之后缺少 `cp_async.bulk.commit_group()` 加 `wait_group(0)`。** 下一个 tile 可能在 store drain 之前 reuse Dsmem。
- **Persistent kernel 在 1024x1024 等小 size 上间歇性失败。** 更大的 size 可以用更长的 K-loop 掩盖 race。重新检查 tile 之间的 phase reset 和 TMA-store commit/wait。
- **`fence.after_thread_sync()` 通常不是 fix。** MMA-completion mbarrier 已经携带 release-acquire semantics。Step 8 和 9 在 writeback edge 保守地添加它，在 `mma2ld.wait` 之后、第一个 `tcgen05.ld` 之前；不要在 TMA-to-MMA edge 常规添加它。

## Correct but Slow

如果 output correct 但 performance 远低于期望，使用相同的 inspection loop：

| Clue | 可能区域 | 首先检查 |
|---|---|---|
| Generated CUDA 没有 `cp.async.bulk.tensor` | Copy 没有 lower 到 TMA | 检查 `dispatch="tma"`、target capability 和 operand layout |
| Generated CUDA 没有 `tcgen05` path | MMA 没有 lower 到 Blackwell Tensor Core instruction | 检查 `dispatch="tcgen05"`、target capability 和 operand layout |
| TMA 和 MMA 不 overlap | Pipeline 太浅或 phase serialize producer/consumer | Inspect generated CUDA 中 wait/arrive/advance 的顺序 |
| 小 shape correctness 好但大 shape speed 差 | Register spill、occupancy 或 staging-buffer pressure | 检查 compiler resource report；减小 tile size、chunk writeback 或降低 pipeline depth |

## Filing a Good Issue

如果 failure 通过了上面的检查，在 [Apache TVM GitHub repository](https://github.com/apache/tvm/issues) 上 filing issue 之前先 reduce 它。包括：

- `tvm.__file__` / `tvm.__version__` output 和 GPU capability；
- 复现 failure 的最小 shape；
- failure 是 compile-time、deadlock、crash、wrong result 还是 correct-but-slow；
- 最小 kernel 或 notebook cell 加上它的 correctness check；
- 保存的 `inspect_source("cuda")` output，或显示可疑 guard、barrier 或 dispatch path 的最小 excerpt。
