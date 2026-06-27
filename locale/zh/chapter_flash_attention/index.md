(chap_flash_attention)=
# Flash Attention 4

:::{admonition} Overview
:class: overview

- Attention 运行两个 MMA，中间夹着 softmax，因此它不能像 GEMM 那样重复一个 MMA。
- kernel 组合了 Part I 中的 hardware primitive（TMA、`tcgen05`、TMEM、barrier）和 Part III 中的 GEMM 技术（warp role、online-softmax rescaling、causal masking、GQA）。
:::

Attention 是决定 transformer 能否运行的 kernel，也是到目前为止我们构建的一切最终需要协同工作的地方。我们为 GEMM 组装的每个组件都带入这里：TMA tile movement、`tcgen05` MMA、TMEM、warpgroup register tile 和 explicit barrier。

挑战在于 attention 不是重复一个 MMA。它是两个 MMA，中间夹着真正的工作：online softmax、causal masking 和保持早期和后期 block 在相同 scale 的 rescaling。

新难度就在那个中间 stage。普通 matmul 只向 accumulator 添加；attention 必须在新 key 和 value 流入时重新访问并 rescale 已经计算的 result。softmax work 本身也在两个 Tensor Core MMA 之间在 CUDA core 上运行，因此 exponential 和 row-wise reduction 直接坐在 critical path 上。

这就是为什么这么多 attention optimization 实际上是 softmax optimization：重构 `exp`，将 softmax 与 MMA overlap 而不是在它上面 stall。

本章的目标不是从头重新推导 Flash Attention。我们将保持足够的 algorithm 可见性使 kernel 可读，然后把注意力集中在真正新的部分：那个 algorithm 如何转化为 TIRx。

最清晰的切入方式是跟随单个 tile 流过 kernel。`Q`、`K` 和 `V` 作为 input tile 进入，从 GMEM 加载到 SMEM。score MMA 将 `Q` 和 `K` 相乘到 TMEM 中的 score tile `S`。Softmax 将 `S` 转为 numerator tile `P`，value MMA 组合 `P` 和 `V` 来更新 output accumulator `O`。

到目前为止这看起来像两个 matmul 粘在一起，但有一个 GEMM 从未遇到过的 twist：每当 running softmax maximum 改变时，到目前为止累积的 `O` 突然处于错误的 scale。它必须在下一个 value MMA 安全地添加之前被 rescale。下面的 section 首先追踪这条 path，然后才展示 TIRx 如何将每个 stage 交给 warpgroup 并将 stage 连接起来。

## Algorithm Shape

在将 tile 放置到 memory 之前，我们需要那些 tile 服务的 algorithm。对于一个 query block，Flash Attention 计算：

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

字面阅读，formula 说先形成完整的 score matrix `S = QKᵀ`，softmax 它，然后乘以 `V`。那是我们不能使用的唯一方法，因为完整的 `S` 巨大。在 seq=4096 时它每 head 持有约 16M 个 element，fp32 约 64 MB，比 SMEM 或单个 128×512 TMEM region 大几个数量级。片上根本没有地方放它。Flash Attention 的回答是根本不 materialize `S`。相反它按 block 流式传输 `K/V`，并携带三个 per-row running state 来总结到目前为止看到的一切：

- `row_max`：到目前为止看到的最大 score。
- `row_sum`：softmax 的 running denominator。
- `O`：running output accumulator。

streaming update 是保持那些 state 在 new block 到达时正确的机制。微妙之处在于每次我们处理一个 block 时，running max 可能上升，一旦上升，我们在旧 max 下计算的一切现在处于错误的 scale。因此在添加新 contribution 之前，我们首先将旧 state 拉回到新 scale：

```text
S = Q_block @ K_block.T
m_new = max(row_max, rowmax(S))
scale = exp((row_max - m_new) / sqrt(d))
P = exp((S - m_new) / sqrt(d))
row_sum = row_sum * scale + rowsum(P)
O = O * scale + P @ V_block
row_max = m_new
```

单个 `scale` factor 在这里一箭双雕：它同时 rescale running denominator 和 running output，使来自早期和后期 block 的 contribution 最终用相同 scale 衡量。

上面的 pseudocode 用自然的 `exp` 和显式的 `/sqrt(d)` 编写，因为那样最容易阅读，但 kernel 走了更便宜的路。它将 `1/sqrt(d)` 和 `log2(e)` 折叠为一个 constant `scale_log2 = log2(e)/sqrt(d)`，并用 hardware `exp2` 在 raw score 上评估每个 exponential，使用恒等式 `exp(x/sqrt(d)) = exp2(x · scale_log2)`。动机仅仅是 `exp2` 比自然 `exp` 在这个 hardware 上更快。

在我们继续之前，有一点值得明确：这里的 `P` *不是* 最终归一化的 attention matrix。它只是当前 K/V block 的 softmax numerator。归一化被有意推迟，只有在最后一个 block 之后 kernel 才写 `O / row_sum`。

对于 TIRx，知道 algorithm 计算什么只是一半 picture。另一半是*每个 tile 在 kernel 运行时 lives 在哪里*，因为这决定了 layout 和 barrier code。`S`、`P` 和 `O` 都是 tile value，每个都有一个 home：

- `S` 是 score tile。score MMA 将它写到 TMEM。
- `P` 是 softmax numerator tile。Softmax 从 TMEM 读 `S` 到 register，计算 `P = exp((S - m_new) / sqrt(d))`，然后将 `P` 写回 TMEM。
- `O` 是 output accumulator tile。value MMA 从 TMEM 读 `P`、从 SMEM 读 `V`，然后累积到 TMEM 中的 `O`。

我们之前标记的 rescale 也是 tile operation，不是 scalar bookkeeping：当 `row_max` 改变时，旧的 `O` 从 TMEM 读取，在 register 中相乘，然后在下一个 value MMA 累积之前写回 TMEM。后面的每个 section 都遵循相同的结构：tile placement、hardware path 和证明下一个 consumer 可以运行的 barrier。

## Tile-Primitive Graph

有了 running state 和它们的 home，我们可以将 algorithm 布局为具体的 tile move 序列。对于一个 K/V block，kernel 从上到下走这条 tile path：

```text
Q, K, V 在 GMEM 中
  -> Q, K, V 在 SMEM 中        通过 TMA load
  -> S 在 TMEM 中              通过 score MMA: QK^T
  -> P 在 TMEM 中              通过 softmax numerator: TMEM -> RF -> TMEM
  -> O 在 TMEM 中              通过 value MMA: P V
  -> O 在 GMEM 中              通过归一化、SMEM staging 和 TMA store
```

与 GEMM 的差异归结为一行。GEMM 是重复的 MMA chain；FA4 有两个 MMA phase，softmax 坐在 chain 中间。几乎所有的后续内容都是那一个额外 stage 的后果。

如果将短 path 展开为 explicit producer-consumer edge，我们得到完整 graph：

| Stage | Tile movement 或 compute | TIRx primitive | Hardware path |
|-------|--------------------------|----------------|---------------|
| Load Q/K/V | GMEM tile -> SMEM tile | `Tx.copy_async(..., dispatch="tma")` | TMA load |
| Score MMA | SMEM 中的 Q 和 K -> TMEM 中的 score tile `S` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| Softmax read | TMEM 中的 `S` -> warpgroup register tile | `Tx.wg.copy_async(reg, tmem)` | `tcgen05.ld` |
| Softmax write | register 中的 numerator tile `P` -> fp16 TMEM view | `Tx.copy_async(tmem_as_f16, reg)` | TMEM store，后跟 `tcgen05.wait.st()` |
| Value MMA | TMEM 中的 `P` 和 SMEM 中的 V -> TMEM 中的 output accumulator `O` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | 带 TMEM operand 的 `tcgen05.mma` |
| Correction | TMEM 中的 `O` -> register -> TMEM 中的 `O` | TMEM readback, register multiply, TMEM store | `tcgen05.ld` / TMEM store |
| Epilogue | 最终 TMEM 中的 `O` -> register -> SMEM -> GMEM | TMEM readback, `Tx.copy`, TMA store | `tcgen05.ld` + TMA store |

新的 row 是 softmax 和 correction。两者都添加 TMEM -> register -> TMEM traffic，两者都在 score MMA 和 value MMA 之间创建额外的 handoff。

**与你的 agent 一起尝试**：要求它只追踪上面的短 path。对于每个箭头，命名 producer stage、consumer stage、source tile、destination tile 和 hardware path。然后询问哪些箭头在 GEMM chapter 中不存在。

## Warp Role 和 Scope

data path 确定后，自然的下一个问题是实际运行每个 stage 的是谁。每个 CTA 这里有 4 个 warpgroup，共 512 个 thread，它们不是按触碰什么 data 来划分，而是按 warpgroup 做*什么类型的工作*来划分：

- WG3 驱动 hardware engine：TMA load、MMA 和 TMA store。
- WG0、WG1 和 WG2 做那些 engine call 之间发生的 register-heavy math：softmax、correction 和 epilogue。

精确的 role table 是：

| Owner | Role | 做什么 |
|-------|------|--------|
| WG3, warp 1 | TMA load | 将 Q、K 和 V tile 从 GMEM 加载到 SMEM |
| WG3, warp 0 | MMA | 发起 score MMA 和 value MMA |
| WG3, warp 2 | TMA store | 将最终 O tile 从 SMEM 存储到 GMEM |
| WG0 | Q stage 0 的 Softmax | 从 TMEM 读 S，计算 P，将 P 写到 TMEM |
| WG1 | Q stage 1 的 Softmax | 对第二个 Q pipeline stage 做相同工作 |
| WG2 | Correction 和 Epilogue | Rescale TMEM 中的 O，归一化，staging output |

容易将"两个 Q stage"误读为两个 attention head，但它们不是。它们仅仅是 Q pipeline 中的两个 slot，WG0 拥有一个，WG1 拥有另一个，使两个 Q tile 可以同时 in flight。这就是 softmax work 出现两次的原因，一次在 WG0 上，一次在 WG1 上。

code 用 symbolic coordinate 挑选这些 role：

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

阅读 kernel 时，先找到 role branch。它告诉你哪个 team 拥有嵌套在其中的每个 tile primitive。

- WG3 warp 1 启动 TMA load command。一个 elected lane issue copy，TMA engine 移动 tile。
- WG3 warp 0 issue `tcgen05.mma` instruction。
- WG0 和 WG1 在 full warpgroup scope 下运行 softmax。
- WG2 在 full warpgroup scope 下运行 correction 和 epilogue work。

一个不对称性最终塑造了整个 barrier graph：*每个* MMA（score 和 value）都只从 WG3 warp 0 发出。WG0 和 WG1 从不 issue MMA。它们只消费 score tile，运行 softmax，将 `P` 写回 TMEM。

这种分离正是 softmax 需要 barrier 的原因。`s_ready` 将 score tile 从 MMA warp 带到 softmax；`p_o_rescale` 携带 `P` 和安全的 `O` slot 供 value MMA 使用（已经 rescale 或因为不需要 rescale 而 release）。本章剩余部分我们将反复回到这两个名字。

## Reading the Fragment

本章的 fragment 是 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py) 的摘录，因此不可避免地引用了我们不 reproduce 的 kernel 部分中定义的 name。自描述的（`wg_id`、`warp_id`、`BLK_M`/`BLK_N`、`HEAD_DIM`、`kv_stage`、`SMEM_PIPE_DEPTH_*` / `TMEM_PIPE_DEPTH` depth、`should_accumulate` 和 `CTA_GROUP`（这里为 1））我们在下面它们首次重要的地方引入。其余的在这里的表中获得一行 gloss，这样当 fragment 把 unfamiliar name 放在你面前时，你有一个地方可以查找：

| Name | 含义 |
|------|------|
| `q_stage`, `i_q` | Q pipeline stage，0 或 1，即哪个 Q tile slot（`SMEM_PIPE_DEPTH_Q = 2`）。在 WG0/WG1 softmax 内部，warpgroup 自己的 `wg_id`（0 或 1）*就是* 同一个 stage index，因此 `S_region[q_stage]`、`P_region[wg_id]` 和 `O_region[i_q]` 都选择同一个 Q stage |
| `MMA_N` | TMEM 列中 score/output tile 宽度（128） |
| `MMA_K` | `P`/`V` 列中 MMA inner-K step（16）；`K_SPLIT = 6 * MMA_K = 96` |
| `K_SPLIT` | value-MMA schedule 的 split point（见 *两个 MMA Phase*）；第一个 value MMA 覆盖列 `0:K_SPLIT`（`6 * MMA_K = 96`） |
| `should_rescale` | WG2 per-row flag：在下一个 value MMA 之前旧 `O` 是否需要 rescaling（用 `any_sync` 跨 warpgroup reduce） |
| `rescale_threshold` | 跳过小 row-max 变化的阈值；当前 kernel 使用 `8.0`，跳过的 rescale 将 `acc_scale` 设为精确 `1.0` |
| `scale_log2` | log2 单位的 softmax scale，`log2(e)/√d`，因此 `P = exp2((S - m) · scale_log2)` |
| `acc_scale` | softmax 通过 SMEM mailbox 传递给 WG2 的 per-row rescale factor |
| `chunk_start`/`chunk_end`, `p_start`/`p_end` | 正在读取/写入的 32-wide softmax chunk 的列 range |

## 两个 MMA Phase

对于每个 streamed K/V tile，Flash Attention 运行两个 MMA phase，softmax 桥接它们：

```text
Q, K -> score MMA -> S
S    -> softmax   -> P
P, V -> value MMA -> O
```

把这看作三个 producer 的 pipeline。第一个 MMA 产生 attention score `S`，softmax 将 `S` 转为 numerator `P`，第二个 MMA 消费 `P` 来更新 output accumulator `O`。按 `row_sum` 的归一化推迟到 epilogue，在所有 K/V tile 都说完之后。

下面的每个 tile op 获得与 GEMM step 相同的 **scope / layout / dispatch** card，外加一行 **Handoff**，命名将 tile 传递给下一个 role 的 barrier。

compute code 从不使用 raw TMEM 列号。相反 kernel 将单个 TMEM allocation 切割为 per-stage view（`S_region`、`P_region`、`O_region`），用 pipeline stage 索引它们（`S_region[q_stage]`、`O_region[i_q]`、`P_region[i_q, 0:K_SPLIT]`）。那些 view 在 [TMEM Layout 和 Reuse](#tmem-layout-and-reuse) section 中用 `T.TMEMStages` 定义；现在只需将每个 region 视为同一物理 TMEM 的 named slice。

### Score MMA

两个 phase 中的第一个是 score MMA，每个 K/V iteration 打开的 matmul。它计算：

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

并将 `128 x 128` score tile 写到 TMEM：

```python
Tx.warp.gemm_async(
    S_region[q_stage],
    Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
    K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

我们可以问 GEMM chapter 对每个 tile op 问的同样四个问题：谁运行它、tile lives 在哪里、如何 dispatch、如何 handoff：

> **Tile-primitive readout: Score MMA**
> - Scope：WG3 warp 0 issue 它；一个 elected lane arrive `s_ready`。
> - Layout：Q、K 在 SMEM 中 → `S` 在 TMEM 中（`S_region[q_stage]`）。
> - Dispatch：`tcgen05`。
> - Handoff：`s_ready`（→ softmax）。

在 `s_ready` 上 arrive 的单个 elected thread 就是整个 handoff。它宣布这个 score tile 完成，softmax warpgroup 现在可以自由读取它。

### MMA 之间的 Softmax

两个 MMA 之间是 softmax，将 score tile `S` 转为 numerator tile `P` 的 stage。它的 readout card 是：

> **Tile-primitive readout: Softmax**
> - Scope：WG0（Q stage 0）/ WG1（Q stage 1），full warpgroup。
> - Layout：TMEM 中的 `S` → register → fp16 TMEM 中的 `P`（`P_region[wg_id]`）。
> - Dispatch：`tcgen05.ld` 读取，TMEM store 写入；两者之间在 register 中做 row-wise math。
> - Handoff：wait `s_ready`；arrive `p_o_rescale`（前 96 列）和 `p_ready_2`（最后 32 列）。

这个 stage 完全没有 GEMM counterpart。WG0/WG1 wait score tile 在 `s_ready` 上到达，然后以 register-sized chunk 从 TMEM 读出来：

```python
Tx.copy_async(
    s_chunk[:, chunk_start : chunk_end],
    S_region[wg_id, chunk_start : chunk_end],
)
```

那是 warpgroup scope 下的 TMEM-to-register tile read。现在 score 坐在 register 中，softmax warpgroup 按顺序做三件事：

1. 计算 row max 和 row sum，
2. 计算 softmax numerator tile `P`，
3. 将 `P` 作为 fp16 写回 TMEM。

最后一步看起来像：

```python
Tx.copy_async(
    P_region[wg_id, p_start : p_end],
    p_chunk[:, p_start : p_end],
)
```

为什么将 `P` 写回 TMEM，当我们在 register 中刚计算完它？因为 value MMA 需要 `P` 作为*tile operand*，MMA 不能将 scattered per-thread scalar register 作为 matrix 读取。这个 kernel 中 `P` 的 MMA-readable form 是 `P_region`，跨越 fp16 TMEM alias `tmem_as_f16` 的 view。因此 writeback 不是冗余 motion；它是将 `P` 放入下一个 MMA 实际可以消费的唯一形状。

### Value MMA

第二个 phase，也是关闭每个 K/V iteration 的 phase，是 value MMA。它计算：

$$O = O + P_{\text{block}}V_{\text{block}}$$

到这个 MMA 运行时，`O` 已经为当前 K/V block 放入正确的 state，在第一个 block 上初始化，在后续 block 上 rescale，因此 MMA 只需要累积。使它区别于 GEMM 的是 operand lives 在哪里：A operand 是 TMEM 中的 `P`，B operand 是 SMEM 中的 `V`，accumulator `O` 也在 TMEM 中：

```python
# 第一个 sub-MMA：列 0:K_SPLIT（P 的前 96 / V 的 row）。
Tx.warp.gemm_async(
    O_region[i_q],
    P_region[i_q, 0:K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True,
    accum=should_accumulate,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
# 第二个 sub-MMA（相同 form，accum=True，gated on p_ready_2）覆盖
# 剩余的列 K_SPLIT:BLK_N。
```

> **Tile-primitive readout: Value MMA**
> - Scope：WG3 warp 0。
> - Layout：TMEM 中的 `P` + SMEM 中的 V → TMEM 中的 `O`（`O_region[i_q]`）。
> - Dispatch：带 TMEM operand 的 `tcgen05`。
> - Handoff：wait `p_o_rescale`、`p_ready_2`、`kv_load.full`；arrive `o_ready`（→ epilogue）。

这个 operand placement 是两个 MMA 之间的 hardware 差异：

- Score MMA 从 SMEM 读两个 operand：Q 和 K。
- Value MMA 从 TMEM 读一个 operand `P`。
- Value MMA 从 SMEM 读另一个 operand V。
- result 累积到 TMEM 中的 `O`。

`accum=should_accumulate` flag 实现了 algorithm 中的"初始化或添加"选择：在 query block 的第一个 K/V tile 上为 false，之后的每个 tile 上为 true。

你可能还注意到 value MMA 不是一次运行，而是拆分为 `96 + 32` schedule：

1. Softmax 以四个 32 列 chunk 写入 `P`。
2. 一旦前三个 chunk 就绪，value MMA 开始在 `P` 的前 96 列和 V 的匹配 row 上运行。
3. 最后 32 列 wait `p_ready_2`。
4. 第二个 MMA 消费最后一个 chunk 并完成 tile。

拆分的原因是保持 Tensor Core busy。将 value MMA 作为单个 instruction 运行，整个 phase 将 stall 直到所有四个 32 列 `P` chunk 完成 exponentiation 和 store。通过立即在第一个三个 chunk 上触发，kernel 将最后一个 chunk 的 `exp` 和 TMEM write 与已经在 flight 的 96-wide MMA overlap，将原本 idle 的时间转为有用的 work。

## TMEM Layout 和 Reuse

所有 `S`、`P` 和 `O` 必须共享一个 `128 x 512` TMEM allocation，它们被打包进其中的方式正是 barrier 和 layout 在这个 kernel 中不可分割的原因：

下图直接展示了那个 packing：score slot、numerator slot 和 output slot 共享一个 TMEM allocation，barrier protocol 使 reuse 合法。

![TMEM Layout](../../img/tmem_layout_v3.png)

图读作一组 tile slot：

- Score slot 持有 `S = QK^T`。
- Numerator slot 在 softmax exponentiation step 之后持有 `P` tile。
- Output slot 持有 fp32 `O` accumulator。

它们不是独立的 buffer。它们是*同一个* allocation 的 region，sharing 不是 stylistic choice 而是 forced 的。Q-pipeline depth 2 时，两个 `S` slot（2 × MMA_N = 256 列）和两个 `O` slot（2 × MMA_N = 256 列）已经占用了全部 512 个 fp32 列。没有剩余给 `P`，因此 `P` 别无选择，只能通过更窄的 fp16 view alias 相同的 byte。这安全的唯一原因是每个 region 严格在其前一个 consumer 完成之后 reuse，而那个 timing 正是 barrier 保证的。因此在 FA4 中 barrier 不仅仅是 scheduling；它们首先使 layout 合法。

TIRx 用 `T.TMEMStages` 使那些 view explicit：

```python
# TMEM 区域：score, numerator, output 共享同一 128×512 分配
tmem_stage = T.TMEMStages(
    tmem,
    S_region=(TMEM_PIPE_DEPTH, MMA_N),       # 2×128, fp32
    P_region=(TMEM_PIPE_DEPTH, MMA_N),       # 2×128, fp16 (alias)
    O_region=(TMEM_PIPE_DEPTH, MMA_N),       # 2×128, fp32
)
```

`P_region` 的 fp16 alias 是关键：它使 `P` 可以写入与 `S`/`O` 相同的 byte，只要 timing 正确。barrier 确保 timing 正确。

## Barrier Flow

barrier 在 FA4 中比在 GEMM 中更密集，因为 softmax 在 score MMA 和 value MMA 之间插入了额外的 handoff。每个 barrier 仍然遵循相同的 rule：producer arrive，consumer wait。

主要 path 中的 barrier 是：

| Barrier | Producer -> consumer | 什么变为安全 |
|---------|----------------------|--------------|
| `q_load.full` | TMA load -> score MMA | Q SMEM tile 可以 feed MMA |
| `q_load.empty` | 这个 Q stage 的所有 score MMA -> TMA load | Q SMEM stage 可以 reuse 给下一个 task |
| `kv_load.full` | TMA load -> score/value MMA | K 或 V SMEM tile 可以 feed MMA |
| `kv_load.empty` | score/value MMA -> TMA load | K/V SMEM stage 可以 reuse |
| `s_ready` | score MMA -> softmax | S TMEM tile 可以读取 |
| `p_o_rescale` | softmax + WG2 -> value MMA | `P` 的前 96 列在 TMEM 中，`O` slot 对 value MMA 安全 |
| `p_ready_2` | softmax -> value MMA | `P` 的最后四分之一在 TMEM 中 |
| `o_ready` | value MMA -> epilogue | 最终 O accumulator 就绪 |
| `softmax_corr.full` | softmax -> WG2 | `acc_scale` 或最终 `row_sum` 在 SMEM mailbox 中就绪 |
| `softmax_corr.empty` | WG2 -> softmax | WG2 读取后同一个 SMEM mailbox slot 可以 reuse |
| `corr_epi.full` | epilogue -> TMA store | O_smem 就绪可以 store |
| `corr_epi.empty` | TMA store -> epilogue | O_smem stage 可以 reuse |

与 GEMM 一样，你可以从谁产生 signal 预测 barrier 的类型：

- TMA load 使用 `TMABar`，因为 TMA engine byte-count 自己的 completion。
- MMA completion 使用 `TCGen05Bar`，因为 `tcgen05.commit` signal completion group。
- 纯 thread-to-thread handoff 使用 `MBarrier`，参与 thread explicit arrive。

split softmax-to-value handoff 值得更仔细观察。它使用两个 gate：

- `p_o_rescale` 让 value MMA 在 `P` 的前 96 列写入且 `O` tile 安全累积到之后启动。
- `p_ready_2` 释放 `P` 的最后 32 列，匹配上一节的 `96 + 32` value-MMA schedule。

第一个 K/V block 是简单 case。WG2 pre-arrive `p_o_rescale`，因为还没有旧 `O` tile 需要 rescale。

后续 block 需要更小心。WG2 只有在跳过不必要的 rescale 或完成旧 `O` 的 rescaling 之后才 arrive 到 `p_o_rescale`。skip test 故意保守：softmax 计算 log2-scaled delta `(m_old - m_new) * scale_log2`；如果那个值仍在 `-rescale_threshold` 之上，new max 没有移动足够远来 justify rescaling，因此 kernel 保持旧 max 并将 `acc_scale` 设为精确 1.0。只有更大的 max jump 走 `exp2` path 并要求 WG2 rescale `O`。

WG2 然后用 `any_sync` 跨 warpgroup reduce `should_rescale`。如果没有 row 需要 update，它不动 `O`。那个 skip 很重要，因为 rescaling `O` 是整个 accumulator 的完整 TMEM -> RF -> TMEM read-modify-write，当 threshold logic 已经将 `acc_scale` 保持在 1.0 时是纯粹的 wasted work。

注意所有新的 barrier 聚集在一个地方。`s_ready`、`p_o_rescale`、`p_ready_2` 和 softmax/correction pair 都是 softmax 周围的 barrier。它们存在的原因只有一个：score MMA 和 value MMA 不再相邻。register math、TMEM rewrite 和 output rescaling 现在坐在它们之间，每个步骤都需要自己的 handoff。

**与你的 agent 一起尝试**：要求它追踪一个 K/V block 通过 `s_ready`、`p_o_rescale`、`p_ready_2` 和 `o_ready`。对于每个 barrier，询问谁 wait、谁 arrive、什么 tile 变为可读、以及之后什么 storage 可以 reuse。

## Pipelining Structure

barrier 告诉我们在 role 消费 tile 之前什么必须*就绪*。它们没有告诉我们什么实际*并发*运行，那是我们现在转向的问题。两者确实不同：correctness gate 可以在 producer 实际运行之前或之后很久满足。

这里没有单一的 pipeline depth，因为不同的 tile stream 以不同速度移动。kernel 因此为每个 stream 保持单独的 ring：

- Q pipeline depth 2：一个 CTA 处理两个 Q stage。WG0 处理一个 stage，WG1 处理另一个。
- KV pipeline depth 3：K 和 V block 在 inner loop 中流式传输，同时相同的 Q stage 被 reuse。
- TMEM pipeline depth 2：每个 Q stage 有自己的 S/P/O TMEM slot，那些 slot 在匹配的 barrier fire 之后 reuse。

下图从 correctness gate 切换到 timeline view，展示那些 separate ring in flight 后哪些 role 可以在大致相同时间 active。

![Flash Attention 4 Pipeline Structure](../../img/flash_attention_pipeline_v2.png)

将这读作 timeline 而不是 barrier graph。它展示哪些 role 在大致相同时刻 active，而早期的 barrier-flow 图是检查精确 producer-consumer wait 的地方。两者一起，两个图回答了我们在本节开头提出的两个不同问题。

每个 row 匹配 code 的一个 role branch：

- WG3 warp 1 issue TMA load。
- WG3 warp 0 issue score MMA 和 value MMA。
- WG0 和 WG1 为两个 Q stage 运行 softmax。
- WG2 release 或 rescale `O`，然后在最后归一化最终 output。
- WG3 warp 2 issue TMA store。

从左到右跟随图追踪一个 representative pipeline wave。load warp 从 `Q0`、`K[n-1]`、`Q1`、`V[n-1]` 开始，然后继续流式传输更低 index 的 K/V block。MMA warp issue 第一个 score MMA 产生 `S0` 和 `S1`，WG0/WG1 将它们转为 `P0` 和 `P1`。

重要的是 MMA warp *不* 运行所有 score MMA 然后运行所有 value MMA。一旦两个 Q stage primed，它 interleaves 两种：当前 `V` block 的 value MMA，然后下一个 `K` block 的 score MMA，以此类推：

```text
score Q0*K[n-1]
score Q1*K[n-1]
value P0*V[n-1]
score Q0*K[n-2]
value P1*V[n-1]
score Q1*K[n-2]
value P0*V[n-2]
...
```

这种 interleaving 正是 score、softmax、correction 和 value row 在图中重叠而不是整齐连续运行的原因。

WG2 row 标记为 `release / rescale`，两 half 对应我们见过的两个 case。在第一个 K/V block 上还没有旧 `O`，因此 WG2 只参与让 value MMA 继续的 handoff；在后续 block 上它可能在 value MMA 累积之前 rescale 旧 `O`。Normalization 和 TMA store 恰好发生一次，在 attention task 的最后一个 K/V block 之后。

没有单个 GEMM-style pipeline 可以描述 FA4，因为 Q、K/V 和 TMEM slot 都按独立 schedule 前进。TIRx 保持那些 schedule explicit，作为 separate tile buffer、`PipelineState` cursor 和 barrier phase，而不是将 kernel 隐藏在一个 monolithic primitive 后面。代价是更多 moving part，但好处是 complexity 保持 visible 和 inspectable。

## Rescaling 和 Writeback

rescale 是 mandatory 的，不是我们可以丢弃的 optimization。Online softmax 可以随每个新 score tile 提高 per-row maximum，每当它这样做时，从早期 block 累积的 `O` 由*旧* maximum scale。这使每个早期 term 大了 `exp(m_new - m_old)` 的因子。跳过 correction 那些 block 会被 over-weighted，最终 output 简单地错误。fix 是 TMEM → register → TMEM tile operation：

$$O_{\text{old}} \leftarrow O_{\text{old}} \cdot e^{(m_{\text{old}} - m_{\text{new}}) / \sqrt{d}}$$

work 拆分到两个 role。Softmax 计算 per-row scale 并将其放入 SMEM mailbox；WG2 wait `softmax_corr.full`，从 TMEM 读当前 `O`，乘以那个 scale，然后写回 `O`：

```python
RESCALE_TILE = T.meta_var(16)
o_row = T.wg_reg_tile(RESCALE_TILE)
Tx.copy_async(o_row, O_region[i_q, d_start : d_start + RESCALE_TILE])
Tx.mul(o_row, o_row, acc_scale)
Tx.copy_async(O_region[i_q, d_start : d_start + RESCALE_TILE], o_row)
T.ptx.tcgen05.wait.st()
```

值得强调这是整个 `O` accumulator 的完整 TMEM → register → TMEM tile operation，不是 scalar bookkeeping，它携带与其他 stage 相同的 readout card：

> **Tile-primitive readout: Correction（rescale）**
> - Scope：WG2，full warpgroup。
> - Layout：TMEM 中的 `O` → register → TMEM 中的 `O`（`O_region[i_q]`）。
> - Dispatch：`tcgen05.ld` 读取，TMEM store 写入；两者之间 register multiply。
> - Handoff：wait `softmax_corr.full`；arrive `p_o_rescale`（→ value MMA）和 `softmax_corr.empty`（→ softmax）。

end-to-end 追踪 synchronization：

1. Softmax 将 scale value 写入 SMEM。
2. WG2 wait `softmax_corr.full`。
3. WG2 rescale TMEM 中的 `O`。
4. WG2 arrive 到 `p_o_rescale`。
5. WG3 的 value MMA 现在可以消费 `P` 并累积到 rescaled `O` tile。

当 `softmax_corr.empty` 在 WG2 读取之后释放 SMEM slot 时 loop 关闭，这使 softmax 可以在下一次迭代 reuse mailbox。

K/V loop 结束后，WG2 从 correction 切换到 epilogue。它 wait 最终 `row_sum` 和 `o_ready`，从 TMEM 读最终 `O`，乘以 `1 / row_sum`（我们在最开始推迟的归一化），cast 到 fp16，写入 `O_smem`。WG3 的 TMA store warp 然后将 `O_smem` 带回 GMEM。

对于计划扩展这个 kernel 的人来说，有一个 limitation 值得标记。它只计算 forward output，而 training forward pass 通常还会存储 backward pass 需要的 log-sum-exp（LSE）。添加那个需要注意一个 scaling detail：这个 kernel 将 `row_max` 保持为*raw*、未 scaled 的 `QK^T` score 的 maximum，而 `row_sum` 累积 `exp((S - row_max) / sqrt(d))`。因此在形成 natural-log LSE 时必须对 `row_max` 重新应用 `1/\sqrt{d}` factor：

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + \mathrm{row\_max}_i / \sqrt{d}$$

这个 implementation 仅 forward-output，不写 LSE。

## Causal Masking

Causal attention 添加一个 constraint（query 只能 attend 到它自己 position 或之前的 key），kernel 用两种互补的方式遵守它，一个便宜，一个精确。

便宜的方式是完全跳过 work。许多 K/V block 完全在对角线上方，对给定 Q block 没有贡献，因此 `get_n_block_max(...)` 计算 block 可能需要的最后一个 block，loop 根本不加载或计算其余部分。

精确的方式处理 straddle 对角线的 block，其中一些列有效，一些无效。那些 block 仍然运行 score MMA，但 softmax 在 exponentiation 之前 mask out 无效列。对于每个 row 它从 row 的 query position 和 block offset 派生列 limit，保持在那个 limit 及以下的所有列，将之后的每列在 register 中设为 `-inf`，因此那些列对 row max 或 `exp2` numerator 都没有贡献。

实现不用逐 element branch，而是用 `mask_r2p(...)` 应用 limit，将它转为整个 32-wide score chunk 上的 bit mask 并一次 mask 整个 chunk。完全在对角线以下的 block 保持所有列，根本不需要 mask。

从 tile-primitive view 看，causal mode 根本不 rewrite data path。它只修剪 K/V trip count 并在 register-resident softmax 中插入 masking step，在 score MMA 和 `P` writeback 之间。

## GQA Support

Grouped Query Attention 让多个 query head 共享单个 K/V head。这节省 memory bandwidth，但提出了一个 packing question：如何在只保持一个 K/V tile 的同时让多个 query head 通过它？kernel 的回答是一次处理一整组 query head 对抗一个 scheduled `kv_head_idx`：

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

技巧是重新解释 128 Q-tile row。对于 `GQA_RATIO=4` 它们不再代表 128 个 sequence position；它们代表 32 个 sequence position 乘以 4 个 query head，打包在一起使所有四个 head 搭乘同一个 K/V tile。row decoding 是：

```text
seq_pos = row // GQA_RATIO
q_head  = row % GQA_RATIO
```

Q load 用 3D view 表达这个 packing。source 是自然的 `Q[batch, seq, qo_head, dim]` layout，destination 是 score MMA 稍后将读取为 flat `128 x HEAD_DIM` operand 的同一个 SMEM tile。view 调和了两者，且不需要任何 copying：

```python
Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
Tx.copy_async(
    Q_smem_3d[i_q, :, :, :],
    Q[batch_idx,
      m_start : m_start + SEQ_Q_PER_TILE,
      kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
      :],
    **tma_copy_q,
)
```

K 和 V 在 memory 中从不 expand，这正是 GQA 的要点：`kv_head_idx` 的单个 K/V tile 被打包到 Q row 中的所有 `GQA_RATIO` query head reuse。output side 镜像 input，用匹配的 3D view 在 epilogue 后将 packed row 存储回 `O[batch, seq, qo_head, dim]`。

后果是 GQA 完全 lives 在 Q-load 和 O-store boundary。在 compute path 内部 score MMA 仍然看到普通的 `128 x HEAD_DIM` Q tile，其余的 tile-primitive graph 不受影响。

## Tile Scheduling

scheduler 的工作是将每个 CTA 映射到 `(batch, kv_head, m_block)` attention task，正确的 strategy 取决于 masking 是否使那些 task 成本相等：

- Non-causal mode 使用 `FlashAttentionLinearScheduler`。每个 task 做相同量的 work，因此固定 CTA pool 按 `num_ctas` 前进就足以均匀分布它们。
- Causal mode 使用 `FlashAttentionLPTScheduler`，因为 causal masking 使 work 极度不均：靠近开始的 Q block attend 约一个 K/V block，而靠近结束的 attend 所有 block。naive split 将使一些 CTA 在完成时远远落后于其他，因此 longest-processing-time scheduler 将 heavy block front-load 来均匀 finish time，同时保持 nearby batch/head task 在一起以获得 L2 locality。

尽管有所有差异，两个 scheduler 暴露相同的 loop interface：

```python
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # 处理一个 Q block 对抗它的 K/V block range
    scheduler.next_tile()
```

唯一 behavioral difference 在于 `next_tile()` 做什么：在 non-causal mode 中它将 CTA 前进到另一个 task，而在 causal mode 中它在当前 task 之后结束 loop。无论如何这纯粹是 scheduling decision：它选择 CTA 拥有*哪个* attention tile，从不决定那个 tile 如何计算。在 loop 内部相同的 local primitive 不管如何运行：TMA load、score MMA、softmax、value MMA、correction、TMA store。

## Compile 和 Verify

上面 everything 都是摘录，因此为了将它们组装并实际运行 kernel，我们从 `tirx-kernels` import 真实的东西，compile 它，并与 torch reference 检查。完整 kernel，本章 walkthrough 的每个组件组装到一个文件中，是 `tirx-kernels` repository 中的 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py)。两件事与 GEMM verify cell 不同：Flash Attention 有更丰富的 entry point（`get_flash_attention4_kernel`），它需要额外的 `profiler_buf` argument 供内置 profiler 使用。这是整个 chapter 要运行的一个 cell：

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import (
    get_flash_attention4_kernel, PROFILER_BUFFER_SIZE)

B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128   # GQA: 32 个 query head 共享 8 个 KV head
Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")
prof = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
ex.mod(Q, K, V, O, prof)   # ex.mod 直接接受 torch tensor，与其他 chapter 一样
torch.cuda.synchronize()

# torch reference；enable_gqa 让 32 个 query head 共享 8 个 KV head
qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
print(f"FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}, non-causal -> PASS")
```

**Expected output**：`... -> PASS`。kernel 在 fp32 中累积 online softmax，但仍有几个 distinct approximation 使它的 result 与 high-precision reference 分离。有 input 和 operand 的 fp16 storage 和 rounding；`exp2`-based softmax reformulation（`scale_log2 = log2(e)/√d` 重构每个 exponential）；online-softmax reordering 和 per-row rescaling，它以 running scale 求和 block 而不是一次性求和；最后 writeback 时 `O` 的 fp16 cast。这里选择的 `rtol`/`atol`，与 source kernel 自己 test 使用的相同 tolerance，尺寸覆盖所有这些对抗 torch reference，而不只是 fp16 rounding。因此如果你在这里看到真正的 failure，而不只是 borderline near-miss，把它读作指向 softmax path 的路标：丢失的 `s_ready` / `p_o_rescale` / `p_ready_2` wait，或 rescale step 未能应用的 `row_max` / `row_sum` update。那些正是本章 barrier 关注的 handoff。

## 与 GEMM 的差异

下表在改变的轴上比较 FA4 与 GEMM：

| 方面 | GEMM | Flash Attention 4 |
|------|------|-------------------|
| MMA phase | 重复一个 MMA | score MMA 和 value MMA |
| MMA 之间的 work | 除了 pipeline handoff 外无 | online softmax、masking 和 O rescaling |
| Running state | 只有 accumulator | row max、row sum、O accumulator |
| 主要 intermediate | accumulator TMEM tile | S、P 和 O TMEM tile region |
| Warp role | TMA producer、MMA consumer、writeback | TMA load、MMA、softmax、correction、TMA store |
| Barrier | 主要是 load/compute/writeback handoff | 额外的 score/softmax/value/correction handoff |
| Scheduling unit | output matrix tile | attention task：`(batch, kv_head, m_block)` |

这些差异中的每一个都追溯到我们打开 chapter 时的 structural change：第二个 MMA，softmax 夹在两个之间。另一方面，底层 TIRx contract 从未改变：

- tile primitive 说明什么 tile move 或 compute，
- 周围 scope 说明哪些 thread 合作，
- layout 说明 tile lives 在哪里，
- barrier 说明下一个 role 何时可以消费它。

因此 FA4 比 GEMM 难不是因为它依赖不同的 hardware，而是因为 simply 有更多 tile value 和它们之间更多 handoff。

## Exercises

1. 与 GEMM 相比，FA4 中两个 MMA phase 之间出现了什么新的 tile handoff？命名 producer、TMEM tile 和 consumer。
2. 为什么 softmax 将 numerator tile `P` 写回 TMEM 而不是只保留在 register 中供 value MMA 使用？
3. 选择 `p_o_rescale` 或 `p_ready_2`。barrier 精确证明了什么，如果 value MMA 跳过那个 wait 会发生什么？

**与你的 agent 一起尝试**：选择一个未注释的 tile primitive，例如 epilogue `Tx.copy_async`、fp32 -> fp16 `Tx.cast` 或第二个 `gemm_pv` sub-MMA。要求它的 scope / layout / dispatch / handoff card，然后将答案与 source guard、allocation 和 wait 检查。
