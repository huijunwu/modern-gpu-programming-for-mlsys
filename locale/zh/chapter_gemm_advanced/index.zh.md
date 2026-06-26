(chap_gemm_advanced)=
# 高级 GEMM

:::{admonition} Overview
:class: overview

- warp specialization 将 producer、MMA consumer 和 writeback 分离到不同 warpgroup。
- CTA cluster 使两个 CTA 合作在一个 larger MMA tile。
- multi-consumer 使用多个 consumer warpgroup 同时计算 tile 的不同 part。
:::

pipelined GEMM 是 good，但不是全部。warp specialization 和 CTA cluster 是 two 更多 optimization 将 pipelined kernel 推向 SOTA。

## Warp Specialization

在 pipelined kernel，所有 thread 做所有 work：load、compute 和 writeback。那 simple，但它使 thread 在不同 role 间切换。

warp specialization 将不同 role 分离到不同 warp。producer warp 专用于 TMA load。consumer warp 专用于 MMA。writeback warp 专用于 epilogue 和 store。

那有几个好处：
- producer warp 可 keep pipeline full 而不与 consumer 竞争 register；
- consumer warp 可 dedicate 所有 register 到 MMA fragment；
- writeback warp 可 dedicate 所有 register 到 epilogue。

```python
@tirx.script
def gemm_kernel(A, B, C):
    # Warp specialization: separate producer, consumer, writeback
    if warpgroup_id == 0:
        # Producer warp: TMA load
        while True:
            stage = k % num_stage
            T.tma_load(A[stage], Asmem[stage], mbarrier=bar[stage])
    elif warpgroup_id == 1:
        # Consumer warp: MMA
        while True:
            T.mbarrier_wait(bar[prev_stage], phase=phase)
            tcgen05.mma(acc, Asmem[prev_stage], Bsmem[prev_stage], kind=...)
            tcgen05.commit()
    else:
        # Writeback warp: epilogue and store
        while True:
            T.mbarrier_wait(bar[prev_stage], phase=phase)
            acc = T.tmem_load(acc)
            acc = T.cast(acc, "float16")
            T.copy(acc, C[prev_stage])
```

那是 warp specialized kernel 的 basic structure。每个 warpgroup 专用于 single role。

## CTA Cluster

在 `cta_group::1`，一个 CTA 拥有 MMA。在 `cta_group::2`，两个 CTA 合作在一个 MMA tile。

那是 M=256 GEMM 的关键。两个 CTA 各持 128 M row。MMA 将两个 CTA 的 SMEM 组合在一个 tile。

```python
@tirx.script
def gemm_kernel(A, B, C):
    # cta_group::2: two CTA cooperate on one MMA tile
    if cta_id % 2 == 0:
        # Even CTA: rows 0-127
        tcgen05.mma(acc, Asmem[0:128], Bsmem, kind="cta_group::2")
    else:
        # Odd CTA: rows 128-255
        tcgen05.mma(acc, Asmem[128:256], Bsmem, kind="cta_group::2")
```

## Multi-Consumer

multi-consumer 使用多个 consumer warpgroup 同时计算 tile 的不同 part。

在 128×256 accumulator，两个 consumer warpgroup 各持 64 row。那使 compute density 加倍。

```python
@tirx.script
def gemm_kernel(A, B, C):
    # Multi-consumer: two consumer warpgroup compute different parts
    for i in range(num_consumers):
        # Each consumer computes its own part of the tile
        tcgen05.mma(acc[i*64:(i+1)*64], Asmem, Bsmem, kind=...)
```

## 完整 Advanced Kernel

combined，warp specialization、CTA cluster 和 multi-consumer 使 pipelined GEMM 推向 SOTA。

```python
@tirx.script
def gemm_kernel(A, B, C):
    # Warp specialization
    if warpgroup_id == 0:
        # Producer warp
        while True:
            stage = k % num_stage
            T.tma_load(A[stage], Asmem[stage], mbarrier=bar[stage])
    elif warpgroup_id < num_consumer + 1:
        # Consumer warp
        while True:
            T.mbarrier_wait(bar[prev_stage], phase=phase)
            tcgen05.mma(acc[my_range], Asmem[prev_stage], Bsmem[prev_stage], kind=...)
            tcgen05.commit()
    else:
        # Writeback warp
        while True:
            T.mbarrier_wait(bar[prev_stage], phase=phase)
            acc = T.tmem_load(acc)
            acc = T.cast(acc, "float16")
            T.copy(acc, C[prev_stage])
```

那是 advanced GEMM kernel。它 exercise full hardware path 用 warp specialization、CTA cluster 和 multi-consumer。
