(chap_gemm_async)=
# Async GEMM: TMA 和 Software Pipelining

:::{admonition} Overview
:class: overview

- TMA 替代 thread copy 用于 GMEM ↔ SMEM tile movement。它释放 warp 用于 compute。
- software pipelining 使用多 SMEM stage 使 next K tile 的 data movement 可与 current K tile 的 Tensor Core compute overlap。
- barrier 和 phase 使 stage reuse safe。
:::

Step 1 的 GEMM（{ref}`chap_gemm_basics`）是 correct，但它 leave 大部分 hardware idle。thread 合作 copy A 和 B tile 从 GMEM 到 SMEM、compute MMA、epilogue 将 result store 到 GMEM。每 step 顺序：copy A、copy B、compute、store。

这个 chapter 关闭那个 gap 通过引入 TMA 和 software pipelining。

## TMA 替代 Thread Copy

Step 1 使用 thread copy 将 A 和 B 从 GMEM 移到 SMEM。每个 thread 计算它应 load 的 element、load 它、store 它在 SMEM。那 correct 但消耗 thread instruction 和 register。

TMA 将那个 work 移到 hardware copy engine。一个 thread 发出 tile copy。copy engine 然后在 GMEM 和 SMEM 间 asynchronous move 一个 rectangular tile。当 engine move byte 时，CTA 其余部分可 continue other work。

```python
# Step 1: thread copy
T.copy(A, Asmem)

# This chapter: TMA
T.tma_load(A, Asmem, mbarrier=bar, byte_count=byte_count)
```

TMA load 是 asynchronous。issuing thread 不 wait 完成。相反，TMA engine 在 byte arrive 时更新 barrier。barrier phase flip 当 arrival count 和 byte count 都满足。

那是 key：TMA 使 kernel 在 Tensor Core compute 时 begin loading next K tile。

## Software Pipelining

pipelined GEMM 使用多 SMEM stage。每个 stage 持有一个 A 和 B tile。stage 在 loop advance 时 rotate。

```text
Stage 0: TMA load K+1, compute K-1, store K-2
Stage 1: TMA load K+2, compute K,   store K-1
Stage 2: TMA load K+3, compute K+1, store K
```

每 stage 有独立 SMEM buffer 和 barrier。MMA path 在 barrier wait 前读 stage。TMA path 在 barrier signal 写 stage。

```python
for k in range(num_k):
    stage = k % num_stage
    
    # TMA load next tile
    T.tma_load(A[stage], Asmem[stage], mbarrier=bar[stage], byte_count=byte_count)
    
    # Wait for previous tile to be ready
    T.mbarrier_wait(bar[prev_stage], phase=phase[stage])
    
    # Compute
    tcgen05.mma(acc, Asmem[prev_stage], Bsmem[prev_stage], kind=...)
    
    # Advance phase
    phase[stage] = 1 - phase[stage]
```

那使 data movement 与 compute overlap。Tensor Core 计算 current tile 时，TMA engine load next tile。barrier 确保 MMA 仅在 tile ready 时读。

## Barrier 和 Phase

每 stage 有 barrier 和 phase。barrier 跟踪 TMA load completion。phase 使 barrier reuse 在多 K iteration 中 safe。

在每 loop iteration，kernel：
1. 在 barrier wait 当前 phase；
2. 在 phase 完成时 toggle phase。

那使 barrier 在 reuse 前从新 iteration 隔离。

## Full Pipelined Kernel

pipelined GEMM kernel 结合 TMA、software pipelining 和 barrier。

```python
@tirx.script
def gemm_kernel(A, B, C):
    BLK_M, BLK_N, BLK_K, K = ...
    num_stage = 3
    
    pool = T.SMEMPool()
    Asmem = pool.alloc((num_stage, BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((num_stage, BLK_N, BLK_K), b_type, layout=B_layout)
    bar = pool.alloc((num_stage,), "uint64", align=8)
    pool.commit()
    
    acc = tcgen05.alloc(tmem_addr, shape=(BLK_M, BLK_N), dtype="float32")
    
    for k in range(num_k):
        stage = k % num_stage
        
        # TMA load next tile
        T.tma_load(A[stage], Asmem[stage], mbarrier=bar[stage], byte_count=byte_count)
        
        # Wait for previous tile
        T.mbarrier_wait(bar[prev_stage], phase=phase[stage])
        
        # Compute
        tcgen05.mma(acc, Asmem[prev_stage], Bsmem[prev_stage], kind=...)
        tcgen05.commit()
        
        # Advance phase
        phase[stage] = 1 - phase[stage]
    
    # Epilogue
    acc = T.tmem_load(acc)
    acc = T.cast(acc, "float16")
    T.copy(acc, C)
    T.tmem_free(acc)
```

那是 pipelined GEMM。它 exercise full hardware path 但用 TMA 和 software pipelining 使 data movement 与 compute overlap。

那是 next chapter 的基础，它添加 warp specialization 和 CTA cluster。
