(chap_flash_attention)=
# Flash Attention 4

:::{admonition} Overview
:class: overview

- Flash Attention 将 attention 计算保持在片上，避免 score matrix 写 HBM。
- 它使用与 GEMM 相同的 technique：TMA pipelining、warp specialization、software pipelining。
- Flash Attention 4 添加 Blackwell 特性：tcgen05、TMEM 和 cluster。
:::

attention 是现代 transformer 的核心。standard attention 将 score matrix 写入 HBM 并后来读回，消耗大量 memory bandwidth。Flash Attention 将 attention 计算保持在片上，避免那个 HBM round trip。

## Attention 问题

standard attention 是：

```text
score = Q @ K.T
weight = softmax(score)
output = weight @ V
```

score matrix shape 是 (S, S)。对于 S=8192，那是 64G。那需要写入 HBM 并后来读回，消耗大量 bandwidth。

Flash Attention 避免那通过将 attention 计算 tiled 在片上。它 compute Q、K、V tile 在 SMEM 上，compute score tile 在 TMEM 上，然后 compute output tile 并 store 到 GMEM。

## Online Softmax

standard softmax 需要知道所有 score 来 normalize。Flash Attention 使用 online softmax，它在 tiled 方式 compute softmax。

online softmax 保持 running sum 和 running max。当新 tile arrive，它 rescale previous result 并 update running sum。

```python
for i in range(num_k):
    # Compute score tile
    score = Q @ K[i].T
    
    # Online softmax
    score_max = max(score)
    score_exp = exp(score - score_max)
    score_sum = sum(score_exp)
    
    # Rescale previous output
    scale = exp(prev_score_max - score_max)
    output = output * scale + score_exp @ V[i]
    
    # Update running sum
    running_sum += score_sum * exp(prev_score_max - score_max)
    prev_score_max = score_max
```

## Causal Masking

causal masking 使 score 只 attend 之前 position。那在 autoregressive model 中 important。

```python
for i in range(num_k):
    # Compute score tile
    score = Q @ K[i].T
    
    # Apply causal mask
    for m in range(M):
        for n in range(N):
            if k_offset + n > q_offset + m:
                score[m, n] = -inf
```

## GQA

GQA（grouped-query attention）减少 K 和 V head 数量。那减少 compute 和 memory。

```python
# GQA: K and V are grouped
num_kv_head = num_q_head // num_group
```

## 完整 Flash Attention 4 Kernel

combined，Flash Attention 4 kernel 使用：
- TMA pipelining 将 Q、K、V tile load 到 SMEM
- tcgen05.mma 计算 score 和 output
- TMEM 存储 score accumulator
- online softmax rescale
- causal masking
- GQA

```python
@tirx.script
def flash_attention_kernel(Q, K, V, O):
    # Warp specialization
    if warpgroup_id == 0:
        # Producer warp: TMA load Q, K, V
        while True:
            stage = k % num_stage
            T.tma_load(Q[stage], Qsmem[stage], mbarrier=bar[stage])
            T.tma_load(K[stage], Ksmem[stage], mbarrier=bar[stage])
            T.tma_load(V[stage], Vsmem[stage], mbarrier=bar[stage])
    elif warpgroup_id < num_consumer + 1:
        # Consumer warp: MMA and softmax
        while True:
            T.mbarrier_wait(bar[prev_stage], phase=phase)
            
            # Compute score
            score = tcgen05.mma(Qsmem[prev_stage], Ksmem[prev_stage], kind=...)
            
            # Apply causal mask
            score = causal_mask(score, k_offset, prev_stage)
            
            # Online softmax
            score_exp, running_sum, running_max = softmax_update(score, running_sum, running_max)
            
            # Compute output
            output = tcgen05.mma(score_exp, Vsmem[prev_stage], kind=...)
            
            # Rescale previous output
            scale = exp(prev_running_max - running_max)
            output = output + prev_output * scale
            
            tcgen05.commit()
    else:
        # Writeback warp: store output
        while True:
            T.mbarrier_wait(bar[prev_stage], phase=phase)
            output = T.tmem_load(output)
            T.copy(output, O[prev_stage])
```

那是 Flash Attention 4 kernel。它 exercise full hardware path 用 TMA pipelining、warp specialization 和 tcgen05。
