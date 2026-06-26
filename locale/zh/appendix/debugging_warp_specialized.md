# Debugging Warp Specialized Kernels

warp specialization 是 GEMM optimization 的关键 technique（{ref}`chap_gemm_advanced`）。它将 producer 和 consumer 分离到不同 warp，使它们在同一 CTA 内 concurrent 运行。

然而，warp specialized kernel 比同步 kernel 更难 debug。当 producer 和 consumer 在相同 CTA 中并行时，bug 可能来自：

```text
- 错误的 barrier synchronization
- 错误的 resource 分配
- producer 和 consumer 之间的 timing issue
```

## Barrier Synchronization

warp specialized kernel 通常使用 barrier 在 producer 和 consumer 之间 synchronize。barrier 必须正确配置，否则 consumer 可能读取 incomplete data 或 producer 可能 overwrite consumer 正在使用的 data。

确保：

```text
- producer 在写 buffer 前 wait 适当 barrier
- consumer 在读 buffer 前 wait 适当 barrier
- phase tracking 正确 toggle
```

## Resource Allocation

warp specialized kernel 必须将 SMEM 和 register 分配给 producer 和 consumer。resource allocation 必须确保 producer 和 consumer 不 conflict。

确保：

```text
- producer 和 consumer 的 SMEM buffer 不 overlap
- register 分配不 conflict
- TMEM allocation 正确
```

## Timing Issue

warp specialized kernel 必须确保 producer 和 consumer 在正确 time 运行。timing issue 可能导致 consumer 在 producer 完成前开始或 producer 在 consumer 开始前完成。

确保：

```text
- producer 和 consumer 的 scheduling 正确
- barrier 在正确 point 设置
- pipeline depth 适当
```

## 调试技巧

一些 helpful debugging 技巧：

```text
- 使用 print 或 trap 指令检查 barrier state
- 使用 nsight-compute 检查 resource usage
- 逐步增加 pipeline depth 并检查 correctness
```
