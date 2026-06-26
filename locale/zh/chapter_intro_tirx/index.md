(chap_tirx_primer)=
# TIRx Overview

:::{admonition} Overview
:class: overview

- TIRx（Tensor IR neXt）是一个 Python DSL 用于编写 GPU kernel。它将 computation 描述为 scope、layout 和 dispatch 的组合。
- scope 定义哪些 thread 或 CTA 参与。layout 描述 data 如何排列。dispatch 决定 computation 如何 execute。
- TIRx 用 Python function 表达 kernel 和 decorator 声明 scope 和 scheduling。
- TIRx compiler 将 DSL code 翻译为 PTX 并 assemble kernel。
:::

在 Part I 中，我们介绍了 GPU hardware 的 building block：thread hierarchy、memory space、compute engine 和 asynchronous coordination。我们学习了那些 kernel 需要 organize work 以 keep those engine busy 的方式。

现在我们需要一种方式表达那种 organization。那正是 TIRx。

TIRx（Tensor IR neXt）是一个 Python DSL 用于编写 GPU kernel。它不试图抽象 hardware。相反，它提供直接表达 scope、layout 和 dispatch 的语言——那些反复出现的 design element。

## Scope, Layout, Dispatch

在 {ref}`chap_background` 中，我们引入三个 recurring design element。它们重新出现。

scope 回答"谁"。它定义哪些 thread、warp、warp group 或 CTA 参与 operation。scope 可以是 single thread、single warp、warpgroup、single CTA 或 cluster。

layout 回答"哪里"。它描述 data 如何排列。layout 决定 data 在 memory 或 register 中的物理 arrangement。它影响 coalescing、bank conflict 和 engine 能否 read tile。

dispatch 回答"如何"。它决定 operation 如何 execute。dispatch 将 computation 分配给 thread 或 CTA。

这三个 element 不是 independent。它们必须 agree。scope 决定参与 computation 的 thread。layout 决定 data 如何到达那些 thread。dispatch 决定 thread 以什么 order 和 manner execute work。

## A Simple GEMM in TIRx

让我们看一个 TIRx 中的 simple GEMM example。

```python
import tvm
from tvm import tirx
from tvm.ir.module import IRModule

@tirx.script
def gemm(A, B, C):
    # scope: one CTA
    # layout: A and B are in global memory, C is in global memory
    # dispatch: one MMA per warpgroup
    M, K = A.shape
    N, _ = B.shape
    # ... kernel body
```

`@tirx.script` decorator 标记 function 作为 TIRx script。TIRx compiler 解析那个 function 并 generate kernel。

function parameter 是 tensor。它们描述 kernel 的 input 和 output。

## TIRx Compilation Model

TIRx 将 DSL code 翻译为 PTX。compilation process 包括几个步骤：

1. parsing：TIRx parser 解析 Python function 并 extract scope、layout 和 dispatch specification。
2. lowering：TIRx lower 将 DSL construct 翻译为 PTX。
3. optimization：compiler 应用 optimization 以 improve performance。
4. code generation：compiler 生成 final PTX。

final PTX 可以 assemble 和 run 在 supported GPU。

## TIRx 与 PyTorch 的关系

TIRx 是 kernel writing language。它不是 deep learning framework。它不 attempt 抽象 hardware 或 hide implementation detail。

然而，TIRx kernel 可以与 PyTorch tensor 一起 use。TIRx 可以 read 和 write PyTorch tensor。那使 TIRx kernel 可以 integrate 与 PyTorch workflow。

## 什么使 TIRx 有用

TIRx 的主要 value 是它提供直接表达 scope、layout 和 dispatch 的语言。那些 design element 是 write high-performance GPU kernel 的关键。

没有 TIRx，kernel programmer 必须 work 在 PTX 或 CUDA C++ 水平。那要求手动处理 scope、layout 和 dispatch。TIRx 使那些 concept 成为 programming model 的第一 class construct。

## 下一章

下一章介绍 TIRx layout API。layout API 是 express data layout 的语言。它是 {ref}`chap_data_layout` 中 layout notation 的 TIRx implementation。
