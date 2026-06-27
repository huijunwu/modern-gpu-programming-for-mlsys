# 面向 MLSys 的现代 GPU 编程

机器学习系统是现代 AI 工作负载的核心。在这些系统中，性能往往取决于少数几个 GPU kernel 的质量。Attention kernel、LLM prefill 和 decode kernel、低精度 block-scaled GEMM、fused MoE layer 以及其他大型 fused kernel 都直接决定了训练和推理时端到端的运行速度。

然而，要让这些 kernel 运行得更快，光有一堆优化技巧是不够的。现代 GPU 不再只是对旧设计的简单变体。最近的架构引入了更丰富的 memory space、新的访问模式，以及越来越专业化的执行单元。要很好地编程它们，我们需要清晰的硬件 mental model，以及对于如何构建高性能 kernel 的实际理解。本书的目的就是同时培养这两方面的能力。

本书遵循一个简单的递进路线：首先理解 GPU 硬件，然后学习我们将要使用的编程模型，最后逐步构建 state-of-the-art 的 kernel。我们的主要目标是 Blackwell 架构，主要的运行示例是高速矩阵乘法（GEMM）和 FlashAttention。在此过程中，我们还将研究 GPU 优化背后的核心要素：data layout、asynchronous data movement 和 asynchronous coordination。

本书的材料脱胎于卡内基梅隆大学的 [Machine Learning Systems](https://mlsyscourse.org/) 课程系列。为了让这些概念更容易学习和运行，本书使用 **TIRx** Python DSL 来逐步构建真实的 GPU kernel 示例。TIRx 贴近硬件，使我们可以在通过可运行代码学习的同时，对底层控制进行推理。

## 本书组织

- **Part I, Understanding the GPU.** 本部分介绍 GPU 的整体组织、编写快速 kernel 的一般方法，以及 data layout、asynchronous memory operation 和 coordination 等关键概念。它构建了本书其余部分所依赖的硬件直觉。
- **Part II, TIRx Overview.** 本部分介绍 TIRx 的关键元素，它们构成了全书代码示例的基础。
- **Part III, GEMM: Tiled to SOTA.** 一份完整的 tiled GEMM 优化指南，通过 TMA pipelining、persistent scheduling、warp specialization 和 2-CTA cluster 逐步构建。
- **Part IV, Flash Attention 4.** 一个基于 Part III 技术的完整 attention kernel：两个 MMA 之间夹着 softmax，online-softmax rescaling、causal masking 和 GQA。
- **Reference.** TIRx 语言参考和 compiler internals。

```{toctree}
:caption: Part I, Understanding the GPU
:maxdepth: 1

chapter_background/index
chapter_performance/index
chapter_data_layout/index
chapter_layout_generations/index
chapter_tma/index
chapter_tensor_cores/index
chapter_tmem/index
chapter_async_barriers/index
chapter_clc/index
```

```{toctree}
:caption: Part II, TIRx Overview
:maxdepth: 1

chapter_intro_tirx/index
chapter_tirx_layout_api/index
```

```{toctree}
:caption: Part III, GEMM: Tiled to SOTA
:maxdepth: 2

chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
```

```{toctree}
:caption: Part IV, Flash Attention 4
:maxdepth: 2

chapter_flash_attention/index
```

```{toctree}
:caption: Reference
:maxdepth: 1

appendix/index
appendix/debugging_warp_specialized
tirx_guide/arch/index
tirx_guide/language_reference/index
```
