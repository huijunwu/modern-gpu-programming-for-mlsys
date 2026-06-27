..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

TIRx lowering pipeline
======================

``tvm.compile(mod, target, tir_pipeline="tirx")`` 运行一个 authored TIRx module
通过 **tirx pipeline** — 一个有序的 TIR passes 序列，将
你写的高层构建（tile primitive、``TileLayout``-typed buffer、
execution-scope id）转换为 split **host** + **device** function，然后 CUDA
backend 将其渲染为 source。该 pipeline 定义在
``python/tvm/tirx/compilation_pipeline.py``（``tirx_pipeline``）；本页 walkthrough
passes in order。

位置
----

``tvm.compile`` 首先 bind target，运行 **tirx pipeline**（module-level
pass below），然后分别对 host 和
device function 应用 **finalization** pass，最后将每个 device function 交给 CUDA code
generator：

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                        │
                                        └──────────▶  device func ──device finalize──▶  CUDA

passes
------

``tirx_pipeline`` module pass 按顺序应用这个 exact 序列（一些由
``PassContext`` config gate）：

.. list-table::
   :header-rows: 1
   :widths: 6 32 62

   * - #
     - Pass
     - 做什么
   * - 1
     - ``LowerTIRx``
     - 核心 lowering — 见下文 `Inside LowerTIRx`_
   * - 2
     - ``UnifyThreadBinding``
     - 合并等效 thread-axis binding，使每个 ``threadIdx`` / ``blockIdx``
       axis 只 declare 一次
   * - 3
     - ``StmtSimplify``
     - statement-level arithmetic simplification（arith analyzer）
   * - 4
     - ``LowerTIRxOpaque``
     - 将剩余 opaque TIRx construct lower 为 plain TIR
   * - 5
     - ``FlattenBuffer``
     - 将 multi-dimensional ``BufferLoad`` / ``BufferStore`` flatten 为 1-D
   * - 6
     - ``BF16ComputeLegalize``
     - 重写 ``bfloat16`` compute 为合法（f32-up-cast）form
   * - 7
     - ``NarrowDataType(32)``
     - 在 provably safe 时将 index/loop ``PrimExpr`` dtype narrow 到 32-bit
   * - 8
     - ``VectorizeLoop``
     - 将 ``T.vectorized`` loop 转为 vector op（如果
       ``tir.disable_vectorize`` 则 skip）
   * - 9
     - ``UnrollLoop``
     - 展开标记为 ``T.unroll`` 的 loop（和 small constant loop）
   * - 10
     - ``StmtSimplify``
     - 再次 simplify，此时 vectorize/unroll 已 expose constant
   * - 11
     - ``CommonSubexprElim``
     - 将重复 subexpression hoist 到 temporary（如果
       ``tir.disable_cse_tir`` 则 skip）
   * - 12
     - ``FP8ComputeLegalize``
     - 重写 ``float8`` compute 为合法 form
   * - 13
     - ``VerifyMemory``
     - 检查无 host-side code 直接 dereference device memory（safety gate）
   * - 14
     - ``AnnotateEntryFunc``
     - 标记单个 PrimFunc 为 module entry point
   * - 15
     - ``SplitHostDevice``
     - 将每个 kernel split 为 **host** function 和 **device** function，在
       ``launch_thread`` boundary
   * - 16
     - ``MakePackedAPI``
     - 重写 host function 为 packed-func ABI（launcher TVM calls）
   * - 17
     - ``FP8StorageLegalize``
     - legalize ``float8`` storage（pack 为 supported container type）
   * - 18
     - ``BF16StorageLegalize``
     - legalize ``bfloat16`` storage

**Finalization** 然后 per function kind 运行：

- **host**：``LowerTVMBuiltin``（lower ``tvm_*`` builtin），``LowerIntrin``
  （target-specific intrinsic）
- **device**：``LowerWarpMemory``（warp-scoped buffer → shuffle），``StmtSimplify``，
  ``LowerIntrin``

Inside LowerTIRx
----------------

``LowerTIRx`` 本身是一个 small 序列（``src/tirx/transform/lower_tirx.cc``）：

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** 用 selected backend
  dispatch 发出的 body 替换每个 ``TilePrimitiveCall``（``copy``、
  ``gemm``、``reduction``、…）— 其 variant-selection 和 codegen。
- **``LowerTIRxCleanup``** 运行 ``LayoutApplier``：它 resolve 每个
  ``TileLayout``-typed buffer access 为 concrete physical address arithmetic
  （``addr = data + elem_offset + layout.apply(coord)``），flatten buffer，并
  lower execution-scope id（``T.cta_id`` / ``T.thread_id`` / … →
  ``blockIdx`` / ``threadIdx`` via ``launch_thread``）。

因此在 ``LowerTIRx`` 后 module 是 plain TIR：无 tile primitive，无
``TileLayout`` indirection，scope id resolve 到 thread axis。

Worked example
--------------

看一个 one-line scale kernel：

.. code-block:: python

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = A[tx] * T.float32(2.0)

**After ``LowerTIRx``** scope id 是 real thread axis 且 layout 已 apply
（``A_1`` / ``B_1`` 是 flattened 1-D view）：

.. code-block:: python

    with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        bx: T.let = blockIdx_x
        tx: T.let = threadIdx_x
        B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)

**After ``SplitHostDevice`` + ``MakePackedAPI``** 一个 function 变成两个 —
一个 host launcher 和一个 device kernel：

.. code-block:: python

    @I.ir_module
    class Module:
        def main(...):          # host: packed-API launcher（compute grid/block，launch）
            ...
        def scale_kernel(...):  # device: __global__ body，在 GPU 上运行

CUDA backend 然后将 ``scale_kernel`` 渲染为 ``__global__`` function
（``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f``）。

Reproduce it yourself
---------------------

你可以手工运行 pipeline 的任意 prefix 来 inspect 一个 stage — 这就是
这些 docs 中 IR snippet 的产生方式：

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda")
    mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
    mod = TT.LowerTIRx()(mod)         # tile primitive dispatch，layout apply
    print(mod.script())               # inspect lowered TIRx IR

或 compile 整个 module 并读取生成的 CUDA：

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    print(exe.mod.imports[0].inspect_source())
