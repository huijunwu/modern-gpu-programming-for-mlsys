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

CUDA C++/PTX intrinsic
======================

当没有 tile primitive 覆盖你的需求时，两个 escape hatch 直接到达 hardware：
**call backend intrinsic**（``T.cuda.*`` / ``T.ptx.*`` namespace
来自 ``tvm.backend.cuda``），或 **inline raw CUDA** source。

Calling backend intrinsic
-------------------------

``T.cuda.*`` 和 ``T.ptx.*`` 直接暴露 CUDA backend 的 device intrinsic——
synchronization、mbarrier、reduction 和 PTX data-movement / MMA family：

.. code-block:: python

    T.cuda.cta_sync()                    # block barrier（__syncthreads）
    T.cuda.warp_sync()                   # __syncwarp
    T.cuda.warpgroup_sync(8)             # warpgroup barrier
    T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))   # block-level reduction

    bar = T.alloc_shared((1,), "uint64")
    T.ptx.mbarrier.init(bar.data, 1)     # async completion 的 mbarrier
    T.ptx.mbarrier.try_wait(bar.data, phase)

完整、可运行的 example——通过 ``T.tvm_warp_shuffle_xor`` 的 warp all-reduce：

.. code-block:: python

    @T.prim_func
    def warp_reduce(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32,), "float32", align=16)
        T.device_entry()
        cta_id = T.cta_id([1]); warp_id = T.warp_id([1]); lane_id = T.lane_id([32])
        v = T.alloc_local((1,), "float32"); i = T.alloc_local((1,), "int32")
        v[0] = T.float32(31 - lane_id)
        i[0] = 16
        while i[0] >= 1:
            v[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
            i[0] = i[0] // 2
        A[lane_id] = v[0]

shuffle 直接 lower 到 ``__shfl_xor_sync``：

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

``T.ptx.*`` / ``T.cuda.*`` 下的其他 family：``cp_async``（LDGSTS）、
``cp_async.bulk.tensor``（TMA）、``ldmatrix`` / ``stmatrix``、``tcgen05.*``
（Blackwell MMA）、``atomic_add``、``fence`` … 完整的 ``tvm.backend.cuda`` reference 见 backend API reference。

Synchronization semantics
-------------------------

四个 synchronization mechanism 在 GEMM 和 Flash Attention kernel 中不断出现。
因为它们控制 asynchronous engine 和 parallel thread group，
滥用其中任何一个通常导致 silent corruption 或 deadlock。

**Mbarrier Phase。** Mbarrier 用单个 internal phase bit 跟踪 arrival。
``T.ptx.mbarrier.try_wait(bar, phase)`` intrinsic block 直到 barrier 的
internal phase 与 caller 提供的 ``phase`` argument *不同*。
因此，在 loop iteration 间 reuse barrier 时，caller 必须在每次 wait 后 flip
它的 local phase tracker（``phase ^= 1``）。不这样做使后续 wait 立即返回，
允许 engine 读取 half-written memory。:ref:`chap_gemm_basics`  walkthrough 完整的 phase-tracking table。

**Election。** ``T.ptx.elect_sync()`` elect *warp 内单个 active lane*，
不是 lane 0，也不是 CTA 每 thread 一个。要将 issuer 缩小到恰好一个 thread，
你必须将它与 warp-level guard 配对。``if warp_id == 0:``
后跟 ``if T.ptx.elect_sync():`` 的 pattern 在 :ref:`chap_gemm_basics` 中用于
issue ``Tx.gemm_async`` 和 ``tcgen05.commit``。

**Named Warpgroup Barrier。** ``T.cuda.cta_sync()`` 映射到 ``__syncthreads()`` 且
要求*每个* CTA thread 到达。一旦 warpgroup 专化到不同
code path，在 warpgroup branch 内放置 ``cta_sync()`` 使 kernel deadlock
因为其他 warpgroup 永远达不到它。hardware 提供 16 个 named
barrier（ID 0 到 15）；``T.cuda.warpgroup_sync(10)`` 只同步一个
warpgroup 的 thread。不同 warpgroup 使用不同 ID（例如
``warpgroup_sync(wg_id + 10)``）使它们永远不会在相同 hardware barrier 上冲突。
见 :ref:`chap_gemm_advanced`。

**Fence。** Fence 在 consumer（通常是
asynchronous engine）读取之前 order producer 的 write：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Fence
     - Order
   * - ``T.ptx.fence.proxy_async("shared::cta")``
     - thread-written shared memory 在 async proxy（TMA store / MMA）读取之前
   * - ``T.ptx.fence.mbarrier_init()``
     - mbarrier initialization 在后续 arrival 或 wait 使用 barrier 之前
   * - ``T.ptx.tcgen05.fence.after_thread_sync()``
     - ``tcgen05`` writeback edge 上的 conservative ordering fence（Step 8 和 9 添加它；TMA-to-MMA path 不需要它）

Inlining raw CUDA
-----------------

对于完全没有 intrinsic 的东西，用 ``T.cuda.func_call(name, *args, source_code=..., return_type=...)``
从 source string 注入 ``__device__`` function：

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """

    @T.prim_func
    def k(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = T.cuda.func_call("my_relu", A[tx], source_code=SRC, return_type="float32")

source verbatim emit，call wired in：

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
