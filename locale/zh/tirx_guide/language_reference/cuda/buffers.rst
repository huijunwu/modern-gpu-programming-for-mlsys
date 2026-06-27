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

Buffer 和 memory
================

Parameter buffer 用 ``T.match_buffer`` bind；scratch buffer 在 body 中用两个
declaration API 之一创建（下面）。用 ``A[i, j]`` 索引 buffer，用
``A[m0:m0+BM, 0:BK]`` 切片它（``BufferRegion``），用 ``A.ptr_to([i, j])`` 或
raw data pointer ``A.data`` 获取 pointer。

Declaring buffer
----------------

两个 fundamental API 创建 buffer：

- ``T.alloc_buffer(shape, dtype, scope=..., ...)`` — **分配新 storage**
  （emit ``AllocBuffer`` node）并返回 ``Buffer``。``T.alloc_shared`` /
  ``T.alloc_local`` 只是 ``alloc_buffer`` 带 ``scope="shared"`` /
  ``scope="local"``。
- ``T.decl_buffer(shape, dtype, data=..., ...)`` — **声明 view** 在现有
  pointer ``data`` 上（无 allocation）；用它来 alias 或 reinterpret
  storage——pool 的 sub-region 或 tensor-memory address。当 ``data=None``
  时它分配，像 ``alloc_buffer``。

buffer 的 ``data`` pointer 是 immutable ``Var``（``alloc_buffer`` 定义它；
``decl_buffer`` 接受一个）。用 pointer *expression* 支撑 buffer，首先 bind 它——
见 :doc:`data_types`。

两者共享一个 descriptor；最重要的 parameter：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Parameter
     - 含义
   * - ``dtype``
     - element type——``"float32"``, ``"float16"``, ``"float4_e2m1fn"``, …
   * - ``shape``
     - logical shape（extent 的 tuple）
   * - ``layout``
     - physical mapping（:ref:`TileLayout <chap_tirx_layout_api>`）；``"default"`` = dense
       row-major
   * - ``elem_offset`` / ``allocated_addr``
     - ``elem_offset``（或 ``byte_offset``）将 *view* 放置在 ``data`` 中的 offset；
       ``allocated_addr`` 携带 pre-assigned address（tensor memory）
   * - ``align``
     - data pointer 的 alignment，以 byte 为单位

``scope`` argument 选择 memory space：

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Scope
     - Shorthand
     - Memory
   * - ``"global"``
     - （default）
     - device global memory
   * - ``"shared"``
     - ``T.alloc_shared``
     - static shared memory（``__shared__``）
   * - ``"shared.dyn"``
     - （pool）
     - dynamic shared memory（pooled——见下面）
   * - ``"local"``
     - ``T.alloc_local``
     - per-thread register
   * - ``"tmem"``
     - （TMEM pool）
     - Blackwell tensor memory（见下面）

.. code-block:: python

    A = T.match_buffer(A_ptr, (M, K), "float16", align=16)   # parameter buffer
    As = T.alloc_shared((BM, BK), "float16")                 # 新 shared tile
    acc = T.alloc_local((4,), "float32")                     # register accumulator
    view = T.decl_buffer((BM, BK), "float16", data=As.data)  # As 上的 view

**基于 ptr 的 buffer 只是 pointer 上的 metadata。** 对于任何非 tmem buffer，
declaration 是 pointer 加 layout，indexing 解析为 address::

    addr(buffer[coord]) = buffer.data + elem_offset + layout.apply(coord, shape=shape)["m"]

（``layout.apply`` 返回 per-axis mapping；它的 ``"m"`` component 是
element offset。）因此*相同*的 logical access 编译为不同的 address
arithmetic，纯粹取决于 buffer 的 metadata。在 4×8 region 上写
``B[i, j] = A[i, j] + 1``，``B`` 用四种方式声明：

.. code-block:: python

    from tvm.tirx.layout import TileLayout, S

    B = T.match_buffer(p, (4, 8), "float32")                                       # row-major
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(1, 4)]))  # column-major
    B = T.match_buffer(p, (4, 8), "float32", elem_offset=64)                       # shifted view
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(16, 1)])) # row stride 16

每个使 ``B[i, j]`` lower 为 generated CUDA 中不同的 index（
``A[i, j]`` load 保持 ``i*8 + j``——只有 ``B`` 的 metadata 改变）：

.. code-block:: c++

    B_ptr[((i * 8) + j)]        = ...;   // row-major：        i*8 + j
    B_ptr[((j * 4) + i)]        = ...;   // column-major：     j*4 + i
    B_ptr[(((i * 8) + j) + 64)] = ...;   // elem_offset=64：   i*8 + j + 64
    B_ptr[((i * 16) + j)]       = ...;   // row stride 16：    i*16 + j

Shared memory
-------------

Shared memory 有两种 flavor——**static**（compile time fixed）和
**dynamic**（launch 时 sized）——加上管理 dynamic case 的 pool helper。

Static
~~~~~~

最简单的 shared buffer 是**static**的——``T.alloc_shared``（即
``scope="shared"``），compile time sized。将 data staging 到其中，``cta_sync`` 使
整个 block 看到 write，然后读回来：

.. code-block:: python

    @T.prim_func
    def smem_demo(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128,), "float32")
        B = T.match_buffer(B_ptr, (128,), "float32")
        T.device_entry()
        bx = T.cta_id([1])
        tx = T.thread_id([128])
        sm = T.alloc_shared((128,), "float32")   # static shared memory
        sm[tx] = A[tx]
        T.cuda.cta_sync()
        B[tx] = sm[tx] * T.float32(2.0)

它 lower 为普通 ``__shared__`` array（generated CUDA，boilerplate 省略）：

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(128)
    smem_demo_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tx = ((int)threadIdx.x);
      __shared__ alignas(64) float sm_ptr[128];      // T.alloc_shared
      sm_ptr[tx] = A_ptr[tx];
      __syncthreads();                               // T.cuda.cta_sync()
      B_ptr[tx] = sm_ptr[tx] * 2.0f;
    }

Dynamic
~~~~~~~

**Dynamic** shared memory（``scope="shared.dyn"``）per launch sized（
``sharedMemBytes`` launch parameter），不是在 compile time。kernel 可能只有**一个**
dynamic-shared allocation——*arena*。因此你分配一次，``decl`` 每个 buffer 作为
其中的 view：``T.decl_buffer`` 带 ``data=`` arena pointer 和 ``elem_offset``：

.. code-block:: python

    arena = T.alloc_buffer((128,), "float32", scope="shared.dyn")   # 一个 arena
    As = T.decl_buffer((64,), "float32", data=arena.data, scope="shared.dyn")                 # offset 0
    Bs = T.decl_buffer((64,), "float32", data=arena.data, elem_offset=64, scope="shared.dyn") # offset 64
    As[tx] = A[tx]
    Bs[tx] = B[tx]
    T.cuda.cta_sync()
    C[tx] = As[tx] + Bs[tx]

两个 view 共享单个 ``extern __shared__`` arena（generated CUDA，
boilerplate 省略；arena 命名为 ``smem`` 以清晰）：

.. code-block:: c++

    extern __shared__ __align__(64) float smem[];   // 一个 dynamic-shared arena
    smem[tx]      = A_ptr[tx];                       // As——offset 0 的 view
    smem[tx + 64] = B_ptr[tx];                       // Bs——offset 64 的 view
    __syncthreads();
    C_ptr[tx] = smem[tx] + smem[tx + 64];

（两个 separate ``alloc_buffer(scope="shared.dyn")`` call 是 error——*只允许一个
dynamic shared memory allocation*。）因此 static shared memory 在 compile time sized
（``__shared__ T x[N];``）；dynamic shared memory 是这个 launch-sized arena，
view 在它内部的 offset 处 decl'd。

.. note::

   **TVM 如何 annotate dynamic-shared size。** arena 的 size 在 compile time 已知
   （这里 ``128`` float = ``512`` byte）。lowering 期间 TVM 将
   ``"tirx.use_dyn_shared_memory"`` tag append 到 device kernel 的
   ``tirx.kernel_launch_params``，host launcher 计算总 byte 并作为最后 launch argument 传递：

   .. code-block:: python

       # device kernel attribute：
       "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]

       # host-side launch call（..., gridDim.x, blockDim.x, dyn_shared_bytes）：
       T.call_packed("dyn_kernel", A.data, B.data, C.data, 1, 64, 512)

   在 run time 那个 ``512`` 成为 ``cuLaunchKernelEx`` call 中的
   ``config.sharedMemBytes``。你从不手动设置它——它从
   ``shared.dyn`` allocation 的 size 派生。

Pool sugar
~~~~~~~~~~

``T.SMEMPool`` 自动化那个 arena bookkeeping——它 bump-allocate offset 使
你不必手动 ``decl`` view。除了 ``alloc`` / ``commit``，它提供
per-buffer ``align=``、构建 MMA-compatible swizzle layout 的
``alloc_mma`` helper、以及 ``move_base_to`` 来 rewind cursor 和 reuse space：

.. code-block:: python

    pool = T.SMEMPool()                          # shared.dyn 上的 bump allocator
    As = pool.alloc((BM, BK), "float16", align=128)   # carve tile
    Bs = pool.alloc((BK, BN), "float16", align=128)
    Cs = pool.alloc_mma((BM, BN), "float16")     # MMA-compatible，swizzle inferred
    pool.commit()                                 # finalize pool size
    # pool.move_base_to(offset) rewind cursor 来 reuse space

TMEM pool（下面的 `Tensor memory`_）层叠在 ``SMEMPool`` 之上。

Register
--------

Per-thread scratch lives 在 register 中。用 ``T.alloc_local(shape, dtype)`` 分配它
（即 ``scope="local"``）：它对每个 thread private 且 lower 为
保持在 register 中的 local array。

.. code-block:: python

    r = T.alloc_local((4,), "float32")   # per-thread register array
    for k in T.unroll(4):
        r[k] = A[tx, k]
    # ... 在 r[0..3] 上 compute ...

.. code-block:: c++

    alignas(64) float r_ptr[4];          # per-thread，register-resident
    r_ptr[0] = A_ptr[tx * 4 + 0];
    r_ptr[1] = A_ptr[tx * 4 + 1];
    // ...

.. note::

   ``alignas(64)`` 是*default* buffer alignment——buffer 的
   ``data_alignment`` default 为 ``runtime::kAllocAlignment``（64 byte），
   CUDA codegen 将它 stamp 到每个 allocation 上，包括 per-thread ``local``
   array（在那里它无意义）。对于这些 register-resident array 它**没有
   performance impact**：带 statically-resolvable index 的 thread-local array 被
   nvcc/ptxas 提升到 register（scalar replacement of aggregates, SROA），因此
   它从不 lives 在 addressable local memory 中且 alignment 是 no-op。（溢出到
   local memory 的 dynamically-indexed array 将实际 pickup 那个
   over-alignment，但那是 unusual case。）register local 的
   over-alignment 是我们计划修复的 known rough edge（对 ``local`` scope 使用
   dtype 的 natural alignment）。

Scalar
~~~~~~

scalar 只是**一个 element** 的 register array——严格来说，你不需要
separate concept。你可以分配 size-1 ``local`` buffer 并索引 ``[0]``：

.. code-block:: python

    phase = T.alloc_local((1,), "int32")   # 1-element register array
    phase[0] = 0
    while phase[0] < 4:
        acc = acc + A[tx, phase[0]]
        phase[0] += 1

但在到处写 ``phase[0]`` 很笨拙，因此**scalar** 恰好是这个的 sugar——
一个你通过**名字**读写的一 element register buffer：

.. code-block:: python

    phase: T.int32 = 0                 # mutable scalar（上面那个的 sugar）
    while phase < 4:
        acc = acc + A[tx, phase]
        phase += 1

    s = T.local_scalar("int32")        # explicit form；by name assign（s = ...，不是 s[0]）
    acc: T.float32 = 0.0               # type-annotated assignment 也创建一个

两者不仅仅是相似——它们 parse 为**结构上相同的 TIRx**。
sugar 完全在 parser 中解决：``phase: T.int32`` *就是* 那个一 element
``local`` buffer，``phase`` / ``phase += 1`` *就是* ``phase[0]`` /
``phase[0] += 1``。``tvm.ir.assert_structural_equal`` 在两个 kernel 上 pass，
printer 甚至将 explicit ``alloc_local`` + ``[0]`` form **render 回** scalar form——
因此一旦 parsing 完成完全没有区别。两者因此 lower 为相同的 ``alignas(64) int phase_ptr[1];``；
scalar 只是让你省略 ``[0]``。（``T.local_scalar`` / ``T.shared_scalar`` / ``T.alloc_scalar`` 选择
scope explicitly。）

.. note::

   **为什么不** ``Var``\ **？** TIRx ``Var`` 是 *immutable*——single static
   binding（它正是 ``T.let`` 产生的，下面）。scalar 需要是
   *mutable*——你在 loop 和 accumulator 中 reassign 它——因此它必须由
   你可以重复 store 的一 element buffer 支撑，而不是 ``Var``。

``let``
~~~~~~~

``T.let`` binding 是**immutable**——single ``LetStmt``（named value，不是
buffer）。用它来 derived constant：

.. code-block:: python

    n: T.let = M * K               # immutable binding（LetStmt）
    half: T.let[T.int32] = N // 2  # ... 带 explicit type

它 lower 为**普通 scalar C variable**——不是 buffer（无 array，无 ``[0]``）。
对于 ``half: T.let = m * 2``（带 runtime ``m``）：

.. code-block:: c++

    int half = m * 2;     # `let` -> const-like local

因为 value 是 immutable，simplifier 可以自由 propagate 和 CSE 它，因此
在 use site 你经常看到 ``m * 2`` 直接 substituted（或通过
common-subexpression temporary shared）而不是对 ``half`` 的 reference。

.. note::

   **为什么有 immutable binding？** 因为 value 不能改变，
   arithmetic analyzer 将 var bind 到它（简化 ``LetStmt`` 时
   ``analyzer.Bind(var, value)``），因此关于 value 证明的 fact——
   constant bound、modular set（divisibility / alignment）、range——
   **通过每个 use propagate**。那 feed index simplification、bounds-check elimination 和
   alignment/vectorization decision。*mutable* scalar 是 memory load
   （``buf[0]``）：analyzer 不能假设它保持 constant，因此那些
   properties 不 carry through。``let`` 也是 pure value——无 allocation，
   自由 inline / substitute / CSE——而 scalar 是带 load/store semantics 的一 element buffer。

Tensor memory
-------------

Blackwell *tensor memory* 不是普通 scratch scope：它必须用 warp-uniform
``T.ptx.tcgen05.alloc`` / ``tcgen05.dealloc`` intrinsic explicit
reserve 和 free，每个 tensor 是用
``T.decl_buffer(..., scope="tmem", allocated_addr=<column>, layout=<tmem layout>)``
声明的 view。``allocated_addr``（column offset）是 mandatory——tensor-core dispatch
assert 它——因此 ``T.alloc_buffer(scope="tmem")``（**不**设置它）将不起作用。
不同于 shared memory，tensor memory 不直接 addressable：它只通过 ``tcgen05``
``mma`` / ``ld`` / ``st`` / ``cp`` 读取和写入。

手动，一个 warp 将 allocation issue 到 shared slot，你在 column offset 处 ``decl`` 每个
tensor 作为 view，一个 warp 在最后 free 它：

.. code-block:: python

    addr = T.alloc_shared((1,), "uint32")             # allocated base 的 slot
    if warp_id == alloc_warp:                         # tcgen05.alloc 是 warp-uniform
        T.ptx.tcgen05.alloc(T.address_of(addr), n_cols=512, cta_group=cta_group)
    acc = T.decl_buffer((CTA_M, 512), "float32", scope="tmem",
                        allocated_addr=0, layout=tmem_layout)   # column 0 的 view
    # ... 将 acc 用作 gemm_async / copy_async operand ...
    if warp_id == alloc_warp:
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
        T.ptx.tcgen05.dealloc(addr, n_cols=512, cta_group=cta_group)

你自己管理 column offset 和 ``tmem_layout``（datapath D/F layout）。
这精确是下面 pool emit 的 sequence。

Pool
~~~~

``T.TMEMPool`` 包装所有那个——warp-uniform alloc/dealloc、column
bump-allocation 和 datapath layout：

.. code-block:: python

    tmem_addr = pool.alloc((1,), "uint32")          # pool = kernel 的 smem pool
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=cta_group,
                           tmem_addr=tmem_addr)
    acc = tmem_pool.alloc((CTA_M, 512), "float32")  # allocated_addr 为你设置
    tmem_pool.commit()                               # emit tcgen05.alloc（一个 warp）
    # ... 使用 acc ...
    tmem_pool.dealloc()                              # emit tcgen05.dealloc（一个 warp）

见 Part III GEMM kernel 获取完整 example。

Buffer API
----------

``Buffer`` 是 pointer 上的 metadata（见上面的 *Declaring buffer*），因此它的大多数
方法是 *compile-time* reshape/reinterpret，改变 index arithmetic
或给你 pointer——它们不 emit 自己的 runtime op。常见的：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - 是什么
   * - ``B.data``
     - raw data pointer（``Var``）；print 为 ``B_ptr``
   * - ``B.ptr_to([i, j])``
     - 到 element 的 typed pointer（``address_of``）；print 为 ``&B_ptr[…]``
   * - ``B.vload([i], dtype="float32x4")`` / ``B.vstore([i], v)``
     - vectorized load / store；print 为 ``*(float4*)(B_ptr + …)``
   * - ``B.view(*shape, layout=…)``
     - 在新 shape/layout 下 reinterpret 相同 storage（无 copy）
   * - ``B.local(*shape, layout=…)``
     - calling thread 的 ``local`` buffer 的 private register slice
   * - ``B.permute(*dims)``
     - 轴 permuted 的 view（transposed layout）
   * - ``B.access_ptr(mask, …)``
     - masked access pointer（``tvm_access_ptr`` builtin），用于将
       region 传递给 intrinsic

**Pointer——``ptr_to`` / ``data``。** ``ptr_to`` 是你如何给 element address
到 intrinsic 或 inline function；``data`` 是 base pointer：

.. code-block:: python

    B[tx] = T.cuda.func_call("ld", A.ptr_to([tx]), source_code=SRC, return_type="float32")

.. code-block:: c++

    B_ptr[tx] = ld(&A_ptr[tx]);          # ptr_to([tx]) -> &A_ptr[tx]；  A.data -> A_ptr

**Vectorized access——``vload`` / ``vstore``。** 将几个 element 作为一个 wide
transfer 移动（也见 :doc:`data_types`）：

.. code-block:: python

    B.vstore([tx * 4], A.vload([tx * 4], dtype="float32x4"))

.. code-block:: c++

    *(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);

**Reshape / reinterpret——``view`` / ``permute``。** 两者都是 pure metadata；
data pointer 不变，只有 index arithmetic 不同。``A.view(64, 4)``
将 256-element buffer 视为 ``64×4``；``A.permute(1, 0)`` transpose 轴：

.. code-block:: python

    A2 = A.view(64, 4);     y = A2[tx, 0] + A2[tx, 3]   # A2[tx, j] -> A_ptr[tx*4 + j]
    At = A.permute(1, 0);   z = At[i, j]                # At[i, j]  -> A_ptr[j*4 + i]

.. code-block:: c++

    A2_ptr[tx * 4]  /* +3 */                 # view：row-major 64x4 index
    At_ptr[(j * 4) + i]                       # permute：swapped stride

**Register——``local``。** 将 thread-axis ``local`` layout decompose 为
calling thread 的 flat register bundle（tile primitive 广泛使用）：

.. code-block:: python

    R  = T.alloc_buffer((32, 8), "float32", scope="local", layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)]))
    Rl = R.local(8)          # 这个 lane 的 8 register

.. code-block:: c++

    alignas(64) float Rl_ptr[8];             # lane 的 private register
