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

Control flow
============

Control flow 是 ``if``、loop family 和 ``while``——每个映射到直观的
CUDA。

if
--

Python ``if`` / ``else`` 变为 CUDA ``if`` / ``else``。用
thread/lane comparison guard work，或用 ``T.ptx.elect_sync()``
elect 单个 issuing thread：

.. code-block:: python

    if tx < 128:
        A[tx] = A[tx] * T.float32(2.0)
    else:
        A[tx] = A[tx] + T.float32(1.0)

    if T.ptx.elect_sync():
        ...                              # 一个 elected lane（例如 issue TMA/MMA）

.. code-block:: c++

    if (((int)threadIdx.x) < 128) {
      A_ptr[tx] = A_ptr[tx] * 2.0f;
    } else {
      A_ptr[tx] = A_ptr[tx] + 1.0f;
    }

对于 expression-level choice（无 branch），使用 ``T.if_then_else(cond, a, b)``。它
lower 为 ternary，因此不引入 control-flow divergence：

.. code-block:: c++

    O_ptr[tx] = (A_ptr[tx] > 0.0f) ? A_ptr[tx] : 0.0f;

Uniform vs. divergent control flow
----------------------------------

Per-thread guard 如 ``if tx < 128`` 对普通 work 没问题，但
**collective** operation 必须被它们 synchronize 的每个 thread *uniformly* 到达。

例如，``T.cuda.cta_sync()`` 映射到 ``__syncthreads()``，需要 thread block 中所有
thread。它绝不应坐在 thread- 或
warpgroup-divergent branch 内部：如果放在 ``if wg_id == 0:`` 内部，其他
warpgroup 永远不会到达且 kernel 将 deadlock。当只有单个 warpgroup
需要 synchronize 时，使用 warpgroup-scoped ``T.cuda.warpgroup_sync(id)``（见
:ref:`chap_gemm_advanced` 和 :doc:`threads_sync`）。

相同 caution 适用于 barrier setup。``mbarrier`` ``.init()`` lower 为
single-thread guard（``if (threadIdx.x < 1)``）。将它嵌套在另一个 divergent
branch 内部可以使 barrier uninitialized，导致 unspecified launch failure。

loop
----

Loop 有四种 flavor；普通 Python ``range`` 变为 ``T.serial``：

- ``T.serial(n)`` — sequential loop（ptxas 仍可能 unroll 它）。
- ``T.unroll(n)`` — 完全 unrolled（展开为 straight-line statement）。
- ``T.vectorized(n)`` — vectorized loop。
- ``T.grid(*extents)`` — nested loop nest。

``break`` / ``continue`` 在 loop 内工作。

.. code-block:: python

    for i, j in T.grid(8, 8):
        B[i, j] = T.max(A[i, j], T.float32(0.0))

.. code-block:: c++

    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        B_ptr[i * 8 + j] = max(A_ptr[i * 8 + j], 0.0f);

``T.unroll(4)`` 展开为四个 straight-line statement，无 loop。

while
-----

``while`` loop 运行直到它的 condition 为 false。使用 mutable scalar counter
（见 :doc:`buffers`）：

.. code-block:: python

    i: T.int32 = 0
    while i < 64:
        A[i] = A[i] + T.float32(1.0)
        i += 1

它 lower 为带 early-exit ``break`` 的 ``while (1)``（counter 是
一 element register buffer）：

.. code-block:: c++

    int i_ptr[1];
    i_ptr[0] = 0;
    while (1) {
      if (!(i_ptr[0] < 64)) { break; }
      A_ptr[i_ptr[0]] = A_ptr[i_ptr[0]] + 1.0f;
      i_ptr[0] = i_ptr[0] + 1;
    }
