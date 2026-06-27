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

Data type 和 expression
=======================

每个 TIRx expression 携带低级别 **dtype** 和高级别 **type**。

Expression dtype
----------------

``PrimExpr`` 的 ``.dtype`` 是它的 scalar（或 vector）element type——``float32``、
``float16``、``bfloat16``、``int32``、``uint8``、``bool``、低精度
``float8_e4m3fn`` / ``float4_e2m1fn`` …、``handle``（pointer）和 vector form
如 ``float32x4``。每个 print 到匹配的 CUDA type。跨几个 dtype 分配 local 和
shared buffer，加上 vectorized ``float32x4`` load/store：

.. code-block:: python

    @T.prim_func
    def dtypes(A_ptr: T.handle, O_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        O = T.match_buffer(O_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([64])
        f16  = T.alloc_local((1,), "float16")        # register scalar ...
        bf16 = T.alloc_local((1,), "bfloat16")
        i32  = T.alloc_local((1,), "int32")
        u8   = T.alloc_local((1,), "uint8")
        b1   = T.alloc_local((1,), "bool")
        sm   = T.alloc_shared((64,), "float16")      # ... 和 shared tile
        v    = T.alloc_local((1,), "float32x4")      # vector-dtype register（float4）
        v[0] = A.vload([tx * 4], dtype="float32x4")  # vectorized load
        O.vstore([tx * 4], v[0])                     # vectorized store
        # ... (use f16/bf16/i32/u8/b1/sm) ...

lower 为（generated CUDA，省略）：

.. code-block:: c++

    half          f16_ptr[1];               // float16
    nv_bfloat16   bf16_ptr[1];              // bfloat16
    int           i32_ptr[1];               // int32
    uchar         u8_ptr[1];                // uint8
    signed char   b1_ptr[1];                // bool
    __shared__ alignas(64) half sm_ptr[64]; // shared float16
    float4        v_ptr[1];                 // float32x4  (vector)
    v_ptr[0]                  = *(float4*)(A_ptr + tx * 4);   // vectorized load
    *(float4*)(O_ptr + tx * 4) = v_ptr[0];                   // vectorized store

buffer 的 dtype 本身可以是 **vector type**：``T.alloc_local((1,), "float32x4")``
直接声明 ``float4`` register（你索引它如 ``v[0]``），``float32x4`` ``vload`` / ``vstore``
然后作为一次 16-byte access 移动它。vector dtype 不绑定到 ``vload``——任何 buffer 或 scalar 可以携带它。

因此 dtype → CUDA mapping 是：

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - dtype → CUDA
     - dtype → CUDA
     - dtype → CUDA
   * - ``float32`` → ``float``
     - ``float16`` → ``half``
     - ``bfloat16`` → ``nv_bfloat16``
   * - ``int32`` → ``int``
     - ``uint8`` → ``uchar``
     - ``bool`` → ``signed char``
   * - ``float32x4`` → ``float4``
     - ``handle`` → ``T*``（pointer）
     - （vector dtype → CUDA vector type）

dtype vs type
-------------

``dtype`` 是*低级别的*——它说"什么 bit"。分开地，value 有高级别 **type**：
scaler 的 ``PrimType(dtype)`` 或 pointer 的
``PointerType(PrimType(dtype), scope)``。大多数 expression 是 scalar
（``PrimType``）；type system 主要对 **pointer** 重要。

Pointer（``handle``）
---------------------

buffer 的 ``data``——它的 pointer——是指针 type 的 ``Var``，且它是
**immutable**（pointer 从不 reassign）。这塑造了你如何获取它：

- ``T.alloc_buffer(...)`` 分配 storage **且**定义它的 ``data`` pointer。
- ``T.decl_buffer(..., data=ptr)`` 在现有 pointer ``Var`` ``ptr`` 上声明 buffer。
- 用 pointer **expression** 支撑 buffer——例如 ``T.ptx.map_shared_rank``
  （PTX ``mapa``）给出另一个 cluster CTA 的 shared address——你必须首先将
  那个 expression bind 到 pointer ``Var``（``data`` 必须是 ``Var``，不是
  expression），使用 ``PointerType`` 的 ``T.let``：

  .. code-block:: python

      from tvm.ir.type import PointerType, PrimType

      ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = \
          T.reinterpret("handle", T.ptx.map_shared_rank(mbar.ptr_to([0]), 0))
      remote_mbar = T.decl_buffer([1], "uint64", data=ptr, scope="shared")
