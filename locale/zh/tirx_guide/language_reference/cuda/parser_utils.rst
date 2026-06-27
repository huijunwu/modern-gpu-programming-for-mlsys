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

Parser 工具
===========

几个 helper 在**parse time**（TVMScript 转为 TIRx 时）起作用，使你可以
inline Python 计算的 value、提取 reusable fragment 和打包 parser-side state。

``T.meta_var`` — inline Python value
------------------------------------

``T.meta_var(x)`` 告诉 parser 将 ``x``（在 **Python** 中计算的 value）
作为 compile-time *meta* value 处理并直接 inline 到 IR 中，而不是
将它 parse 为 script variable。它避免了 throwaway local，并驱动
metaprogramming：对 meta value 的普通 Python ``for`` 在 parser 中 unroll。

.. code-block:: python

    n = T.meta_var(4)              # n 是 Python int，inline
    for j in range(n):            # parse time unroll
        acc[0] = acc[0] + A[tx, j]

``@T.inline`` — inline function
-------------------------------

``@T.inline`` 定义一个 function，它的 body 在 parsing 时**在每个 call site inline**——
生成的 code 中没有 call。它遵循 Python 的 lexical（LEGB）scoping 和 late binding，
因此 parameter 可以 shadow enclosing variable：

.. code-block:: python

    @T.inline
    def add_into(acc, x):
        acc[0] = acc[0] + x

    add_into(acc, A[tx, j])       # inline -> acc[0] = acc[0] + A[tx, j]

``@T.meta_class`` — parser-side state object
--------------------------------------------

``@T.meta_class`` 标记一个普通 Python class，它的**instance 是 parser meta
value**：它们的 field 可以持有 buffer 和 scalar，因此你可以将相关的
allocation 和 state 打包到一个 object 中并在 kernel body 中使用它。

.. code-block:: python

    @T.meta_class
    class State:
        def __init__(self, smem):
            self.acc = T.alloc_local([1], "float32")
            self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

    s = State(smem.data)
    s.acc[0] = T.float32(0.0)     # 像普通 buffer 一样使用它的 field
    # ... s.buf[i] ...

这对于分组 kernel 的 pipeline state（barrier、accumulator、
scratch view）很有用，而不是在 body 中 threading 多个 separate local。

``T.constexpr``
---------------

``T.constexpr`` 标记 compile-time kernel parameter，由 ``@T.jit`` 的
``.specialize(...)`` 烘焙进去。详见 :ref:`chap_tirx_primer`。
