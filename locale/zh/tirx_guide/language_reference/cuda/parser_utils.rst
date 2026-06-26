.. _chap_parser_utils:

Parser Utility
==============

TIRx 使用 Python AST（abstract syntax tree）来解析 kernel source。

Parser utility 是 ``T.prim_func`` 和 ``T.script`` decorator。它们使你能以 Python function 的形式写 TIRx kernel。

``@T.prim_func``
----------------

``@T.prim_func`` decorator 标记一个 Python function 为 TIRx prim_func。

.. code-block:: python

    @T.prim_func
    def my_kernel(A: T.handle, B: T.handle, C: T.handle):
        ...

TIRx parser 解析那个 Python function 并 generate TIR module。

``@T.script``
-------------

``@T.script`` decorator 是 ``@T.prim_func`` 的 alternative。它标记 function 为 TIR script。

.. code-block:: python

    @T.script
    def my_kernel(A, B, C):
        ...

``@T.script`` 提供比 ``@T.prim_func`` 更 flexible 的 syntax。
