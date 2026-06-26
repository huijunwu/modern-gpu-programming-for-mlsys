.. _chap_buffers:

Buffer 和 Memory
================

TIRx 用 ``T.match_buffer()`` 声明 buffer：

.. code-block:: python

    A = T.match_buffer(A_ptr, (M, K), "float16")
    B = T.match_buffer(B_ptr, (N, K), "float16")
    C = T.match_buffer(C_ptr, (M, N), "float16")

``T.match_buffer()`` 将 buffer 的 shape 和 dtype 绑定到 pointer。

使用 ``T.decl_buffer()`` 声明 internal buffer：

.. code-block:: python

    A_shared = T.decl_buffer((BLK_M, BLK_K), "float16", scope="shared")
    B_shared = T.decl_buffer((BLK_N, BLK_K), "float16", scope="shared")
