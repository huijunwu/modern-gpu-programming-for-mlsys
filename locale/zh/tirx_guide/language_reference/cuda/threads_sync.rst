.. _chap_threads_sync:

Thread 和 Synchronization
==========================

TIRx 用 ``T.launch_thread()`` 声明 thread：

.. code-block:: python

    T.launch_thread("threadIdx.x", 256)
    T.launch_thread("threadIdx.y", 1)
    T.launch_thread("blockIdx.x", grid_size)

用 ``T.thread_id()`` 获取 thread id：

.. code-block:: python

    tx = T.thread_id("threadIdx.x")
    ty = T.thread_id("threadIdx.y")
    bx = T.thread_id("blockIdx.x")

用 ``T.thread_extent()`` 获取 thread extent：

.. code-block:: python

    tx_extent = T.thread_extent("threadIdx.x")

Synchronization
---------------

用 ``T.barrier()`` 做 thread block barrier：

.. code-block:: python

    T.barrier("__syncthreads")

用 ``T.mbarrier_wait()`` 做 mbarrier wait：

.. code-block:: python

    T.mbarrier_wait(bar, phase)
