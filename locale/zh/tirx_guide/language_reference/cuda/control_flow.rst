.. _chap_control_flow:

Control Flow
============

TIRx 支持 standard control flow construct：

``for``
------

.. code-block:: python

    for i in range(n):
        T.evaluate(...)

``if``
------

.. code-block:: python

    if T.float16(1.0) > T.float16(0.5):
        T.evaluate(...)

``while``
---------

.. code-block:: python

    while condition:
        T.evaluate(...)
