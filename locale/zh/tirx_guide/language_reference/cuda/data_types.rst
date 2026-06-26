.. _chap_data_types:

Data Types
==========

TIRx 支持多种 data type：

- **float16** (``"float16"``): 16-bit floating-point
- **float32** (``"float32"``): 32-bit floating-point
- **float64** (``"float64"``): 64-bit floating-point
- **bfloat16** (``"bfloat16"``): 16-bit brain floating-point
- **int8** (``"int8"``): 8-bit signed integer
- **int32** (``"int32"``): 32-bit signed integer
- **int64** (``"int64"``): 64-bit signed integer
- **uint8** (``"uint8"``): 8-bit unsigned integer
- **uint32** (``"uint32"``): 32-bit unsigned integer
- **uint64** (``"uint64"``): 64-bit unsigned integer

使用 ``T.dtype()`` 构造 dtype：

.. code-block:: python

    T.dtype("float16")
    T.dtype("float32")
    T.dtype("int32")

使用 ``T.float16()``, ``T.float32()``, ``T.int32()`` 等构造 constant：

.. code-block:: python

    T.float16(1.0)
    T.float32(2.0)
    T.int32(3)
