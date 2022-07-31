<h2 align="center"> ⌚︎👀 </h2>
<p align="center">
    <b>Awaiting array API implementations</b>
</p>


While einexpr is ready for the world, it seems the Python ecosystem is not yet ready for einexpr.

In the interest of future-readiness, einexpr is built with first-class support for Python array API standard-conforming libraries. Since the standard is so new, none of the major tensor libraries have finished implementing it yet.

Don't worry, though; your favourite tensor library developers are almost certainly hard at work implementing it right now! Check back soon! 😁

If you're super keen to experience einexpr, you can try it out right now by installing NumPy 1.24.dev0 using:

```bash
pip install git+https://github.com/numpy/numpy
```

<h1 align="center"> <font face = "Comic sans MS"> einexpr </font> </h1>

<p align="center">
    <b> einexpr makes it easy to write fast, maintainable, interpretable ML code </b>
</p>

Inspired by Einstein summation notation, einexpr encourages explicit dimension naming.

How many times have you had to switch back and forth between your `__init__` and  `forward` methods to figure out what your a layer is doing?

Gone are the days of tediously tracking the dimensions of your tensors throughout your code.

```python
from einexpr import einexpr, einfunc

X = einarray([[1, 2], [3, 4]]) # pass backend='[numpy|jax|torch]' (default: numpy)
Y = einarray([[5, 6], [7, 8]])
a = einarray([1,2])
b = einarray([3,4])

# Dot product
x = a['i'] + b['i']
print(x[''])

# Outer product
x = a['i'] * b['j']
print(x['i j'])

# Matrix-vector multiplication
x = X['i j'] * a['j']
print(x['i'])

# Matrix-matrix multiplication
x = X['i j'] * Y['j k']
print(x['i, k'])
```

## Installation

```bash
pip install einexpr
```

## What is 