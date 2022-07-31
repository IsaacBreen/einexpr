<style>
  @font-face {
    font-family: 'PT Sans';
    font-style: normal;
    font-weight: normal;
    src: local('PT Sans'), local('PTSans-Regular'),
      url(data:application/font-woff2;charset=utf-8;base64,d09GRgABAAAAAHowABMAAAAA+OAA) format('woff2');
  }
</style>


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

<h1 align="center"> <font face = "Courier New"> einexpr </font> </h1>

<p align="center">
    <b> einexpr makes it easy to write fast, maintainable, interpretable ML code </b>
</p>

Inspired by Einstein summation notation, einexpr encourages explicit dimension naming.

This doesn't enable you to do anything you couldn't before, but it does dramatically reduce the mental overhead of working with multi-dimensional arrays.

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