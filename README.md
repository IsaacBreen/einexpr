<h2 align="center"> ⏰👀 </h2>
<p align="center">
    <b>Awaiting array API implementations</b>
</p>


While <font face = "Monaco"> einexpr </font> is ready for the world, it seems the Python ecosystem is not yet ready for <font face = "Monaco"> einexpr </font>.

In the interest of future-readiness, <font face = "Monaco"> einexpr </font> is built with first-class support for Python array API standard-conforming libraries. Since the standard is so new, none of the major tensor libraries have finished implementing it yet.

Don't worry, though; your favourite tensor library developers are almost certainly hard at work implementing it right now! Check back soon! 😁

If you're super keen to experience <font face = "Monaco"> einexpr </font>, you can try it out right now by installing NumPy `1.24.dev0` using:

```bash
pip install git+https://github.com/numpy/numpy
```

<h1 align="center"> <font face = "Monaco"> einexpr </font> </h1>

<p align="center">
    <b> <font face = "Monaco"> einexpr </font> makes it easy to write fast, maintainable, interpretable ML code </b>
</p>

Inspired by Einstein summation notation, <font face = "Monaco"> einexpr </font> encourages explicit dimension naming.

This doesn't enable you to do anything you couldn't before, but it does dramatically reduce the mental overhead of working with multi-dimensional arrays.

```python
import einexpr as ei

X = ei.array([[1, 2], [3, 4]])
Y = ei.array([[5, 6], [7, 8]])
a = ei.array([1,2])
b = ei.array([3,4])

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