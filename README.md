<h2 align="center"> ⏰ 👀 </h2>
<p align="center">
    <b>Awaiting array API implementations</b>
</p>


While `einexpr` is ready for the world, it seems the Python ecosystem is not yet ready for `einexpr`.

In the interest of future-readiness, `einexpr` is built with first-class support for Python array API standard-conforming libraries. Since the standard is so new, none of the major tensor libraries have finished implementing it yet.

Don't worry, though; your favourite tensor library developers are almost certainly hard at work implementing it right now! Check back soon! 😁

If you're super keen to experience `einexpr`, you can try it out right now by installing NumPy `1.24.dev0` using:

```bash
pip install --upgrade git+https://github.com/numpy/numpy
```

# einexpr

<p align="center">
    <b> einexpr makes it easy to write fast, maintainable, interpretable ML code </b>
</p>

Inspired by Einstein summation notation, `einexpr` encourages explicit dimension naming.

Being explicit about how operations map over dimensions dramatically reduce the mental overhead of working with multi-dimensional arrays.

## In action

The dimension naming convention is designed to be as intuitive, flexible, and unambiguous.

```python
import einexpr as ei

# Define some vectors
a = ei.array([1,2])
b = ei.array([3,4])
# (So far, pretty normal...)

# Dot product
x = a['i'] + b['i']
assert x[''] == ei.vecdot(a, b)

# Outer product
x = a['i'] * b['j']
assert x['i j'] == ei.outer(a, b)

# Define some matrices
X = ei.array([[1, 2], [3, 4]])
Y = ei.array([[5, 6], [7, 8]])

# Matrix-vector multiplication
x = X['i j'] * a['j']
assert x['i'] == ei.matmul(X, a)

# Matrix-matrix multiplication
x = X['i j'] * Y['j k']
assert x['i k'] == ei.matmul(X, Y)
```



## Installation

```bash
pip install einexpr
```