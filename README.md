<!-- Display logo -->
![einexpr logo](docs/static/images/logo.png|width=100px)

<h2 align="center"> ⏰ 👀 </h2>
<p align="center">
    <b>Awaiting array API implementations</b>
</p>


While `einexpr` is ready for the world, it seems the Python ecosystem is not yet ready for `einexpr`.

In the interest of future-readiness, `einexpr` is built with first-class support for Python array API standard-conforming libraries. Since the standard is so new, none of the major tensor libraries have finished implementing it yet.

Don't worry, though; your favourite tensor library developers are almost certainly hard at work implementing it right now! Check back soon! 😁

If you're super keen to experience `einexpr`, you can try it out today by installing NumPy `1.24.dev0` using:

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

The convention for specifying dimension structure is designed to be intuitive, flexible, and unambiguous.

```python
import einexpr as ei
import numpy as np

# Define some vectors
a = ei.array([1,2])
b = ei.array([3,4])
# (So far, pretty normal...)

# Dot product
x = a['i'] * b['i']
assert np.dot(a, b) == x['']  # <-- x[''] collapses (by summing) along dimensions i, turning x['i'] into a scalar

# Outer product
x = a['i'] * b['j']
assert np.all(np.outer(a, b) == x['i j'])

# Define some matrices
X = ei.array([[1, 2], [3, 4]])
Y = ei.array([[5, 6], [7, 8]])

# Flatten
x = X['i j']
assert np.all(np.reshape(X, (-1,)) == x['(i,j)'])

# Matrix-vector multiplication
x = X['i j'] * a['j']
assert np.all(np.matmul(X, a) == x['i'])
```

You can also treat dimensions as first-class objects.

```python
i, j = ei.dims(2) # Unnamed object
k, l = ei.dims('k l') # Named objects

# Matrix-matrix multiplication
x = X[i, j] * Y[j, k]
assert np.all(np.matmul(X, Y) == x[i, k])
```

## Installation

```bash
pip install einexpr
```