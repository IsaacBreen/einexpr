<!-- Display logo centered -->
<p align="center">
  <img width="460" src="docs/static/images/logo-black-border.png">
</p>

<p align="center">
    <b> Fast, maintainable, interpretable ML code </b>
</p>


# Introducing einexpr

`einexpr` brings Einstein summation notation to Python.

Einstein summation notation encourages explicit dimension naming.

Explicitly naming dimensions makes it much clearer what operations over those dimensions are actually doing. This can **dramatically reduce the mental overhead of working with multi-dimensional arrays**.

## In action

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

`einexpr`'s dimension structure conventions are designed to be intuitive, flexible, and unambiguous.

In the above examples, the **string** inside the square brackets specifies the dimension structure.

You can also pass dimensions around as **first-class objects**:

```python
i, j, k = ei.dims(3)

x = X[i, j] * Y[j, k]
assert np.all(np.matmul(X, Y) == x[i, k])
```

There's no trade-off: feel free to combine strings and objects in any way that you find convenient.

```python
i, j, k = ei.dims('i j k')

Z = X[i, j] * Y[j, k]

# Equivalent ways of flattening along j and k
Z['i (j k)']
Z[i,(j,k)]
Z['i',('j','k')]
jk = (j,k)
Z['i', jk]

# We can check that the dimensions are correct using the `dims` property
assert Z['i (j k)'].dims == ('i', ('j', 'k'))
...
```

# Installation

```bash
pip install einexpr
```

## Awaiting array API implementations ⏰ 👀


While `einexpr` is ready for the world, it would seem that the Python ecosystem is not yet ready for `einexpr`.

`einexpr` is built with first-class support for Python array API standard-conforming libraries. This means that `einexpr` will have excellent long-term support for NumPy, PyTorch, TensorsFlow, JAX, and other array libraries. However, since standard is rather new, none of these libraries have finished implementing it yet.

Don't worry, though: the devs are probably working on it right now! (e.g. [PyTorch](https://github.com/pytorch/pytorch/issues/58743)). Check back soon.

### For the impatient

If you're super keen to experience `einexpr` right this moment, you can try it out by installing NumPy `1.24.dev0`:

```bash
pip install --upgrade git+https://github.com/numpy/numpy
```