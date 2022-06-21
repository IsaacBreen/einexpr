## What is einexpr?

einexpr brings true Einstein summation notation to Python, allowing you to convert your gory tensor manipulations into aesthetically pleasing and easy-to-follow arithmetic expressions with an idiomatic Python style (using standard operators: +, -, /, *, **) and support for your favourite np.* functions. Think einsum + addition + exponentiation + seamless interoperation with Jax. (PyTorch and TensorFlow support on the way.)

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