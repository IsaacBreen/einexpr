## What is einexpr?

einexpr brings true Einstein summation notation to Python, allowing you to convert your gory tensor manipulations into aesthetically pleasing and easy-to-follow arithmetic expressions with an idiomatic Python style (using standard operators: +, -, /, *, **) and support for your favourite np.* functions. Think einsum + addition + exponentiation + seamless interoperation with Jax. (PyTorch and TensorFlow support on the way.)

```python
import numpy as np
from einexpr import einexpr, einfunc

X = einarray([[1, 2], [3, 4]])
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

# Linear transformation
@einfunc
def linear(x, W, b):
    return x['i'] * W['i j'] + b['j']

x_transformed = linear(x=np.array([1,2]), W=np.array([[1,2],[3,4]]), b=np.array([5,6]))
print(x_transformed['j'])
```

## Installation

```bash
pip install einexpr
```