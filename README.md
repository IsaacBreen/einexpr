einexpr takes einsum further by allowing you to write arbitrary arithmetic expressions using standard Python operators.
```python
import numpy as np
from einexpr import einexpr, einfunc

X = einexpr(np.array([[1, 2], [3, 4]]))
Y = einexpr(np.array([[5, 6], [7, 8]]))
a = einexpr(np.array([1,2]))
b = einexpr(np.array([3,4]))

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
    print(W)
    return x['i'] * W['i j'] + b['j']

x_transformed = linear(x=np.array([1,2]), W=np.array([[1,2],[3,4]]), b=np.array([5,6]))
print(x_transformed['j'])
```
