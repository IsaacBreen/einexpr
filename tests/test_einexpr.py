from einexpr import __version__
from einexpr import einexpr, einfunc, binary_arithmetic_operations, additive_operations, EinsteinExpression
import numpy as np
import math
import jax
import jax.numpy as jnp
from collections import namedtuple
import string
import pprint
import pytest

pp = pprint.PrettyPrinter(indent=4)

tolerance = 1e-3

def test_pow():
    w = einexpr(np.random.rand(4,2))
    x = einexpr(np.random.rand(2))
    expr = w['b,i'] * x['i']
    expr = expr['']
    print(expr)
    assert np.allclose(expr.__array__(), np.einsum('bi,i->', w.__array__(), x.__array__()), tolerance)

def test_simple_expr1():
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[5, 6], [7, 8]])
    a = np.array([1,2])
    b = np.array([3,4])

    Xe = einexpr(X)
    Ye = einexpr(Y)

    # MULTIPLICATION
    assert np.allclose(np.einsum('ij,jk->ik', X, Y), (Xe['i j'] * Ye['j k'])['i k'], tolerance)

    # ADDITION
    assert np.allclose(np.sum(X[:, :, np.newaxis] + Y[np.newaxis, :, :], axis=1), (Xe['i j'] + Ye['j k'])['i k'], tolerance)

    # LINEAR TRANSFORMATION
    def linear(x, W, b):
        return np.einsum("i,ij->j", x, W) + b

    @einfunc
    def linear_ein(x, W, b):
        return x['i'] * W['i j'] + b['j']

    assert np.allclose(linear(a, X, b), linear_ein(a, X, b)['j'], tolerance)

RandomExpressionData = namedtuple("RandomExpressionData", ["expr", "expr_json", "var"])

@pytest.fixture
def make_random_expr(seed, np_like, max_indices=8, max_index_size=5, max_vars=8, p_leaf=0.8):
    np.random.seed(seed)
    num_indices = np.random.randint(2, max_indices)
    num_indices_out = np.random.randint(0,num_indices)
    index_names = list(string.ascii_lowercase[:num_indices])
    index_names_out = list(string.ascii_lowercase[:num_indices_out])
    index_sizes = {i:s for i,s in zip(index_names, np.random.randint(1, max_index_size, size=num_indices))}

    n_vars = np.random.randint(2, max_vars)
    per_var_indices = [list(np.random.choice(index_names, size=np.random.randint(1, num_indices), replace=False)) for _ in range(n_vars)]
    vars = [np.random.rand(*(index_sizes[i] for i in var_indices)) for var_indices in per_var_indices]
    vars = [np_like.array(v) for v in vars]
    ops = binary_arithmetic_operations
    
    def _make_random_expr(vars, per_var_indices, index_names, mul_indices, ops, p_leaf):
        if np.random.rand() < p_leaf:
            var_num = np.random.randint(0, n_vars)
            var = vars[var_num]
            expr = einexpr(var)[",".join(per_var_indices[var_num])]
            expr_json = {"type": "leaf", "value": var, "indices": set(per_var_indices[var_num]), "shape": per_var_indices[var_num]}
            return expr, expr_json
        else:
            op = np.random.choice(list(ops))
            expr_lhs, expr_json_lhs = _make_random_expr(vars, per_var_indices, index_names, mul_indices, ops, p_leaf)
            expr_rhs, expr_json_rhs = _make_random_expr(vars, per_var_indices, index_names, mul_indices, ops, p_leaf)
            expr = getattr(expr_lhs, op)(expr_rhs)
            indices = expr_json_lhs["indices"] | expr_json_rhs["indices"]
            expr_json = {"type": "op", "op": op, "lhs": expr_json_lhs, "rhs": expr_json_rhs, "indices": indices}
            return expr, expr_json

    def _eval_expr_json(expr_json, non_collapsable_indices):
        # In additive subexpressions, divide each term by the size of those indices what are about to be broadcasted over and which are also in non_collapsable_indices.
        # Another explaination: need to divide by size of indices that are about to be broadcast but which aren't subsequently collapsed over (those that appear in 
        # non_collapsable_indices).
        if expr_json["type"] == "op":
            child_non_collapsable_indices = non_collapsable_indices
            if expr_json["op"] in additive_operations:
                child_non_collapsable_indices |= expr_json["lhs"]["indices"] | expr_json["rhs"]["indices"]
            lhs = _eval_expr_json(expr_json["lhs"], child_non_collapsable_indices)
            rhs = _eval_expr_json(expr_json["rhs"], child_non_collapsable_indices)
            if expr_json["op"] in additive_operations:
                collapsable_broadcast_indices_lhs = expr_json["lhs"]["indices"] - expr_json["rhs"]["indices"] - non_collapsable_indices
                collapsable_broadcast_indices_rhs = expr_json["rhs"]["indices"] - expr_json["lhs"]["indices"] - non_collapsable_indices
                lhs = lhs / np.prod([index_sizes[i] for i in collapsable_broadcast_indices_lhs])
                rhs = rhs / np.prod([index_sizes[i] for i in collapsable_broadcast_indices_rhs])
                assert np.prod([index_sizes[i] for i in collapsable_broadcast_indices_lhs]) == np.prod([index_sizes[i] for i in collapsable_broadcast_indices_rhs]) == 1
            return getattr(lhs, expr_json["op"])(rhs)
        elif expr_json["type"] == "leaf":
            # TODO: is the below line necessary?
            var_normshaped = np.einsum(f"{''.join(i for i in expr_json['shape'])}->{''.join(i for i in index_names if i in expr_json['indices'])}", expr_json["value"])
            expand_indices = [n for n, i in enumerate(index_names) if i not in expr_json["indices"]]
            return np.expand_dims(var_normshaped, expand_indices)
                
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        expr, expr_json = _make_random_expr(vars, per_var_indices, index_names, [], ops, p_leaf)
        expr = expr[",".join(index_names_out)]
        var = _eval_expr_json(expr_json, set(index_names_out))
        var = np.sum(var, axis=tuple(n for n, i in enumerate(index_names) if i not in index_names_out))
    return RandomExpressionData(expr, expr_json, var)

# @pytest.mark.skip
@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("np_like", [np, jnp])
def test_random_expr(seed, np_like, make_random_expr):
    expr, expr_json, var = make_random_expr
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if not np.allclose(expr.__array__(), var, tolerance) and np.all([~np.isnan(expr.__array__()), ~np.isnan(var)]):
            print(expr)
            pp.pprint(expr_json)
            print(expr.get_shape())
            print(var.shape)
            raise ValueError(f"Values do not match: {expr.__array__()} != {var}")


# @pytest.mark.skip
@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("np_like", [jnp])
def test_random_expr_jax_jit(seed, make_random_expr):
    expr, expr_json, var = make_random_expr
    assert not any(x is None for x in make_random_expr), f"One of the expressions from {make_random_expr} is None"
    
    # @jax.jit
    def eval_expr(expr):
        return expr.__array__()
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        assert eval_expr(expr) is not None
        if not np.allclose(eval_expr(expr), var, tolerance) and np.all([~np.isnan(eval_expr(expr)), ~np.isnan(var)]):
            print(expr)
            pp.pprint(expr_json)
            print(expr.get_shape())
            print(var.shape)
            raise ValueError(f"Values do not match: {eval_expr(expr)} != {var}")


def test_simple_expr_jax_jit():
    @jax.jit
    def add(x,y):
        x = einexpr(x)
        y = einexpr(y)
        z = x['i'] + y['i']
        return z[''].__array__()
    x = np.random.rand(2)
    y = np.random.rand(2)
    add(x,y)

    @jax.jit
    def add_and_reduce(x,y):
        z = x['i'] + y['i']
        return z['']
    x = einexpr(np.random.rand(2))['i']
    y = einexpr(np.random.rand(2))['i']
    add_and_reduce(x,y)


def test_intro():
    X = einexpr(np.array([[1, 2], [3, 4]]))
    Y = einexpr(np.array([[5, 6], [7, 8]]))
    a = einexpr(np.array([1,2]))
    b = einexpr(np.array([3,4]))

    # Dot product
    x = a['i'] + b['i']
    print(x[''].__array__())

    # Outer product
    x = a['i'] * b['j']
    print(x['i,j'].__array__())

    # Matrix-vector multiplication
    x = X['i,j'] * a['j']
    print(x['i'].__array__())

    # Matrix-matrix multiplication
    x = X['i,j'] * Y['j,k']
    print(x['i,k'].__array__())

    # Linear transformation
    @einfunc
    def linear(x, W, b):
        print(W)
        return x['i'] * W['i,j'] + b['j']

    x_transformed = linear(x=np.array([1,2]), W=np.array([[1,2],[3,4]]), b=np.array([5,6]))
    print(x_transformed['j'].__array__())
