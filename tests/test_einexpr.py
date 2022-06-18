from einexpr import __version__
from einexpr import einexpr, einfunc, binary_arithmetic_operations, additive_operations, multiplicitave_operations, power_operations, EinsteinExpression
import numpy as np
import jax
import jax.numpy as jnp
from collections import namedtuple
import string
import pprint
import pytest

pp = pprint.PrettyPrinter(indent=4)

tolerance = 1e-6
default_dtype = None

default_binary_attr_ops = binary_arithmetic_operations
default_unary_ops = [np.abs]

RandomExpressionData = namedtuple("RandomExpressionData", ["expr", "expr_json", "var"])

N_TRIALS_MULTIPLIER=100
# N_TRIALS_MULTIPLIER=0

@pytest.fixture
def X():
    return np.array([[1, 2], [3, 4]])
    
@pytest.fixture
def Y():
    return np.array([[5, 6, 7], [7, 8, 9]])
    
@pytest.fixture
def x():
    return np.array([1,2])
    
@pytest.fixture
def y():
    return np.array([3,4,5])


def test_pow():
    rng = np.random.default_rng(2022)
    w = einexpr(rng.uniform(size=(4,2)))
    x = einexpr(rng.uniform(size=2))
    expr = w['b,i'] * x['i']
    expr = expr['']
    print(expr)
    assert np.allclose(expr.__array__(), np.einsum('bi,i->', w.__array__(), x.__array__()), tolerance)

def test_simple_expr1(X, Y, x, y):
    Xe, Ye, xe, ye = (einexpr(val) for val in (X, Y, x, y))

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

    assert np.allclose(linear(x, Y, y), linear_ein(x, Y, y)['j'], tolerance)

def test_commonly_failed1(X, Y, x, y):    
    Xe, Ye, xe, ye = (einexpr(val) for val in (X, Y, x, y))
    
    ze = xe['i'] ** (Xe['j i'] ** xe['i'])
    assert np.allclose(ze['j i'].__array__(), x ** (X ** x), tolerance)
    
    z = xe['i'] ** (xe['j'] + xe['j'])
    assert np.allclose(z['i j'].__array__(), x[:, None] ** (x[None, :] + x[None, :]), tolerance)
    print(z.coerce_into_shape('i'))
    assert np.allclose(z['i'], np.sum(x[:, None] ** (x[None, :] + x[None, :]), axis=1), tolerance)

def test_numpy_ufunc_override1(X, Y, x, y):
    Xe, Ye, xe, ye = (einexpr(val) for val in (X, Y, x, y))
    Z = X ** np.abs(-x)
    Ze = Xe['i j'] ** np.abs(-xe['j'])
    assert np.allclose(Z, Ze['i j'].__array__(), tolerance)

    Xe, Ye, xe, ye = (einexpr(val) for val in (X, Y, x, y))
    Z = X ** np.abs(-x)
    Ze = Xe['i j'] ** np.abs(-xe['j'])
    assert np.allclose(Z, Ze['i j'].__array__(), tolerance)

@pytest.fixture
def make_random_expr(seed, np_like, unary_ops=default_unary_ops, binary_attr_ops=default_binary_attr_ops, max_indices=5, max_indices_per_var=3, max_index_size=5, max_vars=8, E_num_nodes=2, max_nodes=10, p_binary_op_given_nonleaf=0.9, low=1, high=100, max_exponent=3):
    assert E_num_nodes >= 1, f"E_num_nodes must not be less than one; got {E_num_nodes}"
    p_leaf_given_not_unary = E_num_nodes / (2*E_num_nodes - 1)
    x = p_binary_op_given_nonleaf
    y = p_leaf_given_not_unary
    p_unary_op = 1 - x / (1 + x*y - y)
    p_leaf = y*(1-p_unary_op)
    p_binary_op = 1 - p_unary_op - p_leaf
    
    rng = np.random.default_rng(seed)
    max_index_size = min(max_index_size, max_indices)
    num_indices = rng.integers(2, max_indices)
    num_indices_out = rng.integers(0,num_indices)
    index_names = list(string.ascii_lowercase[:num_indices])
    index_names_out = list(string.ascii_lowercase[:num_indices_out])
    index_sizes = {i:s for i,s in zip(index_names, rng.integers(1, max_index_size, size=num_indices))}

    n_vars = rng.integers(2, max_vars)
    per_var_indices = [list(rng.choice(index_names, size=rng.integers(1, max_indices_per_var), replace=False)) for _ in range(n_vars)]
    vars = [rng.integers(low, high, size=[index_sizes[i] for i in var_indices]) for var_indices in per_var_indices]
    # vars = [rng.uniform(low, high, [index_sizes[i] for i in var_indices]) for var_indices in per_var_indices]
    vars = [np_like.array(v, dtype=default_dtype) for v in vars]
    
    def _make_random_expr(max_nodes):
        if max_nodes >= 2:
            node_type = rng.choice(["leaf", "unary_op", "binary_attr_op"], p=[p_leaf, p_unary_op, p_binary_op])
        elif max_nodes == 1:
            p_norm = p_leaf + p_unary_op
            node_type = rng.choice(["leaf", "unary_op"], p=[p_leaf/p_norm, p_unary_op/p_norm])
        else:
            node_type = "leaf"
        if node_type == "leaf":
            var_num = rng.integers(0, n_vars)
            var = vars[var_num]
            expr = einexpr(var)[",".join(per_var_indices[var_num])]
            expr_json = {"type": "leaf", "value": var, "indices": set(per_var_indices[var_num]), "shape": per_var_indices[var_num], "num_nodes": 1}
            return expr, expr_json
        elif node_type == "unary_op":
            op = rng.choice(list(unary_ops))
            expr, expr_json = _make_random_expr(max_nodes)
            newexpr = op(expr)
            expr_json = {"type": "unary_op", "op": op, "indices": expr_json["indices"], "operand": expr_json, "num_nodes": expr_json["num_nodes"]}
            return newexpr, expr_json
        elif node_type == "binary_attr_op":
            op = rng.choice(list(binary_attr_ops))
            expr_lhs, expr_json_lhs = _make_random_expr(max_nodes)
            expr_rhs, expr_json_rhs = _make_random_expr(max_nodes - expr_json_lhs["num_nodes"])
            if op in power_operations:
                expr_rhs = np.abs(expr_rhs)
                # expr_rhs = np.clip(expr_rhs, 0, max_exponent)
                expr_json_rhs = {"type": "unary_op", "op": np.abs, "indices": expr_json_rhs["indices"], "operand": expr_json_rhs, "num_nodes": expr_json_rhs["num_nodes"]}
                # expr_json_rhs = {"type": "unary_op", "op": lambda x: np.clip(x, 0, max_exponent), "indices": expr_json_rhs["indices"], "operand": expr_json_rhs, "num_nodes": expr_json_rhs["num_nodes"]}
            expr = getattr(expr_lhs, op)(expr_rhs)
            expr['']
            indices = expr_json_lhs["indices"] | expr_json_rhs["indices"]
            expr_json = {"type": "binary_attr_op", "op": op, "lhs": expr_json_lhs, "rhs": expr_json_rhs, "indices": indices, "num_nodes": 1 + expr_json_lhs["num_nodes"] + expr_json_rhs["num_nodes"]}
            return expr, expr_json

    def _eval_expr_json(expr_json, non_collapsable_indices):
        # In additive subexpressions, divide each term by the size of those indices what are about to be broadcasted over and which are also in non_collapsable_indices.
        # Another explaination: need to divide by size of indices that are about to be broadcast but which aren't subsequently collapsed over (those that appear in 
        # non_collapsable_indices).
        if expr_json["type"] == "unary_op":
            return expr_json["op"](_eval_expr_json(expr_json["operand"], non_collapsable_indices))
        elif expr_json["type"] == "binary_attr_op":
            child_non_collapsable_indices = non_collapsable_indices.copy()
            if expr_json["op"] in additive_operations:
                child_non_collapsable_indices |= expr_json["lhs"]["indices"] | expr_json["rhs"]["indices"]
            lhs = _eval_expr_json(expr_json["lhs"], child_non_collapsable_indices)
            rhs = _eval_expr_json(expr_json["rhs"], child_non_collapsable_indices)
            if expr_json["op"] in additive_operations:
                collapsable_broadcast_indices_lhs = expr_json["rhs"]["indices"] - expr_json["lhs"]["indices"] - non_collapsable_indices
                collapsable_broadcast_indices_rhs = expr_json["lhs"]["indices"] - expr_json["rhs"]["indices"] - non_collapsable_indices
                lhs = lhs / np.prod([index_sizes[i] for i in collapsable_broadcast_indices_lhs])
                rhs = rhs / np.prod([index_sizes[i] for i in collapsable_broadcast_indices_rhs])
            return getattr(lhs, expr_json["op"])(rhs)
        elif expr_json["type"] == "leaf":
            # TODO: is the below line necessary?
            var_normshaped = np.einsum(f"{''.join(i for i in expr_json['shape'])}->{''.join(i for i in index_names if i in expr_json['indices'])}", expr_json["value"])
            expand_indices = [n for n, i in enumerate(index_names) if i not in expr_json["indices"]]
            return np.expand_dims(var_normshaped, expand_indices)
                
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        expr, expr_json = _make_random_expr(max_nodes)
        expr = expr[",".join(index_names_out)]
        var = _eval_expr_json(expr_json, set(index_names_out))
        var = np.sum(var, axis=tuple(n for n, i in enumerate(index_names) if i not in index_names_out))
    return RandomExpressionData(expr, expr_json, var)

# @pytest.mark.xfail
@pytest.mark.parametrize("seed", range(1*N_TRIALS_MULTIPLIER))
@pytest.mark.parametrize("np_like", [np])
def test_random_expr(seed, np_like, make_random_expr):
    expr, expr_json, var = make_random_expr
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if not np.allclose(expr.__array__(), var, tolerance) and np.all([~np.isnan(expr.__array__()), ~np.isnan(var)]):
            print(expr)
            pp.pprint(expr_json)
            print(expr.get_shape())
            print(var.shape)
            raise ValueError(f"Values do not match: {expr.__array__()} != {var}")


@pytest.mark.xfail
@pytest.mark.parametrize("seed", range(1*N_TRIALS_MULTIPLIER))
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
    rng = np.random.default_rng(seed=0)
    @jax.jit
    def add(x,y):
        x = einexpr(x)
        y = einexpr(y)
        z = x['i'] + y['i']
        return z[''].__array__()
    x = rng.uniform(size=2)
    y = rng.uniform(size=2)
    add(x,y)

    @jax.jit
    def add_and_reduce(x,y):
        z = x['i'] + y['i']
        return z['']
    x = einexpr(rng.uniform(size=2))['i']
    y = einexpr(rng.uniform(size=2))['i']
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
