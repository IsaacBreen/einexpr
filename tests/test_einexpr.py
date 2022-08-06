import warnings
from argparse import ArgumentError

warnings.filterwarnings("ignore", category=UserWarning, message="JAX on Mac ARM machines is experimental and minimally tested.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="the imp module is deprecated in favour of importlib and slated for removal in .*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered.*")

import pprint
import string
from collections import namedtuple
from typing import List

import einexpr
import numpy as np
import numpy.array_api as npa
import pytest

pp = pprint.PrettyPrinter(indent=4)

N_TRIALS_MULTIPLIER = 10
TOLERANCE = 1e-12
DEFAULT_DTYPE = npa.float64


class WrappedFunction:
    def __init__(self, name, func, nparams: int = None):
        self.name = name
        self.func = func
        self.__name__ = name
        self.nparams = nparams
        
    def __call__(self, *args, **kwargs):
        if self.nparams is not None and len(args) != self.nparams:
            raise ArgumentError(None, f"{self} requires {self.nparams} arguments but got {len(args)}")
        return self.func(*args, **kwargs)
    
    def __repr__(self):
        return f"WrappedFunction({self.name})"
    
    def __str__(self):
        return WrappedFunction.__repr__(self)


class MagicCaller:
    """
    Represents a call to a magic method of the first argument.
    """
    def __init__(self, name: str, nparams: int = None):
        self.name = name
        self.__name__ = name
        self.nparams = nparams
    
    def __call__(self, *args, **kwargs):
        if self.nparams is not None and len(args) != self.nparams:
            raise ArgumentError(None, f"{self} requires {self.nparams} arguments but got {len(args)}")
        return getattr(args[0], self.name)(*args[1:], **kwargs)
    
    def __repr__(self):
        return f"MagicCaller({self.name})"
    
    def __str__(self):
        return f"MagicCaller({self.name})"


class ArrayNamespaceCaller:
    """
    Represents a call to a method in the array namespace of the first argument.
    """
    def __init__(self, name: str, nparams: int = None):
        self.name = name
        self.__name__ = name
        self.nparams = nparams
    
    def __call__(self, *args, **kwargs):
        if self.nparams is not None and len(args) != self.nparams:
            raise ArgumentError(None, "{self} requires {self.nparams} arguments but got {len(args)}")
        return getattr(args[0].__array_namespace__(), self.name)(*args, **kwargs)
    
    def __repr__(self):
        return f"ArrayNamespaceCaller({self.name})"
    
    def __str__(self):
        return f"ArrayNamespaceCaller({self.name})"


unary_arithmetic_magics_str = {
    "__neg__",
    "__pos__",
    "__invert__"
}

binary_magic_strs = dict(
    multiply = {"__mul__", "__truediv__"},
    add = {"__add__", "__sub__"},
    power = {"__pow__"},
)

binary_magics = {key: {MagicCaller(op) for op in ops} for key, ops in binary_magic_strs.items()}
# Soft clip exponents between 1/3 and 3 + 1/3
binary_magics['__pow__'] = {
    WrappedFunction("Soft clip", lambda x, y: MagicCaller('__pow__')(x, ArrayNamespaceCaller('abs')(3/(1+npa.e**-y) + 3/10)))
}

unary_elementwise_array_namespace_op_strs = """
    abs acos acosh asin asinh atan atanh ceil cos cosh exp expm1 floor log log1p
    log2 log10 negative positive round sign sin sinh square sqrt tan tanh trunc
    """.split()

binary_elementwise_array_namespace_op_strs = """
    add atan2 divide floor_divide  logaddexp multiply pow remainder subtract
    """.split()

# Arguments with non-float argument dtype reqiurements that we've ignored for now:
# isfinite isinf isnan
# bitwise_and bitwise_left_shift bitwise_invert bitwise_or bitwise_right_shift bitwise_xor 
# logical_not logical_and logical_or logical_xor
# greater greater_equal less less_equal equal not_equal

default_unary_ops = set()
default_unary_ops |= {ArrayNamespaceCaller(op) for op in unary_elementwise_array_namespace_op_strs}

default_binary_ops = set()
default_binary_ops |= {op for ops in binary_magics.values() for op in ops}
default_binary_ops |= {ArrayNamespaceCaller(op) for op in binary_elementwise_array_namespace_op_strs}

RandomExpressionData = namedtuple("RandomExpressionData", ["expr", "expr_json", "var"])

@pytest.fixture
def X():
    return npa.asarray([[1, 2], [3, 4]])
    
@pytest.fixture
def Y():
    return npa.asarray([[5, 6, 7], [7, 8, 9]])
    
@pytest.fixture
def x():
    return npa.asarray([1,2])
    
@pytest.fixture
def y():
    return npa.asarray([3,4,5])

@pytest.mark.skip
def test_pow():
    rng = np.random.default_rng(2022)
    w = einexpr.einarray(rng.uniform(size=(4,2)), dims='b,i')
    x = einexpr.einarray(rng.uniform(size=2), dims='i')
    expr = w['b,i'] * x['i']
    expr = expr['']
    print(expr)
    assert np.allclose(expr.__array__(), np.einsum('bi,i->', w.__array__(), x.__array__()), TOLERANCE)

@pytest.mark.skip
def test_simple_expr1(X, Y, x, y):
    Xe = einexpr.einarray(X, dims='i j')
    Ye = einexpr.einarray(Y, dims='j k')
    xe = einexpr.einarray(x, dims='i')
    ye = einexpr.einarray(y, dims='j')
    
    Ye['j']

    # MULTIPLICATION
    assert np.allclose(np.einsum('ij,jk->ik', X, Y), (Xe['i j'] * Ye['j k'])['i k'].a, TOLERANCE)

    # ADDITION
    assert np.allclose(npa.sum(X[:, :, npa.newaxis] + Y[npa.newaxis, :, :], axis=1), (Xe['i j'] + Ye['j k'])['i k'].a, TOLERANCE)

    # LINEAR TRANSFORMATION
    def linear(x, W, b):
        return np.einsum("i,ij->j", x, W) + b

    # def linear_ein(x, W, b):
    #     return x['i'] * W['i j'] + b['j']

    # assert np.allclose(linear(x, Y, y), linear_ein(x, Y, y)['j'], TOLERANCE)


@pytest.mark.skip
def test_commonly_failed1(X, Y, x, y):    
    Xe = einexpr.einarray(X, dims='i j')
    xe = einexpr.einarray(x, dims='i')
    ye = einexpr.einarray(x, dims='j')
    
    ze = xe['i'] ** (Xe['j i'] ** xe['i'])
    assert np.allclose(ze['j i'].__array__(), x ** (X ** x), TOLERANCE)
    
    z = xe['i'] ** (xe['j'] + xe['j'])
    assert np.allclose(z['i j'].__array__(), x[:, None] ** (x[None, :] + x[None, :]), TOLERANCE)
    print(z.coerce_into_shape('i'))
    assert np.allclose(z['i'].a, npa.sum(x[:, None] ** (x[None, :] + x[None, :]), axis=1), TOLERANCE)

@pytest.mark.skip
def test_numpy_ufunc_override1(X, Y, x, y):
    Xe = einexpr.einarray(X, dims='i j')
    xe = einexpr.einarray(x, dims='j')

    Z = X ** npa.abs(-x)
    Ze = Xe['i j'] ** npa.abs(-xe['j'])
    assert np.allclose(Z, Ze['i j'].__array__(), TOLERANCE)

    Z = X ** npa.abs(-x)
    Ze = Xe['i j'] ** npa.abs(-xe['j'])
    assert np.allclose(Z, Ze['i j'].__array__(), TOLERANCE)


@pytest.fixture
def random_expr_json(
    seed,
    unary_ops=default_unary_ops,
    binary_ops=default_binary_ops,
    max_indices=8,
    max_indices_per_var=4,
    max_index_size=7,
    max_vars=8,
    E_num_nodes=2,
    max_nodes=10,
    p_binary_op_given_nonleaf=0.9,
    low=1,
    high=100,
    max_exponent=3,
    softmax=10
):
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
    max_indices_per_var = min(max_indices_per_var, num_indices)
    index_names = list(string.ascii_lowercase[:num_indices])
    index_sizes = {i: rng.integers(1, max_index_size) for i in index_names}

    n_vars = rng.integers(2, max_vars)
    per_var_indices = [list(rng.choice(index_names, size=rng.integers(1, max_indices_per_var), replace=False)) for _ in range(n_vars)]
    vars = [rng.integers(low, high, size=[index_sizes[i] for i in var_indices]) for var_indices in per_var_indices]
    # vars = [rng.uniform(low, high, [index_sizes[i] for i in var_indices]) for var_indices in per_var_indices]
    vars = [npa.asarray(v, dtype=DEFAULT_DTYPE) for v in vars]
    
    def _make_random_expr_json(max_nodes):
        if max_nodes >= 2:
            node_type = rng.choice(["leaf", "unary_op", "binary_op"], p=[p_leaf, p_unary_op, p_binary_op])
        elif max_nodes == 1:
            p_norm = p_leaf + p_unary_op
            node_type = rng.choice(["leaf", "unary_op"], p=[p_leaf/p_norm, p_unary_op/p_norm])
        else:
            node_type = "leaf"
        if node_type == "leaf":
            var_num = rng.integers(0, n_vars)
            return {"type": "leaf", "value": vars[var_num], "indices": set(per_var_indices[var_num]), "shape": per_var_indices[var_num], "num_nodes": 1}
        elif node_type == "unary_op":
            op = rng.choice(sorted(unary_ops, key=id))
            expr_json = _make_random_expr_json(max_nodes)
            return {"type": "unary_op", "op": op, "indices": expr_json["indices"], "operand": expr_json, "num_nodes": expr_json["num_nodes"]}
        elif node_type == "binary_op":
            op = rng.choice(sorted(binary_ops, key=id))
            expr_json_lhs = _make_random_expr_json(max_nodes)
            expr_json_rhs = _make_random_expr_json(max_nodes - expr_json_lhs["num_nodes"])
            indices = expr_json_lhs["indices"] | expr_json_rhs["indices"]
            expr_json = {
                "type": "binary_op",
                "op": op,
                "lhs": expr_json_lhs,
                "rhs": expr_json_rhs,
                "indices": indices,
                "num_nodes": 1 + expr_json_lhs["num_nodes"] + expr_json_rhs["num_nodes"]
            }
            return expr_json

    expr_json = _make_random_expr_json(max_nodes)
    index_names_used = list(expr_json["indices"])
    num_indices_out = rng.integers(0, len(index_names_used))
    index_names_out = list(rng.choice(index_names_used, size=num_indices_out, replace=False))
    def get_leaves(x):
        if x["type"] == "leaf":
            return [x]
        elif x["type"] == "unary_op":
            return get_leaves(x["operand"])
        elif x["type"] == "binary_op":
            return get_leaves(x["lhs"]) + get_leaves(x["rhs"])
        else:
            raise ValueError(f"Unknown node type: {x['type']}")
    return {"expr_json": expr_json, "dims": index_names_used, "out_dims": tuple(index_names_out), "dim_sizes": index_sizes, 'leaves': get_leaves(expr_json)}


def json_eval(expr_json, non_collapsable_indices, index_names: List[str], index_sizes):
    # In additive subexpressions, divide each term by the size of those indices what are about to be broadcasted over and which are also in non_collapsable_indices.
    # Another explaination: need to divide by size of indices that are about to be broadcast but which aren't subsequently collapsed over (those that appear in 
    # non_collapsable_indices).
    if expr_json["type"] == "leaf":
        # TODO: is the below line necessary?
        var_normshaped = np.einsum(f"{''.join(i for i in expr_json['shape'])}->{''.join(i for i in index_names if i in expr_json['indices'])}", expr_json["value"])
        expand_indices = [n for n, i in enumerate(index_names) if i not in expr_json["indices"]]
        return npa.expand_dims(npa.asarray(var_normshaped), axis=expand_indices)
    elif expr_json["type"] == "unary_op":
        val = expr_json["op"](json_eval(expr_json["operand"], non_collapsable_indices, index_names, index_sizes))
        expr_json['value'] = val
        return npa.asarray(val)
    elif expr_json["type"] == "binary_op":
        child_non_collapsable_indices = non_collapsable_indices.copy()
        lhs = json_eval(expr_json["lhs"], child_non_collapsable_indices, index_names, index_sizes)
        rhs = json_eval(expr_json["rhs"], child_non_collapsable_indices, index_names, index_sizes)
        rhs = rhs.__array_namespace__().astype(rhs, lhs.dtype)
        val = expr_json["op"](lhs, rhs)
        expr_json['value'] = val
        return npa.asarray(val)
    else:
        raise ValueError(f"Unknown expression type: {expr_json['type']}")


@pytest.fixture
def random_expr_value(random_expr_json):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        try:
            val = json_eval(random_expr_json["expr_json"], set(random_expr_json["out_dims"]), random_expr_json["dims"], random_expr_json["dim_sizes"])
            val = npa.sum(val, axis=tuple(n for n, i in enumerate(random_expr_json["dims"]) if i not in random_expr_json["out_dims"]))
            dims_after_sum = [i for i in random_expr_json["dims"] if i in random_expr_json["out_dims"]]
            val = npa.permute_dims(val, tuple(dims_after_sum.index(dim) for dim in random_expr_json["out_dims"]))
        except Exception as e:
            print(expr_json_to_str(random_expr_json))
            raise e
    return val


def json_to_einexpr(expr_maker, expr_json):
    if expr_json["type"] == "leaf":
        return expr_maker(expr_json["value"], dims=expr_json["shape"])
    elif expr_json["type"] == "unary_op":
        operand = json_to_einexpr(expr_maker, expr_json["operand"])
        return expr_json['op'](operand)
    elif expr_json["type"] == "binary_op":
        lhs = json_to_einexpr(expr_maker, expr_json["lhs"])
        rhs = json_to_einexpr(expr_maker, expr_json["rhs"])
        rhs = rhs.__array_namespace__().astype(rhs, lhs.dtype)
        return expr_json['op'](lhs, rhs)
    else:
        raise ValueError(f"Unknown expression type: {expr_json['type']}")


@pytest.fixture
def random_einexpr(expr_maker, random_expr_json):
    try:
        expr = json_to_einexpr(expr_maker, random_expr_json["expr_json"])
    except Exception as e:
        print(expr_json_to_str(random_expr_json))
        raise e
    out = expr[random_expr_json["out_dims"]]
    if hasattr(out, 'dims') and out.dims != tuple(random_expr_json["out_dims"]):
        raise ValueError("dims and out_dims are not compatible")
    return out


@pytest.mark.skip
@pytest.mark.parametrize("seed", [0, 1])
def test_myexpr(random_expr_value):
    print(random_expr_json)


@pytest.mark.skip
def test_expr2():
    x = einexpr.einarray(npa.asarray([1, 2, 3]), dims=['i'])
    y = einexpr.einarray(npa.asarray([4, 5, 6]), dims=['i'])
    x = einexpr.einarray(npa.asarray([[1, 2, 3]]), dims=['j', 'i'])
    y = einexpr.einarray(npa.asarray([[4], [5], [6]]), dims=['i', 'k'])

    print(x+y)
    print(x*y)
    print(npa.matmul(x, y))
    print((x+y).coerce([], set()))
    print((x+y)[''])
    assert len((x+y)[''].dims) == 0


def expr_json_to_str(expr_json):
    names = iter(string.ascii_letters)
    var_names = {id(v['value']): next(names) for v in expr_json['leaves']}
    def helper(j):
        if j["type"] == "leaf":
            return str(var_names[id(j["value"])])
        elif j["type"] == "unary_op":
            op_name = j['op'].__name__ + (f"[{j['op_name']}]" if 'op_name' in j else '')
            return f"{op_name}({helper(j['operand'])})"
        elif j["type"] == "binary_op":
            op_name = j['op'].__name__ + (f"[{j['op_name']}]" if 'op_name' in j else '')
            return f"{op_name}({helper(j['lhs'])}, {helper(j['rhs'])})"
        else:
            raise ValueError(f"Unknown expression type: {j['type']}")
    return helper(expr_json['expr_json']) + ' where ' + ', '.join(f"{var_names[id(v['value'])]}[{','.join(v['indices'])}] = {v['value']}" for v in expr_json['leaves'])


# @pytest.mark.skip
@pytest.mark.parametrize("seed", range(1*N_TRIALS_MULTIPLIER))
@pytest.mark.parametrize("expr_maker", [einexpr.einarray])
def test_random_expr(seed, random_expr_json, random_expr_value, random_einexpr):
    val = random_expr_value
    expr = random_einexpr
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if expr.__array__().shape != val.shape:
            raise ValueError(f"Shape mismatch: {expr.__array__().shape} != {val.shape}")
        if not np.allclose(expr.__array__(), val, TOLERANCE) and npa.all(~npa.isnan(expr.__array__())) and npa.all(~npa.isnan(val)):
            print(f"val: {val}")
            print(f"expr: {expr}")
            # print(f"expr_json: {pp.pformat(random_expr_json)}")
            print(f"expr_str: {expr_json_to_str(random_expr_json)}")
            if val.shape == ():
                raise ValueError(f"Values do not match: {expr.__array__()} != {val}")


@pytest.mark.skip
def test_list_to_einarray():
    x = einexpr.einarray([1,2,3], dims='i')
    y = x+x
    y.dims


@pytest.mark.skip
def test_ambiguous_matmul():
    a = einexpr.einarray([1,2,3], dims=['i'])
    b = einexpr.einarray([4,5,6,7], dims=['j'])
    c = einexpr.einarray([8,9,10,11], dims=['j'])
    X = a*b
    try:
        Y = npa.matmul(X, a)
    except einexpr.AmbiguousDimensionException:
        pass
    else:
        raise Exception("Expected ValueError for ambiguous dims")
    try:
        Y = npa.matmul(a, X)
    except einexpr.AmbiguousDimensionException:
        pass
    else:
        raise Exception("Expected ValueError for ambiguous dims")


@pytest.mark.skip
def test_concatenate(X, Y, x, y):
    X = einexpr.einarray(X, dims='i j')
    Y = einexpr.einarray(Y, dims='i k')
    x = einexpr.einarray(x, dims='i')
    y = einexpr.einarray(y, dims='k')
    
    z = X*x
    z.a
    assert z.dims == ('i', 'j')
    
    # Concatenate X to itself along the first dimension
    z = npa.concatenate([X, X], axis=0)
    
    # Concatenate X to Y
    z = npa.concatenate([X, Y], axis=1)
    
    # mean
    z0 = npa.mean(X, axis=0)
    z1 = npa.mean(X, axis='i')
    z2 = npa.mean(X, axis=['i'])
    assert npa.all(z0 == z1)
    assert npa.all(z0 == z2)


@pytest.mark.skip
def test_commonly_failed_2():
    y = einexpr.einarray([29., 38.], dims='i')
    
    def f(y):
        return 1/(1+npa.e**-(getattr(y, '__truediv__')(y)))
    z = f(y)
    assert all(z['i'].a == f(y.a))


@pytest.mark.skip
def test_commonly_failed_3():
    y = einexpr.einarray([[190.,156.],[58.,64.]], dims='i j')
    
    def f(y):
        return 1/(1+npa.e**-y)
    z = f(y)
    assert z[''].a == npa.sum(f(y.a))


def test_full_reduction(X):
    X = einexpr.einarray(X, dims='i j')
    assert X[''].dims == ()
    assert X[''].shape == ()

# @pytest.mark.skip
def test_reshape():
    x = einexpr.einarray([[1,2,3],[4,5,6]], dims='i j')
    assert x.shape == (2,3)
    assert x['(i->n) j'].shape == (2,3)
    assert x['(i j)'].shape == (6,)
    assert x['(i j) -> n'].shape == (6,)
    assert x['j i'].shape == (3,2)
    assert x['(j->m) (i->n)'].shape == (3,2)


def test_named_axis(X):
    X = einexpr.einarray(X, dims='i j')
    assert einexpr.sum(X, axis='i').dims == einexpr.sum(X, axis=0).dims == ('j',)
    assert einexpr.sum(X, axis='j').dims == einexpr.sum(X, axis=1).dims == ('i',)
    
    Y = X['(i j)']
    assert Y.dims == (('i', 'j'),)
    assert einexpr.sum(Y, axis='i').dims == (('j',),)

    
def test_ellipsis(X):
    X = einexpr.einarray(X, dims='i j')
    X['...']
    X['i ...']
    X['... j']
    X['j ...']
    X['... i']
    
    einexpr.ones((1,2,3))['i j k']['i j ...']


def test_dim_binding(X):
    X = einexpr.einarray(X)
    assert X['i j'].dims == ('i', 'j')
    
    X = einexpr.ones((1,2,3))
    X['_ _ _']
    X['i _ _']
    X['_ j _']
    X['_ _ k']
    X['i j _']
    X['i _ k']
    X['_ j k']
    X['i j k']


@pytest.mark.skip
def test_dim_errors(X):
    X = einexpr.einarray(X)


def test_positional_dims(X):
    X = einexpr.einarray(X)
    Y = einexpr.einarray(X)
    Z = einexpr.einarray(X, dims='i j')
    
    
    assert len(R := (X*X).dims) == 2 and all(isinstance(dim, einexpr.dimension.AbsorbingDimension) for dim in R)
    assert len(R := (X*Y).dims) == 2 and all(isinstance(dim, einexpr.dimension.AbsorbingDimension) for dim in R)
    assert (X*Z).dims == ('i', 'j')


# TODO: This is a terrible hack and should be fixed. To see why, try the following:
# print(AbsorbingDimensionMatcher() == einexpr.array_api.dimension.AbsorbingDimension())  # True
# print(einexpr.array_api.dimension.AbsorbingDimension() == AbsorbingDimensionMatcher())  # False
class AbsorbingDimensionMatcher:
    def __eq__(self, other):
        return isinstance(other, einexpr.AbsorbingDimension)


def test_positional_dims_and_rename(X):
    X = einexpr.einarray(X)

    assert ('i', AbsorbingDimensionMatcher()) == X['i _'].dims
    assert (AbsorbingDimensionMatcher(), 'n') == X['_ _->n'].dims
    assert ('i', 'n') == X['_->i j->n'].dims


def test_positional_dims_and_ellipsis_and_rename(X):
    X = einexpr.einarray(X)

    assert (AbsorbingDimensionMatcher(), 'n') == X['... _->n'].dims


def test_first_class_dims(X):
    X = einexpr.einarray(X)
    Y = einexpr.einarray(X)
    i, j = einexpr.dims(2)
    assert X[i, j].dims == (i, j)
    assert X[i, ...].dims[0] == i
    assert X[..., j].dims[1] == j


def test_tricky_reshapes_l1():
    x = einexpr.ones((2,3,4), dims='_ _ k')
    
    assert ('k', 'i', AbsorbingDimensionMatcher()) == x['k i _'].dims
    assert ('k', 'i', AbsorbingDimensionMatcher()) == x['k i ...'].dims
    
    assert ('i', 'j', 'k') == x['i j _'].dims
    assert ('i', 'j', 'k') == x['i j ...'].dims
    
    assert ('i', AbsorbingDimensionMatcher(), 'k') == x['i _ _'].dims
    assert ('i', AbsorbingDimensionMatcher(), 'k') == x['i ... _'].dims
    assert ('i', 'k', 'j') == x['i ... j'].dims

    assert (AbsorbingDimensionMatcher(), 'k', 'i') == x['... i'].dims
    assert ('i', AbsorbingDimensionMatcher(), 'k') == x['i ...'].dims


def test_tricky_reshapes_l2():
    x = einexpr.ones((2,3,4), dims='i j k')
    x = x['(i j k)']
    assert ('i', ('j', 'k')) == x['i ...'].dims
    assert ('j', ('i', 'k')) == x['j ...'].dims
    assert ('k', ('i', 'j')) == x['k ...'].dims
    
    assert (('j', 'k'), 'i') == x['... i'].dims
    assert (('i', 'k'), 'j') == x['... j'].dims
    assert (('i', 'j'), 'k') == x['... k'].dims
    
    assert ('k', ('j',), 'i') == x['k ... i'].dims
    assert ('k', ('i',), 'j') == x['k ... j'].dims
    assert ('i', ('k',), 'j') == x['i ... j'].dims
    
    x = einexpr.ones((2,3,4,5,6,7,8,9,10), dims='i j k l m n o p q')
    x = x['i j k (l m n) o p q']


def test_tutorial():
    # ---------------------------------- PART 1 ---------------------------------- #
    import einexpr as ei
    import numpy as np

    # Define some vectors
    a = ei.array([1,2])
    b = ei.array([3,4])
    # (So far, pretty normal...)

    # Dot product
    x = a['i'] * b['i']
    assert np.dot(a, b) == x['']

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


    # ---------------------------------- PART 2 ---------------------------------- #
    i, j = ei.dims(2) # Unnamed object
    k, l = ei.dims('k l') # Named objects

    # Matrix-matrix multiplication
    x = X[i, j] * Y[j, k]
    assert np.all(np.matmul(X, Y) == x[i, k])