from __future__ import annotations

import functools
import inspect
import warnings
from collections import Counter
from itertools import chain, combinations
from typing import *

import numpy as np


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def enhanced_decorator(decorator_to_enhance):
    def enchanced_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and len(kwargs) == 0:
            return decorator_to_enhance(args[0])
        elif len(args) == 0:
            def decorator_wrapper(func):
                return decorator_to_enhance(func, **kwargs)
            return decorator_wrapper
        else:
            raise TypeError(
                f"decorator_maker() takes either a one positional argument (the function to decorate) or zero or more keyword arguments (got positional arguments {args} and keyword arguments {kwargs}")
    return enchanced_decorator


def powerset(iterable, reverse: bool = False) -> Iterator[Set[Any]]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    rng = range(len(s)+1)
    if reverse:
        rng = reversed(rng)
    return chain.from_iterable(combinations(s, r) for r in rng)


def assert_outputs(assertion: Callable[Any, bool], make_message=None):
    def wrapper(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            result = func(*args, **kwargs)
            if make_message:
                assert assertion(result), make_message(result)
            else:
                assert assertion(
                    result), f"Output assertion failed. Result: {result}. Assertion: {inspect.getsource(assertion)}Call: {func.__name__}({','.join(map(repr, args))}, {','.join(f'{k}={v!r}' for k,v in kwargs.items())})."
            return result
        return wrapped_func
    return wrapper


# Function to return all SCS of substrings `X[0…m-1]`, `Y[0…n-1]`
def SCS(X, Y, m, n, lookup):
 
    # if the end of the first string is reached, create a list
    # containing the second substring and return
    if m == 0:
        return {Y[:n]}
 
    # if the end of the second string is reached, create a list
    # containing the first substring and return
    elif n == 0:
        return {X[:m]}
 
    # if the last character of `X` and `Y` is the same, it should occur
    # only one time in SSC
    if X[m - 1] == Y[n - 1]:
 
        # find all SCS of substring `X[0…m-2]` and `Y[0…n-2]`
        # and append the current character `X[m-1]` or `Y[n-1]` to all SCS of
        # substring `X[0…m-2]` and `Y[0…n-2]`
 
        scs = SCS(X, Y, m - 1, n - 1, lookup)
        return {(*s, X[m - 1]) for s in scs}
 
    # we reach here when the last character of `X` and `Y` don't match
 
    # if a top cell of the current cell has less value than the left cell,
    # then append the current character of string `X` to all SCS of
    # substring `X[0…m-2]`, `Y[0…n-1]`
 
    if lookup[m - 1][n] < lookup[m][n - 1]:
        scs = SCS(X, Y, m - 1, n, lookup)
        return {(*s, X[m - 1]) for s in scs}
 
    # if a left cell of the current cell has less value than the top cell,
    # then append the current character of string `Y` to all SCS of
    # substring `X[0…m-1]`, `Y[0…n-2]`
 
    if lookup[m][n - 1] < lookup[m - 1][n]:
        scs = SCS(X, Y, m, n - 1, lookup)
        return {(*s, Y[n - 1]) for s in scs}
 
    # if the top cell value is the same as the left cell, then go in both
    # top and left directions
 
    # append the current character of string `X` to all SCS of
    # substring `X[0…m-2]`, `Y[0…n-1]`
    top = SCS(X, Y, m - 1, n, lookup)
 
    # append the current character of string `Y` to all SCS of
    # substring `X[0…m-1]`, `Y[0…n-2]`
    left = SCS(X, Y, m, n - 1, lookup)
 
    return set([(*s, X[m - 1]) for s in top] + [(*s, Y[n - 1]) for s in left])
 
 
# Function to fill the lookup table by finding the length of SCS of
# sequences `X[0…m-1]` and `Y[0…n-1]`
def SCSLength(X, Y, m, n, lookup):
 
    # initialize the first column of the lookup table
    for i in range(m + 1):
        lookup[i][0] = i
 
    # initialize the first row of the lookup table
    for j in range(n + 1):
        lookup[0][j] = j
 
    # fill the lookup table in a bottom-up manner
    for i in range(1, m + 1):
 
        for j in range(1, n + 1):
            # if the current character of `X` and `Y` matches
            if X[i - 1] == Y[j - 1]:
                lookup[i][j] = lookup[i - 1][j - 1] + 1
            # otherwise, if the current character of `X` and `Y` don't match
            else:
                lookup[i][j] = min(lookup[i - 1][j] + 1, lookup[i][j - 1] + 1)
 
 
# Function to find all shortest common supersequence of string `X` and `Y`
def get_all_scs_with_unique_elems_pairwise(X, Y):
 
    m = len(X)
    n = len(Y)
 
    # lookup[i][j] stores the length of SCS of substring `X[0…i-1]` and `Y[0…j-1]`
    lookup = [[0 for x in range(n + 1)] for y in range(m + 1)]
 
    # fill lookup table
    SCSLength(X, Y, m, n, lookup)

    # find and return all shortest common supersequence
    return SCS(X, Y, m, n, lookup)
 

def get_all_scs_with_unique_elems(*Xs) -> Set:
    """
    Filters out all SCSs that contain duplicate elements.
    """
    Xs = [tuple(x) if isinstance(x, list) else x for x in Xs]
    SCS = {Xs[0]}
    for X in Xs[1:]:
        SCS = {new_substr for current_substr in SCS for new_substr in get_all_scs_with_unique_elems_pairwise(current_substr, X)}
        n_unique = len(set(next(iter(SCS))))
        SCS = {substr for substr in SCS if len(substr) == n_unique}
    return SCS


def remove_duplicates(lst):
    """
    Removes duplicate elements from an iterable.
    """
    lst2 = []
    for x in lst:
        if x not in lst2:
            lst2.append(x)
    return tuple(lst2)


def get_all_scs_with_nonunique_elems_removed(*Xs) -> Set:
    """
    Removes duplicate elements from SCSs.
    """
    Xs = [tuple(x) if isinstance(x, list) else x for x in Xs]
    SCS = {Xs[0]}
    for X in Xs[1:]:
        SCS = {remove_duplicates(new_substr) for current_substr in SCS for new_substr in get_all_scs_with_unique_elems_pairwise(current_substr, X)}
    return SCS


def list_to_english(lst: List) -> str:
    lst = list(lst)
    match len(lst):
        case 0:
            return ''
        case 1:
            return str(lst[0])
        case default:
            return ', '.join(str(item) for item in lst[:-1]) + ' and ' + str(lst[-1])


def pytree_mapreduce(tree, map_func, reduce_func=lambda x: x):
    """
    Mapreduce a tree of Python objects.
    """
    if isinstance(tree, list):
        return reduce_func([pytree_mapreduce(item, map_func, reduce_func) for item in tree])
    elif isinstance(tree, tuple):
        return reduce_func(tuple(pytree_mapreduce(item, map_func, reduce_func) for item in tree))
    elif isinstance(tree, dict):
        return reduce_func({key: pytree_mapreduce(value, map_func, reduce_func) for key, value in tree.items()})
    elif isinstance(tree, set):
        return reduce_func({pytree_mapreduce(item, map_func, reduce_func) for item in tree})
    else:
        return map_func(tree)