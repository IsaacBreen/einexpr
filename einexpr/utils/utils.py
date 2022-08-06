from __future__ import annotations
import collections
from dataclasses import dataclass, field

import functools
import itertools
from collections.abc import Iterable, Iterator
import inspect
import warnings
from collections import Counter
from itertools import chain, combinations, repeat, starmap
from typing import *
import warnings

import numpy as np


T = TypeVar('T')


def enhanced_decorator(decorator_to_enhance):
    @functools.wraps(decorator_to_enhance)
    def enchanced_decorator(*args, **kwargs):
        if len(args) == 1 and (callable(args[0]) or isinstance(args[0], type)) and not kwargs:
            return decorator_to_enhance(args[0])
        elif len(args) == 0:
            def decorator_wrapper(func):
                return decorator_to_enhance(func, **kwargs)
            return decorator_wrapper
        else:
            raise TypeError(
                f"Enhanced decorator takes either a one positional argument (the function or class to decorate) or zero or more keyword arguments. Got positional arguments {args} and keyword arguments {kwargs}.")
    return enchanced_decorator


@enhanced_decorator
def deprecated(item, /, *, message=None):
    """
    Use this decorator to mark functions and classes as deprecated. It will emit a warning upon use of the deprecated item.
    """
    message = "" if message is None else ' ' + message
    if isinstance(item, type):
        # Warn whenever the class is instantiated or subclassed
        class DeprecatedClass(item):
            def __new__(cls, *args, **kwargs):
                warnings.warn(f"Instantiation of deprecated class {item.__name__}." + message, DeprecationWarning)
                return super().__new__(cls)
            
            def __init_subclass__(cls, **kwargs):
                warnings.warn(f"Subclassing of deprecated class {item.__name__}." + message, DeprecationWarning)
                super().__init_subclass__(**kwargs)
                
            def __repr__(self):
                return f"<Deprecated {super().__repr__()}>"
            
            def __str__(self):
                return super().__str__()
                
        # Wrap all instances of classmethod and staticmethod to emit a warning
        # TODO: this doesn't work for some reason... 
        for name, method in inspect.getmembers(item, inspect.isfunction):
            if isinstance(method, classmethod):
                setattr(DeprecatedClass, name, classmethod(deprecated(message=message)(method.__func__)))
            elif isinstance(method, staticmethod):
                setattr(DeprecatedClass, name, staticmethod(deprecated(message=message)(method.__func__)))
            
        DeprecatedClass.__deprecated__ = True
        return DeprecatedClass
    elif callable(item):
        @functools.wraps(item)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn(f"Call to deprecated function {item.__name__}." + message,
                        category=DeprecationWarning,
                        stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return item(*args, **kwargs)
    else:
        raise TypeError(f"deprecated decorator takes either a function or class. Got argument {item} of type {type(item)}.")
    return new_func


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


def deep_str(tree):
    """
    Convert a tree of Python objects to a string.
    """
    return pytree_mapreduce(tree, lambda x: str(x))


def minimalistic_str(sequence, outer_brackets: bool = True):
    """
    Convert a tree of Python objects to a string where, for example, lists and tuples are formatted without commas and strings are formatted with quotes. Items with spaces are still wrapped in quotes.
    """
    if isinstance(sequence, (list, tuple, set)):
        inner_str = ' '.join(minimalistic_str(item, True) for item in sequence)
    elif isinstance(sequence, dict):
        inner_str = ' '.join(f'{minimalistic_str(key, True)}: {minimalistic_str(value, True)}' for key, value in sequence.items())
    else:
        item = str(sequence)
        if ' ' in item:
            # Add quotes around the item and escape quotes inside it
            item = item.replace('"', '\\"')
            item = f'"{item}"'
        return item
    bracket_strs = {
        list: '[{}]',
        tuple: '({})',
        set: '{}',
        dict: '{}',
    }
    return bracket_strs[type(sequence)].format(inner_str) if outer_brackets else inner_str


def pedantic_dict_merge(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries into a single dictionary, raising an error if any duplicate keys have different values.
    """
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result and result[key] != value:
                raise ValueError(f'Duplicate key {key} with different values: {result[key]} and {value}')
            result[key] = value
    return result


@dataclass(frozen=True)
class ExpandSentinel:
    """
    A sentinel object that indicates that an iterable should be expanded into its parent.
    """
    content: Iterable = field(default_factory=list)
    
    def __iter__(self):
        return iter(self.content)
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        return self.content[index]
    
    def __contains__(self, item):
        return item in self.content


@dataclass(frozen=True)
class RemoveSentinel:
    """
    A sentinel object that indicates that an element should be removed from its parent.
    """
    pass


def pytree_mapreduce(tree, map_func=lambda x: x, reduce_func=lambda x: x):
    """
    Mapreduce a tree of Python objects.
    """
    if isinstance(tree, list):
        return reduce_func(list(_item for item in tree if not isinstance(_item := pytree_mapreduce(item, map_func, reduce_func), RemoveSentinel)))
    elif isinstance(tree, tuple):
        return reduce_func(tuple(_item for item in tree if not isinstance(_item := pytree_mapreduce(item, map_func, reduce_func), RemoveSentinel)))
    elif isinstance(tree, dict):
        return reduce_func({key: _value for key, value in tree.items() if not isinstance(_value := pytree_mapreduce(value, map_func, reduce_func), RemoveSentinel)})
    elif isinstance(tree, set):
        return reduce_func({_item for item in tree if not isinstance(_item := pytree_mapreduce(item, map_func, reduce_func), RemoveSentinel)})
    else:
        return map_func(tree)


def deep_remove(tree: T, match_func: Callable[Any, bool]) -> T:
    """
    Remove elements from a tree of Python objects.
    """
    def remover(item):
        return RemoveSentinel() if match_func(item) else item
    return pytree_mapreduce(tree, remover, remover)


def get_position(tree, item):
    if isinstance(tree, collections.abc.Sequence) and not isinstance(tree, str):
        for i, subtree in enumerate(tree):
            if subtree == item:
                return (i,)
            elif (child_position := get_position(subtree, item)) is not None:
                return (i,) + child_position
    return None


def split_at(tree: Sequence, position: int):
    if len(position) == 1:
        return tree[:position[0]], tree[position[0]+1:]
    elif isinstance(tree, list):
        lhs, rhs = split_at(tree[position[0]], position[1:])
        return [*tree[:position[0]], lhs], [rhs, *tree[position[0]+1:]]
    elif isinstance(tree, tuple):
        lhs, rhs = split_at(tree[position[0]], position[1:])
        return (*tree[:position[0]], lhs), (rhs, *tree[position[0]+1:])
    else:
        raise TypeError(f'Cannot split {type(tree)} at {position}')

def deep_iter(iterable: Iterable, /, *, dict_mode: Literal['keys', 'values', 'both'] = 'values', yield_iterables: bool = False) -> Iterator:
    """
    Iterate over a tree of Python objects.
    """
    if yield_iterables:
        yield iterable
    if isinstance(iterable, dict):
        if dict_mode in ('keys', 'both'):
            yield from deep_iter(iterable.keys(), dict_mode=dict_mode, yield_iterables=yield_iterables)
        if dict_mode in ('values', 'both'):
            yield from deep_iter(iterable.values(), dict_mode=dict_mode, yield_iterables=yield_iterables)
    elif isinstance(iterable, Iterable):
        for item in iterable:
            if item is iterable:
                # Avoid infinite recursion for iterables that yield themselves (e.g. strings of length 1) by skipping the item
                pass
            else:
                if isinstance(item, Iterable):
                    yield from deep_iter(item, dict_mode=dict_mode, yield_iterables=yield_iterables)
                else:
                    yield item


def tree_contains(tree, item):
    """
    Check if an item is in a tree of Python objects.
    """
    if tree == item:
        return True
    for _item in deep_iter(tree, yield_iterables=True):
        if _item == item:
            return True
    return False


def deprecated_guard(func):
    """
    Validate that the inputs and outputs of a function are not deprecated.
    """
    @functools.wraps(func)
    def deprecated_guard_wrapper(*args, **kwargs):
        for arg in deep_iter(args, dict_mode='both', yield_iterables=True):
            if hasattr(arg, '__deprecated__') and arg.__deprecated__:
                raise ValueError(f'Argument {arg} is deprecated')
        for kw, arg in kwargs.items():
            if hasattr(arg, '__deprecated__') and arg.__deprecated__:
                raise ValueError(f'Keyword argument {kw}={arg} is deprecated')
        output = func(*args, **kwargs)
        if isinstance(output, Iterator):
            output, output_tee = itertools.tee(output)
        else:
            output_tee = output
        for output_item in deep_iter(output_tee, dict_mode='both', yield_iterables=True):
            if hasattr(output_item, '__deprecated__') and output_item.__deprecated__:
                raise ValueError(f'Output {output_item} is deprecated')
        return output
    return deprecated_guard_wrapper


class EmptySentinel:
    """
    A sentinel object that indicates that an iterable is empty.
    """
    pass


def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))