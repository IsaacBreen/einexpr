import warnings
import functools
from typing import Callable, Iterable, Iterator, Sequence, Set, Any, TypeVar, Union
from itertools import chain, combinations
import inspect
from __future__ import annotations


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


def lcs(a, b):
    # generate matrix of length of longest common subsequence for substrings of both words
    lengths = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

    # read a substring from the matrix
    result = ''
    j = len(b)
    for i in range(1, len(a)+1):
        if lengths[i][j] != lengths[i-1][j]:
            result += a[i-1]

    return result


T = TypeVar('T')


def shortest_common_supersequence_pairwise(seq1: Sequence[T], seq2: Sequence[T]) -> Sequence[T]:
    """
    From https://github.com/kamyu104/LeetCode-Solutions/blob/master/Python/shortest-common-supersequence.py
    """
    dp = [[0 for _ in range(len(seq2)+1)] for _ in range(2)]
    bt = [[None for _ in range(len(seq2)+1)] for _ in range(len(seq1)+1)]
    for i, c in enumerate(seq1):
        bt[i+1][0] = (i, 0, c)
    for j, c in enumerate(seq2):
        bt[0][j+1] = (0, j, c)
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if dp[i % 2][j+1] > dp[(i+1) % 2][j]:
                dp[(i+1) % 2][j+1] = dp[i % 2][j+1]
                bt[i+1][j+1] = (i, j+1, seq1[i])
            else:
                dp[(i+1) % 2][j+1] = dp[(i+1) % 2][j]
                bt[i+1][j+1] = (i+1, j, seq2[j])
            if seq1[i] != seq2[j]:
                continue
            if dp[i % 2][j]+1 > dp[(i+1) % 2][j+1]:
                dp[(i+1) % 2][j+1] = dp[i % 2][j]+1
                bt[i+1][j+1] = (i, j, seq1[i])
    i, j = len(seq1), len(seq2)
    result = []
    while i != 0 or j != 0:
        i, j, c = bt[i][j]
        result.append(c)
    result.reverse()
    if isinstance(seq1, str) and isinstance(seq2, str):
        return ''.join(result)
    else:
        return result


def shortest_common_supersequence(seqs: Sequence[Sequence[T]]) -> Sequence[T]:
    # TODO: not sure if this is actually valid.
    superseq = seqs[0]
    for seq in seqs[1:]:
        superseq = shortest_common_supersequence_pairwise(superseq, seq)
    return superseq


class SLLNode:
    def __repr__(self):
        return f"{__class__.__name__}({', '.join(map(repr, self))})"


class ConcreteSLLNode(SLLNode):
    def __init__(self, value: T, *next_values: T, next_node: SLLNode = None):
        assert bool(next_values) != bool(next_node), "One and only one next_values or next_node must be specified."
        self.value = value
        self.next = ConcreteSLLNode(*next_values) if next_values else None

    def __iter__(self):
        yield self.value
        if self.next:
            yield from self.next

    def __repr__(self):
        return f"{__class__.__name__}({', '.join(map(repr, self))})"

    def contains(self, value: T, stop_at: ConcreteSLLNode = None) -> bool:
        return value == self.value or self.next and stop_at is not self.next and self.next.contains(value)

    def __contains__(self, value: T) -> bool:
        return self.contains(value)
    
    def matches(self, value: T, stop_at: ConcreteSLLNode = None) -> bool:
        return self.value == value


class AmbiguousSLLNode(SLLNode):
    def __init__(self, value: T = None, next_node: T = None):
        assert isinstance(value, set)
        self.values = value or {}
        self.next = next_node

    def add(self, value):
        self.values.add(value)

    def __iter__(self):
        yield from self.values
        if self.next:
            yield from self.next

    def contains(self, value: T, stop: ConcreteSLLNode = None) -> bool:
        return value in self.values or self.next and stop is not self.next and self.next.contains(value)

    def __contains__(self, value: T) -> bool:
        return self.contains(value)
    
    def matches(self, value: T) -> bool:
        return value in self.values


def disambiguify_sll(chain: ConcreteSLLNode, value: T, stop: ConcreteSLLNode=None):
    if isinstance(chain, AmbiguousSLLNode):
        if value in chain.values:
            chain.values.remove(value)
        else:
            return
    elif isinstance(chain, ConcreteSLLNode):
        if chain.value == value:
            raise ValueError("Cannot disambiguify a value when a node unambiguiously contains that value.")
    else:
        raise ValueError(f"Unrecognized node type: {type(chain)}")
    if chain.next and chain.next is not stop:
        disambiguify_sll(chain.next, value, stop)


def ambiguify_sll(chain: ConcreteSLLNode, value: T, stop: ConcreteSLLNode=None):
    if isinstance(chain, AmbiguousSLLNode):
        if value not in chain.values:
            chain.values.add(value)
        else:
            return
    elif isinstance(chain, ConcreteSLLNode):
        if chain.value == value:
            raise ValueError("Cannot ambiguify a value when a node unambiguiously contains that value.")
        if chain.next is None or isinstance(chain.next, ConcreteSLLNode):
            # Insert a new ambiguous node.
            chain.next = AmbiguousSLLNode(next_node=chain.next)
    else:
        raise ValueError(f"Unrecognized node type: {type(chain)}")
    if chain.next and chain.next is not stop:
        ambiguify_sll(chain.next, value, stop)


def strictly_totally_ordered_union(*iterables: Sequence[T]) -> Iterator[T]:
    head = ConcreteSLLNode(*iterables[0])
    for iterable in iterables[1:]:
        tail = head
        it = iter(iterable)
        value = next(it)
        while True:
            if isinstance(tail, ConcreteSLLNode):
                if tail.next is None:
                    # tail is the last node in the chain. value must come after it, so insert a new node.
                    tail.next = ConcreteSLLNode(value, None)
                    disambiguify_sll(head, value, tail.next)
                    value = next(it)
                    tail = tail.next
                elif tail.next.value == value:
                    # tail.next.value matches value, so move on to the next node.
                    value = next(it)
                    tail = tail.next
                elif value in tail.next:
                    # value is already in the tail somewhere, so move on until we reach that point.
                    while not tail.value.matches(value):
                        tail = tail.next
                elif value not in tail.next:
                    # value is not in the tail.next, so it could be anywhere from tail to the node matching the next value in the chain.
                    ambiguous_values = {value}
                    next_value = next(it)
                    next_node = tail.next
                    while True:
                        if isinstance(next_node, AmbiguousSLLNode) and next_node.matches(next_value) and next_node.next and next_value not in next_node.next:
                            break
                        elif isinstance(next_node, ConcreteSLLNode) and next_value == next_node.value:
                            break
                        elif next_node.next is None:
                            # next_value is ambiguous too!
                            ambiguous_values.add(next_value)
                            try:
                                # Try a new one and start over.
                                next_value = next(it)
                                next_node = tail.next
                            except StopIteration:
                                # There are no more values to try. Use all of them and ambuguify to the end.
                                next_node = None
                                next_value = None
                                break
                        elif next_node.next is not None:
                            next_node = next_node.next
                        else:
                            raise ValueError("Should never get here.")
                    if next_node is None:
                        # We've exhausted all of the values to try. Ambiguify to the end.
                        ambiguify_sll(tail.next, ambiguous_values)
                    else:
                        # We've found a node that matches the next value. Ambiguify to that node.
                        ambiguify_sll(tail, ambiguous_values, stop=next_node)
                        value = next_value
                        tail = next_node
                else:
                    raise ValueError("Should never get here.")
            elif isinstance(tail, AmbiguousSLLNode):
                if value in tail.values:
                    value = next(it)
                else:
                    tail.add(value)
                    value = next(it)