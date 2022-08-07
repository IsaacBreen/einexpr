import importlib
from importlib.metadata import entry_points, EntryPoint
import sys
from collections import namedtuple
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple, Union
import einexpr


_BACKENDS = None
_DEFAULT_BACKEND = None


def conforms_to_array_api(array: Any, strict: bool = False) -> bool:
    """
    Returns True if the given array conforms to the array API.
    """
    if strict:
        return hasattr(array, '__array_namespace__')
    else:
        return any((
            hasattr(array, '__array_namespace__'),
            array.__class__.__module__.split('.')[0] in ('numpy', 'torch', 'jax', 'tensorflow', 'mxnet', 'ivy') # Ivy
        ))


def _discover_array_api_entry_points() -> Dict[str, EntryPoint]:
    """
    Discovers all array APIs registered in the current environment and returns
    a dictionary mapping the name of the array API to the entry point.
    """
    global _BACKENDS
    if _BACKENDS is None:
        try:
            _BACKENDS = {ep.name: ep for ep in entry_points(group='array_api')}
        except TypeError:
            _BACKENDS = {ep.name: ep for ep in entry_points().get('array_api', [])}
    return _BACKENDS


def _discover_default_array_api() -> ModuleType:
    """
    Returns the default array API for the current environment.
    """
    global _BACKENDS, _DEFAULT_BACKEND
    if _BACKENDS is None:
        _discover_array_api_entry_points()
    preferences = ['numpy', 'torch', 'cupy', 'jax', 'tensorflow', 'mxnet']
    for preference in preferences:
        if preference in _BACKENDS:
            _DEFAULT_BACKEND = _BACKENDS[preference].load()
            break
    else:
        # If no preference is found, return the first entry point.
        _DEFAULT_BACKEND = list(_BACKENDS.values()).pop().load()


def get_array_api_backend(*, name: Optional[str] = None, array: Optional['RawArray' | Tuple['RawArray']] = None) -> ModuleType:
    """
    Returns the array API backend with the given name.
    """
    if _BACKENDS is None:
        _discover_array_api_entry_points()
    if sum(1 for x in [name, array] if x is not None) > 1:
        raise ValueError("Can only specify one of name or array.")
    if name is not None:
        if name in _BACKENDS:
            return _BACKENDS[name].load()
        else:
            if name in ('numpy', 'torch', 'jax', 'tensorflow', 'mxnet', 'ivy'):
                # Try to import Ivy using importlib
                ivy_spec = importlib.util.find_spec("ivy")
                if ivy_spec is not None:
                    import ivy
                    return ivy
            raise Exception(f"No array API backend with the name '{name}' found.")
    if array is not None:
        if isinstance(array, (tuple, list)):
            array = array[0]
        if not conforms_to_array_api(array, strict=True):
            if array.__class__.__module__.split('.')[0] in ('numpy', 'torch', 'jax', 'tensorflow', 'mxnet', 'ivy'):
                # Try to import Ivy using importlib
                ivy_spec = importlib.util.find_spec("ivy")
                if ivy_spec is not None:
                    import ivy
                    return ivy
            raise Exception(f"The given array does not conform to the array API standard.")
        return array.__array_namespace__()
    else:
        # Return the default array API.
        if _DEFAULT_BACKEND is None:
            _discover_default_array_api()
        return _DEFAULT_BACKEND


def get_asarray(*, name: Optional[str] = None, array: Optional['RawArray' | Tuple['RawArray']] = None) -> Callable:
    """
    Returns a function that converts the given array to the array API backend with the given name.
    """
    if _BACKENDS is None:
        _discover_array_api_entry_points()
    if all(x is None for x in [name, array]):
        return get_array_api_backend().asarray
    if sum(1 for x in [name, array] if x is not None) > 1:
        raise ValueError("Can only specify one of name or array.")
    if array is not None:
        if conforms_to_array_api(array, strict=True):
            return get_array_api_backend(array=array).asarray
        elif conforms_to_array_api(array, strict=False):
            # Try to import Ivy using importlib
            array_spec = importlib.util.find_spec(array.__class__.__module__)
            ivy_spec = importlib.util.find_spec("ivy")
            if array_spec is not None and ivy_spec is not None:
                import ivy
                module = importlib.import_module(array_spec.name)
                return lambda *args, **kwargs: module.asarray(ivy.to_native(*args, **kwargs))
        raise Exception(f"The given array does not conform to the array API standard.")
    elif name is not None:
        if name in _BACKENDS:
            return _BACKENDS[name].load().asarray
        elif name in ('numpy', 'torch', 'jax', 'tensorflow', 'mxnet', 'ivy'):
            array_spec = importlib.util.find_spec(name)
            ivy_spec = importlib.util.find_spec("ivy")
            if array_spec is not None and ivy_spec is not None:
                import ivy
                module = importlib.import_module(array_spec.name)
                if name == 'jax':
                    module = module.numpy.asarray
                return lambda *args, **kwargs: module.asarray(ivy.to_native(*args, **kwargs))
        raise Exception(f"No array API backend with the name '{name}' found.")
    raise einexpr.exceptions.InternalError("This should never happen.")