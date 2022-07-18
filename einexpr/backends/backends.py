from importlib.metadata import entry_points, EntryPoint
import sys
from collections import namedtuple
from types import ModuleType
from typing import Any, Callable, Dict, Union
import einexpr


_BACKENDS = None
_DEFAULT_BACKEND = None


def conforms_to_array_api(array: Any):
    """
    Returns True if the given array conforms to the array API.
    """
    return hasattr(array, '__array_namespace__')


def _discover_array_api_entry_points() -> Dict[str, EntryPoint]:
    """
    Discovers all array APIs registered in the current environment and returns
    a dictionary mapping the name of the array API to the entry point.
    """
    try:
        return {ep.name: ep for ep in entry_points(group='array_api')}
    except TypeError:
        return {ep.name: ep for ep in entry_points().get('array_api', [])}


def _discover_default_array_api() -> ModuleType:
    """
    Returns the default array API for the current environment.
    """
    if _BACKENDS is None:
        raise einexpr.exceptions.InternalError("Backends have not been discovered yet.")
    preferences = ['numpy', 'torch', 'cupy', 'jax', 'tensorflow', 'mxnet']
    for preference in preferences:
        if preference in _BACKENDS:
            return _BACKENDS[preference].load()
    # If no preference is found, return the first entry point.
    return list(_BACKENDS.values()).pop().load()


def get_array_api_backend(name: Union[str, None]) -> ModuleType:
    """
    Returns the array API backend with the given name.
    """
    if _BACKENDS is None:
        raise einexpr.exceptions.InternalError("Backends have not been discovered yet.")
    if name is None:
        if _DEFAULT_BACKEND is None:
            raise einexpr.exceptions.InternalError("No default backend has been set.")
        return _DEFAULT_BACKEND
    if name not in _BACKENDS:
        raise Exception(f"No array API backend with the name '{name}' found.")
    return _BACKENDS[name].load()


_BACKENDS = _discover_array_api_entry_points()
_DEFAULT_BACKEND = _discover_default_array_api()