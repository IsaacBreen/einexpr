import warnings
import functools

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
            raise TypeError(f"decorator_maker() takes either a one positional argument (the function to decorate) or zero or more keyword arguments (got positional arguments {args} and keyword arguments {kwargs}")
    return enchanced_decorator
