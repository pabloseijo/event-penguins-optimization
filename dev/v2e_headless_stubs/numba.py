"""No-op decorators for v2e paths that are not used without video rendering."""


def _decorate(function):
    return function


def jit(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return _decorate


def njit(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return _decorate
