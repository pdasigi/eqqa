from typing import Callable

import inspect
import importlib
import json
import hashlib


def filter_params(params: dict, method: callable) -> dict:
    method_params = inspect.signature(method).parameters
    return {p: p_val for p, p_val in params.items() if p in method_params}


def filter_params_by_prefix(params: dict, prefix: str, trim: bool) -> dict:
    params = {p: p_val for p, p_val in params.items() if p.startswith(prefix)}
    if trim:
        params = {p[len(prefix):]: p_val for p, p_val in params.items()}
    return params

def generate_uuid(content, indent: int = 2) -> str:
    """Deterministic uuid generator of the `content`."""
    content = json.dumps(content, sort_keys=True, indent=indent).encode("utf-8")
    return hashlib.md5(content).hexdigest()


def import_method(fullpath: str) -> Callable:
    """Import a specific method or class given the corresponding path."""
    if fullpath is None:
        raise ValueError(f"Cannot import method {fullpath}")

    paths = fullpath.rsplit(".", 1)

    if len(paths) == 1:
        module, method = "__main__", paths
    else:
        module, method = paths

    try:
        module = importlib.import_module(module)
    except:
        module = import_method(module)
    return getattr(module, method)


def method_name(method: Callable) -> str:
    """Determines the fully qualified name of a method or a class."""
    module = method.__module__
    qualname = method.__qualname__

    # FIXME - Filter for builtins (e.g., method_name(str))
    return f"{module}.{qualname}"
