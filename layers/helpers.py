import inspect
import copy
from . import lazylayers as L


def split_kwargs(kwargs, *classes):
    remaining_kwargs = copy.deepcopy(kwargs)
    class_kwargs = [dict() for C in classes]
    for k, v in list(remaining_kwargs.items()):
        for C_kwargs, C in zip(class_kwargs, classes):
            params = inspect.signature(C).parameters
            if k in params:
                C_kwargs[k] = v
        remaining_kwargs.pop(k)
    return class_kwargs, remaining_kwargs


def _join_layer(join):
    switch = {
        "add": L.Add,
        "mul": L.Multiply,
        "avg": L.Average,
        "drop": L.DropAvg,
        "concat": L.Concatenate,
    }
    return switch[join]


def _Activation(act):
    if isinstance(act, str):
        if act != "linear":
            return L.Activation(activation = act)
        else:
            return lambda x: x
    else:
        return act