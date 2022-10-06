import functools
import itertools
from collections import namedtuple
from typing import Iterable, Any

experiment_params = "day topic count budget method"
Config = namedtuple("Config", experiment_params)
ParamConfig = namedtuple("ParamConfig", " ".join(experiment_params.split()[1:]))
TweetHashtagKeyPath = namedtuple("TweetHashtagKeyPath", experiment_params + " th_key")
HashtagTweetKeyPath = namedtuple("HashtagTweetKeyPath", experiment_params + " ht_key")
th_key, ht_key = "hashtag_source_dict", "source_hashtag_dict"


def rget(d, *keys):
    """Recursive get."""
    # https://stackoverflow.com/a/28225747
    return functools.reduce(lambda c, k: c.get(k, {}), keys, d)


def srget(d, *keys):
    """Recursive get & set."""
    subd = rget(
        d,
        *keys,
    )
    if not subd:
        rget(d, *keys[:-1])[keys[-1]] = {}
        return rget(d, *keys)
    return subd


def rinit(keys, d=None, default=None) -> dict:
    """Recursively initialize a dictionary with a list of keys."""
    d = d or {}

    if not default:
        fn = lambda c, k: c.setdefault(k, {})
    elif len(default) == 2:
        fn = lambda c, k: c.setdefault(k, {default[0]: default[1]})
    elif len(default) == 3 and default[2][0] == "path":
        default_paths = get_keys(*(keys, default[2][1]))
        fn = lambda c, k: c.setdefault(
            k, {default[0]: rget(default[1], *default_paths)}
        )
    else:
        raise ValueError("Invalid default value")

    if isinstance(keys, list):
        for key_set in keys:
            functools.reduce(fn, key_set, d)
    else:
        functools.reduce(fn, keys, d)
    return d


def rset(d, path, key, value):
    """Recursive set."""
    elements = rget(d, *path)


def kset(d, value, *path):
    subd = srget(d, *path[:-1])
    subd[path[-1]] = value


def rsetm(d, paths, key, value):
    """Recursive set multiple."""
    elements = [srget(d, *path) for path in paths]

    if isinstance(key, list) and isinstance(value, list):
        for element, _key, _value in zip(elements, key, value):
            element[_key] = _value
    elif isinstance(key, list) and not isinstance(value, list):
        for element, _key in zip(elements, key):
            element[_key] = value
    elif not isinstance(key, list) and isinstance(value, list):
        for element, _value in zip(elements, value):
            element[key] = _value
    else:
        for element in elements:
            element[key] = value


def get_keys(*arg_matrix):
    return list(itertools.product(*[*arg_matrix]))


def rinsertm(iterable: Iterable[Iterable], value: Any) -> tuple:
    """Recursive insert."""
    return tuple(tuple([*element] + [value]) for element in iterable)


def rinsert(iterable: Iterable, value: Any) -> tuple:
    return tuple(*iterable, value)
