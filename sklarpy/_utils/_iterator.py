from tqdm import tqdm
from typing import Iterable

__all__ = ['get_iterator']


def get_iterator(x: Iterable, show_progress: bool, msg: str):
    if not isinstance(x, Iterable):
        raise TypeError("x must be a Iterable.")

    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean.")

    if not isinstance(msg, str):
        raise TypeError("msg must be a string.")

    if show_progress:
        return tqdm(x, msg)
    return x
