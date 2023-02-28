from tqdm import tqdm
from typing import Iterable

__all__ = ['get_iterator']


def get_iterator(x: Iterable, show_progress: bool, msg: str):
    if show_progress:
        return tqdm(x, msg)
    return x
