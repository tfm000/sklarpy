# Contains code for communicating progress to the user when iterating
from tqdm import tqdm
from typing import Iterable

__all__ = ['get_iterator']


def get_iterator(x: Iterable, show_progress: bool, msg: str) -> Iterable:
    """Allows the user to receive messages on the progress of a given iteration
    in the logs.

    Parameters
    -----------
    x: Iterable
        The object to iterate over.
    show_progress: bool
        Whether to show the progress of the iteration.
    msg: str
        The message to display in the logs.
    """
    # checking arguments
    if not isinstance(x, Iterable):
        raise TypeError("x must be a Iterable.")

    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean.")

    if not isinstance(msg, str):
        raise TypeError("msg must be a string.")

    # returning iterator
    return tqdm(x, msg) if show_progress else x
