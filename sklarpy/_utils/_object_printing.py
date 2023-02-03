# Containing functions for making printed objects easier to read
from numpy import set_printoptions, inf
from pandas import options

__all__ = ['print_full_arrays_and_dataframes']


def print_full_arrays_and_dataframes() -> None:
    """Ensures numpy arrays and pandas DataFrames are printed in a full and readable format."""
    set_printoptions(linewidth=inf)
    options.display.width = 0
