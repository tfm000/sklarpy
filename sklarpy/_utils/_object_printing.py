# Containing code for making printed objects easier to read
import numpy as np
import pandas as pd

__all__ = ['print_full_arrays_and_dataframes']


def print_full_arrays_and_dataframes() -> None:
    """Ensures numpy arrays and pandas DataFrames are printed in a full and readable format."""
    np.set_printoptions(linewidth=np.inf)
    pd.options.display.width = 0
