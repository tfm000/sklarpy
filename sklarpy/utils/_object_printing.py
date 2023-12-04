# Containing code for making printed objects easier to read
import numpy as np
import pandas as pd

__all__ = ['print_full']


def print_full() -> None:
    """Ensures numpy arrays and pandas DataFrames are printed in a full and
    readable format."""
    np.set_printoptions(linewidth=np.inf)
    pd.options.display.width = 0
