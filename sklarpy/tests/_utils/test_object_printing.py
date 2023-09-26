# Contains code for testing SklarPy's object printing functions
import numpy as np
import pandas as pd

from sklarpy import print_full


def test_print_full():
    """Testing print_full function."""
    num: int = 10 ** 5

    arrA: np.ndarray = np.linspace(0, 1000, num)
    arrB: np.ndarray = np.linspace(-1000, 1000, num)
    df: pd.DataFrame = pd.DataFrame({'A': arrA, 'B': arrB}).T

    # testing print_full
    print_full()
