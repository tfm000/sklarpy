# Contains useful functions
import numpy as np

__all__ = ['Utils']


class Utils:
    @staticmethod
    def data_type(arr: np.ndarray, dist_type: bool = True) -> tuple:
        """Checks whether an array contains float or integer values.

        Parameters
        ===========
        arr: np.ndarray
            Your array of data. Must only contain float or integer values.

        Returns
        =======
        arr_tuple: tuple
            arr, type
        """
        if not ((arr.dtype == float) or (arr.dtype == int)):
            raise TypeError("Array must only contain integer or float values.")

        arr_int = arr.astype(int)
        if np.all((arr - arr_int) == 0):
            if dist_type:
                return arr_int, int, 'discrete'
            return arr_int, int
        if dist_type:
            return arr, float, 'continuous'
        return arr, float
