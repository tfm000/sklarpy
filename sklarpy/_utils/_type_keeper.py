import pandas as pd
import numpy as np
from typing import Union

from sklarpy._utils import dataframe_or_array

__all__ = ['TypeKeeper']


class TypeKeeper:
    _HANDLED_TYPES: tuple = (pd.DataFrame, np.ndarray, type(None))

    def _check_valid_type(self, user_input):
        if type(user_input) not in self._HANDLED_TYPES:
            raise TypeError("Invalid type entered into TypeKeeper")

    def _get_required_info(self, user_input) -> dict:
        # returns a tuple, first index type, second contains the shape of the original input
        info_dict: dict = {}
        user_input_type = type(user_input)
        if user_input_type == pd.DataFrame:
            ndims: int = len(user_input.columns)
            shape: tuple = len(user_input.index), ndims
            other: dict = {'cols': user_input.columns, 'index': user_input.index}
        elif user_input_type == np.ndarray:
            shape: tuple = user_input.shape
            ndims: int = shape[1]
            other: dict = {}
        elif user_input is None:
            ndims: int = 0
            shape: tuple = (0, 0)
            other: dict = {}
        return {'type': user_input_type, 'ndims': ndims, 'shape': shape, 'other': other}

    def __init__(self, original_input: Union[dataframe_or_array, None]):
        self._check_valid_type(original_input)
        self._original_info: dict = self._get_required_info(original_input)

    def _type_keep_dataframe_from_1d_array(self, array: np.ndarray, col_name: list = None, add_index: bool = False) -> pd.DataFrame:
        if col_name is None:
            col_name = [0]

        df: pd.DataFrame = pd.DataFrame(array, columns=col_name)
        if add_index:
            if len(array) == self._original_info['shape'][0]:
                df.index = self._original_info['other']['index']
            else:
                raise ValueError("array is not the same length as original index.")
        return df

    def type_keep_from_1d_array(self, array: np.ndarray, match_datatype: bool = True, **kwargs):
        if not isinstance(match_datatype, bool):
            raise TypeError("match_datatype must be a boolean.")

        if not match_datatype:
            return array

        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy array")
        elif array.ndim != 1:
            raise ValueError("array must be 1d")

        if self._original_info['type'] == pd.DataFrame:
            return self._type_keep_dataframe_from_1d_array(array, **kwargs)
        else:
            # original type is None or np.ndarray
            return array

    def _type_keep_dataframe_from_2d_array(self, array: np.ndarray, add_index: bool = False) -> pd.DataFrame:
        return self._type_keep_dataframe_from_1d_array(array, self._original_info['other']['cols'], add_index)

    def check_dimensions(self, x: dataframe_or_array):
        self._check_valid_type(x)
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x_dims: int = 1
            else:
                x_dims: int = x.shape[1]
        elif isinstance(x, pd.DataFrame):
            x_dims: int = len(x.columns)
        elif x is None:
            x_dims: int = 0

        if (x_dims != self._original_info['ndims']) and (self._original_info['type'] != type(None)):
            raise ValueError("dimensions of x do not match original input.")

    def type_keep_from_2d_array(self, array: np.ndarray, match_datatype: bool = True, **kwargs):
        if not match_datatype:
            return array

        self.check_dimensions(array)
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy array")

        if self._original_info['type'] == pd.DataFrame:
            return self._type_keep_dataframe_from_2d_array(array, **kwargs)
        else:
            # original type is None or np.ndarray
            return array

    def _match_secondary_dataframe(self, df: pd.DataFrame):
        for column in self._original_info['other']['cols']:
            if column not in df.columns:
                raise ValueError(f"{column} in original input, but not in secondary input")
        return df[self._original_info['other']['cols']]

    def match_secondary_input(self, x: dataframe_or_array) -> dataframe_or_array:
        # checking input
        self._check_valid_type(x)
        self.check_dimensions(x)

        x_type = type(x)
        if x_type != self._original_info['type']:
            return x
        elif x_type == pd.DataFrame:
            return self._match_secondary_dataframe(x)
        return x

    def match_square_matrix(self, square_matrix: dataframe_or_array) -> dataframe_or_array:
        if square_matrix is None:
            raise TypeError("square_matrix cannot be None")

        square_matrix = self.match_secondary_input(square_matrix)
        square_matrix = self.match_secondary_input(square_matrix.T)
        return square_matrix

    @property
    def original_info(self):
        return self._original_info.copy()

    @property
    def original_type(self):
        return self.original_info['type']


if __name__ == '__main__':
    data = np.random.normal(size=(100, 2))
    data_df = pd.DataFrame(data, columns=['a', 'b'])

    tk_none = TypeKeeper(None)
    tk_arr = TypeKeeper(data)
    tk_df = TypeKeeper(data_df)

    new_data = np.array([[1, 2], [3,4]])
    new_1d_data = np.array([1,2,3,4,5])
    new_df_data = pd.DataFrame(new_data, columns=['a', 'b'], index=['a', 'b'])
    for tk in (tk_none, tk_arr, tk_df):
        print(tk._original_info['type'])
        print(tk.type_keep_from_2d_array(new_data))
        print(tk.type_keep_from_1d_array(new_1d_data, col_name=['pdf']))
        print(tk.match_square_matrix(new_df_data))