# Contains variable types used throughout SklarPy
from typing import Union, Iterable
from numpy import ndarray
from pandas import Series, DataFrame

__all__ = ['numeric', 'num_or_array', 'data_iterable', 'str_or_iterable', 'all_user_input_types',
           'dataframe_or_array', 'none_or_array']

numeric = Union[float, int]
num_or_array = Union[float, int, ndarray]
data_iterable = Union[DataFrame, Series, ndarray, Iterable]
str_or_iterable = Union[str, Iterable]
all_user_input_types = Union[DataFrame, Series, ndarray, Iterable, float, int]
dataframe_or_array = Union[DataFrame, ndarray]
none_or_array = Union[None, ndarray]
