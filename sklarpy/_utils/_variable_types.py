# Contains variable types used throughout SklarPy
from typing import Union, Iterable
from numpy import ndarray
from pandas import Series, DataFrame

__all__ = ['num_or_array', 'data_iterable', 'str_or_iterable']


num_or_array = Union[float, int, ndarray]
data_iterable = Union[DataFrame, Series, ndarray, Iterable]
str_or_iterable = Union[str, Iterable]
