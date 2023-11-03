# Contains code for converting function output type to match the user's
# inputted data type
import pandas as pd
import numpy as np
from typing import Union

__all__ = ['TypeKeeper']


class TypeKeeper:
    """Class used for converting function output type to match the user's
    inputted data type."""
    _HANDLED_TYPES: tuple = (pd.DataFrame, np.ndarray, type(None))

    def _check_valid_type(self,
                          user_input: Union[pd.DataFrame, np.ndarray, None]) \
            -> None:
        """Checks if the user input's input is a type supported by TypeKeeper.

        Parameters
        ----------
        user_input : Union[pd.DataFrame, np.ndarray, None]
            The original data inputted by the user.
        """
        if type(user_input) not in self._HANDLED_TYPES:
            raise TypeError("Invalid type entered into TypeKeeper")

    def _get_required_info(self,
                           user_input: Union[pd.DataFrame, np.ndarray, None]) \
            -> dict:
        """Extracts information needed from the user's input to preserve type.

        Parameters
        ----------
        user_input : Union[pd.DataFrame, np.ndarray, None]
            The original data inputted by the user.

        Returns
        -------
        original_info: dict
            A dictionary containing information on the user's original input
        """
        user_input_type = type(user_input)
        if user_input_type == pd.DataFrame:
            ndims: int = len(user_input.columns)
            shape: tuple = (len(user_input.index), ndims)
            other: dict = {'cols': user_input.columns,
                           'index': user_input.index}
        elif user_input_type == np.ndarray:
            shape: tuple = user_input.shape
            ndims: int = shape[-1]
            other: dict = {}
        elif user_input is None:
            ndims: int = 0
            shape: tuple = (0, 0)
            other: dict = {}
        return {'type': user_input_type, 'ndims': ndims, 'shape': shape,
                'other': other}

    def __init__(self, user_input: Union[pd.DataFrame, np.ndarray, None]):
        """Object used for converting function output type to match the user's
         inputted data type.

        Parameters
        ----------
        user_input : Union[pd.DataFrame, np.ndarray, None]
            The original data inputted by the user.
        """
        self._check_valid_type(user_input)
        self._original_info: dict = self._get_required_info(user_input)

    def _type_keep_dataframe_from_1d_array(
            self, array: np.ndarray, col_name: list = None,
            add_index: bool = False) -> pd.DataFrame:
        """Converts a numpy array into a dataframe whilst incorporating
        information about the user's original data input.

        User's original data input must have been a pandas dataframe to use
        this method.

        Parameters
        ----------
        array: np.ndarray
            numpy array to convert into a dataframe.
        col_name: list
            Optional. A list of column names to use in the dataframe.
            As the array is 1d, this should contain a single element.
            If None parsed, the column is labeled as 0.
            Default is None.
        add_index: bool
            Optional. True to use the index of the user's original dataframe
            for this output.

        Returns
        -------
        df: pd.DataFrame
            The array transformed into a pd.DataFrame object.
        """
        if col_name is None:
            col_name = [0]

        df: pd.DataFrame = pd.DataFrame(array, columns=col_name)
        if add_index:
            if len(array) == self._original_info['shape'][0]:
                df.index = self._original_info['other']['index']
            else:
                raise ValueError("array is not the same length as original "
                                 "index.")
        return df

    def type_keep_from_1d_array(
            self, array: np.ndarray, match_datatype: bool = True, **kwargs) \
            -> Union[np.ndarray, pd.DataFrame]:
        """Converts a numpy array into the user's original datatype.

        Parameters
        ----------
        array : np.ndarray
            numpy array to convert into the user's original datatype.
        match_datatype: bool
            Optional. True to convert the user's datatype to match the
            original.
            Default is True.
        kwargs:
            Keyword arguments to pass to _type_keep_dataframe_from_1d_array.

        Returns
        -------
        res: Union[np.ndarray, pd.DataFrame]
            The array transformed into the user's original datatype,
            if desired.
        """
        # checking arguments
        if not isinstance(match_datatype, bool):
            raise TypeError("match_datatype must be a boolean.")

        if not match_datatype:
            return array

        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy array")
        elif array.ndim != 1:
            raise ValueError("array must be 1d")

        # converting to original datatype
        if self._original_info['type'] == pd.DataFrame:
            return self._type_keep_dataframe_from_1d_array(array, **kwargs)
        else:
            # original type is None or np.ndarray
            return array

    def _type_keep_dataframe_from_2d_array(
            self, array: np.ndarray, add_index: bool = False) -> pd.DataFrame:
        """Converts a numpy array into a dataframe whilst incorporating
        information about the user's original data input.

        User's original data input must have been a pandas dataframe to use
        this method.

        Parameters
        ----------
        array: np.ndarray
            numpy array to convert into a dataframe.
        add_index: bool
            Optional. True to use the index of the user's original dataframe
            for this output.

        Returns
        -------
        df: pd.DataFrame
            The array transformed into a pd.DataFrame object.
        """
        return self._type_keep_dataframe_from_1d_array(
            array, self._original_info['other']['cols'], add_index)

    def check_dimensions(self, x: Union[pd.DataFrame, np.ndarray]) -> None:
        """Checks whether the dimensions of x match those of the original
        datatype.

        Raises ValueError if dimensions do not match.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            pandas dataframe or numpy array whose dimension to check.
        """
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

        if (x_dims != self._original_info['ndims']) and \
                (self._original_info['type'] != type(None)):
            raise ValueError("dimensions of x do not match original input.")

    def type_keep_from_2d_array(
            self, array: np.ndarray, match_datatype: bool=True, **kwargs) \
            -> Union[np.ndarray, pd.DataFrame]:
        """Converts a numpy array into the user's original datatype.

        Parameters
        ----------
        array : np.ndarray
            numpy array to convert into the user's original datatype.
        match_datatype: bool
            Optional. True to convert the user's datatype to match the
            original.
            Default is True.
        kwargs:
            Keyword arguments to pass to _type_keep_dataframe_from_2d_array.

        Returns
        -------
        res: Union[np.ndarray, pd.DataFrame]
            The array transformed into the user's original datatype,
            if desired.
        """
        # checking arguments
        if not match_datatype:
            return array

        self.check_dimensions(array)
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy array")

        # converting to original datatype
        if self._original_info['type'] == pd.DataFrame:
            return self._type_keep_dataframe_from_2d_array(array, **kwargs)
        else:
            # original type is None or np.ndarray
            return array

    def _match_secondary_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures x matches the same dimensions and
        column order as the original input.

        User's original data input must have been a pandas
        dataframe to use this method.

        Parameters
        ----------
        x : pd.DataFrame
            secondary input, to match.

        Returns
        -------
        res: pd.DataFrame
            secondary input, which now matches the original data.
        """
        for column in self._original_info['other']['cols']:
            if column not in df.columns:
                raise ValueError(f"{column} in original input, but not in "
                                 f"secondary input")
        return df[self._original_info['other']['cols']]

    def match_secondary_input(self, x: Union[pd.DataFrame, np.ndarray]) \
            -> Union[pd.DataFrame, np.ndarray]:
        """Ensures x matches the same dimensions and (for dataframes)
        column order as the original input.

        Parameters
        ----------
        x : Union[pd.DataFrame, np.ndarray]
            secondary input, to match.

        Returns
        -------
        res: Union[pd.DataFrame, np.ndarray]
            secondary input, which now matches the original data.
        """
        # checking input
        self._check_valid_type(x)
        self.check_dimensions(x)

        x_type = type(x)
        if x_type != self._original_info['type']:
            return x
        elif x_type == pd.DataFrame:
            return self._match_secondary_dataframe(x)
        return x

    def match_square_matrix(
            self, square_matrix: Union[pd.DataFrame, np.ndarray])\
            -> Union[pd.DataFrame, np.ndarray]:
        """Ensures a square matrix matches the same dimensions and
        (for dataframes) column order as the original input.

        Parameters
        ----------
        x : Union[pd.DataFrame, np.ndarray]
            secondary input, to match.

        Returns
        -------
        res: Union[pd.DataFrame, np.ndarray]
            square matrix, which now matches the original data.
        """
        if square_matrix is None:
            raise TypeError("square_matrix cannot be None")

        square_matrix = self.match_secondary_input(square_matrix)
        square_matrix = self.match_secondary_input(square_matrix.T).T
        return square_matrix

    @property
    def original_info(self) -> dict:
        """Returns information on the user's original input."""
        return self._original_info.copy()

    @property
    def original_type(self):
        """Returns the user's original input type."""
        return self.original_info['type']
