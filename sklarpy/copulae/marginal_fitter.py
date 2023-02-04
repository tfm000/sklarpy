# Class used to fit univariate marginal distributions to each random variable in a multivariate sample
import numpy as np
import pandas as pd
from typing import Union

from sklarpy._utils import str_or_iterable, data_iterable, check_multivariate_data, FitError, Savable
from sklarpy.univariate import UnivariateFitter

__all__ = ['MarginalFitter']


class MarginalFitter(Savable):
    """Class used to fit multiple distributions to data easily and to calculate cdf values to use with copulas."""

    def __init__(self, data: data_iterable, name: str = 'MarginalFitter'):
        """Used for fitting the best univariate distributions to each variable in a given multivariate dataset.

        Parameters
        ===========
        data: data_iterable
            The dataset containing a sample of your random variables. Can be a pd.DataFrame, np.ndarray or any
            multi-dimensional iterable. Data may be continuous or discrete or both.
        name: str
            The name of your MarginalFitter object. Default is 'MarginalFitter'.
            This is used when saving if a file paths is not specified.
        """
        # must take in multivariate data in the form of a numpy or dataframe in init.
        res: tuple = check_multivariate_data(data)
        self._data: np.ndarray = res[0]
        self._data_columns_indices: dict = res[1]
        self._num_variables: int = len(self._data_columns_indices)
        if not isinstance(name, str):
            raise TypeError('name of MarginalFitter must be a string')
        self._name: str = name
        self._fitted: bool = False
        self._fitted_marginals: dict = None
        self._summary: pd.DataFrame = None

    def __str__(self):
        return f"{self.name}(fitted={self._fitted})"

    def __repr__(self):
        return self.__str__()

    def _check_univariate_fitter_options(self, univariate_fitter_options: dict) -> dict:
        """Standardising and checking the user provided univariate_fitter_options is in the required format.

        Parameters
        ===========
        univariate_fitter_options: dict
            User provided arguments to use in our UnivariateFitter objects when fitting each marginal distribution.
            The relevant arguments are those found in the .fit and .get_best methods for UnivariateFitter.
            Default value is None, which means the default arguments for UnivariateFitter are used.
            The user can choose to use the same arguments for each marginal distribution by using a dictionary with
            the arguments as keys and the argument values as values.
            Alternatively, the user can provide different arguments for each marginal distribution when fitting, by
            providing a dictionary with the column names (if dataframe) / index (if np.ndarray/other iterable) of
            your variables ad keys and nested dictionaries of the form described above as values. Note, that if you
            choose to provide different arguments for different variables, all variables must be specified. If you wish
            to use the default arguments for a particular variable, pass an empty dictionary for it.

        See Also
        ========
        UnivariateFitter

        Returns
        ========
        univariate_fitter_options: dict
            Standardised univariate_fitter_options dictionary.
        """
        if univariate_fitter_options is None:
            # No user provided univariate_fitter_options
            return {col: {} for col in self._data_columns_indices.keys()}
        elif not isinstance(univariate_fitter_options, dict):
            raise TypeError('univariate_fitter_options must be a dictionary')

        at_least_one_col: bool = False
        all_present: bool = True
        for col in self._data_columns_indices.keys():
            if col in univariate_fitter_options.keys():
                at_least_one_col = True
            else:
                all_present = False

        if at_least_one_col and all_present:
            # user inputted options for each variable
            return univariate_fitter_options
        elif (not at_least_one_col) and (not all_present):
            # user inputted a single dictionary to use for each variable
            return {col: univariate_fitter_options for col in self._data_columns_indices.keys()}
        raise ValueError('univariate_fitter_options cannot contain options for only a subset of variables.')

    def _fit_single_marginal(self, column: Union[int, str], univariate_fitter_options: dict):
        """Selects the probability distribution which best fits a given marginal.

        Parameters
        ==========
        column: Union[int, str]
            The column name (if pd.DataFrame provided by user) or index (if np.ndarray/other iterable provided by user)
            of the variable we are fitting.
        univariate_fitter_options: dict
            The arguments of UnivariateFitter to use for this particular variable.

        Returns
        ========
        marginal_dict
            The best fitted distribution for our variable.
        """
        get_best_options: dict = {'significant': True, 'pvalue': 0.05}
        for option in get_best_options:
            if option in univariate_fitter_options:
                get_best_options[option] = univariate_fitter_options.pop(option)

        index: int = self._data_columns_indices[column]
        variable_data: np.ndarray = self._data[:, index]
        fitter: UnivariateFitter = UnivariateFitter(variable_data)
        fitter.fit(**univariate_fitter_options)
        marginal_dist = fitter.get_best(**get_best_options)
        return marginal_dist

    def fit(self, univariate_fitter_options: dict = None):
        """Fits the best univariate distributions to each variable in a given multivariate dataset

        Parameters
        ===========
        univariate_fitter_options: dict
            User provided arguments to use in our UnivariateFitter objects when fitting each marginal distribution.
            The relevant arguments are those found in the .fit and .get_best methods for UnivariateFitter.
            Default value is None, which means the default arguments for UnivariateFitter are used.
            The user can choose to use the same arguments for each marginal distribution by using a dictionary with
            the arguments as keys and the argument values as values.
            Alternatively, the user can provide different arguments for each marginal distribution when fitting, by
            providing a dictionary with the column names (if dataframe) / index (if np.ndarray/other iterable) of
            your variables ad keys and nested dictionaries of the form described above as values. Note, that if you
            choose to provide different arguments for different variables, all variables must be specified. If you wish
            to use the default arguments for a particular variable, pass an empty dictionary for it.

        See Also
        ========
        UnivariateFitter

        Returns
        ========
        self:
            self
        """
        univariate_fitter_options: dict = self._check_univariate_fitter_options(univariate_fitter_options)
        summaries: list = []
        self._fitted_marginals = {}

        for col in self._data_columns_indices.keys():
            marginal_dist = self._fit_single_marginal(col, univariate_fitter_options[col])
            self._fitted_marginals[col] = marginal_dist
            summaries.append(marginal_dist.summary)

        self._summary = pd.concat(summaries, axis=1)
        self._fitted = True
        return self

    def _pdfs_cdf_ppfs_logpdfs_inputs(self, x: np.ndarray, func: str) -> np.ndarray:
        """implements pdf, cdf, ppf and logpdf for our marginal distributions

        Parameters
        ===========
        x: np.ndarray
            Our input for our marginal function
        func: str
            The name of the function to be used as a string.

        Returns
        =======
        values: np.ndarray
            Function values
        """
        if not self._fitted:
            raise FitError('MarginalFitter object has not been fitted.')

        if x is None:
            x = self._data
        else:
            x, _ = check_multivariate_data(x, self._num_variables)
        x_shape = x.shape

        values: np.ndarray = np.full(x_shape, np.NaN, dtype=float)
        for col, index in self._data_columns_indices.items():
            marginal_dist = self._fitted_marginals[col]
            values[:, index] = eval(f"marginal_dist.{func}(x[:, index])")
        return values

    def marginal_pdfs(self, x: np.ndarray = None) -> np.ndarray:
        """Calculates the pdf values for each univariate marginal distribution for a given set of observations x.

        Parameters
        ===========
        x: np.ndarray
            The values to calculate the marginal pdf values of.
            Must be the same dimension as the number of variables.
            If None passes, the pdf values of the sample used in the original dataset are returned.

        Returns
        ========
        marginal_pdf_values: np.ndarray
            An array of marginal pdf values.
        """
        return self._pdfs_cdf_ppfs_logpdfs_inputs(x, 'pdf')

    def marginal_cdfs(self, x: np.ndarray = None) -> np.ndarray:
        """Calculates the cdf values for each univariate marginal distribution for a given set of observations x.

        Parameters
        ===========
        x: np.ndarray
            The values to calculate the marginal cdf values, P(X<=x), of.
            Must be the same dimension as the number of variables.
            If None passes, the cdf values of the sample used in the original dataset are returned.

        Returns
        ========
        marginal_cdf_values: np.ndarray
            An array of marginal cdf values.
        """
        return self._pdfs_cdf_ppfs_logpdfs_inputs(x, 'cdf')

    def marginal_ppfs(self, q: np.ndarray) -> np.ndarray:
        """Calculates the ppf values for each univariate marginal distribution for a given set of observations x.

        Parameters
        ===========
        q: np.ndarray
            The quartile values to calculate the marginal ppf values, i.e. cdf^-1(q), of.
            Must be the same dimension as the number of variables.

        Returns
        ========
        marginal_ppf_values: np.ndarray
            An array of marginal ppf values.
        """
        return self._pdfs_cdf_ppfs_logpdfs_inputs(q, 'ppf')

    def marginal_logpdfs(self, x: np.ndarray = None) -> np.ndarray:
        """Calculates the pdf values for each univariate marginal distribution for a given set of observations x.

        Parameters
        ===========
        x: np.ndarray
            The values to calculate the marginal log-pdf values of.
            Must be the same dimension as the number of variables.
            If None passes, the log-pdf values of the sample used in the original dataset are returned.

        Returns
        ========
        marginal_logpdf_values: np.ndarray
            An array of marginal log-pdf values.
        """
        return self._pdfs_cdf_ppfs_logpdfs_inputs(x, 'logpdf')

    def marginal_rvs(self, size: int) -> np.ndarray:
        """Randomly samples values from the marginal distributions.

        Parameters
        ===========
        size: int
            The number of samples to generate from each marginal distribution.

        Returns
        ========
        marginal_rvs_values: np.ndarray
            A random sample with shape (size, num_variables)
        """
        if not self._fitted:
            raise FitError('MarginalFitter object has not been fitted.')

        if not (isinstance(size, int) and size > 0):
            raise TypeError('size must be a positive integer')

        rvs_values: np.ndarray = np.full((size, self._num_variables), np.NaN, dtype=float)
        for col, index in self._data_columns_indices.items():
            marginal_dist = self._fitted_marginals[col]
            rvs_values[:, index] = marginal_dist.rvs((size, ))
        return rvs_values

    @property
    def marginals(self) -> dict:
        """Returns a dict with marginals as values and the indices (if np.ndarray / other iterable) or column names
        (if dataframe) as keys
        """
        if not self._fitted:
            raise FitError('MarginalFitter object has not been fitted.')
        return self._fitted_marginals

    @property
    def summary(self) -> pd.DataFrame:
        """A pd.DataFrame containing a summary of the fitted marginals"""
        if not self._fitted:
            raise FitError('MarginalFitter object has not been fitted.')
        return self._summary

    @property
    def num_variables(self) -> int:
        """The number of variables present in the original dataset."""
        return self._num_variables

    @property
    def name(self) -> str:
        """The name of the MarginalFitter object."""
        return self._name
