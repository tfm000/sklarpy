# Contains code for the MarginalFitter class, which fits univariate
# distributions to each random variable in a multivariate sample
import numpy as np
import pandas as pd
from typing import Union, Iterable

from sklarpy.utils._errors import FitError
from sklarpy.utils._type_keeper import TypeKeeper
from sklarpy.utils._input_handlers import check_multivariate_data
from sklarpy.utils._iterator import get_iterator
from sklarpy.utils._serialize import Savable
from sklarpy.univariate import UnivariateFitter
from sklarpy.plotting._pair_plot import pair_plot

__all__ = ['MarginalFitter']


class MarginalFitter(Savable):
    """Class used to fit multiple distributions to data easily and to calculate
    cdf/distribution values to use with copulas."""
    _OBJ_NAME = 'MarginalFitter'

    def __init__(self, data: Union[pd.DataFrame, np.ndarray],
                 name: str = None):
        """Used for fitting univariate distributions to each random variable
        in a given multivariate dataset.

        Parameters
        -----------
        data: Union[pd.DataFrame, np.ndarray]
            The dataset containing a sample of your random variables.
            Can be a pd.DataFrame or np.ndarray. Data may be continuous or
            discrete or both.
        name: str
            The name of your MarginalFitter object.
            Used when saving, if a file path is not specified and/or for
            additional identification purposes.
            Default is 'MarginalFitter'.
        """
        Savable.__init__(self, name)

        self._data: np.ndarray = check_multivariate_data(data)
        self._num_variables: int = data.shape[1]
        self._fitted: bool = False
        self._typekeeper: TypeKeeper = TypeKeeper(data)

        self._fitted_marginals: dict = None
        self._summary: pd.DataFrame = None
        self._cdf_data: np.ndarray = None

    def __str__(self):
        return f"{self.name}(fitted={self._fitted})"

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return self.num_variables

    def _check_univariate_fitter_options(self, univariate_fitter_options: dict
                                         ) -> dict:
        """Standardising and checking the user provided
        univariate_fitter_options is in the required format.

        Parameters
        ----------
        univariate_fitter_options: dict
            User provided arguments to use in our UnivariateFitter objects
            when fitting each marginal distribution. The relevant arguments
            are those found in the .fit and .get_best methods for
            UnivariateFitter.
            Default value is None, which means the default arguments for
            UnivariateFitter are used. The user can choose to use the same
            arguments for each marginal distribution by using a dictionary
            with the arguments as keys and the argument values as values.
            Alternatively, the user can provide different arguments for each
            marginal distribution when fitting, by providing a dictionary with
            the column indexes of your variables as keys and nested
            dictionaries of the form described above as values.
            Note, that if you choose to provide different arguments for
            different variables, all variables must be specified.
            If you wish to use the default arguments for a particular variable,
            pass an empty dictionary, {}, for it.

        See Also
        --------
        UnivariateFitter

        Returns
        -------
        univariate_fitter_options: dict
            Standardised univariate_fitter_options dictionary.
        """
        if univariate_fitter_options is None:
            # No user provided univariate_fitter_options.
            # Using default arguments
            return {i: {} for i in range(self._num_variables)}

        elif not isinstance(univariate_fitter_options, dict):
            raise TypeError('univariate_fitter_options must be a dictionary')

        at_least_one_index: bool = False
        all_present: bool = True
        for i in range(self._num_variables):
            if i in univariate_fitter_options.keys():
                at_least_one_index = True
            else:
                all_present = False

        if at_least_one_index and all_present:
            # user has inputted options for each variable
            return univariate_fitter_options
        elif (not at_least_one_index) and (not all_present):
            # user inputted a single dictionary of arguments
            # to use for each variable
            return {
                i: univariate_fitter_options
                for i in range(self._num_variables)
            }
        raise ValueError('univariate_fitter_options cannot contain '
                         'options for only a subset of variables.')

    def _fit_single_marginal(self, index: int,
                             univariate_fitter_options: dict):
        """Selects the probability distribution which best fits a
        given marginal.

        Parameters
        ----------
        index: int
            The index of the variable we are fitting.
        univariate_fitter_options: dict
            The arguments of UnivariateFitter to use for this particular
            variable.

        Returns
        -------
        marginal_dict
            The best fitted distribution for our variable.
        """
        variable_data: np.ndarray = self._data[:, index]
        fitter: UnivariateFitter = UnivariateFitter(variable_data)
        fitter.fit(**univariate_fitter_options)
        marginal_dist = fitter.get_best(**univariate_fitter_options)
        return marginal_dist

    def fit(self, univariate_fitter_options: dict = None, **kwargs):
        """Fits the best univariate distributions to each variable in a given
        multivariate dataset

        Parameters
        ----------
        univariate_fitter_options: dict
            User provided arguments to use in our UnivariateFitter objects
            when fitting each marginal distribution. The relevant arguments
            are those found in the .fit and .get_best methods for
            UnivariateFitter.
            Default value is None, which means the default arguments for
            UnivariateFitter are used. The user can choose to use the same
            arguments for each marginal distribution by using a dictionary
            with the arguments as keys and the argument values as values.
            Alternatively, the user can provide different arguments for each
            marginal distribution when fitting, by providing a dictionary with
            the column indexes of your variables as keys and nested
            dictionaries of the form described above as values.
            Note, that if you choose to provide different arguments for
            different variables, all variables must be specified.
            If you wish to use the default arguments for a particular variable,
            pass an empty dictionary, {}, for it.
        kwargs:
            See below

        Keyword arguments
        ------------------
        show_progress: bool
            Whether to show the progress of your fitting.

        See Also
        --------
        UnivariateFitter

        Returns
        --------
        self:
            self
        """
        univariate_fitter_options: dict = \
            self._check_univariate_fitter_options(univariate_fitter_options)
        summaries: list = []
        self._fitted_marginals = {}
        self._cdf_data: np.ndarray = np.full(self._data.shape, np.nan, float)

        iterator = get_iterator(
            range(self._num_variables), kwargs.get('show_progress', False),
            'MarginalFitter Progress: '
        )
        for index in iterator:
            marginal_dist = self._fit_single_marginal(
                index, univariate_fitter_options[index]
            )
            self._fitted_marginals[index] = marginal_dist
            summaries.append(marginal_dist.summary)
            self._cdf_data[:, index] = marginal_dist.cdf(self._data[:, index])

        summary: pd.DataFrame = pd.concat(summaries, axis=1)
        if self._typekeeper.original_type == pd.DataFrame:
            index: pd.Index = summary.index
            summary = self._typekeeper.type_keep_from_2d_array(
                np.asarray(summary)
            )
            summary.index = index
        self._summary = summary
        self._fitted = True
        return self

    def _fit_check(self) -> None:
        """raises an error if MarginalFitter object has not been fitted
        to data."""
        if not self._fitted:
            raise FitError('MarginalFitter object has not been fitted.')

    def _pdfs_cdf_ppfs_logpdfs_inputs(self, func_str: str,
                                      x: Union[pd.DataFrame, np.ndarray],
                                      match_datatype: bool = True) \
            -> Union[pd.DataFrame, np.ndarray]:
        """Utility function able to implement pdf, cdf, ppf and logpdf
        methods without duplicate code.

        Parameters
        ----------
        func_str: str
            The name of the method to implement.
        x: Union[pd.DataFrame, np.ndarray]
            Our input for our marginal function.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.

        Returns
        -------
        values: Union[pd.DataFrame, np.ndarray]
            implemented method values.
        """
        self._fit_check()

        if x is None:
            x = self._cdf_data if func_str == 'ppf' else self._data
        else:
            check_multivariate_data(x, self._num_variables)
            x = self._typekeeper.match_secondary_input(x)
            x = check_multivariate_data(x, self._num_variables)

        values: np.ndarray = np.full(x.shape, np.NaN, dtype=float)
        for index in range(self._num_variables):
            marginal_dist = self._fitted_marginals[index]
            values[:, index] = eval(f"marginal_dist.{func_str}(x[:, index])")

        if match_datatype:
            return self._typekeeper.type_keep_from_2d_array(values)
        return values

    def marginal_pdfs(self, x: Union[pd.DataFrame, np.ndarray] = None,
                      match_datatype: bool = True) \
            -> Union[pd.DataFrame, np.ndarray]:
        """Calculates the pdf values for each univariate marginal distribution
        for a given set of observations x.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the marginal pdf values at.
            Must be the same dimension as the number of variables.
            If None passes, the pdf values of the sample used in the original
            dataset are returned.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.

        Returns
        --------
        marginal_pdf_values: Union[pd.DataFrame, np.ndarray]
            Marginal pdf values.
        """
        return self._pdfs_cdf_ppfs_logpdfs_inputs('pdf', x, match_datatype)

    def marginal_cdfs(self, x: Union[pd.DataFrame, np.ndarray] = None,
                      match_datatype: bool = True) \
            -> Union[pd.DataFrame, np.ndarray]:
        """Calculates the cdf values for each univariate marginal
        distribution for a given set of observations x.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the marginal cdf P(X<=x), at.
            Must be the same dimension as the number of variables.
            If None passes, the cdf values of the sample used in the original
            dataset are returned.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.

        Returns
        --------
        marginal_cdf_values: Union[pd.DataFrame, np.ndarray]
            Marginal cdf values.
        """
        return self._pdfs_cdf_ppfs_logpdfs_inputs('cdf', x, match_datatype)

    def marginal_ppfs(self, q: Union[pd.DataFrame, np.ndarray] = None,
                      match_datatype: bool = True) \
            -> Union[pd.DataFrame, np.ndarray]:
        """Calculates the ppf values for each univariate marginal
        distribution for a given set of observations x.

        Parameters
        -----------
        q: Union[pd.DataFrame, np.ndarray]
            The quartile values to evaluate the marginal ppf, cdf^-1(q), at.
            Must be the same dimension as the number of variables.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.

        Returns
        -------
        marginal_ppf_values: Union[pd.DataFrame, np.ndarray]
            Marginal ppf values.
        """
        return self._pdfs_cdf_ppfs_logpdfs_inputs('ppf', q, match_datatype)

    def marginal_logpdfs(self, x: Union[pd.DataFrame, np.ndarray] = None,
                         match_datatype: bool = True) \
            -> Union[pd.DataFrame, np.ndarray] :
        """Calculates the pdf values for each univariate marginal distribution
        for a given set of observations x.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the marginal log-pdf at.
            Must be the same dimension as the number of variables.
            If None passes, the log-pdf values of the sample used in the
            original dataset are returned.
        match_datatype: bool
            True to output the same datatype as the input.
            False to output a np.ndarray.
            Default is True.

        Returns
        -------
        marginal_logpdf_values: Union[pd.DataFrame, np.ndarray]
            Marginal log-pdf values.
        """
        return self._pdfs_cdf_ppfs_logpdfs_inputs('logpdf', x, match_datatype)

    def marginal_rvs(self, size: int, match_datatype: bool = True) \
            -> Union[pd.DataFrame, np.ndarray]:
        """Randomly samples values from the marginal distributions.

        Parameters
        ----------
        size: int
            The number of samples to generate from each marginal distribution.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.

        Returns
        ---------
        marginal_rvs_values: Union[pd.DataFrame, np.ndarray]
            A random sample with shape (size, num_variables)
        """
        self._fit_check()

        if not (isinstance(size, int) and size > 0):
            raise TypeError('size must be a positive integer')

        rvs_values: np.ndarray = np.full((size, self._num_variables), np.NaN,
                                         dtype=float)
        for index in range(self._num_variables):
            marginal_dist = self._fitted_marginals[index]
            rvs_values[:, index] = marginal_dist.rvs((size, ))

        if match_datatype:
            return self._typekeeper.type_keep_from_2d_array(rvs_values)
        return rvs_values

    def pairplot(self, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True,
                 axes_names: Iterable = None, plot_kde: bool = True,
                 num_generate: int = 10 ** 3, show: bool = True) -> None:
        """Produces a pair-plot of each fitted marginal distribution.

        Parameters
        ----------
        color : str
            The matplotlib.pyplot color to use in your plots.
            Default is 'royalblue'.
        alpha : float
            The matplotlib.pyplot alpha to use in your plots.
            Default is 1.0
        figsize: tuple
            The matplotlib.pyplot figsize tuple to size the overall figure.
            Default figsize is (8,8)
        grid : bool
            True to include a grid in each pair-plot. False for no grid.
            Default is True.
        axes_names: Iterable
            The names of your axes / variables to use in your plots.
            If provided, must be an iterable with the same length as the number
            of variables.
        plot_kde: bool
            True to plot the KDE of your marginal distributions in the
            diagonal plots
            Default is True.
        num_generate: int
            The number of random variables to generate from each marginal
            distribution, to use as data for the pair-plots.
            Default is 1000.
        show: bool
            True to display the pair-plots when the method is called.
            Default is True.
        """
        # checks
        self._fit_check()

        # checking arguments
        if axes_names is None:
            pass
        elif not (isinstance(axes_names, Iterable) and
                  len(axes_names) == self._num_variables):
            raise TypeError("invalid argument type in pairplot. "
                            "check axes_names is None or a tuple with an "
                            "element for each variable.")

        if (not isinstance(num_generate, int)) or num_generate <= 0:
            raise TypeError("num_generate must be a posivie integer")

        # data for plot
        plot_df: pd.DataFrame = self.marginal_rvs(num_generate, True)
        if not isinstance(plot_df, pd.DataFrame):
            plot_df = pd.DataFrame(plot_df)
        if axes_names is not None:
            plot_df.columns = axes_names

        # plotting
        title: str = f"{self.name} Data Pair-Plot"
        pair_plot(plot_df, title, color, alpha, figsize, grid, plot_kde, show)

    @property
    def marginals(self) -> dict:
        """Returns a dict with univariate marginal distributions as values
        and indices as keys."""
        if not self._fitted:
            raise FitError('MarginalFitter object has not been fitted.')
        return self._fitted_marginals.copy()

    @property
    def summary(self) -> pd.DataFrame:
        """A pd.DataFrame containing a summary of the fitted marginals"""
        if not self._fitted:
            raise FitError('MarginalFitter object has not been fitted.')
        return self._summary.copy()

    @property
    def num_variables(self) -> int:
        """The number of variables present in the original dataset."""
        return self._num_variables

    @property
    def fitted(self) -> bool:
        """True if the MarginalFitter object has been fitted to data.
        False otherwise."""
        return self._fitted
