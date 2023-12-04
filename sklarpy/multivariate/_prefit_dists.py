# Contains code for pre-fitted multivariate models
from typing import Union, Callable, Tuple, Iterable
import numpy as np
import pandas as pd
from abc import abstractmethod
from collections import deque
import scipy.integrate
from scipy.optimize import differential_evolution

from sklarpy.utils._type_keeper import TypeKeeper
from sklarpy.utils._iterator import get_iterator
from sklarpy.utils._not_implemented import NotImplementedBase
from sklarpy.utils._params import Params
from sklarpy.utils._input_handlers import check_multivariate_data
from sklarpy.utils._errors import FitError
from sklarpy.plotting._pair_plot import pair_plot
from sklarpy.plotting._threeD_plot import threeD_plot
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy.misc import CorrelationMatrix

__all__ = ['PreFitContinuousMultivariate']


class PreFitContinuousMultivariate(NotImplementedBase):
    """A pre-fit continuous multivariate model."""
    _DATA_FIT_METHODS: Tuple[str] = ('mle', )

    def __init__(self, name: str, params_obj: Params, num_params: int,
                 max_num_variables: int):
        """A pre-fit continuous multivariate model.

        Parameters
        ----------
        name : str
            The name of the multivariate object.
            Used when saving, if a file path is not specified and/or for
            additional identification purposes.
        params_obj: Params
            The SklarPy Params object associated with this specific
            multivariate probability distribution.
        num_params: int
            The number of parameters which define this distribution.
        max_num_variables: int
            The maximum number of variables this multivariate probability
            distribution is defined for.
            i.e. 2 for bivariate models.
        """
        self._name: str = name
        self._params_obj: Params = params_obj
        self._num_params: int = num_params
        self._max_num_variables: int = max_num_variables

    def __str__(self) -> str:
        return f"PreFitContinuous{self.name.title()}Distribution"

    def __repr__(self) -> str:
        return self.__str__()

    def _get_x_array(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Converts the user's data input into a numpy array and performs
        checks.

        Raises errors if data is not in the correct format / type.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            User provided data values to check and convert.

        Returns
        -------
        x_array: np.ndarray
            numpy array of the multivariate data.
        """
        x_array: np.ndarray = check_multivariate_data(x, allow_1d=True,
                                                      allow_nans=True)

        if x_array.shape[1] > self._max_num_variables:
            raise ValueError(f"too many variables for {self.name}")

        return x_array

    def _check_loc_shape(self, loc: np.ndarray, shape: np.ndarray,
                         definiteness: str, ones: bool) -> None:
        """Checks the location vector and shape matrix parameters are valid,
        raising errors if they are not.

        Parameters
        ----------
        loc: np.ndarray
            The location vector of a multivariate model.
        shape: np.ndarray
            The square, symmetric shape matrix of a multivariate model.
        definiteness: str
            Performs checks on the definiteness of the shape matrix.
            See CorrelationMatrix._check_matrix for specifics.
        ones: bool
            True to require the shape matrix to contain all ones in the
            diagonal. See CorrelationMatrix._check_matrix for specifics.
        """
        # checking location vector
        loc_error: bool = False
        if not isinstance(loc, np.ndarray):
            loc_error = True
        num_variables: int = loc.size
        if num_variables <= 0:
            loc_error = True
        if loc_error:
            raise TypeError(
                "loc vector must be a numpy array with non-zero size.")

        # checking shape matrix
        if not isinstance(shape, np.ndarray):
            raise TypeError("shape matrix must be a numpy array.")
        elif (shape.shape[0] != num_variables) or \
                (shape.shape[1] != num_variables):
            raise ValueError("shape matrix of incorrect dimension.")

        CorrelationMatrix._check_matrix('Shape', definiteness, ones,
                                        shape, True)

    @abstractmethod
    def _check_params(self, params: tuple, **kwargs) -> None:
        """Checks the parameters of the multivariate model and raises an error
        if one or more is not valid.

        Parameters
        ----------
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        kwargs:
            Model specific kwargs.
        """
        if len(params) != self.num_params:
            raise ValueError("Incorrect number of params given by user")

    def _get_params(self, params: Union[Params, tuple], **kwargs) -> tuple:
        """Converts the user's params input into tuple form and then checks
        the parameters of the multivariate model and raises an error if one or
        more is invalid.

        Parameters
        ----------
        params : Union[Params, tuple]
            The parameters which define the multivariate model.
            Can be either the SklarPy Params object associated with this
            specific multivariate probability distribution, or a tuple
            containing parameter values in the correct index order.
        kwargs:
            See below.

        Keyword Arguments
        -----------------
        check_params : bool
            True to call _check_params on the params tuple, False to not.
            Default is True.
        kwargs:
            Model specific kwargs to pass to _check_params.

        Returns
        -------
        params_tuple: tuple
            The parameters which define the multivariate model, in tuple form.
        """
        if isinstance(params, self._params_obj):
            params = params.to_tuple
        elif not isinstance(params, tuple):
            raise TypeError("params must be a valid params object or a tuple.")

        if kwargs.get('check_params', True):
            self._check_params(params, **kwargs)
        return params

    @abstractmethod
    def _get_dim(self, params: tuple) -> int:
        """Retrieves the number of variables implied by the given parameters.

        Parameters
        ----------
        params : tuple
            The parameters which define the multivariate model, in tuple form.

        Return
        -------
        d: int
            The number of variables implied by the given parameters.
        """

    def _check_dim(self, data: np.ndarray, params: tuple) -> None:
        """Checks the dimensions of the dataset match that of the parameters.
        Raises an error if the dimensions do not match.

        Parameters
        ----------
        data : np.ndarray
            The dataset provided by the user.
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        """
        if data.shape[1] != self._get_dim(params):
            raise ValueError("Dimensions implied by parameters do not match "
                             "those of the dataset.")

    def _singlular_cdf(self, num_variables: int, xrow: np.ndarray,
                       params: tuple) -> float:
        """Returns the cumulative distribution function value for a single set
        of variable observations, by numerically integrating the multivariate
        pdf.

        Parameters
        ----------
        num_variables : int
            The number of variables.
        xrow : np.ndarray
            A single set of variable observations.
        params : tuple
            The parameters which define the multivariate model, in tuple form.

        Returns
        -------
        single_cdf : float
            The cumulative distribution function value for a single set
            of variable observations
        """
        def integrable_pdf(*xrow) -> float:
            xrow = np.asarray(xrow, dtype=float)
            return float(self.pdf(xrow, params, match_datatype=False))

        ranges = [[-np.inf, float(xrow[i])] for i in range(num_variables)]
        res: tuple = scipy.integrate.nquad(integrable_pdf, ranges)
        return res[0]

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        """The cumulative distribution function of the multivariate
        distribution.

        If a closed form expression for the multivariate distribution exists,
        overwrite this method with it. If not, the cdf values will be
        calculated by numerically integrating the multivariate pdf.

        Parameters
        ----------
        x: np.ndarray
            numpy array of multivariate data.
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        kwargs:
            See below.

        Keyword Arguments
        ------------------
        show_progress: bool
            True to display the progress of the cdf calculations.
            Default is False.

        Returns
        -------
        cdf_array: np.ndarray
            numpy array of cumulative distribution values.
        """
        num_variables: int = x.shape[1]

        show_progress: bool = kwargs.get('show_progress', False)
        iterator = get_iterator(x, show_progress, "calculating cdf values")

        try:
            return np.array([self._singlular_cdf(num_variables, xrow, params)
                             for xrow in iterator], dtype=float)
        except NotImplementedError:
            self._not_implemented('cdf')

    def _logpdf(self, x: np.ndarray, params: tuple,  **kwargs) -> np.ndarray:
        """The log of the probability density function of the multivariate
        distribution.

        To be overwritten by child classes.

        Parameters
        ----------
        x: np.ndarray
            numpy array of multivariate data.
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        kwargs:
            Model specific kwargs.

        Returns
        -------
        logpdf_array : np.ndarray
            numpy array of log-pdf values.
        """
        self._not_implemented('log-pdf')

    def _pdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        """The probability density function of the multivariate distribution.

        To be overwritten by child classes.

        Parameters
        ----------
        x: np.ndarray
            numpy array of multivariate data.
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        kwargs:
            Model specific kwargs.

        Returns
        -------
        pdf_array : np.ndarray
            numpy array of pdf values.
        """
        self._not_implemented('pdf')

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        """The random variable generator function of the multivariate
        distribution.

        To be overwritten by child classes.

        Parameters
        ----------
        size: int
            How many multivariate random samples to generate from the
            multivariate distribution.
        params : tuple
            The parameters which define the multivariate model, in tuple form.

        Returns
        -------
        rvs_array: np.ndarray
            Multivariate array of random variables, sampled from the
            multivariate distribution.
        """
        self._not_implemented('rvs')

    def _logpdf_pdf_cdf(self, func_str: str,
                        x: Union[pd.DataFrame, np.ndarray],
                        params: Union[Params, tuple],
                        match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """Utility function able to implement logpdf, pdf and cdf methods
        without duplicate code.

        Parameters
        ----------
        func_str: str
            The name of the method to implement.
        x: Union[pd.DataFrame, np.ndarray]
            Our input data for our functions of the multivariate distribution.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        match_datatype: bool
            Optional. True to convert the user's datatype to match the
            original.
            Default is True.
        kwargs:
            kwargs to pass to the implemented method.

        Returns
        -------
        output: Union[pd.DataFrame, np.ndarray]
            Implemented method values, transformed into the user's original
            datatype, if desired.
        """
        # checking arguments
        if func_str not in ('logpdf', 'pdf', 'cdf'):
            raise ValueError("func_name invalid")
        x_array: np.ndarray = self._get_x_array(x)
        params_tuple: tuple = self._get_params(params, **kwargs)
        self._check_dim(data=x_array, params=params_tuple)

        shape: tuple = x_array.shape

        # only calculating for non-nan rows
        output: np.ndarray = np.full((x_array.shape[0], ), np.nan)
        mask: np.ndarray = np.isnan(x_array).any(axis=1)
        if mask.sum() == shape[0]:
            # all provided data is nans
            return output

        # evaluating method
        values: np.ndarray = eval(f"self._{func_str}(x_array[~mask], "
                                  f"params_tuple, **kwargs)")
        output[~mask] = values
        return TypeKeeper(x).type_keep_from_1d_array(
            output, match_datatype=match_datatype, col_name=[func_str])

    def logpdf(self, x: Union[pd.DataFrame, np.ndarray],
               params: Union[Params, tuple], match_datatype: bool = True,
               **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """The log of the multivariate probability density function.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the multivariate log-pdf function at.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        match_datatype: bool
            Optional. True to convert the user's datatype to match the
            original.
            Default is True.

        Returns
        -------
        logpdf: Union[pd.DataFrame, np.ndarray]
            log-pdf values, transformed into the user's original
            datatype, if desired.
        """
        try:
            # using logpdf when possible
            return self._logpdf_pdf_cdf("logpdf", x, params, match_datatype,
                                        **kwargs)
        except NotImplementedError:
            # using pdf if possible
            try:
                pdf_values: np.ndarray = self._logpdf_pdf_cdf("pdf", x, params,
                                                              False, **kwargs)
                logpdf_values: np.ndarray = np.log(pdf_values)
                return TypeKeeper(x).type_keep_from_1d_array(
                    logpdf_values, match_datatype, col_name=['logpdf'])
            except NotImplementedError:
                # raising a function specific exception
                self._not_implemented('logpdf')

    def pdf(self, x: Union[pd.DataFrame, np.ndarray],
            params: Union[Params, tuple], match_datatype: bool = True,
            **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """The multivariate probability density function.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the multivariate pdf function at.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        match_datatype: bool
            Optional. True to convert the user's datatype to match the
            original.
            Default is True.

        Returns
        -------
        pdf: Union[pd.DataFrame, np.ndarray]
            pdf values, transformed into the user's original
            datatype, if desired.
        """
        try:
            # using logpdf when possible
            logpdf_values: Union[np.ndarray, pd.DataFrame] = \
                self._logpdf_pdf_cdf("logpdf", x, params,
                                     match_datatype, **kwargs)
            return np.exp(logpdf_values)
        except NotImplementedError:
            # using pdf values if possible
            return self._logpdf_pdf_cdf("pdf", x, params,
                                        match_datatype, **kwargs)

    def cdf(self, x: Union[pd.DataFrame, np.ndarray],
            params: Union[Params, tuple], match_datatype: bool = True,
            **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """The multivariate cumulative density function.

        This may take time to evaluate for certain distributions, due to
        d-dimensional numerical integration. In these cases, mc_cdf will likely
        evaluate faster.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the multivariate cdf function at.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        match_datatype: bool
            Optional. True to convert the user's datatype to match the
            original.
            Default is True.

        Returns
        -------
        cdf: Union[pd.DataFrame, np.ndarray]
            cdf values, transformed into the user's original
            datatype, if desired.
        """
        return self._logpdf_pdf_cdf("cdf", x, params, match_datatype, **kwargs)

    def mc_cdf(self, x: Union[pd.DataFrame, np.ndarray],
               params: Union[Params, tuple], match_datatype: bool = True,
               num_generate: int = 10 ** 4, show_progress: bool = False,
               **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """The monte-carlo numerical approximation of the multivariate cdf
        function. The standard cdf function may take time to evaluate for
        certain distributions, due to d-dimensional numerical integration. In
        these cases, mc_cdf will likely evaluate faster.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the multivariate cdf function at.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        match_datatype: bool
            Optional. True to convert the user's datatype to match the
            original.
            Default is True.
        num_generate: int
            The number of random numbers to generate to use when numerically
            approximating the multivariate cdf using monte-carlo.
        show_progress: bool
            True to display the progress of the mc-cdf calculations.
            Default is False.

        Returns
        -------
        mc_cdf: Union[pd.DataFrame, np.ndarray]
            numerical cdf values transformed into the user's original datatype,
            if desired.
        """
        # checking arguments
        x_array: np.ndarray = self._get_x_array(x)
        params_tuple: tuple = self._get_params(params, **kwargs)
        self._check_dim(data=x_array, params=params_tuple)
        if not isinstance(num_generate, int) or (num_generate <= 0):
            raise TypeError("num_generate must be a positive integer")

        shape: tuple = x_array.shape

        # only calculating for non-nan rows
        output: np.ndarray = np.full((x_array.shape[0], ), np.nan)
        mask: np.ndarray = np.isnan(x_array).any(axis=1)
        if mask.sum() == shape[0]:
            # all provided data is nans
            return output

        # whether to show progress
        iterator = get_iterator(x_array[~mask], show_progress,
                                "calculating monte-carlo cdf values")

        # generating rvs
        rvs = kwargs.get("rvs", None)
        rvs_array: np.ndarray = self.rvs(num_generate, params) if rvs is None \
            else check_multivariate_data(rvs, num_variables=x_array.shape[1])

        # calculating cdf values via mc
        mc_cdf_values: deque = deque()
        for row in iterator:
            mc_cdf_val: float = np.all(rvs_array <= row, axis=1).sum() / \
                                num_generate
            mc_cdf_values.append(mc_cdf_val)
        mc_cdf_values: np.ndarray = np.asarray(mc_cdf_values)

        output[~mask] = mc_cdf_values
        return TypeKeeper(x).type_keep_from_1d_array(
            output, match_datatype, col_name=['mc cdf'])

    def rvs(self, size: int, params: Union[Params, tuple]) -> np.ndarray:
        """The random variable generator function.

        Parameters
        ----------
        size: int
            How many multivariate random samples to generate from the
            multivariate distribution.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.

        Returns
        -------
        rvs: np.ndarray
            Multivariate array of random variables, sampled from the
            multivariate distribution.
        """
        # checks
        if not isinstance(size, int):
            raise TypeError("size must be a positive integer")
        elif size <= 0:
            raise ValueError("size must be a positive integer")

        params_tuple: tuple = self._get_params(params)

        # returning rvs
        return self._rvs(size, params_tuple)

    def likelihood(self, data: Union[pd.DataFrame, np.ndarray],
                   params: Union[Params, tuple]) -> float:
        """The likelihood function.

        Parameters
        ----------
        data: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the likelihood function at.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.

        Returns
        -------
        likelihood: float
            likelihood function value.
        """
        try:
            loglikelihood: float = self.loglikelihood(data, params)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('likelihood')
        return np.exp(loglikelihood)

    def loglikelihood(self, data: Union[pd.DataFrame, np.ndarray],
                      params: Union[Params, tuple], **kwargs) -> float:
        """The log-likelihood function.

        Parameters
        ----------
        data: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the log-likelihood function at.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.

        Returns
        -------
        loglikelihood: float
            log-likelihood function value.
        """
        try:
            logpdf_values: np.ndarray = self.logpdf(data, params, False,
                                                    **kwargs)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('log-likelihood')

        mask: np.ndarray = np.isnan(logpdf_values)
        if np.any(np.isinf(logpdf_values)):
            # returning -np.inf instead of nan
            return -np.inf
        elif mask.sum() == mask.size:
            # all logpdf values are nan, so returning nan
            return np.nan
        return float(np.sum(logpdf_values[~mask]))

    def aic(self, data: Union[pd.DataFrame, np.ndarray],
            params: Union[Params, tuple], **kwargs) -> float:
        """The Akaike Information Criterion (AIC) function.

        Parameters
        ----------
        data: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the AIC function at.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.

        Returns
        -------
        aic: float
            AIC function value.
        """
        try:
            loglikelihood: float = self.loglikelihood(data, params)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('aic')
        data_array: np.ndarray = self._get_x_array(data)
        return 2 * (self.num_scalar_params(data_array.shape[1], **kwargs)
                    - loglikelihood)

    def bic(self, data: Union[pd.DataFrame, np.ndarray],
            params: Union[Params, tuple], **kwargs) -> float:
        """The Bayesian Information Criterion (BIC) function.

        Parameters
        ----------
        data: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the BIC function at.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.

        Returns
        -------
        bic: float
            BIC function value.
        """
        try:
            loglikelihood: float = self.loglikelihood(data, params)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('bic')

        data_array: np.ndarray = self._get_x_array(data)
        num_data_points, d = data_array.shape
        num_data_points -= np.isnan(data_array).sum()
        return -2 * loglikelihood \
               + np.log(num_data_points) * self.num_scalar_params(d, **kwargs)

    def marginal_pairplot(self, params: Union[Params, tuple],
                          color: str = 'royalblue', alpha: float = 1.0,
                          figsize: tuple = (8, 8), grid: bool = True,
                          axes_names: tuple = None, plot_kde: bool = True,
                          num_generate: int = 10 ** 3, show: bool = True,
                          **kwargs) -> None:
        """Produces a pair-plot of each marginal distribution of the
        multivariate distribution.

        Parameters
        ----------
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
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
            of variables. If None provided, axes will not be labeled.
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
        # checking arguments
        if (not isinstance(num_generate, int)) or num_generate <= 0:
            raise TypeError("num_generate must be a posivie integer")

        rvs: np.ndarray = self.rvs(num_generate, params)  # data for plot
        plot_df: pd.DataFrame = pd.DataFrame(rvs)

        if axes_names is None:
            pass
        elif not (isinstance(axes_names, Iterable) and
                  len(axes_names) == rvs.shape[1]):
            raise TypeError("invalid argument type in pairplot. check "
                            "axes_names is None or a iterable with an "
                            "element for each variable.")

        if axes_names is not None:
            plot_df.columns = axes_names

        # plotting
        title: str = f"{self.name.replace('_', ' ').title()} " \
                     f"Marginal Pair-Plot"
        pair_plot(plot_df, title, color, alpha, figsize, grid, plot_kde, show)

    def _threeD_plot(self, func_str: str, var1_range: np.ndarray,
                     var2_range: np.ndarray, params: Union[Params, tuple],
                     color: str, alpha: float, figsize: tuple, grid: bool,
                     axes_names: Iterable, zlim: tuple, num_generate: int,
                     num_points: int, show_progress: bool, show: bool,
                     mc_num_generate: int = None) -> None:
        """Utility function able to implement pdf_plot, cdf_plot and
        mc_cdf_plot methods without duplicate code.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        Parameters
        ----------
        func_str: str
            The name of the method to implement
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            determined by generating a multivariate random sample of size
            num_generate and then taking the observed min and max values of
            each variable from this sample.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            determined by generating a multivariate random sample of size
            num_generate and then taking the observed min and max values of
            each variable from this sample.
        color : str
            The matplotlib.pyplot color to use in your plot.
            Default is 'royalblue'.
        alpha : float
            The matplotlib.pyplot alpha to use in your plot.
            Default is 1.0
        figsize: tuple
            The matplotlib.pyplot figsize tuple to size the overall figure.
            Default figsize is (8,8)
        grid : bool
            True to include a grid in the 3D plot. False for no grid.
            Default is True.
        axes_names: Iterable
            The names of your axes / variables to use in your plot.
            If provided, must be an iterable with the same length as the
            number of variables (length 2). If None provided, the axes will
            be labeled as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the multivariate
            distribution, to determine the bounds of our variables, if
            necessary. See var1_range and var2_range for more information.
            Default is 1000.
        num_points: int
            The number of points to use in your evenly spaced var1_range and
            var2_range arrays, if they are not provided by the user.
            Default is 100.
        show_progress: bool
            Whether to show the progress of the plotting.
            Default is True.
        show: bool
            True to display the plot when the method is called.
            Default is True.
        mc_num_generate: int
            For mc_cdf_plot only.
            The number of multivariate random variables to generate when
            evaluating monte-carlo functions.
            Default is 10,000.
        """
        # checking arguments
        if (var1_range is not None) and (var2_range is not None):
            for var_range in (var1_range, var2_range):
                if not isinstance(var_range, np.ndarray):
                    raise TypeError("var1_range and var2_range must be None "
                                    "or numpy arrays.")

        else:
            rvs_array: np.ndarray = self.rvs(num_generate, params)
            if rvs_array.shape[1] != 2:
                raise NotImplementedError(f"{func_str}_plot is not "
                                          f"implemented when the number of "
                                          f"variables is not 2.")
            xmin, xmax = rvs_array.min(axis=0), rvs_array.max(axis=0)
            var1_range: np.ndarray = np.linspace(xmin[0], xmax[0], num_points)
            var2_range: np.ndarray = np.linspace(xmin[1], xmax[1], num_points)

        if axes_names is None:
            axes_names = ('variable 1', 'variable 2')

        if (mc_num_generate is None) and ('mc' in func_str):
            raise ValueError("mc_num_generate cannot be none for a "
                             "monte-carlo function.")

        # title and name of plot to show user
        plot_name: str = func_str.replace('_', ' ').upper()
        title: str = f"{self.name.replace('_', ' ').title()} {plot_name} Plot"

        # func kwargs
        if 'mc' in func_str:
            rvs = self.rvs(mc_num_generate, params)
        else:
            rvs = None
        func_kwargs: dict = {'params': params, 'match_datatype': False,
                             'show_progress': False, 'rvs': rvs}
        func: Callable = eval(f"self.{func_str}")

        # plotting
        threeD_plot(
            func=func, var1_range=var1_range, var2_range=var2_range,
            func_kwargs=func_kwargs, func_name=plot_name, title=title,
            color=color, alpha=alpha, figsize=figsize, grid=grid,
            axis_names=axes_names, zlim=zlim, show_progress=show_progress,
            show=show)

    def pdf_plot(self, params: Union[Params, tuple],
                 var1_range: np.ndarray = None, var2_range: np.ndarray = None,
                 color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True,
                 axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100,
                 show_progress: bool = True, show: bool = True, **kwargs
                 ) -> None:
        """Produces a 3D plot of the multivariate distribution's pdf / density
        function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        Parameters
        ----------
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            determined by generating a multivariate random sample of size
            num_generate and then taking the observed min and max values of
            each variable from this sample.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            determined by generating a multivariate random sample of size
            num_generate and then taking the observed min and max values of
            each variable from this sample.
        color : str
            The matplotlib.pyplot color to use in your plot.
            Default is 'royalblue'.
        alpha : float
            The matplotlib.pyplot alpha to use in your plot.
            Default is 1.0
        figsize: tuple
            The matplotlib.pyplot figsize tuple to size the overall figure.
            Default figsize is (8,8)
        grid : bool
            True to include a grid in the 3D plot. False for no grid.
            Default is True.
        axes_names: Iterable
            The names of your axes / variables to use in your plot.
            If provided, must be an iterable with the same length as the
            number of variables (length 2). If None provided, the axes will
            be labeled as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the multivariate
            distribution, to determine the bounds of our variables, if
            necessary. See var1_range and var2_range for more information.
            Default is 1000.
        num_points: int
            The number of points to use in your evenly spaced var1_range and
            var2_range arrays, if they are not provided by the user.
            Default is 100.
        show_progress: bool
            Whether to show the progress of the plotting.
            Default is True.
        show: bool
            True to display the plot when the method is called.
            Default is True.
        """
        self._threeD_plot(
            func_str='pdf', var1_range=var1_range, var2_range=var2_range,
            params=params, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show)

    def cdf_plot(self, params: Union[Params, tuple],
                 var1_range: np.ndarray = None, var2_range: np.ndarray = None,
                 color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True,
                 axes_names: tuple = None, zlim: tuple = (0, 1),
                 num_generate: int = 1000, num_points: int = 100,
                 show_progress: bool = True, show: bool = True, **kwargs
                 ) -> None:
        """Produces a 3D plot of the multivariate distribution's cdf /
        cumulative density function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        This may take time to evaluate for certain distributions,
        due to d-dimensional numerical integration. In these cases,
        mc_cdf_plot will likely evaluate faster.

        Parameters
        ----------
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            determined by generating a multivariate random sample of size
            num_generate and then taking the observed min and max values of
            each variable from this sample.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            determined by generating a multivariate random sample of size
            num_generate and then taking the observed min and max values of
            each variable from this sample.
        color : str
            The matplotlib.pyplot color to use in your plot.
            Default is 'royalblue'.
        alpha : float
            The matplotlib.pyplot alpha to use in your plot.
            Default is 1.0
        figsize: tuple
            The matplotlib.pyplot figsize tuple to size the overall figure.
            Default figsize is (8,8)
        grid : bool
            True to include a grid in the 3D plot. False for no grid.
            Default is True.
        axes_names: Iterable
            The names of your axes / variables to use in your plot.
            If provided, must be an iterable with the same length as the
            number of variables (length 2). If None provided, the axes will
            be labeled as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the multivariate
            distribution, to determine the bounds of our variables, if
            necessary. See var1_range and var2_range for more information.
            Default is 1000.
        num_points: int
            The number of points to use in your evenly spaced var1_range and
            var2_range arrays, if they are not provided by the user.
            Default is 100.
        show_progress: bool
            Whether to show the progress of the plotting.
            Default is True.
        show: bool
            True to display the plot when the method is called.
            Default is True.
        """
        self._threeD_plot(
            func_str='cdf', var1_range=var1_range, var2_range=var2_range,
            params=params, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show)

    def mc_cdf_plot(self, params: Union[Params, tuple],
                    var1_range: np.ndarray = None,
                    var2_range: np.ndarray = None,
                    mc_num_generate: int = 10 ** 4, color: str = 'royalblue',
                    alpha: float = 1.0, figsize: tuple = (8, 8),
                    grid: bool = True, axes_names: tuple = None,
                    zlim: tuple = (0, 1), num_generate: int = 1000,
                    num_points: int = 100, show_progress: bool = True,
                    show: bool = True, **kwargs) -> None:
        """Produces a 3D plot of the multivariate distribution's cdf /
        cumulative density function, using monte-carlo numerical approximation.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        This may take time to evaluate for certain distributions,
        due to d-dimensional numerical integration. In these cases,
        mc_cdf_plot will likely evaluate faster.

        Parameters
        ----------
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            determined by generating a multivariate random sample of size
            num_generate and then taking the observed min and max values of
            each variable from this sample.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            determined by generating a multivariate random sample of size
            num_generate and then taking the observed min and max values of
            each variable from this sample.
        mc_num_generate: int
            The number of multivariate random variables to generate when
            evaluating monte-carlo functions.
            Default is 10,000.
        color : str
            The matplotlib.pyplot color to use in your plot.
            Default is 'royalblue'.
        alpha : float
            The matplotlib.pyplot alpha to use in your plot.
            Default is 1.0
        figsize: tuple
            The matplotlib.pyplot figsize tuple to size the overall figure.
            Default figsize is (8,8)
        grid : bool
            True to include a grid in the 3D plot. False for no grid.
            Default is True.
        axes_names: Iterable
            The names of your axes / variables to use in your plot.
            If provided, must be an iterable with the same length as the
            number of variables (length 2). If None provided, the axes will
            be labeled as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the multivariate
            distribution, to determine the bounds of our variables, if
            necessary. See var1_range and var2_range for more information.
            Default is 1000.
        num_points: int
            The number of points to use in your evenly spaced var1_range and
            var2_range arrays, if they are not provided by the user.
            Default is 100.
        show_progress: bool
            Whether to show the progress of the plotting.
            Default is True.
        show: bool
            True to display the plot when the method is called.
            Default is True.
        """
        self._threeD_plot(
            func_str='mc_cdf', var1_range=var1_range, var2_range=var2_range,
            params=params, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show,
            mc_num_generate=mc_num_generate)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return self._name

    @property
    def num_params(self) -> int:
        """Returns the number of parameters defining the multivariate model."""
        return self._num_params

    def _num_shape_scalar_params(self, d: int, copula: bool = False) -> int:
        """Calculates the number of unique parameters in a square, symmetric
        shape matrix.

        Parameters
        ----------
        d: int
            The dimension of the matrix / number of variables.
        copula: bool
            True if the distribution is a copula distribution and the shape
            matrix is a correlation matrix. False otherwise.
            Default is False.

        Returns
        -------
        num_shape_scalar_params: int
            The number of unique parameters in a square, symmetric
            shape matrix.
        """
        num_shape_scalar_params = 0.5 * d * (d-1) if copula \
            else 0.5 * d * (d+1)
        return int(num_shape_scalar_params)

    @abstractmethod
    def num_scalar_params(self, d: int, copula: bool, **kwargs) -> int:
        """Calculates the number of unique scalar parameters defining the
        multivariate model.

        Parameters
        ----------
        d: int
            The dimension of the matrix / number of variables.
        copula: bool
            True if the distribution is a copula distribution. False otherwise.

        Returns
        -------
        num_scalar_params: int
            The number of unique scalar parameters defining the multivariate
            model.
        """

    @staticmethod
    def _shape_from_array(arr: np.ndarray, d: int) -> np.ndarray:
        """Converts a flattened array of shape matrix values into a square
        matrix form. Assumes shape matrix is symmetric.

        Parameters
        ----------
        arr: np.ndarray
            A flattened array of shape matrix values.
        d: int
            The dimension of the matrix / number of variables.

        Returns
        -------
        shape: np.ndarray
            Square shape matrix.
        """
        shape: np.ndarray = np.full((d, d), np.nan, dtype=float)
        np.fill_diagonal(shape, arr[:d])
        shape_non_diag: np.ndarray = arr[d:]
        endpoint: int = 0
        for i in range(d - 1):
            startpoint = endpoint
            endpoint = int(0.5 * (i + 1) * (2 * d - i - 2))
            shape[i, i + 1:] = shape_non_diag[startpoint:endpoint]
            shape[i + 1:, i] = shape_non_diag[startpoint:endpoint]
        return shape

    @staticmethod
    def _bounds_dict_to_tuple(bounds_dict: dict, d: int, as_tuple: bool) \
            -> Union[dict, tuple]:
        """Performs checks on user specified bounds for parameters and places
        them in tuple form for our optimizers.

        Parameters
        ----------
        bounds_dict: dict
            User specified bounds in dictionary form. Keys must be the
            parameter names and values the bounds for each parameter.
        d: int
            The dimension / number of variables.
        as_tuple: bool
            True to return the processed bounds as a tuple or dictionary.

        Returns
        -------
        bounds: Union[dict, tuple]
            The bounds to use in parameter fitting / optimization.
        """
        param_err_msg: str = "bounds must be a tuple of length 2 for scalar " \
                             "params or a matrix of shape (d, 2) for vector " \
                             "params."
        bounds_tuples: deque = deque()
        for param, param_bounds in bounds_dict.items():
            if isinstance(param_bounds, np.ndarray) and \
                    param_bounds.shape == (d, 2):
                param_bounds_tuples: tuple = tuple(map(tuple, param_bounds))
                bounds_tuples.extend(param_bounds_tuples)
            elif isinstance(param_bounds, tuple) and len(param_bounds) == 2:
                bounds_tuples.append(param_bounds)
            else:
                raise ValueError(param_err_msg)
        return tuple(bounds_tuples) if as_tuple else bounds_dict

    @abstractmethod
    def _get_bounds(self, default_bounds: dict, d: int, as_tuple: bool,
                    **kwargs) -> Union[dict, tuple]:
        """Combines the default bounds with any user specified bounds.

        Parameters
        ----------
        default_bounds : dict
            The default bounds to use for each parameter for the distribution
            during parameter fitting / optimization.
        d: int
            The dimension / number of variables.
        as_tuple: bool
            True to return the processed bounds as a tuple or dictionary.

        Returns
        -------
        bounds: [Union, dict, tuple]
            The bounds to use in parameter fitting / optimization.
        """
        bounds_dict: dict = kwargs.get('bounds', {})
        bounds_dict = {param: bounds_dict.get(param, default_bounds[param])
                       for param in default_bounds}

        # removing location vector bounds if copula distribution.
        to_remove = ['loc'] if kwargs.get('copula', False) else []
        return self._remove_bounds(bounds_dict, to_remove, d, as_tuple)

    def _remove_bounds(self, bounds_dict: dict, to_remove: list, d: int,
                       as_tuple: bool) -> Union[dict, tuple]:
        """Removed specified bounds from a bounds dictionary.

        Parameters
        ----------
        bounds_dict: dict
            User specified bounds in dictionary form. Keys must be the
            parameter names and values the bounds for each parameter.
        to_remove: list
            A list of bound names to remove.
        d: int
            The dimension / number of variables.
        as_tuple: bool
            True to return the processed bounds as a tuple or dictionary.

        Returns
        -------
        bounds: Union[dict, tuple]
            The bounds to use in parameter fitting / optimization.
        """
        for bound in to_remove:
            if bound in bounds_dict:
                bounds_dict.pop(bound)
        return self._bounds_dict_to_tuple(bounds_dict, d, as_tuple)

    def _theta_to_params(self, theta: np.ndarray, **kwargs) -> tuple:
        """Converts an array of optimizer outputs into distribution parameters.

        Parameters
        -----------
        theta: np.ndarray
            An array of scalar values representing the distribution parameters.
        kwargs:
            model specific keyword arguments.

        Returns
        --------
        params_tuple: tuple
            The parameters which define the multivariate model, in tuple form.
        """

    def _params_to_theta(self, params: tuple, **kwargs) -> np.ndarray:
        """Converts parameters into a theta numpy array for use in
        optimization.

        Parameters
        ----------
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        copula: bool
            True if the distribution is a copula distribution. False otherwise.
        kwargs:
            model specific keyword arguments.

        Returns
        -------
        theta : np.ndarray
            theta array containing parameter info.
        """

    def _get_mle_objective_func_kwargs(self, data: np.ndarray, **kwargs
                                       ) -> dict:
        """Returns any additional arguments (besides theta and data) required
        for the mle objective function.

        Parameters
        ----------
        data: np.ndarray
            An array of multivariate data to optimize parameters over.
        kwargs:
            model specific keyword arguments.

        Returns
        -------
        kwargs: dict
            Additional arguments required for the mle objective function.
        """

    def _mle_objective_func(self, theta: np.ndarray, data: np.ndarray,
                            kwargs: dict) -> float:
        """The objective function to optimize when performing Maximum
        Likelihood Estimation (MLE).

        Parameters
        ----------
        theta: np.ndarray
            An array of scalar values representing the distribution parameters.
        data: np.ndarray
            An array of multivariate data to optimize parameters over.
        kwargs: dict
            model specific keyword arguments, as a dictionary.

        Returns
        -------
        neg_loglikelihood: float
            The negative log-likelihood value associated with the theta array.
        """
        params: tuple = self._theta_to_params(theta=theta, **kwargs)
        loglikelihood: float = self.loglikelihood(data=data, params=params,
                                                  definiteness=None)
        return np.inf if np.isnan(loglikelihood) else -loglikelihood

    def _mle(self, data: np.ndarray, params0: np.ndarray, bounds: tuple,
             maxiter: int, tol: float, show_progress: bool, **kwargs
             ) -> Tuple[tuple, bool]:
        """Performs Maximum Likelihood Estimation (MLE) to fit / estimate the
        parameters of the distribution from the data.

        We reduce the dimension of the problem when possible.

        We use scipy's implementation of differential evolution as our
        non-convex solver.

        Parameters
        ----------
        data: np.ndarray
            An array of multivariate data to optimize parameters over using
            Maximum Likelihood Estimation (MLE).
        params0: np.ndarray
            An initial estimate of the parameters to use when starting the
            optimization algorithm. These can be a Params object of the
            specific multivariate distribution or a tuple containing these
            parameters in the correct order.
        bounds: tuple
            The bounds to use in parameter fitting / optimization, as a tuple.
        maxiter: int
            The maximum number of iterations to perform by the differential
            evolution solver.
        tol: float
            The tolerance to use when determining convergence, by the
            differential evolution solver.
        show_progress: bool
            True to display the progress of the differential evolution
            optimizer calculations.
        kwargs:
            Model specific keyword arguments.

        See Also
        --------
        scipy.optimize.differential_evolution

        Returns
        -------
        res: Tuple[tuple, bool]
            The parameters optimized to fit the data,
            True if convergence was successful false otherwise.
        """
        # getting theta0
        theta0: np.ndarray = self._params_to_theta(params=params0, **kwargs)

        # getting args to pass to the optimizer
        mle_kwargs: dict = self._get_mle_objective_func_kwargs(data=data,
                                                               **kwargs)
        constraints: tuple = mle_kwargs.pop('constraints', tuple())

        # running optimization
        mle_res = differential_evolution(
            self._mle_objective_func, bounds, args=(data, mle_kwargs),
            maxiter=maxiter, tol=tol, x0=theta0, disp=show_progress,
            constraints=constraints)

        # extracting params from results
        theta: np.ndarray = mle_res['x']
        params: tuple = self._theta_to_params(theta=theta, **mle_kwargs)
        converged: bool = mle_res['success']

        if show_progress:
            print(f"MLE Optimisation Complete. Converged= {converged}"
                  f", f(x)= {mle_res['fun']}")
        return params, converged

    def _get_params0(self, data: np.ndarray, bounds: tuple, **kwargs) -> tuple:
        """Generates an initial estimate of parameters to use in optimization,
        which satisfies any bounding constraints.

        Parameters
        ----------
        data: np.ndarray
            An array of multivariate data to optimize parameters over.
        bounds: tuple
            The bounds to use in parameter fitting / optimization, as a tuple.
        kwargs:
            Model specific keyword arguments.

        Returns
        -------
        params0: np.ndarray
            An initial estimate of the parameters.
        """

    @abstractmethod
    def _fit_given_data_kwargs(self, method: str, data: np.ndarray,
                               **user_kwargs) -> dict:
        """Returns the user specified kwargs, combined with any required
        default arguments.

        Parameters
        ----------
        method : str
            The optimization method to use when fitting the distribution to
            the observed data.
        data : np.ndarray
            The multivariate dataset to fit the distribution's parameters too.
        user_kwargs:
            Method specific optional arguments.

        Returns
        -------
        kwargs : dict
            User specified kwargs, combined with any required default
            arguments.
        """
        if method == 'mle':
            bounds: tuple = self._get_bounds(data, True, **user_kwargs)
            user_kwargs.pop('bounds', None)
            copula: bool = user_kwargs.pop('copula', False)
            kwargs: dict = {'copula': copula, 'bounds': bounds,
                            'maxiter': 1000, 'tol': 0.5,
                            'cov_method': 'pp_kendall', 'min_eig': None,
                            'show_progress': False}
            kwargs['params0'] = self._get_params0(
                data=data, **{**kwargs, **user_kwargs})

        else:
            raise ValueError(f'{method} is not a valid method.')
        return kwargs

    def _fit_given_data(self, data: np.ndarray, method: str, **kwargs) \
            -> Tuple[tuple, bool]:
        """Runs a specified optimization algorithm to fit / estimate the
        parameters of the distribution to observed data.

        Parameters
        ----------
        data : np.ndarray
            The multivariate dataset to fit the distribution's parameters too.
        method : str
            The optimization method to use when fitting the distribution to
            the observed data.
        kwargs:
            Method specific optional arguments.

        Returns
        -------
        res: Tuple[tuple, bool]
            The parameters optimized to fit the data,
            True if convergence was successful false otherwise.
        """
        # getting fit method
        cleaned_method: str = method.lower().strip()\
            .replace('-', '_').replace(' ', '_')
        if cleaned_method not in self._DATA_FIT_METHODS:
            raise ValueError(f'{method} is not a valid data fitting method '
                             f'for {self.name}')

        # getting fit method
        data_fit_func: Callable = eval(f"self._{cleaned_method}")

        # getting additional fit args
        default_kwargs: dict = self._fit_given_data_kwargs(cleaned_method,
                                                           data, **kwargs)
        kwargs_to_skip: tuple = ('q2_options', 'bounds')
        for kwarg, value in default_kwargs.items():
            if (kwarg not in kwargs) and (kwarg not in kwargs_to_skip):
                kwargs[kwarg] = value

        for kwarg in kwargs_to_skip:
            if kwarg in default_kwargs:
                kwargs[kwarg] = default_kwargs[kwarg]

        # converting potential params objects into tuples
        if 'params0' in kwargs and kwargs['params0'] is not None:
            kwargs['params0'] = self._get_params(kwargs['params0'])

        # fitting to data
        return data_fit_func(data=data, **kwargs)

    @abstractmethod
    def _fit_given_params_tuple(self, params: tuple, **kwargs) \
            -> Tuple[dict, int]:
        """Performs checks on user given parameters to use when fitting.

        Parameters
        ----------
        params: tuple
            The parameters which define the multivariate model, in tuple form.

        Returns
        -------
        params_dict, d: Tuple[dict, int]
            Dictionary containing model parameters in the correct order,
            Number of variables model has been fitted too.
        """
        if len(params) != self.num_params:
            raise ValueError("Incorrect number of params given by user")

    def fit(self, data: Union[pd.DataFrame, np.ndarray] = None,
            params: Union[Params, tuple] = None, method: str = 'mle',
            **kwargs) -> FittedContinuousMultivariate:
        """Call to fit parameters to a given dataset or to fit the
        distribution object to a set of specified parameters.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Optional. The multivariate dataset to fit the distribution's
            parameters too. Not required if `params` is provided.
        params : Union[Params, tuple]
            Optional. The parameters of the distribution to fit the object
            too. These can be a Params object of the specific multivariate
            distribution or a tuple containing these parameters in the correct
            order.
        method : str
            When fitting to data only.
            The method to use when fitting the distribution to the observed
            data.
            Default depends on the specific distribution.
        kwargs:
            See individual distributions.

        Returns
        --------
        fitted_multivariate: FittedContinuousMultivariate
            A fitted distribution.
        """
        default_kwargs: dict = {'raise_cov_error': True}
        kwargs = {**default_kwargs, **kwargs}

        fit_info: dict = {}
        if (data is None) and (params is None):
            raise ValueError("data and params cannot both be None when "
                             "fitting.")
        elif params is not None:
            # User has provided a params object or tuple
            num_data_points: int = 0

            # saving params
            if isinstance(params, self._params_obj):
                params: tuple = params.to_tuple
            if isinstance(params, tuple) and len(params) == self.num_params:
                params_dict, num_variables = self._fit_given_params_tuple(
                    params, **kwargs)
            else:
                raise TypeError(f"if params provided, must be a "
                                f"{self._params_obj} type or tuple "
                                f"of length {self.num_params}")
            params: Params = self._params_obj(params_dict, self.name,
                                              num_variables)

            # generating random data for fit evaluation stats
            data: np.ndarray = self.rvs(10**3, params)
            data_array: np.ndarray = data
            success: bool = True
        else:
            # user has provided data to fit

            # getting info from data
            data_array: np.ndarray = check_multivariate_data(
                data, allow_1d=True, allow_nans=False)
            num_variables: int = data_array.shape[1]
            if num_variables > self._max_num_variables:
                raise FitError(f"Too many columns in data to interpret "
                               f"as variables for {self.name} distribution.")

            # fitting parameters to data
            try:
                params_tuple, success = self._fit_given_data(data_array,
                                                             method, **kwargs)
            except Exception as e:
                raise FitError(f"The following error occurred while fitting. "
                               f"\n\n{e}\n\nThis may occur if you have "
                               f"insufficient data to perform the parameter "
                               f"optimization.")
            params_dict, _ = self._fit_given_params_tuple(params_tuple)
            params: Params = self._params_obj(params_dict, self.name,
                                              num_variables)

            num_data_points: int = data_array.shape[0]

        # fitting TypeKeeper object
        type_keeper: TypeKeeper = TypeKeeper(data)

        # calculating fit statistics
        fit_info['likelihood'] = self.likelihood(data, params)
        fit_info['loglikelihood'] = self.loglikelihood(data, params)
        fit_info['aic'] = self.aic(data, params, **kwargs)
        fit_info['bic'] = self.bic(data, params, **kwargs)

        # calculating fit bounds
        fitted_bounds: np.ndarray = np.full((num_variables, 2), np.nan,
                                            dtype=float)
        fitted_bounds[:, 0] = data_array.min(axis=0)
        fitted_bounds[:, 1] = data_array.max(axis=0)
        fit_info['fitted_bounds'] = fitted_bounds

        # other fit values
        fit_info['type_keeper'] = type_keeper
        fit_info['params'] = params
        fit_info['num_variables'] = num_variables
        fit_info['success'] = success
        fit_info['num_data_points'] = num_data_points
        fit_info['num_params'] = len(params)
        fit_info['num_scalar_params'] = self.num_scalar_params(num_variables,
                                                               **kwargs)
        return FittedContinuousMultivariate(self, fit_info)
