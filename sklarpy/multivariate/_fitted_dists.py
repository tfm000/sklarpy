# Contains code for fitted multivariate models
import numpy as np
import pandas as pd
from typing import Union, Iterable

from sklarpy.utils._params import Params
from sklarpy.utils._type_keeper import TypeKeeper
from sklarpy.utils._copy import Copyable
from sklarpy.utils._serialize import Savable

__all__ = ['FittedContinuousMultivariate']


class FittedContinuousMultivariate(Savable, Copyable):
    """A fitted continuous multivariate model."""
    def __init__(self, obj, fit_info: dict):
        """
        A fitted continuous multivariate model.

        Parameters
        ----------
        obj: PreFitContinuousMultivariate
            A PreFitContinuousMultivariate object
        fit_info: dict
            A dictionary containing information about the fit.
        """
        self.__obj = obj
        self.__fit_info: dict = fit_info
        Savable.__init__(self, self.__obj.name)

    def __str__(self) -> str:
        return f"FittedContinuous{self.name.title()}Distribution"

    def __repr__(self) -> str:
        return self.__str__()

    def logpdf(self, x: Union[pd.DataFrame, np.ndarray],
               match_datatype: bool = True, **kwargs
               ) -> Union[pd.DataFrame, np.ndarray]:
        """The log of the multivariate probability density function.

       Parameters
       ----------
       x: Union[pd.DataFrame, np.ndarray]
           The values to evaluate the multivariate log-pdf function at.
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
        return self.__obj.logpdf(x, self.params, match_datatype, **kwargs)

    def pdf(self, x: Union[pd.DataFrame, np.ndarray],
            match_datatype: bool = True, **kwargs
            ) -> Union[pd.DataFrame, np.ndarray]:
        """The multivariate probability density function.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the multivariate pdf function at.
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
        return self.__obj.pdf(x, self.params, match_datatype, **kwargs)

    def cdf(self, x: Union[pd.DataFrame, np.ndarray],
            match_datatype: bool = True, **kwargs
            ) -> Union[pd.DataFrame, np.ndarray]:
        """The multivariate cumulative density function.

        This may take time to evaluate for certain distributions, due to
        d-dimensional numerical integration. In these cases, mc_cdf will likely
        evaluate faster.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the multivariate cdf function at.
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
        return self.__obj.cdf(x, self.params, match_datatype, **kwargs)

    def mc_cdf(self, x: Union[pd.DataFrame, np.ndarray],
               match_datatype: bool = True, num_generate: int = 10 ** 4,
               show_progress: bool = False, **kwargs
               ) -> Union[pd.DataFrame, np.ndarray]:
        """The monte-carlo numerical approximation of the multivariate cdf
        function. The standard cdf function may take time to evaluate for
        certain distributions, due to d-dimensional numerical integration. In
        these cases, mc_cdf will likely evaluate faster.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the multivariate cdf function at.
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
        return self.__obj.mc_cdf(
            x, params=self.params, match_datatype=match_datatype,
            num_generate=num_generate, show_progress=show_progress, **kwargs)

    def rvs(self, size: tuple, match_datatype: bool = True
            ) -> Union[pd.DataFrame, np.ndarray]:
        """The random variable generator function.

        Parameters
        ----------
        size: int
            How many multivariate random samples to generate from the
            multivariate distribution.
        match_datatype: bool
            Optional. True to return the generated random variables in the same
            format as the fitted dataset (if the model was fitted to data).
            Default is True.

        Returns
        -------
        rvs: Union[pd.DataFrame, np.ndarray]
            Multivariate array of random variables, sampled from the
            multivariate distribution.
        """
        rvs_array: np.ndarray = self.__obj.rvs(size, self.params)
        type_keeper: TypeKeeper = self.__fit_info['type_keeper']
        return type_keeper.type_keep_from_2d_array(rvs_array, match_datatype)

    def __likelihood_loglikelihood_aic_bic(
            self, func_str: str, data: Union[pd.DataFrame, np.ndarray]
    ) -> float:
        """Utility function able to implement likelihood, loglikelihood, aic
        and bic methods without duplicate code.

        Parameters
        ----------
        func_str: str
            The name of the method to implement.
        data: Union[pd.DataFrame, np.ndarray]
            Our input data for the functions of the multivariate distribution.

        Returns
        -------
        output: float
            Implemented method values.
        """
        if data is None:
            return self.__fit_info[func_str]
        obj = self.__obj
        return eval(f"obj.{func_str}(data, self.params)")

    def likelihood(self, data: Union[pd.DataFrame, np.ndarray] = None
                   ) -> float:
        """The likelihood function.

        Parameters
        ----------
        data: Union[pd.DataFrame, np.ndarray]
            Optional.
            The values to evaluate the likelihood function at.
            If None, the likelihood function values of the distribution fit is
            returned.

        Returns
        -------
        likelihood: float
            likelihood function value.
        """
        return self.__likelihood_loglikelihood_aic_bic('likelihood', data)

    def loglikelihood(self, data: Union[pd.DataFrame, np.ndarray] = None
                      ) -> float:
        """The log-likelihood function.

        Parameters
        ----------
        data: Union[pd.DataFrame, np.ndarray]
            Optional.
            The values to evaluate the loglikelihood function at.
            If None, the loglikelihood function values of the distribution fit
            is returned.

        Returns
        -------
        loglikelihood: float
            log-likelihood function value.
        """
        return self.__likelihood_loglikelihood_aic_bic('loglikelihood', data)

    def aic(self, data: Union[pd.DataFrame, np.ndarray] = None) -> float:
        """The Akaike Information Criterion (AIC) function.

        Parameters
        ----------
        data: Union[pd.DataFrame, np.ndarray]
            Optional.
            The values to evaluate the AIC function at.
            If None, the AIC function values of the distribution fit is
            returned.

        Returns
        -------
        aic: float
            AIC function value.
        """
        return self.__likelihood_loglikelihood_aic_bic('aic', data)

    def bic(self, data: Union[pd.DataFrame, np.ndarray] = None) -> float:
        """The Bayesian Information Criterion (BIC) function.

        Parameters
        ----------
        data: Union[pd.DataFrame, np.ndarray]
            Optional.
            The values to evaluate the AIC function at.
            If None, the BIC function values of the distribution fit is
            returned.

        Returns
        -------
        bic: float
            BIC function value.
        """
        return self.__likelihood_loglikelihood_aic_bic('bic', data)

    def marginal_pairplot(self, color: str = 'royalblue', alpha: float = 1.0,
                          figsize: tuple = (8, 8), grid: bool = True,
                          axes_names: Iterable = None, plot_kde: bool = True,
                          num_generate: int = 10 ** 3, show: bool = True,
                          **kwargs) -> None:
        """Produces a pair-plot of each marginal distribution of the
        multivariate distribution.

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
            If provided, must be an iterable with the same length as the
            number of variables. If None provided, the axes will be labeled
            using the names of the variables provided during the joint
            distribution fit, if possible.
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
        if axes_names is None:
            type_keeper: TypeKeeper = self.__fit_info['type_keeper']
            if type_keeper.original_type == pd.DataFrame:
                axes_names = type_keeper.original_info['other']['cols']

        self.__obj.marginal_pairplot(
            params=self.params, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, plot_kde=plot_kde,
            num_generate=num_generate, show=show)

    def _threeD_plot(self, func_name: str, var1_range: np.ndarray,
                     var2_range: np.ndarray, color: str, alpha: float,
                     figsize: tuple, grid: bool, axes_names: Iterable,
                     zlim: tuple, num_points: int, show_progress, show: bool,
                     mc_num_generate: int = None) -> None:
        """Utility function able to implement pdf_plot, cdf_plot and
        mc_cdf_plot methods without duplicate code.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        Parameters
        ----------
        func_str: str
            The name of the method to implement
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            taken to be the bounds of each variable observed during the joint
            distribution fit - the fitted_bounds.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            taken to be the bounds of each variable observed during the joint
            distribution fit - the fitted_bounds.
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
            The names of your axes / variables to use in your plots.
            If provided, must be an iterable with the same length as the
            number of variables. If None provided, the axes will be labeled
            using the names of the variables provided during the joint
            distribution fit, if possible.
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
        # argument checks
        if axes_names is None:
            type_keeper: TypeKeeper = self.__fit_info['type_keeper']
            if type_keeper.original_type == pd.DataFrame:
                axes_names = type_keeper.original_info['other']['cols']

        if self.num_variables != 2:
            raise NotImplementedError(
                f"{func_name}_plot is not implemented when the number of "
                f"variables is not 2.")

        if (not isinstance(num_points, int)) or (num_points <= 0):
            raise TypeError("num_points must be a strictly positive integer.")

        # creating our ranges
        fitted_bounds: np.ndarray = self.__fit_info['fitted_bounds']
        if var1_range is None:
            var1_range: np.ndarray = np.linspace(
                fitted_bounds[0][0], fitted_bounds[0][1],
                num_points, dtype=float)
        if var2_range is None:
            var2_range: np.ndarray = np.linspace(
                fitted_bounds[1][0], fitted_bounds[1][1],
                num_points, dtype=float)

        # plotting
        self.__obj._threeD_plot(
            func_name, var1_range=var1_range, var2_range=var2_range,
            params=self.params, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim, num_generate=0,
            num_points=num_points, show_progress=show_progress, show=show,
            mc_num_generate=mc_num_generate)

    def pdf_plot(self, var1_range: np.ndarray = None,
                 var2_range: np.ndarray = None, color: str = 'royalblue',
                 alpha: float = 1.0, figsize: tuple = (8, 8),
                 grid: bool = True, axes_names: tuple = None,
                 zlim: tuple = (None, None), num_points: int = 100,
                 show_progress: bool = True, show: bool = True,
                 **kwargs) -> None:
        """Produces a 3D plot of the multivariate distribution's pdf / density
        function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        Parameters
        ----------
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            taken to be the bounds of each variable observed during the joint
            distribution fit - the fitted_bounds.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            taken to be the bounds of each variable observed during the joint
            distribution fit - the fitted_bounds.
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
            'pdf', var1_range=var1_range, var2_range=var2_range, color=color,
            alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names,
            zlim=zlim, num_points=num_points, show_progress=show_progress,
            show=show)

    def cdf_plot(self, var1_range: np.ndarray = None,
                 var2_range: np.ndarray = None, color: str = 'royalblue',
                 alpha: float = 1.0, figsize: tuple = (8, 8),
                 grid: bool = True, axes_names: tuple = None,
                 zlim: tuple = (None, None), num_points: int = 100,
                 show_progress: bool = True, show: bool = True,
                 **kwargs) -> None:
        """Produces a 3D plot of the multivariate distribution's cdf /
        cumulative density function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        This may take time to evaluate for certain distributions,
        due to d-dimensional numerical integration. In these cases,
        mc_cdf_plot will likely evaluate faster.

        Parameters
        ----------
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            taken to be the bounds of each variable observed during the joint
            distribution fit - the fitted_bounds.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            taken to be the bounds of each variable observed during the joint
            distribution fit - the fitted_bounds.
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
            'cdf', var1_range=var1_range, var2_range=var2_range, color=color,
            alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names,
            zlim=zlim, num_points=num_points, show_progress=show_progress,
            show=show)

    def mc_cdf_plot(self, var1_range: np.ndarray = None,
                    var2_range: np.ndarray = None,
                    mc_num_generate: int = 10 ** 4, color: str = 'royalblue',
                    alpha: float = 1.0, figsize: tuple = (8, 8),
                    grid: bool = True, axes_names: tuple = None,
                    zlim: tuple = (None, None), num_points: int = 100,
                    show_progress: bool = True, show: bool = True,
                    **kwargs) -> None:
        """Produces a 3D plot of the multivariate distribution's cdf /
        cumulative density function, using monte-carlo numerical approximation.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        This may take time to evaluate for certain distributions,
        due to d-dimensional numerical integration. In these cases,
        mc_cdf_plot will likely evaluate faster.

        Parameters
        ----------
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            taken to be the bounds of each variable observed during the joint
            distribution fit - the fitted_bounds.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            taken to be the bounds of each variable observed during the joint
            distribution fit - the fitted_bounds.
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
            'mc_cdf', var1_range=var1_range, var2_range=var2_range,
            color=color, alpha=alpha, figsize=figsize, grid=grid,
            axes_names=axes_names, zlim=zlim, num_points=num_points,
            show_progress=show_progress, show=show,
            mc_num_generate=mc_num_generate)

    @property
    def params(self) -> Params:
        """Returns a model specific Params object containing the fitted
        parameters defining the multivariate model."""
        return self.__fit_info.copy()['params']

    @property
    def num_params(self) -> int:
        """Returns the number of parameters defining the multivariate model."""
        return self.__fit_info.copy()['num_params']

    @property
    def num_scalar_params(self) -> int:
        """Returns the number of unique scalar parameters defining the
        multivariate model."""
        return self.__fit_info.copy()['num_scalar_params']

    @property
    def num_variables(self) -> int:
        """Returns the number of variables the distribution was fitted too."""
        return self.__fit_info.copy()['num_variables']

    @property
    def fitted_num_data_points(self) -> int:
        """Returns the number of data points the distribution was fitted too.
        """
        return self.__fit_info['num_data_points']

    @property
    def converged(self) -> bool:
        """Returns True if the distribution fit converged successfully.
        False otherwise."""
        return self.__fit_info.copy()['success']

    @property
    def summary(self) -> pd.DataFrame:
        """A dataframe containing summary information of the distribution fit.
        """
        index: list = [
            'Distribution', '#Variables', '#Params', '#Scalar Params',
            'Converged', 'Likelihood', 'Log-Likelihood', 'AIC', 'BIC',
            '#Fitted Data Points'
        ]

        data: list = [
            self.name, self.num_variables, self.num_params,
            self.num_scalar_params, self.converged, self.likelihood(),
            self.loglikelihood(), self.aic(), self.bic(),
            self.fitted_num_data_points
        ]
        return pd.DataFrame(data, index=index, columns=['summary'])
