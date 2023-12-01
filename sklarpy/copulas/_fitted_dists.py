# Contains code for fitted copula models
import numpy as np
import pandas as pd
from typing import Union, Iterable

from sklarpy.utils._type_keeper import TypeKeeper
from sklarpy.utils._serialize import Savable
from sklarpy.utils._copy import Copyable
from sklarpy.utils._params import Params

__all__ = ['FittedCopula']


class FittedCopula(Savable, Copyable):
    """A fitted copula model"""
    def __init__(self, obj, fit_info: dict):
        """A fitted copula model.

        Parameters
        ----------
        obj: PreFitCopula
            A PreFitCopula object, which defines your FittedCopula object.
        fit_info: dict
            A dictionary containing information about the copula distribution
            fit.
        """
        self.__obj = obj
        self.__fit_info = fit_info
        Savable.__init__(self, self.__obj.name)

    def __str__(self) -> str:
        return f"Fitted{self.name.title()}Copula"

    def __repr__(self) -> str:
        return self.__str__()

    def logpdf(self, x: Union[pd.DataFrame, np.ndarray],
               match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The log-pdf function of the overall joint distribution.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the log-pdf function of the joint
            distribution at.
        match_datatype: bool
            True to output the same datatype as the input. False to output
            a np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's logpdf.

        Returns
        -------
        logpdf: Union[pd.DataFrame, np.ndarray]
            log-pdf values of the joint distribution.
        """
        return self.__obj.logpdf(
            x=x, copula_params=self.copula_params, mdists=self.mdists,
            match_datatype=match_datatype, **kwargs)

    def pdf(self, x: Union[pd.DataFrame, np.ndarray],
            match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The pdf function of the overall joint distribution.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the pdf function of the joint distribution
            at.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's pdf.

        Returns
        -------
        pdf: Union[pd.DataFrame, np.ndarray]
            pdf values of the joint distribution.
        """
        return self.__obj.pdf(
            x=x, copula_params=self.copula_params, mdists=self.mdists,
            match_datatype=match_datatype, **kwargs)

    def cdf(self, x: Union[pd.DataFrame, np.ndarray],
            match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The cdf function of the overall joint distribution.
        This may take time to evaluate for certain copula distributions,
        due to d-dimensional numerical integration.
        In these case, mc_cdf will likely evaluate faster.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the cdf function of the joint distribution
            at.
        match_datatype: bool
            True to output the same datatype as the input. False to output
            a np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's cdf.

        Returns
        -------
        cdf: Union[pd.DataFrame, np.ndarray]
            cdf values of the joint distribution.
        """
        return self.__obj.cdf(
            x=x, copula_params=self.copula_params, mdists=self.mdists,
            match_datatype=match_datatype, **kwargs)

    def mc_cdf(self, x: Union[pd.DataFrame, np.ndarray],
               match_datatype: bool = True, num_generate: int = 10 ** 4,
               show_progress: bool = False, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The monte-carlo numerical approximation of the cdf function of the
        overall joint distribution.
        The standard cdf function may take time to evaluate for certain copula
        distributions, due to d-dimensional numerical integration.
        In these cases, mc_cdf will likely evaluate faster

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the cdf function of the joint distribution
            at.
        match_datatype: bool
            True to output the same datatype as the input. False to output
            a np.ndarray.
            Default is True.
        num_generate: int
            The number of random numbers to generate to use when numerically
            approximating the joint cdf using monte-carlo.
        show_progress: bool
            True to display the progress of the mc-cdf calculations.
            Default is False.
        kwargs:
            kwargs to pass to the multivariate distribution's mc_cdf.

        Returns
        -------
        mc_cdf: Union[pd.DataFrame, np.ndarray]
            numerical cdf values of the joint distribution.
        """
        return self.__obj.mc_cdf(
            x=x, copula_params=self.copula_params, mdists=self.mdists,
            match_datatype=match_datatype, num_generate=num_generate,
            show_progress=show_progress, **kwargs)

    def rvs(self, size: int, ppf_approx: bool = True,
            match_datatype: bool = True) -> np.ndarray:
        """The random variable generator function of the overall joint
        distribution. This requires the evaluation of the ppf / quantile
        function of each marginal distribution, which for certain univariate
        distributions requires the evaluation of an integral and may be
        time-consuming.
        The user therefore has the option to use the ppf_approx / quantile
        approximation function in place of this.

        Parameters
        ----------
        size: int
            How many multivariate random samples to generate from the overall
            joint distribution.
        ppf_approx: bool
            True to use the ppf_approx function to approximate the
            ppf / quantile function, via linear interpolation, when generating
            random variables.
            Default is True.
        match_datatype: bool
        True to output the same datatype as the fitted data, if possible.
        False to output a np.ndarray.
        Default is True.

        Returns
        -------
        rvs: np.ndarray
            Multivariate array of random variables, sampled from the
            joint distribution.
        """
        rvs_array: np.ndarray = self.__obj.rvs(
            size=size, copula_params=self.copula_params, mdists=self.mdists,
            ppf_approx=ppf_approx)
        type_keeper: TypeKeeper = self.__fit_info['type_keeper']
        return type_keeper.type_keep_from_2d_array(
            rvs_array, match_datatype=match_datatype)

    def copula_logpdf(self, u: Union[pd.DataFrame, np.ndarray],
                      match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The log-pdf function of the copula distribution.

        Parameters
        ----------
        u: Union[pd.DataFrame, np.ndarray]
            The cdf / pseudo-observation values to evaluate the log-pdf
            function of the copula distribution at. Each ui must be in the
            range (0, 1) and should be the cdf values of the univariate
            marginal distribution of the random variable xi.
        match_datatype: bool
            True to output the same datatype as the input.
            False to output a np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's logpdf.

        Returns
        -------
        copula_logpdf: Union[pd.DataFrame, np.ndarray]
            log-pdf values of the copula distribution.
        """
        return self.__obj.copula_logpdf(
            u=u, copula_params=self.copula_params,
            match_datatype=match_datatype, **kwargs)

    def copula_pdf(self, u: Union[pd.DataFrame, np.ndarray],
                   match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The pdf function of the copula distribution.

        Parameters
        ----------
        u: Union[pd.DataFrame, np.ndarray]
            The cdf / pseudo-observation values to evaluate the pdf function of
            the copula distribution at. Each ui must be in the range (0, 1)
            and should be the cdf values of the univariate marginal
            distribution of the random variable xi.
        match_datatype: bool
            True to output the same datatype as the input.
            False to output a np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's pdf.

        Returns
        -------
        copula_pdf: Union[pd.DataFrame, np.ndarray]
            pdf values of the copula distribution.
        """
        return self.__obj.copula_pdf(
            u=u, copula_params=self.copula_params,
            match_datatype=match_datatype, **kwargs)

    def copula_cdf(self, u: Union[pd.DataFrame, np.ndarray],
                   match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The cdf function of the copula distribution.
        This may take time to evaluate for certain copula distributions,
        due to d-dimensional numerical integration. In these case,
        copula_mc_cdf will likely evaluate faster.

        Parameters
        -----------
        u: Union[pd.DataFrame, np.ndarray]
            The cdf / pseudo-observation values to evaluate the cdf function
            of the copula distribution at. Each ui must be in the range (0, 1)
            and should be the cdf values of the univariate marginal
            distribution of the random variable xi.
        match_datatype: bool
            True to output the same datatype as the input.
            False to output a np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's cdf.

        Returns
        -------
        copula_cdf: Union[pd.DataFrame, np.ndarray]
            cdf values of the copula distribution.
        """
        return self.__obj.copula_cdf(
            u=u, copula_params=self.copula_params,
            match_datatype=match_datatype, **kwargs)

    def copula_mc_cdf(self, u: Union[pd.DataFrame, np.ndarray],
                      match_datatype: bool = True, num_generate: int = 10 ** 4,
                      show_progress: bool = False, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The monte-carlo numerical approximation of the cdf function of the
        copula distribution. The standard copula_cdf function may take time to
        evaluate for certain copula distributions, due to d-dimensional
        numerical integration. In these cases, copula_mc_cdf will likely
        evaluate faster

        Parameters
        ----------
        u: Union[pd.DataFrame, np.ndarray]
            The cdf / pseudo-observation values to evaluate the cdf function of
            the copula distribution at. Each ui must be in the range (0, 1)
            and should be the cdf values of the univariate marginal
            distribution of the random variable xi.
        match_datatype: bool
            True to output the same datatype as the input.
            False to output a np.ndarray.
            Default is True.
        num_generate: int
            The number of random numbers to generate to use when numerically
            approximating the copula cdf using monte-carlo.
        show_progress: bool
            True to display the progress of the copula mc-cdf calculations.
            Default is False.
        kwargs:
            kwargs to pass to the multivariate distribution's mc_cdf.

        Returns
        -------
        copula_mc_cdf: Union[pd.DataFrame, np.ndarray]
            numerical cdf values of the copula distribution.
        """
        return self.__obj.copula_mc_cdf(
            u=u, copula_params=self.copula_params,
            match_datatype=match_datatype, num_generate=num_generate,
            show_progress=show_progress, **kwargs)

    def copula_rvs(self, size: int) -> np.ndarray:
        """The random variable generator function of the copula distribution.

        Parameters
        ----------
        size: int
            How many multivariate random samples to generate from the copula
            distribution.

        Returns
        -------
        rvs: np.ndarray
            Multivariate array of random variables, sampled from the copula
            distribution. These correspond to randomly sampled cdf /
            pseudo-observation values of the univariate marginals.
        """
        return self.__obj.copula_rvs(
            size=size, copula_params=self.copula_params)

    def num_marginal_params(self) -> int:
        """Calculates the total number of parameters defining the marginal
        distributions.

        Returns
        -------
        num_marginal_params : int
            The total number of parameters defining the marginal distributions.
        """
        return self.__obj.num_marginal_params(mdists=self.mdists)

    def num_copula_params(self) -> int:
        """Calculates the number of parameters defining the multivariate
        distribution of the copula model.

        Returns
        -------
        num_copula_params: int
            The number of parameters defining the multivariate distribution
            of the copula model.
        """
        return self.__obj.num_copula_params(copula_params=self.copula_params)

    def num_scalar_params(self) -> int:
        """Calculates the number of scalar parameters defining the overall
        joint distribution.

        Returns
        -------
        num_scalar_params: int
            The number of scalar parameters defining the overall joint
            distribution.
        """
        return self.__obj.num_scalar_params(mdists=self.mdists)

    def num_params(self) -> int:
        """Calculates the number of parameters defining the overall joint
        distribution.

        Returns
        -------
        num_params: int
            The number of parameters defining the overall joint distribution.
        """
        return self.__obj.num_params(mdists=self.mdists)

    def __likelihood_loglikelihood_aic_bic(
            self, func_str: str, data: Union[pd.DataFrame, np.ndarray] = None)\
            -> float:
        """Utility function able to implement likelihood, loglikelihood, aic
        and bic methods without duplicate code.

        Parameters
        ----------
        func_str: str
            The name of the method to implement.
        data: Union[pd.DataFrame, np.ndarray]
            Our input data for our function of the joint distribution.
            If None passed, the value of the function obtained during the
            joint distribution fit will be returned.

        Returns
        -------
        value: float
            implemented method value.
        """
        if data is None:
            return self.__fit_info[func_str]
        obj = self.__obj
        return eval(f"obj.{func_str}(data=data, "
                    f"copula_params=self.copula_params, mdists=self.mdists)")

    def loglikelihood(self, data: Union[np.ndarray, pd.DataFrame] = None) \
            -> float:
        """The loglikelihood function of the overall joint distribution.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The values to evaluate the loglikelihood function of the joint
            distribution at. If None passed, the loglikelihood value obtained
            during the joint distribution fit will be returned.

        Returns
        -------
        loglikelihood : float
            loglikelihood value of the joint distribution.
        """
        return self.__likelihood_loglikelihood_aic_bic(
            func_str="loglikelihood", data=data)

    def likelihood(self, data: Union[np.ndarray, pd.DataFrame] = None) \
            -> float:
        """The likelihood function of the overall joint distribution.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The values to evaluate the likelihood function of the joint
            distribution at. If None passed, the likelihood value obtained
            during the joint distribution fit will be returned.

        Returns
        -------
        likelihood : float
            likelihood value of the joint distribution.
        """
        return self.__likelihood_loglikelihood_aic_bic(
            func_str="likelihood", data=data)

    def aic(self, data: Union[np.ndarray, pd.DataFrame] = None) -> float:
        """The Akaike Information Criterion (AIC) function of the overall
        joint distribution.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The values to evaluate the AIC function of the joint distribution
            at. If None passed, the AIC value obtained during the joint
            distribution fit will be returned.

        Returns
        -------
        aic : float
            AIC value of the joint distribution.
        """
        return self.__likelihood_loglikelihood_aic_bic(
            func_str="aic", data=data)

    def bic(self, data: Union[np.ndarray, pd.DataFrame] = None) -> float:
        """The Bayesian Information Criterion (BIC) function of the overall
        joint distribution.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The values to evaluate the BIC function of the joint distribution
            at. If None passed, the BIC value obtained during the joint
            distribution fit will be returned.

        Returns
        -------
        bic : float
            BIC value of the joint distribution.
        """
        return self.__likelihood_loglikelihood_aic_bic(
            func_str="bic", data=data)

    def marginal_pairplot(self, ppf_approx: bool = True,
                          color: str = 'royalblue', alpha: float = 1.0,
                          figsize: tuple = (8, 8), grid: bool = True,
                          axes_names: Iterable = None, plot_kde: bool = True,
                          num_generate: int = 10 ** 3, show: bool = True,
                          **kwargs) -> None:
        """Produces a pair-plot of each fitted marginal distribution.

        This requires the sampling of multivariate random variables from the
        joint copula distribution, which requires the evaluation of the
        ppf / quantile function of each marginal distribution, which for
        certain univariate distributions requires the evaluation of an integral
        and may be time-consuming. The user therefore has the option to use
        the ppf_approx / quantile approximation function in place of this.

        Parameters
        ----------
        ppf_approx: bool
            True to use the ppf_approx function to approximate the ppf /
            quantile function, via linear interpolation, when generating
            random variables.
            Default is True.
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
        # checking if possible to label axes
        if axes_names is None:
            type_keeper: TypeKeeper = self.__fit_info['type_keeper']
            if type_keeper.original_type == pd.DataFrame:
                axes_names = type_keeper.original_info['other']['cols']

        # creating marginal pair-plot
        self.__obj.marginal_pairplot(
            copula_params=self.copula_params, mdists=self.mdists,
            ppf_approx=ppf_approx, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, plot_kde=plot_kde,
            num_generate=num_generate, show=show)

    def _threeD_plot(self, func_str: str, ppf_approx: bool,
                     var1_range: np.ndarray, var2_range: np.ndarray,
                     color: str, alpha: float, figsize: tuple, grid: bool,
                     axes_names: tuple, zlim: tuple, num_generate: int,
                     num_points: int, show_progress: bool, show: bool,
                     mc_num_generate: int = None, ranges_to_u: bool = False) \
            -> None:
        """Utility function able to implement pdf_plot, cdf_plot, mc_cdf_plot,
        copula_pdf_plot, copula_cdf_plot and copula_mc_cdf_plot methods without
        duplicate code.

        Note that these plots are only implemented when we have 2-dimensional
        / bivariate distributions.

        This may require the sampling of multivariate random variables from
        the joint copula distribution, which requires the evaluation of the
        ppf / quantile function of each marginal distribution, which for
        certain univariate distributions requires the evaluation of an
        integral and may be time-consuming. The user therefore has the option
        to use the ppf_approx / quantile approximation function in place of
        this.

        Parameters
        ----------
        func_str: str
            The name of the method to implement
        ppf_approx: bool
            True to use the ppf_approx function to approximate the ppf /
            quantile function, via linear interpolation, when generating
            random variables.
            Default is True.
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 / u1 variable
            to plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99) for copula function plots. For joint distribution
            plots, the upper and lower bounds are taken to be the bounds of
            each variable observed during the joint distribution fit - the
            fitted_bounds.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 / u2 variable
            to plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99) for copula function plots. For joint distribution
            plots, the upper and lower bounds are taken to be the bounds of
            each variable observed during the joint distribution fit - the
            fitted_bounds.
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
            number of variables (length 2). If None provided, the axes will be
            labeled using the names of the variables provided during the joint
            distribution fit, if possible. Otherwise, the axes will be labeled
            as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the joint
            distribution, to determine the bounds of our variables,
            if necessary. See var1_range and var2_range for more information.
            Default is 1000.
        num_points: int
            The number of points to use in your evenly spaced var1_range
            and var2_range arrays, if they are not provided by the user.
            Default is 100.
        show_progress: bool
            Whether to show the progress of the plotting.
            Default is True.
        show: bool
            True to display the plot when the method is called.
            Default is True.
        mc_num_generate: int
            For mc_cdf_plot and copula_mc_cdf_plot only.
            The number of multivariate random variables to generate when
            evaluating monte-carlo functions.
            Default is 10,000.
        ranges_to_u: bool
            True to convert user provided var1_range and var2_range arrays
            to marginal distribution cdf / pseudo-observation values.
            Default is False.
        """

        # argument checks
        if axes_names is None:
            type_keeper: TypeKeeper = self.__fit_info['type_keeper']
            if type_keeper.original_type == pd.DataFrame:
                axes_names = type_keeper.original_info['other']['cols']

        if self.num_variables != 2:
            raise NotImplementedError(
                f"{func_str}_plot is not implemented when the number of "
                f"variables is not 2.")

        if (not isinstance(num_points, int)) or (num_points <= 0):
            raise TypeError("num_points must be a strictly positive integer.")

        # creating our ranges
        eps: float = 10 ** -2
        rng_bounds = np.array([[eps, 1-eps], [eps, 1-eps]], dtype=float) \
            if 'copula' in func_str else self.__fit_info['fitted_bounds']
        if np.isnan(rng_bounds.flatten()).sum() != 0:
            raise ArithmeticError(
                f"Cannot plot {func_str} as fitted_bounds contains nans. "
                f"This is likely because the parameters estimated via the "
                f"specified fit method are not valid.")
        if var1_range is None:
            var1_range: np.ndarray = np.linspace(
                rng_bounds[0][0], rng_bounds[0][1], num_points, dtype=float)
        if var2_range is None:
            var2_range: np.ndarray = np.linspace(
                rng_bounds[1][0], rng_bounds[1][1], num_points, dtype=float)

        # plotting
        self.__obj._threeD_plot(
            func_str=func_str, copula_params=self.copula_params,
            mdists=self.mdists, ppf_approx=ppf_approx, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show,
            mc_num_generate=mc_num_generate, ranges_to_u=ranges_to_u)

    def pdf_plot(self, ppf_approx: bool = True, var1_range: np.ndarray = None,
                 var2_range: np.ndarray = None, color: str = 'royalblue',
                 alpha: float = 1.0, figsize: tuple = (8, 8),
                 grid: bool = True, axes_names: tuple = None,
                 zlim: tuple = (None, None), num_generate: int = 1000,
                 num_points: int = 100, show_progress: bool = True,
                 show: bool = True, **kwargs) -> None:
        """Produces a 3D plot of the joint distribution's pdf / density
        function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        This requires the sampling of multivariate random variables from the
        joint copula distribution, which requires the evaluation of the
        ppf / quantile function of each marginal distribution, which for
        certain univariate distributions requires the evaluation of an
        integral and may be time-consuming. The user therefore has the option
        to use the ppf_approx / quantile approximation function in place of
        this.

        Parameters
        ----------
        ppf_approx: bool
            True to use the ppf_approx function to approximate the
            ppf / quantile function, via linear interpolation, when generating
            random variables.
            Default is True.
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
            number of variables (length 2). If None provided, the axes will be
            labeled using the names of the variables provided during the joint
            distribution fit, if possible. Otherwise, the axes will be labeled
            as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the joint
            distribution, to determine the bounds of our variables,
            if necessary. See var1_range and var2_range for more information.
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
            func_str='pdf', ppf_approx=ppf_approx, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show)

    def cdf_plot(self, ppf_approx: bool = True, var1_range: np.ndarray = None,
                 var2_range: np.ndarray = None, color: str = 'royalblue',
                 alpha: float = 1.0, figsize: tuple = (8, 8),
                 grid: bool = True, axes_names: tuple = None,
                 zlim: tuple = (None, None), num_generate: int = 1000,
                 num_points: int = 100, show_progress: bool = True,
                 show: bool = True, **kwargs) -> None:
        """Produces a 3D plot of the joint distribution's cdf / cumulative
        density function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        This may take time to evaluate for certain copula distributions, due
        to d-dimensional numerical integration. In these case, mc_cdf_plot
        will likely evaluate faster.

        This requires the sampling of multivariate random variables from the
        joint copula distribution, which requires the evaluation of the
        ppf / quantile function of each marginal distribution, which for
        certain univariate distributions requires the evaluation of an
        integral and may be time-consuming. The user therefore has the option
        to use the ppf_approx / quantile approximation function in place of
        this.

        Parameters
        ----------
        ppf_approx: bool
            True to use the ppf_approx function to approximate the ppf /
            quantile function, via linear interpolation, when generating
            random variables.
            Default is True.
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
            number of variables (length 2). If None provided, the axes will be
            labeled using the names of the variables provided during the joint
            distribution fit, if possible. Otherwise, the axes will be labeled
            as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the joint
            distribution, to determine the bounds of our variables,
            if necessary. See var1_range and var2_range for more information.
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
            func_str='cdf', ppf_approx=ppf_approx, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show)

    def mc_cdf_plot(self, ppf_approx: bool = True,
                    var1_range: np.ndarray = None,
                    var2_range: np.ndarray = None,
                    mc_num_generate: int = 10 ** 4, color: str = 'royalblue',
                    alpha: float = 1.0, figsize: tuple = (8, 8),
                    grid: bool = True, axes_names: tuple = None,
                    zlim: tuple = (None, None), num_generate: int = 1000,
                    num_points: int = 100, show_progress: bool = True,
                    show: bool = True, **kwargs) -> None:
        """Produces a 3D plot of the joint distribution's cdf / cumulative
        density function, using a monte-carlo numerical approximation.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        The standard cdf function may take time to evaluate for certain copula
        distributions, due to d-dimensional numerical integration. In these
        cases, mc_cdf_plot will likely evaluate faster

        This requires the sampling of multivariate random variables from the
        joint copula distribution, which requires the evaluation of the
        ppf / quantile function of each marginal distribution, which for
        certain univariate distributions requires the evaluation of an
        integral and may be time-consuming. The user therefore has the option
        to use the ppf_approx / quantile approximation function in place of
        this.

        Parameters
        ----------
        ppf_approx: bool
            True to use the ppf_approx function to approximate the
            ppf / quantile function, via linear interpolation, when generating
            random variables.
            Default is True.
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
            evaluating mc_cdf.
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
            number of variables (length 2). If None provided, the axes will be
            labeled using the names of the variables provided during the joint
            distribution fit, if possible. Otherwise, the axes will be labeled
            as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the joint
            distribution, to determine the bounds of our variables,
            if necessary. See var1_range and var2_range for more information.
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
            func_str='mc_cdf', ppf_approx=ppf_approx, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show,
            mc_num_generate=mc_num_generate)

    def copula_pdf_plot(self, var1_range: np.ndarray = None,
                        var2_range: np.ndarray = None,
                        color: str = 'royalblue', alpha: float = 1.0,
                        figsize: tuple = (8, 8), grid: bool = True,
                        axes_names: tuple = None, zlim: tuple = (None, None),
                        num_generate: int = 1000, num_points: int = 100,
                        show_progress: bool = True, show: bool = True,
                        **kwargs) -> None:
        """Produces a 3D plot of the copula distribution's pdf / density
        function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        Parameters
        ----------
        var1_range: np.ndarray
            numpy array containing a range of values for the u1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            (0.01, 0.99).
        var2_range: np.ndarray
            numpy array containing a range of values for the u2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            (0.01, 0.99).
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
            number of variables (length 2). If None provided, the axes will be
            labeled using the names of the variables provided during the joint
            distribution fit, if possible. Otherwise, the axes will be labeled
            as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the joint
            distribution, to determine the bounds of our variables,
            if necessary. See var1_range and var2_range for more information.
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
            func_str='copula_pdf', ppf_approx=True, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show)

    def copula_cdf_plot(self, var1_range: np.ndarray = None,
                        var2_range: np.ndarray = None,
                        color: str = 'royalblue', alpha: float = 1.0,
                        figsize: tuple = (8, 8), grid: bool = True,
                        axes_names: tuple = None, zlim: tuple = (None, None),
                        num_generate: int = 1000, num_points: int = 100,
                        show_progress: bool = True, show: bool = True,
                        **kwargs) -> None:
        """Produces a 3D plot of the copula distribution's cdf / density
        function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        This may take time to evaluate for certain copula distributions, due
        to d-dimensional numerical integration. In these case,
        copula_mc_cdf_plot will likely evaluate faster.

        Parameters
        ----------
        var1_range: np.ndarray
            numpy array containing a range of values for the u1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            (0.01, 0.99).
        var2_range: np.ndarray
            numpy array containing a range of values for the u2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            (0.01, 0.99).
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
            number of variables (length 2). If None provided, the axes will be
            labeled using the names of the variables provided during the joint
            distribution fit, if possible. Otherwise, the axes will be labeled
            as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the
            joint distribution, to determine the bounds of our variables,
            if necessary. See var1_range and var2_range for more information.
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
        ranges_to_u: bool = not (var1_range is None and var2_range is None)
        self._threeD_plot(
            func_str='copula_cdf', ppf_approx=True, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show, ranges_to_u=ranges_to_u)

    def copula_mc_cdf_plot(self, var1_range: np.ndarray = None,
                           var2_range: np.ndarray = None,
                           mc_num_generate: int = 10 ** 4,
                           color: str = 'royalblue', alpha: float = 1.0,
                           figsize: tuple = (8, 8), grid: bool = True,
                           axes_names: tuple = None,
                           zlim: tuple = (None, None),
                           num_generate: int = 1000, num_points: int = 100,
                           show_progress: bool = True, show: bool = True,
                           **kwargs) -> None:
        """Produces a 3D plot of the copula distribution's cdf / density
        function using monte-carlo numerical methods.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        The standard copula_cdf function may take time to evaluate for certain
        copula distributions, due to d-dimensional numerical integration.
        In these cases, copula_mc_cdf_plot will likely evaluate faster

        Parameters
        ----------
        var1_range: np.ndarray
            numpy array containing a range of values for the u1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            (0.01, 0.99).
        var2_range: np.ndarray
            numpy array containing a range of values for the u2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            (0.01, 0.99).
        mc_num_generate: int
            The number of multivariate random variables to generate when
            evaluating copula_mc_cdf.
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
            number of variables (length 2). If None provided, the axes will be
            labeled using the names of the variables provided during the joint
            distribution fit, if possible. Otherwise, the axes will be labeled
            as 'variable 1' and 'variable 2' respectively.
        zlim: tuple
            The matplotlib.pyplot bounds of the z-axis to use in your plot.
            Default is (None, None) -> No z-axis bounds.
        num_generate: int
            The number of random variables to generate from the joint
            distribution, to determine the bounds of our variables,
            if necessary. See var1_range and var2_range for more information.
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
        ranges_to_u: bool = not (var1_range is None and var2_range is None)
        self._threeD_plot(
            func_str='copula_mc_cdf', ppf_approx=True, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show,
            mc_num_generate=mc_num_generate, ranges_to_u=ranges_to_u)

    @property
    def copula_params(self) -> Params:
        """Returns the Params object of the copula distribution."""
        return self.__fit_info.copy()['copula_params']

    @property
    def mdists(self) -> dict:
        """Returns a dict with univariate marginal distributions as values and
        indices as keys."""
        return self.__fit_info.copy()['mdists']

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
        """Returns True if the copula distribution converged successfully.
        False otherwise."""
        return self.__fit_info.copy()['success']

    @property
    def summary(self) -> pd.DataFrame:
        """Returns a dataframe containing summary information on the joint
        distribution and its fit."""
        return self.__fit_info.copy()['summary']
