from scipy.optimize import differential_evolution
import scipy.stats
import numpy as np
import warnings
from collections import deque
from typing import Callable
from functools import partial

from sklarpy.copulas.elliptical._elliptical import Elliptical
from sklarpy.copulas.elliptical._elliptical_params import StudentTCopulaParams
from sklarpy.copulas.elliptical._multivariate_t import multivariate_t_cdf
from sklarpy._utils import numeric, FitError, dataframe_or_array, get_iterator


__all__ = ['StudentTCopula']


class StudentTCopula(Elliptical):
    _OBJ_NAME = "StudentTCopula"
    _MAX_NUM_VARIABLES = np.inf
    _PARAMS_OBJ = StudentTCopulaParams

    def __init__(self, marginals=None, name: str = None):
        Elliptical.__init__(self, marginals, name)
        self._dof: float = None
        self._mc_cdf: Callable = None

    def fit(self, method: str = 'laloux_pp_kendall', params: StudentTCopulaParams = None,
            corr: dataframe_or_array = None, dof: numeric = None, dof_bounds: tuple = (2.01, 100),
            raise_dof_fit_error: bool = False, **kwargs):
        if params is not None:
            self._params_check(params)
            corr = params.corr
            dof = params.dof

        Elliptical._fit_corr(self, method, corr, **kwargs)

        try:
            self._fitting = True
            self._fit_dof(dof, dof_bounds, raise_dof_fit_error)
            self._fitting = False
        except Exception as e:
            self._fitting = False
            raise e

        if self._dof <= 2:
            warnings.warn("dof <= 2")

        self._fitted = True
        self._params = {'corr': self.corr, 'dof': self.dof}
        return self

    def _objective_func(self, dof_array: np.ndarray):
        self._dof = float(dof_array)
        return -self.loglikelihood()

    def _fit_dof(self, dof: float, dof_bounds: tuple, raise_error: bool):
        if dof is not None:
            if not (isinstance(dof, int) or isinstance(dof, float)):
                raise TypeError("if dof given it must be a float or integer")
            if dof <= 0:
                raise ValueError("dof must be strictly positive")
            self._dof = dof
            return 0

        if not isinstance(dof_bounds, tuple):
            raise TypeError("dof_bounds must be a tuple")
        elif len(dof_bounds) != 2:
            raise ValueError("dof_bounds must contain exactly two elements; the lower bound and the upper bound of the"
                             " degrees of freedom parameter")

        new_dof_bounds = []
        eps = 10 ** -9
        for val in dof_bounds:
            if val < 0:
                raise ValueError("dof bounds must all be strictly positive.")
            if val == 0:
                val = eps
            new_dof_bounds.append(val)
        new_dof_bounds = new_dof_bounds
        lb, ub = min(new_dof_bounds), max(new_dof_bounds)
        if lb == ub:
            if lb > eps/2:
                lb -= eps/2
                ub += eps/2
            else:
                ub += eps

        res = differential_evolution(self._objective_func, [(lb, ub)])
        if not res['success']:
            if raise_error:
                raise FitError("failed to successfully converge to optimal dof value.")
            warnings.warn("failed to successfully converge to optimal dof value")

    def _rvs(self, size: int) -> np.ndarray:
        X: np.ndarray = self._generate_multivariate_std_normal_rvs(size)
        xi: np.ndarray = scipy.stats.chi2.rvs(df=self.dof, size=(size, 1))
        t_rvs: np.ndarray = X / np.sqrt(xi/self.dof)
        return scipy.stats.t.cdf(t_rvs, self.dof)

    def _pdf(self, u: np.ndarray, **kwargs) -> np.ndarray:
        x: np.ndarray = scipy.stats.t.ppf(u, (self.dof, ))
        return scipy.stats.multivariate_t.pdf(x, shape=self.corr, df=self.dof) / scipy.stats.t.pdf(x, (self.dof,)).prod(axis=1)

    def _cdf(self, u: np.ndarray, **kwargs) -> np.ndarray:
        show_progress: bool = kwargs.get('show_progress', True)
        if not isinstance(show_progress, bool):
            raise TypeError('show_progress must be a boolean if given as kwarg')
        x: np.ndarray = scipy.stats.t.ppf(u, self.dof)
        cdf_vals: np.ndarray = multivariate_t_cdf(x, shape=self.corr, df=self.dof, show_progress=show_progress)
        return cdf_vals

    def mc_cdf(self, u: np.ndarray, num_generate: int = 10**4, match_datatype: bool = True, show_progress: bool = True) -> np.ndarray:
        # argument checks
        if (not isinstance(num_generate, int)) or (num_generate <= 0):
            raise TypeError("num_generate must be a positive integer.")

        for bool_arg in (match_datatype, show_progress):
            if not isinstance(bool_arg, bool):
                raise TypeError("match_datatype, show_progress must be a boolean.")

        u_array: np.ndarray = self._get_u_array(u)

        # whether to show progress
        iterator = get_iterator(u_array, show_progress, "calculating monte-carlo cdf values")

        # calculating cdf values via mc
        rvs: np.ndarray = self.rvs(num_generate, False)
        mc_cdf_values: deque = deque()
        for row in iterator:
            mc_cdf_val: float = np.all(rvs <= row, axis=1).sum() / num_generate
            mc_cdf_values.append(mc_cdf_val)
        mc_cdf_values: np.ndarray = np.asarray(mc_cdf_values)

        if match_datatype:
            return self._type_keeper.type_keep_from_1d_array(mc_cdf_values, col_name=["MC CDF Values"])
        return mc_cdf_values

    def mc_cdf_plot(self, num_generate: int = 10**4, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (0, None),
                 num_points: int = 100, show_progress: bool = True, show: bool = True):
        self._mc_cdf = partial(self.mc_cdf, num_generate=num_generate)
        self._pdf_cdf_plot('_mc_cdf', color, alpha, figsize, grid, axes_names, zlim, num_points, show_progress, show)
        self._mc_cdf = None

    def optimization_plot(self, dof_bounds: tuple = (2.01, 100), num_points: int = 10**3, color: str = 'royalblue',
                          optimal_color: str = 'red', optimal_size: float = 10.0, figsize: tuple = (8, 8),
                          grid: bool = True, ylim: tuple = None, show: bool = True):
        # checks
        self._fit_check()

        if self._marginals is None:
            raise NotImplementedError("Optimization plot requires marginal cdf values.")

        # checking arguments
        for str_arg in (color, optimal_color):
            if not isinstance(str_arg, str):
                raise TypeError("invalid argument in optimization_plot. check color, optimal_color are strings.")

        for numeric_arg in (optimal_size, num_points):
            if not (isinstance(numeric_arg, float) or isinstance(numeric_arg, int)):
                raise TypeError("invalid argument type in optimization_plot. check optimal_size, num_points are floats"
                                " or integers.")
        optimal_size = float(optimal_size)
        num_points = int(num_points)

        tuple_args = [figsize, dof_bounds]
        if ylim is not None:
            tuple_args.append(ylim)
        for tuple_arg in tuple_args:
            if not (isinstance(tuple_arg, tuple) and len(tuple_arg) == 2):
                raise TypeError("invalid argument type in optimization_plot. check figsize, dof_bounds, ylim are tuples"
                                " of length 2.")

        for bool_arg in (grid, show):
            if not isinstance(bool_arg, bool):
                raise TypeError("invalid argument type in optimization_plot check grid, show are boolean.")

        # data for plot
        optimal_dof: float = self.dof
        x: np.ndarray = np.linspace(*dof_bounds, num_points)
        loglikelihoods: list = [-self._objective_func(xi) for xi in x]
        optimal_loglikelihood = -self._objective_func(optimal_dof)

        # plotting
        fig = plt.figure(f"{self.name} Optimisation Plot", figsize=figsize)
        plt.plot(x, loglikelihoods, color=color, label="log-likelihood")
        plt.scatter(optimal_dof, optimal_loglikelihood, color=optimal_color, s=optimal_size, label=f"optimum={round(self.dof, 2)}")
        plt.xlabel("degrees of freedom")
        plt.ylabel("log-likelihood")
        plt.grid(grid)
        plt.legend(loc='upper right')

        if show:
            plt.show()

    @property
    def dof(self) -> float:
        """The fitted degrees of freedom of the student-t copula"""
        self._fit_check()
        return self._dof


if __name__ == '__main__':
    from sklarpy.univariate import normal, gamma
    from sklarpy.copulas import MarginalFitter
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import norm
    p = 0.6
    my_corr = np.array([[1, p], [p, 1]])

    std5 = StudentTCopula().fit(corr=my_corr, cov=my_corr, dof=3.41652)#.pdf(np.array([[0.4, 0.5]]))

    rvs = std5.rvs(5000)
    std6: StudentTCopula = StudentTCopula(rvs, 'std6').fit()
    # std6.mc_cdf_plot()
    # std6.pdf_plot()
    # std6.marginal_pairplot(alpha=0.2)
    # std6.cdf_plot()

    # print(std6.params.dof)
    # # breakpoint()
    # std6.optimization_plot()
    # std6.pdf_plot(axes_names=('rain fall', 'sun percentage'))
    std6.save()
    # breakpoint()

    # num_samples = 10000
    # np.random.seed(100)
    # my_normal_sample = np.random.normal(0, 1, size=(num_samples, 2)) @ np.linalg.cholesky(my_corr).T
    # my_cop_sample = norm.cdf(my_normal_sample)
    # my_realistic_sample = my_cop_sample.copy()
    # my_realistic_sample[:, 0] = normal.ppf(my_realistic_sample[:, 0], (6, 5))
    # my_realistic_sample[:, 1] = gamma.ppf(my_realistic_sample[:, 1], (4, 3))
    #
    # # testing studentt
    # my_student = StudentTCopula(my_cop_sample)
    # print(my_student)
    # my_student.fit()
    # print(my_student)
    # print(my_student.params.dof)
    # print(my_student.params.corr)
    # # print(my_student.cdf())
    #
    # my_student_sample = my_student.rvs(num_samples)
    # second_student = StudentTCopula(my_student_sample, 'StudentT_2').fit()
    # print(second_student)
    # print(second_student.params.dof)
    # print(second_student.params.corr)
    # # second_student.save()
    # third_student = second_student.copy('hhh')
    # print(third_student)
    # import matplotlib.pyplot as plt
    # # plt.scatter(second_student.params.dof, -second_student.loglikelihood(), s=10, c='red')
    # # xrange = np.linspace(2.01, 10, 100)
    # # objective_func_vals = [second_student._objective_func(np.array([x])) for x in xrange]
    # # plt.plot(xrange, objective_func_vals, c='blue')
    # # argmin = np.argmin(objective_func_vals)
    # # plt.scatter(xrange[argmin], objective_func_vals[argmin], s=10, c='green')
    # # plt.show()
    #
    # # pdf_vals = third_student.pdf()
    # # fig = plt.figure()
    # # ax = plt.axes(projection='3d')
    # # ax.plot_trisurf(my_student_sample[:, 0], my_student_sample[:, 1], pdf_vals, antialiased=False, linewidth=0)
    # # plt.show()
    #
    # # cdf_vals = third_student.cdf(my_student_sample[:100])
    # # fig = plt.figure()
    # # ax = plt.axes(projection='3d')
    # # ax.plot_trisurf(my_student_sample[:100, 0], my_student_sample[:100, 1], cdf_vals)
    # # plt.show()
    # # third_student.marginal_pairplot(alpha=0.1)
    # forth_student = StudentTCopula().fit(params=third_student.params)
    # forth_student.pdf_plot()
    # forth_student.marginal_pairplot()

# for plots, do sub-plots of each marginal, cdf and pdf plots (if bivariate) and of the objective function and the min value
