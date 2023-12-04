# Contains code for pre-fitted copula models
from typing import Union, Iterable, Callable, Dict, List
import numpy as np
import pandas as pd
from collections import deque

from sklarpy.copulas import MarginalFitter
from sklarpy.utils._input_handlers import check_multivariate_data, get_mask
from sklarpy.utils._type_keeper import TypeKeeper
from sklarpy.utils._params import Params
from sklarpy.utils._not_implemented import NotImplementedBase
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate, \
    FittedContinuousMultivariate
from sklarpy.univariate._fitted_dists import FittedUnivariateBase
from sklarpy.plotting._pair_plot import pair_plot
from sklarpy.plotting._threeD_plot import threeD_plot
from sklarpy.copulas._fitted_dists import FittedCopula

__all__ = ['PreFitCopula']


class PreFitCopula(NotImplementedBase):
    """A pre-fit copula model"""
    __MAX_RVS_LOOPS: int = 100

    def __init__(self, name: str, mv_object: PreFitContinuousMultivariate):
        """A pre-fit copula model.

        Parameters
        ----------
        name : str
            The name of the copula object.
            Used when saving, if a file path is not specified and/or for
            additional identification purposes.
        mv_object : PreFitContinuousMultivariate
            The multivariate distribution which defines the copula
            distribution.
        """
        self._name: str = name
        self._mv_object: PreFitContinuousMultivariate = mv_object

    def __str__(self) -> str:
        return f"PreFit{self.name.title()}Copula"

    def __repr__(self) -> str:
        return self.__str__()

    def _get_data_array(self, data: Union[pd.DataFrame, np.ndarray],
                        is_u: bool) -> np.ndarray:
        """Converts the user's data input into a numpy array and performs
        checks.

        Raises errors if data is not in the correct format / type.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            User provided data values to check and convert.
            Can be observations of random variables or marginal cdf /
            pseudo-observation values.
        is_u : bool
            True if the user data is marginal cdf / pseudo-observation values.
            False otherwise.

        Returns
        -------
        data_array: np.ndarray
            numpy array of the multivariate data.
        """
        # checking if data is multivariate and converting to np.ndarray
        data_array: np.ndarray = check_multivariate_data(
            data=data, allow_1d=True, allow_nans=True)
        if is_u:
            # checking if data lies in [0, 1] range
            if not (np.all(data_array >= 0.0) and np.all(data_array <= 1.0) and
                    (np.isnan(data_array).sum() == 0)):
                raise ValueError("Expected all u values to be between 0 and "
                                 "1. Check u contains valid cdf values from "
                                 "your marginal distributions.")
        return data_array

    def _get_mdists(self, mdists: Union[MarginalFitter, dict], d: int,
                    check: bool = True) -> Dict[int, FittedUnivariateBase]:
        """Converts the user's marginal distributions into a standardized
        dictionary format and performs checks.

        Parameters
        ----------
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
        d: int
            The number of variables.
        check: bool
            Whether to perform checks on the mdist object.

        Returns
        --------
        mdists_dict: Dict[int, FittedUnivariateBase]
            A standardized dictionary with numbered indices as values and
            fitted SklarPy univariate distributions as as values.
        """
        if isinstance(mdists, MarginalFitter):
            mdists = mdists.marginals

        if check:
            if isinstance(mdists, dict):
                if len(mdists) != d:
                    raise ValueError("mdists number of distributions and the "
                                     "number of variables are not equal.")

                for index, dist in mdists.items():
                    if not (isinstance(index, int) and
                            issubclass(type(dist), FittedUnivariateBase)):
                        raise ValueError('If mdists is a dictionary, it must '
                                         'be specified with integer keys and '
                                         'SklarPy fitted univariate '
                                         'distributions as values.')
            else:
                raise TypeError("mdists must be a dictionary or a fitted "
                                "MarginalFitter object")
        return mdists

    def _get_copula_params(self, copula_params: Union[Params, tuple],
                           data_array: np.ndarray) -> tuple:
        """Converts the user's copula-params input into tuple form and then
        checks the parameters of the model and raises an error if one or more
        is invalid.

        Parameters
        ----------
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        data_array: np.ndarray
            numpy array of the multivariate data.

        Returns
        -------
        copula_params_tuple: tuple
            The parameters which define the multivariate distribution in the
            copula model.
        """
        params_tuple: tuple = self._mv_object._get_params(params=copula_params)
        self._mv_object._check_dim(data=data_array, params=params_tuple)
        return copula_params if isinstance(copula_params, tuple) \
            else copula_params.to_tuple

    def __mdist_calcs(self, func_strs: List[str], data: np.ndarray,
                      mdists: Union[MarginalFitter, dict], check: bool,
                      funcs_kwargs: dict = None) -> Dict[str, np.ndarray]:
        """Utility function able to evaluate functions of the univariate
        marginal distributions.

        Parameters
        ----------
        func_strs : List[str]
            List containing names of univariate distribution functions to
            implement.
        data: np.ndarray
            numpy array containing the input data for the mdists functions.
        check: bool
            whether to perform checks on the mdists.
        funcs_kwargs: dict
            dictionary of kwargs to pass to the mdists functions.
            Default is None if no kwargs to pass.

        Returns
        -------
        res: Dict[str, np.ndarray]
            A dictionary with func_strs as keys and numpy arrays of outputs as
            values.
        """
        if funcs_kwargs is None:
            funcs_kwargs = {}

        # getting marginal distributions
        d: int = data.shape[1]
        mdists_dict: dict = self._get_mdists(mdists=mdists, d=d, check=check)

        res: dict = {}
        for func_str in func_strs:
            # checking valid method / function
            if func_str not in dir(FittedUnivariateBase):
                raise NotImplementedError(
                    f"{func_str} not implemented in FittedUnivariateBase.")

            # evaluating the function for each variable and its respective
            # marginal dist
            func_kwargs: dict = funcs_kwargs.get(func_str, {})
            func_array: np.ndarray = None
            for index, dist in mdists_dict.items():
                data_i: np.ndarray = data[:, index]
                vals: np.ndarray = eval(
                    f"dist.{func_str}(data_i, **func_kwargs)"
                )

                if func_array is None:
                    n: int = vals.size if isinstance(vals, np.ndarray) else 1
                    func_array = np.full((n, d), np.nan, dtype=float)
                func_array[:, index] = vals

            res[func_str] = func_array
        return res

    def logpdf(self, x: Union[pd.DataFrame, np.ndarray],
               copula_params: Union[Params, tuple],
               mdists: Union[MarginalFitter, dict],
               match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The log-pdf function of the overall joint distribution.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the log-pdf function of the joint
            distribution at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's logpdf.

        Returns
        -------
        logpdf: Union[pd.DataFrame, np.ndarray]
            log-pdf values of the joint distribution.
        """
        # checking data
        x_array: np.ndarray = self._get_data_array(data=x, is_u=False)

        # checking copula params
        copula_params_tuple: tuple = self._get_copula_params(
            copula_params=copula_params, data_array=x_array)

        # getting non-nan data
        mask, masked_data, output = get_mask(data=x_array)

        # calculating u values
        mdists_dict: dict = self._get_mdists(mdists, d=masked_data.shape[1],
                                             check=True)
        res: dict = self.__mdist_calcs(func_strs=['cdf', 'logpdf'],
                                       data=masked_data, mdists=mdists_dict,
                                       check=True)

        # calculating logpdf values
        logpdf_values: np.ndarray = self.copula_logpdf(
            u=res['cdf'], copula_params=copula_params_tuple,
            match_datatype=False, **kwargs) + res['logpdf'].sum(axis=1)

        # converting to correct output datatype
        output[~mask] = logpdf_values
        return TypeKeeper(x).type_keep_from_1d_array(
            array=output, match_datatype=match_datatype, col_name=['logpdf'])

    def pdf(self, x: Union[pd.DataFrame, np.ndarray],
            copula_params: Union[Params, tuple],
            mdists: Union[MarginalFitter, dict], match_datatype: bool = True,
            **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """The pdf function of the overall joint distribution.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the pdf function of the joint distribution
            at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
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
        try:
            logpdf_values: np.ndarray = self.logpdf(
                x=x, copula_params=copula_params, mdists=mdists,
                match_datatype=False, **kwargs)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('pdf')
        pdf_values: np.ndarray = np.exp(logpdf_values)
        return TypeKeeper(x).type_keep_from_1d_array(
            array=pdf_values, match_datatype=match_datatype, col_name=['pdf'])

    def __cdf_mccdf(self, mc_cdf: bool, x: Union[pd.DataFrame, np.ndarray],
                    copula_params: Union[Params, tuple],
                    mdists: Union[MarginalFitter, dict],
                    match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """Utility function able to implement cdf and mc_cdf methods without
        duplicate code.

        Parameters
        ----------
        mc_cdf: bool
            True if mc_cdf is being implemented. False for cdf.
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the cdf / mc_cdf function of the joint
            distribution at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's cdf / mc_cdf.

        Returns
        -------
        cdf: Union[pd.DataFrame, np.ndarray]
            cdf / mc_cdf values of the joint distribution.
        """
        # checking data
        x_array: np.ndarray = self._get_data_array(data=x, is_u=False)

        # checking copula params
        copula_params_tuple: tuple = self._get_copula_params(
            copula_params=copula_params, data_array=x_array)

        # getting non-nan data
        mask, masked_data, output = get_mask(data=x_array)

        # calculating u values
        res: dict = self.__mdist_calcs(
            func_strs=['cdf'], data=masked_data, mdists=mdists, check=True)

        # calculating cdf values
        mc_str: str = "mc_" if mc_cdf else ""
        func: Callable = eval(f"self.copula_{mc_str}cdf")
        copula_cdf_values: np.ndarray = func(
            u=res['cdf'], copula_params=copula_params_tuple,
            match_datatype=False, **kwargs)

        # converting to correct output datatype
        output[~mask] = copula_cdf_values
        return TypeKeeper(x).type_keep_from_1d_array(
            array=output, match_datatype=match_datatype,
            col_name=[f'{mc_str}cdf'])

    def cdf(self, x: Union[pd.DataFrame, np.ndarray],
            copula_params: Union[Params, tuple],
            mdists: Union[MarginalFitter, dict],  match_datatype: bool = True,
            **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """The cdf function of the overall joint distribution.
        This may take time to evaluate for certain copula distributions, due
        to d-dimensional numerical integration. In these case, mc_cdf will
        likely evaluate faster.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the cdf function of the joint distribution
            at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's cdf.

        Returns
        -------
        cdf: Union[pd.DataFrame, np.ndarray]
            cdf values of the joint distribution.
        """
        return self.__cdf_mccdf(
            mc_cdf=False, x=x, copula_params=copula_params, mdists=mdists,
            match_datatype=match_datatype, **kwargs)

    def mc_cdf(self, x: Union[pd.DataFrame, np.ndarray],
               copula_params: Union[Params, tuple],
               mdists: Union[MarginalFitter, dict],
               match_datatype: bool = True, num_generate: int = 10 ** 4,
               show_progress: bool = False, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The monte-carlo numerical approximation of the cdf function of the
        overall joint distribution. The standard cdf function may take time
        to evaluate for certain copula distributions, due to d-dimensional
        numerical integration. In these cases, mc_cdf will likely evaluate
        faster.

        Parameters
        ----------
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the cdf function of the joint distribution
            at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
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
        return self.__cdf_mccdf(
            mc_cdf=True, x=x, copula_params=copula_params, mdists=mdists,
            match_datatype=match_datatype, num_generate=num_generate,
            show_progress=show_progress, **kwargs)

    def rvs(self, size: int, copula_params: Union[Params, tuple],
            mdists: Union[MarginalFitter, dict], ppf_approx: bool = True,
            **kwargs) -> np.ndarray:
        """The random variable generator function of the overall joint
        distribution. This requires the evaluation of the ppf / quantile
        function of each marginal distribution, which for certain univariate
        distributions requires the evaluation of an integral and may be
        time-consuming. The user therefore has the option to use the
        ppf_approx / quantile approximation function in place of this.

        Parameters
        ----------
        size: int
            How many multivariate random samples to generate from the overall
            joint distribution.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
        ppf_approx: bool
            True to use the ppf_approx function to approximate the ppf /
            quantile function, via linear interpolation, when generating
            random variables.
            Default is True.

        Returns
        -------
        rvs: np.ndarray
            Multivariate array of random variables, sampled from the joint
            distribution.
        """
        copula_rvs: np.ndarray = self.copula_rvs(
            size=size, copula_params=copula_params)
        func_str: str = "ppf_approx" if ppf_approx else "ppf"
        res: dict = self.__mdist_calcs(func_strs=[func_str], data=copula_rvs,
                                       mdists=mdists, check=True)
        return res[func_str]

    def _h_logpdf_sum(self, g: np.ndarray,
                      copula_params: Union[Params, tuple]) -> np.ndarray:
        # logpdf of the marginals of G
        return np.full((g.shape[0], ), 0.0, dtype=float)

    def copula_logpdf(self, u: Union[pd.DataFrame, np.ndarray],
                      copula_params: Union[Params, tuple],
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
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters in
             the correct order.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's logpdf.

        Returns
        -------
        copula_logpdf: Union[pd.DataFrame, np.ndarray]
            log-pdf values of the copula distribution.
        """
        # checking data
        u_array: np.ndarray = self._get_data_array(data=u, is_u=True)

        # checking copula params
        copula_params_tuple: tuple = self._get_copula_params(
            copula_params=copula_params, data_array=u_array)

        # calculating copula logpdf
        g: np.ndarray = self._u_to_g(u=u_array,
                                     copula_params=copula_params_tuple)
        g_logpdf: np.ndarray = self._mv_object.logpdf(
            x=g, params=copula_params_tuple, match_datatype=False, **kwargs)
        copula_logpdf_values: np.ndarray = g_logpdf - self._h_logpdf_sum(
            g=g, copula_params=copula_params_tuple)
        return TypeKeeper(u).type_keep_from_1d_array(
            array=copula_logpdf_values, match_datatype=match_datatype,
            col_name=['copula_logpdf'])

    def copula_pdf(self, u: Union[pd.DataFrame, np.ndarray],
                   copula_params: Union[Params, tuple],
                   match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The pdf function of the copula distribution.

        Parameters
        ----------
        u: Union[pd.DataFrame, np.ndarray]
            The cdf / pseudo-observation values to evaluate the pdf function
            of the copula distribution at. Each ui must be in the range (0, 1)
            and should be the cdf values of the univariate marginal
            distribution of the random variable xi.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's pdf.

        Returns
        -------
        copula_pdf: Union[pd.DataFrame, np.ndarray]
            pdf values of the copula distribution.
        """
        try:
            copula_logpdf_values: np.ndarray = self.copula_logpdf(
                u=u, copula_params=copula_params, match_datatype=False,
                **kwargs)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('copula_pdf')
        copula_pdf_values: np.ndarray = np.exp(copula_logpdf_values)
        return TypeKeeper(u).type_keep_from_1d_array(
            array=copula_pdf_values, match_datatype=match_datatype,
            col_name=['copula_pdf'])

    def __copula_cdf_mccdf(self, mc_cdf: bool,
                           u: Union[pd.DataFrame, np.ndarray],
                           copula_params: Union[Params, tuple],
                           match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """Utility function able to implement copula_cdf and copula_mc_cdf
        methods without duplicate code.

        Parameters
        ----------
        mc_cdf: bool
            True if copula_mc_cdf is being implemented. False for copula_cdf.
        x: Union[pd.DataFrame, np.ndarray]
            The values to evaluate the copula_cdf / copula_mc_cdf function of
            the joint distribution at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's cdf / mc_cdf.

        Returns
        -------
        copula_cdf: Union[pd.DataFrame, np.ndarray]
            cdf / mc_cdf values of the copula distribution.
        """
        # checking data
        u_array: np.ndarray = self._get_data_array(data=u, is_u=True)

        # checking copula params
        copula_params_tuple: tuple = self._get_copula_params(
            copula_params=copula_params, data_array=u_array)

        # calculating cdf values
        g: np.ndarray = self._u_to_g(u_array, copula_params_tuple)
        mc_str: str = "mc_" if mc_cdf else ""
        func: Callable = eval(f"self._mv_object.{mc_str}cdf")
        copula_cdf_values: np.ndarray = func(x=g, params=copula_params_tuple,
                                             match_datatype=False, **kwargs)
        return TypeKeeper(u).type_keep_from_1d_array(
            array=copula_cdf_values, match_datatype=match_datatype,
            col_name=[f'{mc_str}cdf'])

    def copula_cdf(self, u: Union[pd.DataFrame, np.ndarray],
                   copula_params: Union[Params, tuple],
                   match_datatype: bool = True, **kwargs) \
            -> Union[pd.DataFrame, np.ndarray]:
        """The cdf function of the copula distribution.
        This may take time to evaluate for certain copula distributions,
        due to d-dimensional numerical integration. In these case,
        copula_mc_cdf will likely evaluate faster.

        Parameters
        ----------
        u: Union[pd.DataFrame, np.ndarray]
            The cdf / pseudo-observation values to evaluate the cdf function
            of the copula distribution at. Each ui must be in the range (0, 1)
            and should be the cdf values of the univariate marginal
            distribution of the random variable xi.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        match_datatype: bool
            True to output the same datatype as the input. False to output a
            np.ndarray.
            Default is True.
        kwargs:
            kwargs to pass to the multivariate distribution's cdf.

        Returns
        -------
        copula_cdf: Union[pd.DataFrame, np.ndarray]
            cdf values of the copula distribution.
        """
        return self.__copula_cdf_mccdf(
            mc_cdf=False, u=u, copula_params=copula_params,
            match_datatype=match_datatype, **kwargs)

    def copula_mc_cdf(self, u: Union[pd.DataFrame, np.ndarray],
                      copula_params: Union[Params, tuple],
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
        return self.__copula_cdf_mccdf(
            mc_cdf=True, u=u, copula_params=copula_params,
            match_datatype=match_datatype, num_generate=num_generate,
            show_progress=show_progress, **kwargs)

    def _u_g_pdf(self, func: Callable, arr: np.ndarray,
                 copula_params: Union[Params, tuple], **kwargs) \
            -> np.ndarray:
        """Utility function able to implement _g_to_u, _u_to_g and
        _h_logpdf_sum methods without duplicate code.

        Parameters
        ----------
        func: Callable
            The univariate distribution function to evaluate for each
            dimension vector of the array matrix. Must take a 1D array,
            univariate distribution parameters and any user specified kwargs
            as arguments.
        arr: np.ndarray
            A multivariate array with shape (num_observations, num_variables),
            which you wish to evaluate.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        kwargs:
            Any additional keyword arguments

        Returns
        -------
        output: np.ndarray
            The output of your specified function
        """

    def _g_to_u(self, g: np.ndarray, copula_params: Union[Params, tuple]) \
            -> np.ndarray:
        # g = mv rv
        # u = copula rv
        # x = overall rv
        # i.e. for gaussian copula, g = ppf(u) and therefore u = cdf(g)
        return g

    def _u_to_g(self, u: np.ndarray, copula_params: Union[Params, tuple]):
        return u

    def copula_rvs(self, size: int, copula_params: Union[Params, tuple],
                   **kwargs) -> np.ndarray:
        """The random variable generator function of the copula distribution.

        Parameters
        ----------
        size: int
            How many multivariate random samples to generate from the copula
            distribution.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.

        Returns
        -------
        rvs: np.ndarray
            Multivariate array of random variables, sampled from the copula
            distribution. These correspond to randomly sampled cdf /
            pseudo-observation values of the univariate marginals.
        """
        num_loops: int = 0
        d: int = self._mv_object._get_dim(
            self._mv_object._get_params(copula_params))
        valid_copula_rvs: deque = deque()
        while size > 0:
            # generating random variables from multivariate distribution
            mv_rvs: np.ndarray = self._mv_object.rvs(size, copula_params)

            # converting to copula rvs
            raw_copula_rvs: np.ndarray = self._g_to_u(mv_rvs, copula_params)

            # filtering out invalid copula rvs (not in [0, 1]^d)
            mask: np.ndarray = ((raw_copula_rvs > 0) & (raw_copula_rvs < 1)
                                ).sum(axis=1) == d
            copula_rvs = raw_copula_rvs[mask]
            valid_copula_rvs.append(copula_rvs)

            # repeating until sample size reached
            size -= copula_rvs.shape[0]
            num_loops += 1
            if num_loops > self.__MAX_RVS_LOOPS:
                raise ArithmeticError(f"Unable to generate valid copula rvs. "
                                      f"Max number of retries reached: "
                                      f"{self.__MAX_RVS_LOOPS}")
        return np.concatenate(valid_copula_rvs, axis=0)

    def _get_components_summary(self,
                                fitted_mv_object: FittedContinuousMultivariate,
                                mdists: dict, typekeeper: TypeKeeper) \
            -> pd.DataFrame:
        """Creates a summary of the marginal distributions and the fitted
        multivariate copula distribution.

        Parameters
        ----------
        fitted_mv_object : FittedContinuousMultivariate
            The fitted multivariate distribution.
        mdists : dict
            The fitted marginal distributions of each random variable.
            Must a dictionary with numbered indices as values and fitted
            SklarPy univariate distributions as values. The dictionary indices
            must correspond to the indices of the variables.
        typekeeper: TypeKeeper
            A TypeKeeper object initialized on the fitted dataset.

        Returns
        -------
        component_summary: pd.DataFrame
            A summary of the marginal distributions and the fitted
            multivariate copula distribution.
        """
        # getting summary of marginal dists
        summaries: list = [dist.summary for dist in mdists.values()]
        summary: pd.DataFrame = pd.concat(summaries, axis=1)
        if typekeeper.original_type == pd.DataFrame:
            index: pd.Index = summary.index
            summary = typekeeper.type_keep_from_2d_array(np.asarray(summary))
            summary.index = index
        mv_summary: pd.DataFrame = fitted_mv_object.summary
        mv_summary.columns = [self.name]
        return pd.concat([mv_summary, summary], axis=1)

    def num_marginal_params(self, mdists: Union[MarginalFitter, dict],
                            **kwargs) -> int:
        """Calculates the total number of parameters defining the marginal
        distributions.

        Parameters
        ----------
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.

        Returns
        -------
        num_marginal_params : int
            The total number of parameters defining the marginal distributions.
        """
        d: int = len(mdists)
        mdists_dict: dict = self._get_mdists(mdists=mdists, d=d, check=True)
        return int(sum([dist.num_params for dist in mdists_dict.values()]))

    def num_copula_params(self, copula_params: Union[Params, dict], **kwargs
                          ) -> int:
        """Calculates the number of parameters defining the multivariate
        distribution of the copula model.

        Parameters
        ----------
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.

        Returns
        -------
        num_copula_params: int
            The number of parameters defining the multivariate distribution
            of the copula model.
        """
        return len(copula_params)

    def num_scalar_params(self, mdists: Union[MarginalFitter, dict], **kwargs
                          ) -> int:
        """Calculates the number of scalar parameters defining the overall
        joint distribution.

        Parameters
        ----------
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.

        Returns
        -------
        num_scalar_params: int
            The number of scalar parameters defining the overall joint
            distribution.
        """
        return self._mv_object.num_scalar_params(
            d=len(mdists), copula=True) + self.num_marginal_params(mdists)

    def num_params(self, mdists: Union[MarginalFitter, dict], **kwargs) -> int:
        """Calculates the number of parameters defining the overall joint
        distribution.

        Parameters
        ----------
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.

        Returns
        -------
        num_params: int
            The number of parameters defining the overall joint distribution.
        """
        return self._mv_object.num_params + self.num_marginal_params(mdists)

    def fit(self, data: Union[pd.DataFrame, np.ndarray, None] = None,
            copula_params: Union[Params, tuple, None] = None,
            mdists: Union[MarginalFitter, dict, None] = None, **kwargs) \
            -> FittedCopula:
        """Fits the overall joint distribution to a given dataset or user
        provided parameters.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray, None]
            The multivariate dataset to fit the distribution's parameters too.
            Not required if `copula_params` and `mdists` provided.
        copula_params: Union[Params, tuple, None]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the
            specific multivariate distribution or a tuple containing these
            parameters in the correct order. If not passed, user must provide
            a dataset to fit too.
        mdists : Union[MarginalFitter, dict, None]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables. If not passed, user must provide
            a dataset to fit too.
        kwargs:
            kwargs to pass to MarginalFitter.fit and / or the relevant
            multivariate distribution's .fit method
            See below

        Keyword Arguments
        -----------------
        univariate_fitter_options: dict
            User provided arguments to use when fitting each marginal
            distribution. See MarginalFitter.fit documentation for more.
        show_progress: bool
            True to show the progress of your fitting.
        method: str
            The method to use when fitting the copula distribution to data.

        Returns
        -------
        fitted_copula: FittedCopula
            A fitted copula.
        """
        if (data is None) and (copula_params is None or mdists is None):
            raise ValueError(
                "copula_params and mdist must be provided if data is not.")

        if mdists is None:
            # fitting marginal distributions
            mdists: MarginalFitter = MarginalFitter(data=data).fit(**kwargs)
        d: int = len(mdists)
        mdists_dict: dict = self._get_mdists(mdists=mdists, d=d, check=True)

        if copula_params is None:
            # calculating u values
            data_array: Union[np.ndarray, None] = \
                self._get_data_array(data=data, is_u=False)
            res: dict = self.__mdist_calcs(func_strs=['cdf'], data=data_array,
                                           mdists=mdists_dict, check=False)
            u_array: Union[np.ndarray, None] = res['cdf']
        else:
            u_array: Union[np.ndarray, None] = None

        # fitting copula
        if 'corr_method' in kwargs:
            kwargs['cov_method'] = kwargs.pop('corr_method')
        kwargs['copula'] = True
        fitted_mv_object: FittedContinuousMultivariate =\
            self._mv_object.fit(data=u_array, params=copula_params, **kwargs)

        if len(mdists_dict) != fitted_mv_object.num_variables:
            raise ValueError("number of variables of for mdist and copula "
                             "params do not match.")

        # generating data to use when calculating statistics
        try:
            data_array: np.ndarray = self.rvs(
                size=10**3, copula_params=fitted_mv_object.params,
                mdists=mdists_dict, ppf_approx=True) if data is None \
                else check_multivariate_data(
                data, allow_1d=True, allow_nans=True)

        except ArithmeticError as e:
            if str(e) != (f"Unable to generate valid copula rvs. Max number "
                          f"of retries reached: {self.__MAX_RVS_LOOPS}"):
                raise
            else:
                data_array: np.ndarray = np.full((10**3, d), np.nan,
                                                 dtype=float)

        # fitting TypeKeeper object
        type_keeper: TypeKeeper = TypeKeeper(data_array)

        # calculating fit statistics
        loglikelihood: float = self.loglikelihood(
            data=data_array, copula_params=fitted_mv_object.params,
            mdists=mdists_dict)
        likelihood = np.exp(loglikelihood)
        aic: float = self.aic(
            data=data_array, copula_params=fitted_mv_object.params,
            mdists=mdists_dict)
        bic: float = self.bic(
            data=data_array, copula_params=fitted_mv_object.params,
            mdists=mdists_dict)

        fit_info: dict = {}
        fit_info['likelihood'] = likelihood
        fit_info['loglikelihood'] = loglikelihood
        fit_info['aic'] = aic
        fit_info['bic'] = bic

        # building summary
        num_params: int = self.num_params(mdists=mdists)
        num_scalar_params: int = self.num_scalar_params(mdists=mdists_dict)
        index: list = ['Distribution', '#Variables', '#Params',
                       '#Scalar Params', 'Converged', 'Likelihood',
                       'Log-Likelihood', 'AIC', 'BIC', '#Fitted Data Points']
        values: list = ['Joint Distribution', fitted_mv_object.num_variables,
                        num_params, num_scalar_params,
                        fitted_mv_object.converged, likelihood, loglikelihood,
                        aic, bic, fitted_mv_object.fitted_num_data_points]
        summary: pd.DataFrame = pd.DataFrame(
            values, index=index, columns=['Joint Distribution'])
        component_summary: pd.DataFrame = self._get_components_summary(
            fitted_mv_object=fitted_mv_object, mdists=mdists_dict,
            typekeeper=type_keeper)
        summary = pd.concat([summary, component_summary], axis=1)
        fit_info['summary'] = summary

        # calculating fit bounds
        num_variables: int = data_array.shape[1]
        fitted_bounds: np.ndarray = np.full((num_variables, 2), np.nan,
                                            dtype=float)
        mask, _, _ = get_mask(data_array)
        if (~mask).sum() != 0:
            fitted_bounds[:, 0] = data_array.min(axis=0)
            fitted_bounds[:, 1] = data_array.max(axis=0)
        fit_info['fitted_bounds'] = fitted_bounds

        # other fit values
        fit_info['type_keeper'] = type_keeper
        fit_info['copula_params'] = fitted_mv_object.params
        fit_info['mdists'] = mdists_dict
        fit_info['num_variables'] = num_variables
        fit_info['success'] = fitted_mv_object.converged
        fit_info['num_data_points'] = fitted_mv_object.fitted_num_data_points
        fit_info['num_params'] = num_params
        fit_info['num_scalar_params'] = num_scalar_params
        return FittedCopula(self, fit_info)

    def likelihood(self, data: Union[np.ndarray, pd.DataFrame],
                   copula_params: Union[Params, tuple],
                   mdists: Union[MarginalFitter, dict]) -> float:
        """The likelihood function of the overall joint distribution.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The values to evaluate the likelihood function of the joint
            distribution at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.

        Returns
        -------
        likelihood : float
            likelihood value of the joint distribution.
        """
        try:
            loglikelihood: float = self.loglikelihood(
                data=data, copula_params=copula_params, mdists=mdists)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('likelihood')
        return np.exp(loglikelihood)

    def loglikelihood(self, data: Union[np.ndarray, pd.DataFrame],
                      copula_params: Union[Params, tuple],
                      mdists: Union[MarginalFitter, dict]) -> float:
        """The log-likelihood function of the overall joint distribution.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The values to evaluate the log-likelihood function of the joint
            distribution at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.

        Returns
        -------
        loglikelihood : float
            loglikelihood value of the joint distribution.
        """
        try:
            logpdf_values: np.ndarray = self.logpdf(
                x=data, copula_params=copula_params, mdists=mdists,
                match_datatype=False)
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
            copula_params: Union[Params, tuple],
            mdists: Union[MarginalFitter, dict]) -> float:
        """The Akaike Information Criterion (AIC) function of the overall
        joint distribution.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The values to evaluate the AIC function of the joint distribution
            at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.

        Returns
        -------
        aic : float
            AIC value of the joint distribution.
        """
        try:
            loglikelihood: float = self.loglikelihood(
                data=data, copula_params=copula_params, mdists=mdists)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('aic')
        return 2 * (self.num_scalar_params(mdists=mdists) - loglikelihood)

    def bic(self, data: Union[pd.DataFrame, np.ndarray],
            copula_params: Union[Params, tuple],
            mdists: Union[MarginalFitter, dict]) -> float:
        """The Bayesian Information Criterion (BIC) function of the overall
        joint distribution.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            The values to evaluate the BIC function of the joint distribution
            at.
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.

        Returns
        -------
        bic : float
            BIC value of the joint distribution.
        """
        try:
            loglikelihood: float = self.loglikelihood(
                data=data, copula_params=copula_params, mdists=mdists)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('bic')
        data_array: np.ndarray = self._get_data_array(data=data, is_u=False)
        num_data_points: int = data_array.shape[0]
        return -2 * loglikelihood + np.log(num_data_points) * \
               self.num_scalar_params(mdists=mdists)

    def marginal_pairplot(self, copula_params: Union[Params, tuple],
                          mdists: Union[MarginalFitter, dict],
                          ppf_approx: bool = True, color: str = 'royalblue',
                          alpha: float = 1.0, figsize: tuple = (8, 8),
                          grid: bool = True, axes_names: tuple = None,
                          plot_kde: bool = True, num_generate: int = 10 ** 3,
                          show: bool = True, **kwargs) -> None:
        """Produces a pair-plot of each fitted marginal distribution.

        This requires the sampling of multivariate random variables from the
        joint copula distribution, which requires the evaluation of the ppf /
        quantile function of each marginal distribution, which for certain
        univariate distributions requires the evaluation of an integral and
        may be time-consuming. The user therefore has the option to use the
        ppf_approx / quantile approximation function in place of this.

        Parameters
        ----------
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond to
            the indices of the variables.
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
            If provided, must be an iterable with the same length as the number
            of variables. If None provided, the axes will not be labeled.
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

        rvs: np.ndarray = self.rvs(
            size=num_generate, copula_params=copula_params, mdists=mdists,
            ppf_approx=ppf_approx)  # data for plot
        plot_df: pd.DataFrame = pd.DataFrame(rvs)

        if axes_names is None:
            pass
        elif not (isinstance(axes_names, Iterable) and
                  len(axes_names) == rvs.shape[1]):
            raise TypeError("invalid argument type in pairplot. check "
                            "axes_names is None or a iterable with an element "
                            "for each variable.")

        if axes_names is not None:
            plot_df.columns = axes_names

        # plotting
        title: str = f"{self.name.replace('_', ' ').title()} Marginal " \
                     f"Pair-Plot"
        pair_plot(plot_df, title, color, alpha, figsize, grid, plot_kde, show)

    def _threeD_plot(self, func_str: str, copula_params: Union[Params, tuple],
                     mdists: Union[MarginalFitter, dict], ppf_approx: bool,
                     var1_range: np.ndarray, var2_range: np.ndarray,
                     color: str, alpha: float, figsize: tuple, grid: bool,
                     axes_names: Iterable, zlim: tuple, num_generate: int,
                     num_points: int, show_progress: bool, show: bool,
                     mc_num_generate: int = None, ranges_to_u: bool = False) \
            -> None:
        """Utility function able to implement pdf_plot, cdf_plot, mc_cdf_plot,
        copula_pdf_plot, copula_cdf_plot and copula_mc_cdf_plot methods
        without duplicate code.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

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
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
        ppf_approx: bool
            True to use the ppf_approx function to approximate the
            ppf / quantile function, via linear interpolation, when
            generating random variables.
            Default is True.
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 / u1 variable
            to plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99) for copula function plots. For joint distribution
            plots, the upper and lower bounds are determined by generating
            a multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 / u2 variable
            to plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99) for copula function plots. For joint distribution
            plots, the upper and lower bounds are determined by generating
            a multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
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
        # checking arguments
        test_rvs: np.ndarray = self.rvs(
            size=2, copula_params=copula_params, mdists=mdists,
            ppf_approx=ppf_approx)
        if test_rvs.shape[1] != 2:
            raise NotImplementedError(
                f"{func_str}_plot is not implemented when the number of "
                f"variables is not 2.")

        if (not isinstance(num_points, int)) or (num_points <= 0):
            raise TypeError("num_points must be a strictly positive integer.")

        if (mc_num_generate is None) and ('mc' in func_str):
            raise ValueError("mc_num_generate cannot be none for a "
                             "monte-carlo function.")

        # creating our ranges
        if (var1_range is not None) and (var2_range is not None):
            for var_range in (var1_range, var2_range):
                if not isinstance(var_range, np.ndarray):
                    raise TypeError("var1_range and var2_range must be "
                                    "None or numpy arrays.")

            if ranges_to_u:
                # converting x to u
                mdists_dict: dict = self._get_mdists(mdists=mdists, d=2,
                                                     check=True)
                var1_range = mdists_dict[0].cdf(var1_range)
                var2_range = mdists_dict[1].cdf(var2_range)

        else:
            if 'copula' in func_str:
                eps: float = 10 ** -2
                xmin, xmax = np.array([[eps, eps], [1-eps, 1-eps]],
                                      dtype=float)
            else:
                rvs_array: np.ndarray = self.rvs(
                    size=num_generate, copula_params=copula_params,
                    mdists=mdists, ppf_approx=ppf_approx)
                xmin, xmax = (rvs_array.min(axis=0), rvs_array.max(axis=0))
            var1_range: np.ndarray = np.linspace(
                xmin[0], xmax[0], num_points, dtype=float)
            var2_range: np.ndarray = np.linspace(
                xmin[1], xmax[1], num_points, dtype=float)

        # getting axes names
        if axes_names is None:
            axes_names = ('variable 1', 'variable 2')

        # title and name of plot to show user
        plot_name: str = func_str.replace('_', ' ').\
            upper().replace('COPULA', 'Copula')
        title: str = f"{self.name.replace('_', ' ').title()} {plot_name} Plot"

        # func kwargs
        func_kwargs: dict = {'copula_params': copula_params, 'mdists': mdists,
                             'match_datatype': False, 'show_progress': False}
        if func_str == 'copula_mc_cdf':
            urvs = self.copula_rvs(size=mc_num_generate,
                                   copula_params=copula_params)
            grvs: np.ndarray = self._u_to_g(urvs, copula_params)
            func_kwargs['rvs'] = grvs
        else:
            func_kwargs['rvs'] = None
        func: Callable = eval(f"self.{func_str}")

        # plotting
        threeD_plot(func=func, var1_range=var1_range, var2_range=var2_range,
                    func_kwargs=func_kwargs, func_name=plot_name, title=title,
                    color=color, alpha=alpha, figsize=figsize, grid=grid,
                    axis_names=axes_names, zlim=zlim,
                    show_progress=show_progress, show=show)

    def pdf_plot(self, copula_params: Union[Params, tuple],
                 mdists: Union[MarginalFitter, dict],
                 ppf_approx: bool = True, var1_range: np.ndarray = None,
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
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
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
            plots, the upper and lower bounds are determined by generating a
            multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 / u2 variable
            to plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99) for copula function plots. For joint distribution
            plots, the upper and lower bounds are determined by generating a
            multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
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
            func_str='pdf', copula_params=copula_params, mdists=mdists,
            ppf_approx=ppf_approx, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show)

    def cdf_plot(self, copula_params: Union[Params, tuple],
                 mdists: Union[MarginalFitter, dict], ppf_approx: bool = True,
                 var1_range: np.ndarray = None, var2_range: np.ndarray = None,
                 color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True,
                 axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100,
                 show_progress: bool = True, show: bool = True, **kwargs
                 ) -> None:
        """Produces a 3D plot of the joint distribution's cdf / cumulative
        density function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        This may take time to evaluate for certain copula distributions,
        due to d-dimensional numerical integration. In these cases,
        mc_cdf_plot will likely evaluate faster.

        This requires the sampling of multivariate random variables from the
        joint copula distribution, which requires the evaluation of the ppf /
        quantile function of each marginal distribution, which for certain
        univariate distributions requires the evaluation of an integral and
        may be time-consuming. The user therefore has the option to use the
        ppf_approx / quantile approximation function in place of this.

        Parameters
        ----------
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
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
            plots, the upper and lower bounds are determined by generating a
            multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 / u2 variable
            to plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99) for copula function plots. For joint distribution
            plots, the upper and lower bounds are determined by generating a
            multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
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
            func_str='copula_cdf', copula_params=copula_params, mdists=mdists,
            ppf_approx=ppf_approx, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show, ranges_to_u=ranges_to_u)

    def mc_cdf_plot(self, copula_params: Union[Params, tuple],
                    mdists: Union[MarginalFitter, dict],
                    ppf_approx: bool = True, var1_range: np.ndarray = None,
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
        joint copula distribution, which requires the evaluation of the ppf /
        quantile function of each marginal distribution, which for certain
        univariate distributions requires the evaluation of an integral and
        may be time-consuming. The user therefore has the option to use the
        ppf_approx / quantile approximation function in place of this.

        Parameters
        ----------
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables.
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
            plots, the upper and lower bounds are determined by generating a
            multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 / u2 variable
            to plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99) for copula function plots. For joint distribution
            plots, the upper and lower bounds are determined by generating a
            multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
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
            number of variables (length 2). If None provided, the axes will
            be labeled as 'variable 1' and 'variable 2' respectively.
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
            func_str='copula_mc_cdf', copula_params=copula_params,
            mdists=mdists, ppf_approx=ppf_approx, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show,
            mc_num_generate=mc_num_generate, ranges_to_u=ranges_to_u)

    def copula_pdf_plot(self, copula_params: Union[Params, tuple],
                        mdists: Union[MarginalFitter, dict],
                        var1_range: np.ndarray = None,
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
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable. Must be
            a fitted MarginalFitter object or a dictionary with numbered
            indices as values and fitted SklarPy univariate distributions as
            values. The dictionary indices must correspond to the indices of
            the variables.
        var1_range: np.ndarray
            numpy array containing a range of values for the x1 / u1 variable
            to plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99) for copula function plots. For joint distribution
            plots, the upper and lower bounds are determined by generating a
            multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
        var2_range: np.ndarray
            numpy array containing a range of values for the x2 / u2 variable
            to plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99) for copula function plots. For joint distribution
            plots, the upper and lower bounds are determined by generating a
            multivariate random sample of size num_generate and then taking
            the observed min and max values of each variable from this sample.
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
            func_str='copula_pdf', copula_params=copula_params, mdists=mdists,
            ppf_approx=True, var1_range=var1_range, var2_range=var2_range,
            color=color, alpha=alpha, figsize=figsize, grid=grid,
            axes_names=axes_names, zlim=zlim, num_generate=num_generate,
            num_points=num_points, show_progress=show_progress, show=show)

    def copula_cdf_plot(self, copula_params: Union[Params, tuple],
                        mdists: Union[MarginalFitter, dict],
                        var1_range: np.ndarray = None,
                        var2_range: np.ndarray = None,
                        color: str = 'royalblue', alpha: float = 1.0,
                        figsize: tuple = (8, 8), grid: bool = True,
                        axes_names: tuple = None, zlim: tuple = (None, None),
                        num_generate: int = 1000, num_points: int = 100,
                        show_progress: bool = True, show: bool = True, **kwargs
                        ) -> None:
        """Produces a 3D plot of the copula distribution's cdf / density
        function.

        Note that these plots are only implemented when we have
        2-dimensional / bivariate distributions.

        This may take time to evaluate for certain copula distributions,
        due to d-dimensional numerical integration. In these case,
        copula_mc_cdf_plot will likely evaluate faster.

        Parameters
        ----------
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        mdists : Union[MarginalFitter, dict]
            The fitted marginal distributions of each random variable. Must be
            a fitted MarginalFitter object or a dictionary with numbered
            indices as values and fitted SklarPy univariate distributions as
            values. The dictionary indices must correspond to the indices of
            the variables.
        var1_range: np.ndarray
            numpy array containing a range of values for the u1 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            (0.01, 0.99) for copula function plots.
        var2_range: np.ndarray
            numpy array containing a range of values for the u2 variable to
            plot across. If None passed, then an evenly spaced array of length
            num_points will be generated, whose upper and lower bounds are
            (0.01, 0.99) for copula function plots.
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
            func_str='copula_cdf', copula_params=copula_params, mdists=mdists,
            ppf_approx=True, var1_range=var1_range, var2_range=var2_range,
            color=color, alpha=alpha, figsize=figsize, grid=grid,
            axes_names=axes_names, zlim=zlim, num_generate=num_generate,
            num_points=num_points, show_progress=show_progress, show=show)

    def copula_mc_cdf_plot(self, copula_params: Union[Params, tuple],
                           mdists: Union[MarginalFitter, dict],
                           var1_range: np.ndarray = None,
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

        The standard copula_cdf function may take time to evaluate for
        certain copula distributions, due to d-dimensional numerical
        integration. In these cases, copula_mc_cdf_plot will likely evaluate
        faster.

        Parameters
        ----------
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the specific
            multivariate distribution or a tuple containing these parameters
            in the correct order.
        var1_range: np.ndarray
            numpy array containing a range of values for the u1 variable to
            plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99).
        var2_range: np.ndarray
            numpy array containing a range of values for the u2 variable to
            plot across. If None passed, then an evenly spaced array of
            length num_points will be generated, whose upper and lower bounds
            are (0.01, 0.99).
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
            number of variables (length 2). If None provided, the axes will
            be labeled as 'variable 1' and 'variable 2' respectively.
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
            func_str='copula_mc_cdf', copula_params=copula_params,
            mdists=mdists, ppf_approx=True, var1_range=var1_range,
            var2_range=var2_range, color=color, alpha=alpha, figsize=figsize,
            grid=grid, axes_names=axes_names, zlim=zlim,
            num_generate=num_generate, num_points=num_points,
            show_progress=show_progress, show=show,
            mc_num_generate=mc_num_generate)

    @property
    def name(self) -> str:
        """Returns the name of the copula model."""
        return self._name
