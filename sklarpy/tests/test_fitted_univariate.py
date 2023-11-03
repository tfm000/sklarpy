# # Contains tests for sklarpy fitted univatiate distributions
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import os
#
# from sklarpy.univariate import distributions_map
# from sklarpy.univariate._fitted_dists import FittedContinuousUnivariate, FittedDiscreteUnivariate
# from sklarpy._utils import near_zero, SaveError, FitError
# from sklarpy.tests.helpers import get_data, get_fitted_dict
#
#
# def test_fitted_pdfs(poisson_data, uniform_data):
#     """Testing the pdf functions of all fitted univariate distributions"""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking pdf values are the correct data-type
#                 pdf_values: np.ndarray = fitted.pdf(data)
#                 assert isinstance(pdf_values, np.ndarray), f"pdf values for {name} are not contained in a numpy " \
#                                                            f"array when {fitted_type}"
#
#                 # checking same number of pdf values as input
#                 assert pdf_values.size == data.size, f"number pdf values for {name} do not match the number of " \
#                                                      f"inputs when {fitted_type}"
#
#                 # checking for nan values
#                 assert np.isnan(pdf_values).sum() == 0, f"nan values present in {name} pdf when {fitted_type}"
#
#                 # checking all pdf values are non-negative
#                 assert np.all(pdf_values >= 0), f"pdf values present in {name} are negative when {fitted_type}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_cdfs(poisson_data, uniform_data):
#     """Testing the cdf functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking cdf values are the correct data-type
#                 cdf_values: np.ndarray = fitted.cdf(data)
#                 assert isinstance(cdf_values, np.ndarray), f"cdf values for {name} are not contained in a numpy " \
#                                                            f"array when {fitted_type}"
#
#                 # checking same number of cdf values as input
#                 assert cdf_values.size == data.size, f"number cdf values for {name} do not match the number of " \
#                                                      f"inputs when {fitted_type}"
#
#                 # checking for nan and inf values
#                 assert (np.isnan(cdf_values).sum() == 0) and (not np.all(np.isinf(cdf_values))), \
#                     f"nan or inf values present in {name} cdf when {fitted_type}"
#
#                 # checking cdf values are non-decreasing
#                 sorted_data: np.ndarray = data.copy()
#                 sorted_data.sort()
#                 sorted_cdf_values: np.ndarray = fitted.cdf(sorted_data)
#                 neighbour_difference: np.ndarray = sorted_cdf_values[1:] - sorted_cdf_values[:-1]
#                 negative_values: np.ndarray = neighbour_difference[np.where(neighbour_difference < 0)]
#                 if negative_values.size > 0:
#                     # we may have negative_values which are very small and likely a float rounding error.
#                     assert np.all(negative_values > -near_zero), f"cdf values of {name} are not monotonically " \
#                                                                  f"increasing when {fitted_type}."
#
#                 # checking extremes
#                 assert np.all(fitted.cdf(np.inf) == 1.0), f"cdf of {name} is not 1.0 at infinity when {fitted_type}"
#                 assert np.all(fitted.cdf(-np.inf) == 0.0), f"cdf of {name} is not 0.0 at -infinity when {fitted_type}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_ppf(poisson_data, uniform_data):
#     """Testing the ppf functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 ppf_values: np.ndarray = fitted.ppf(uniform_data)
#                 # checking correct type
#                 assert isinstance(ppf_values, np.ndarray), f"ppf values for {name} are not contained in a numpy " \
#                                                            f"array when {fitted_type}"
#
#                 # checking same number of ppf values as input
#                 assert ppf_values.size == uniform_data.size, f"number ppf values for {name} do not match the number " \
#                                                              f"of inputs when {fitted_type}"
#
#                 # checking for nan values
#                 assert np.isnan(ppf_values).sum() == 0, f"nan values present in {name} ppf when {fitted_type}"
#         except RuntimeError:
#             continue
#
#
# def test_fitted_rvs(poisson_data, uniform_data):
#     """Testing the rvs functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 num: int = 10
#                 for shape in ((num,), (num, 2), (num, 5), (num, 13, 6)):
#                     rvs_values: np.ndarray = fitted.rvs(shape)
#
#                     # checking correct type
#                     assert isinstance(rvs_values, np.ndarray), f"rvs values for {name} are not contained in an " \
#                                                                f"np.ndarray when {fitted_type}"
#
#                     # checking for nan values
#                     assert np.isnan(rvs_values).sum() == 0, f"nan values present in {name} rvs when {fitted_type}"
#
#                     # checking correct shape
#                     assert rvs_values.shape == shape, f"incorrect shape generated for rvs for {name} when" \
#                                                       f" {fitted_type}. target shape is {shape}, generated shape is " \
#                                                       f"{rvs_values.shape}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_logpdf(poisson_data, uniform_data):
#     """Testing the log-pdf functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking log-pdf values are the correct data-type
#                 logpdf_values: np.ndarray = fitted.logpdf(data)
#                 assert isinstance(logpdf_values, np.ndarray), f"log-pdf values for {name} are not contained in a " \
#                                                               f"numpy array when {fitted_type}"
#
#                 # checking same number of pdf values as input
#                 assert logpdf_values.size == data.size, f"number of log-pdf values for {name} do not match the " \
#                                                         f"number of inputs when {fitted_type}"
#
#                 # checking for nan values
#                 assert np.isnan(logpdf_values).sum() == 0, f"nan values present in {name} log-pdf when {fitted_type}"
#         except RuntimeError:
#             continue
#
#
# def test_fitted_likelihood(poisson_data, uniform_data):
#     """Testing the likelihood functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking likelihood values are the correct type
#                 likelihood: float = fitted.likelihood(data)
#                 assert isinstance(likelihood, float), f"likelihood for {name} is not a float when {fitted_type}"
#
#                 # checking likelihood is a valid number
#                 valid: bool = not (np.isnan(likelihood) or (likelihood < 0))
#                 assert valid, f"likelihood for {name} is is nan or negative when {fitted_type}"
#         except RuntimeError:
#             continue
#
#
# def test_fitted_loglikelihood(poisson_data, uniform_data):
#     """Testing the log-likelihood functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking log-likelihood values are the correct type
#                 loglikelihood: float = fitted.loglikelihood(data)
#                 assert isinstance(loglikelihood, float), f"log-likelihood for {name} is not a float when {fitted_type}"
#
#                 # checking log-likelihood is a valid number
#                 valid: bool = not np.isnan(loglikelihood)
#                 assert valid, f"log-likelihood for {name} is is nan when {fitted_type}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_aic(poisson_data, uniform_data):
#     """Testing the aic functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking aic values are the correct type
#                 aic: float = fitted.aic(data)
#                 assert isinstance(aic, float), f"aic for {name} is not a float when {fitted_type}"
#
#                 # checking aic is a valid number
#                 valid: bool = not np.isnan(aic)
#                 assert valid, f"aic for {name} is is nan when {fitted_type}"
#         except RuntimeError:
#             continue
#
#
# def test_fitted_bic(poisson_data, uniform_data):
#     """Testing the bic functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking bic values are the correct type
#                 bic: float = fitted.bic(data)
#                 assert isinstance(bic, float), f"bic for {name} is not a float when {fitted_type}"
#
#                 # checking bic is a valid number
#                 valid: bool = not np.isnan(bic)
#                 assert valid, f"bic for {name} is is nan when {fitted_type}"
#         except RuntimeError:
#             continue
#
#
# def test_fitted_sse(poisson_data, uniform_data):
#     """Testing the sse functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking sse values are the correct type
#                 sse: float = fitted.sse(data)
#                 assert isinstance(sse, float), f"sse for {name} is not a float when {fitted_type}"
#
#                 # checking sse is a valid number
#                 valid: bool = not (np.isnan(sse) or (sse < 0))
#                 assert valid, f"sse for {name} is is nan or negative when {fitted_type}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_gof(poisson_data, uniform_data):
#     """Testing the gof functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking gof object is a dataframe
#                 gof: pd.DataFrame = fitted.gof(data)
#                 assert isinstance(gof, pd.DataFrame), f"gof for {name} is not a pandas dataframe when {fitted_type}"
#
#                 # checking gof is non-empty
#                 assert len(gof) > 0, f"gof for {name} is empty when {fitted_type}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_plot(poisson_data, uniform_data):
#     """Testing the plot functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking we can plot without errors
#                 fitted.plot(show=False)
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_save(poisson_data, uniform_data):
#     """Testing the save functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 save_location: str = f'{os.getcwd()}/{fitted.name}.pickle'
#                 fitted.save(save_location)
#                 my_fitted_object = Path(save_location)
#                 if my_fitted_object.exists():
#                     my_fitted_object.unlink()
#                 else:
#                     raise SaveError(f"unable to save {fitted.name} when {fitted_type}")
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_name(poisson_data, uniform_data):
#     """Testing the name functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 assert isinstance(fitted.name, str), f"name of {name} distribution is not a string when {fitted_type}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_name_with_params(poisson_data, uniform_data):
#     """Testing the name with parameters functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 assert isinstance(fitted.name_with_params, str), f"name_with_params of {name} distribution is " \
#                                                                  f"not a string when {fitted_type}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_summary(poisson_data, uniform_data):
#     """Testing the summary functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking summary object is a dataframe
#                 summary: pd.DataFrame = fitted.summary
#                 assert isinstance(summary, pd.DataFrame), f"summary for {name} is not a pandas dataframe when " \
#                                                           f"{fitted_type}"
#
#                 # checking gof is non-empty
#                 assert len(summary) > 0, f"summary for {name} is empty when {fitted_type}"
#         except RuntimeError:
#             continue
#
#
# def test_fitted_params(poisson_data, uniform_data):
#     """Testing the parameters functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 params: tuple = fitted.params
#
#                 # checking parameters are in a tuple
#                 assert isinstance(params, tuple), f"params for {name} is not a tuple when {fitted_type}"
#
#                 # checking parameters are not nan or inf
#                 for param in params:
#                     if np.isinf(param) or np.isnan(param):
#                         raise FitError(f"Parameters for {name} are inf or nan when {fitted_type}")
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_support(poisson_data, uniform_data):
#     """Testing the support functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 support: tuple = fitted.support
#
#                 # checking support is a tuple
#                 assert isinstance(support, tuple), f"support values for {name} are not contained in a tuple " \
#                                                    f"when {fitted_type}"
#
#                 # checking for nan values
#                 assert np.isnan(support).sum() == 0, f"nan values present in {name} support when {fitted_type}"
#
#                 # checking only two values in support
#                 assert len(support) == 2, f"incorrect number of values in support for {name} when {fitted_type}"
#
#                 # checking lb < ub
#                 assert support[0] < support[1], f"lb < ub is not satisfied in support for {name} when {fitted_type}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_domain(poisson_data, uniform_data):
#     """Testing the fitted domain functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 fitted_domain: tuple = fitted.fitted_domain
#
#                 # checking fitted domain values are in a tuple
#                 assert isinstance(fitted_domain, tuple), f"fitted_domain for {name} is not a tuple when {fitted_type}"
#
#                 # checking tuple is of length 2 or 0
#                 if fitted_type == 'fitted to data':
#                     req_length: int = 2
#                 else:
#                     req_length: int = 0
#                 assert len(fitted_domain) == req_length, f"length of fitted domain for {name} is not {req_length} " \
#                                                          f"when {fitted_type}"
#
#                 # checking fitted domain values in (min, max) order
#                 if fitted_type == 'fitted to data':
#                     assert fitted_domain[0] <= fitted_domain[1], f"lb <= ub not satisfied for {name}'s fitted domain " \
#                                                                  f"when {fitted_type}"
#
#
#                 # checking fitted domain values are not nan or inf
#                 for val in fitted_domain:
#                     if np.isinf(val) or np.isnan(val):
#                         raise FitError(f"Fitted domain values for {name} are inf or nan when {fitted_type}")
#         except RuntimeError:
#             continue
#
#
# def test_fitted_fitted_to_data(poisson_data, uniform_data):
#     """Testing the fitted to data functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 # checking fitted to data is boolean
#                 fitted_to_data: bool = fitted.fitted_to_data
#                 assert isinstance(fitted_to_data, bool), f"fitted_to_data is not boolean for {name} " \
#                                                                 f"when {fitted_type}"
#
#                 # checking fitted_to_data value is correct
#                 if fitted_type == 'fitted to data':
#                     correct_value: bool = (fitted_to_data == True)
#                 else:
#                     correct_value: bool = (fitted_to_data == False)
#                 assert correct_value, f"incorrect value for fitted_to_data for {name} when {fitted_type}"
#         except RuntimeError:
#             continue
#
#
# def test_fitted_num_params(poisson_data, uniform_data):
#     """Testing the num_params functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 num_params: int = fitted.num_params
#
#                 # checking correct type
#                 assert isinstance(num_params, int), f"num_params is not an integer for {name} when {fitted_type}"
#
#                 # checking non-nan, non-inf and positive
#                 assert num_params >= 0 and (not (np.isnan(num_params) and np.inf(num_params))), f"num_params is nan, " \
#                                                                                                 f"inf or < 0 for " \
#                                                                                                 f"{name} when " \
#                                                                                                 f"{fitted_type}"
#                 # checking valid value
#                 if name in distributions_map['all numerical']:
#                     valid_value: bool = (num_params == 0)
#                 else:
#                     valid_value: bool = (num_params > 0)
#                 assert valid_value, f"incorrect num_params value for {name} when {fitted_type}"
#
#         except RuntimeError:
#             continue
#
#
# def test_fitted_num_data_points(poisson_data, uniform_data):
#     """Testing the num_data_points functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 num_data_points: int = fitted.fitted_num_data_points
#
#                 # checking correct data type
#                 assert isinstance(num_data_points, int), f"num_data_points for {name} is not an integer when " \
#                                                          f"{fitted_type}"
#
#                 # checking non-nan, non-inf and positive
#                 assert num_data_points >= 0 and (not (np.isnan(num_data_points) and np.inf(num_data_points))), \
#                     f"num_data_points is nan, inf or < 0 for {name} when {fitted_type}"
#
#                 # checking valid value
#                 if fitted_type == 'fitted to data':
#                     valid_value: bool = (num_data_points > 0)
#                 else:
#                     valid_value: bool = (num_data_points == 0)
#                 assert valid_value, f"incorrect num_data_points value for {name} when {fitted_type}"
#         except RuntimeError:
#             continue
#
#
# def test_fitted_continuous_or_discrete(poisson_data, uniform_data):
#     """Testing the continuous_or_discrete functions of all fitted univariate distributions."""
#     for name in distributions_map['all']:
#         try:
#             data: np.ndarray = get_data(name, uniform_data, poisson_data)
#             fitted_dict: dict = get_fitted_dict(name, data)
#
#             # testing when fitting to both data and parameters
#             for fitted_type, fitted in fitted_dict.items():
#                 continuous_or_discrete: str = fitted.continuous_or_discrete
#
#                 # checking correct data type
#                 assert isinstance(continuous_or_discrete, str), f"continuous_or_discrete is not a string for {name} " \
#                                                                 f"when {fitted_type}"
#
#                 # checking correct value
#                 correct_value: bool = (isinstance(fitted, FittedContinuousUnivariate) and
#                                        continuous_or_discrete == 'continuous') or \
#                                       (isinstance(fitted, FittedDiscreteUnivariate) and
#                                        continuous_or_discrete == 'discrete')
#                 assert correct_value, f"continuous_or_discrete value is incorrect for {name} when {fitted_type}"
#
#
#         except RuntimeError:
#             continue
#
#
# # def test_fitted_(poisson_data, uniform_data):
# #     """Testing the  functions of all fitted univariate distributions."""
# #     for name in distributions_map['all']:
# #         try:
# #             data: np.ndarray = get_data(name, uniform_data, poisson_data)
# #             fitted_dict: dict = get_fitted_dict(name, data)
# #
# #             # testing when fitting to both data and parameters
# #             for fitted_type, fitted in fitted_dict.items():
# #
# #         except RuntimeError:
# #             continue