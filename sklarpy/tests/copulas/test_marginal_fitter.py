# Contains code for testing SklarPy's MarginalFitter object
import pandas as pd
import pytest
import numpy as np

from sklarpy.copulas import MarginalFitter
from sklarpy.utils._errors import FitError


def test_init(all_mvt_data):
    """Testing whether MarginalFitter initialises without errors."""
    for data in all_mvt_data.values():
        mfitter = MarginalFitter(data)


def test_fit(all_mvt_data):
    """Testing the fit method for MarginalFitter"""
    for dataset_name, data in all_mvt_data.items():
        # testing fit with default arguments
        mfitter1: MarginalFitter = MarginalFitter(data)
        mfitter1.fit()

        # testing fit with non-default arguments affecting all marginals
        univar_opts2: dict = {
            'distributions': 'all numerical',
            'numerical': True,
            'multimodal': True,
            'pvalue': 0.1}
        mfitter2: MarginalFitter = MarginalFitter(data)
        mfitter2.fit(univariate_fitter_options=univar_opts2)

        # testing fit with non-default arguments affecting individual marginals
        mfitter3: MarginalFitter = MarginalFitter(data)
        univar_opts3: dict = {i: univar_opts2
                              for i in range(mfitter3.num_variables)}
        mfitter3.fit(univariate_fitter_options=univar_opts3)

        mfitter4: MarginalFitter = MarginalFitter(data)
        with pytest.raises(ValueError):
            univar_opts4 = univar_opts3.copy()
            univar_opts4.pop(0)
            mfitter4.fit(univar_opts4)


def test_pdfs_cdfs_ppfs_logpdf(all_mvt_data, all_mvt_uniform_data):
    """Testing the cdfs method for MarginalFitter"""
    for dataset_name in ('mvt_mixed', 'pd_mvt_mixed'):
        data = all_mvt_data[dataset_name]
        mfitter: MarginalFitter = MarginalFitter(data)
        for func in ('marginal_pdfs', 'marginal_cdfs', 'marginal_ppfs',
                     'marginal_logpdfs'):
            # testing func when not fit
            with pytest.raises(FitError,
                               match='MarginalFitter object has not been '
                                     'fitted.'):
                eval(f"mfitter.{func}()")
        mfitter.fit()

        for func in ('marginal_pdfs', 'marginal_cdfs', 'marginal_ppfs',
                     'marginal_logpdfs'):
            # testing func when fit, without data
            all_values: list = []
            if func != 'marginal_ppfs':
                values = eval(f"mfitter.{func}()")
                all_values.append((values, data))

            datasets = all_mvt_uniform_data.values() \
                if func == 'marginal_ppfs' \
                else (all_mvt_data['mvt_mixed'], all_mvt_data['pd_mvt_mixed'])

            for func_data in datasets:
                # testing func when fit, with data
                values = eval(f"mfitter.{func}(func_data)")
                all_values.append((values, func_data))

            for values, func_data in all_values:
                # checking match datatype
                assert type(values) == type(data), 'match_datatype failed'

                # checking same shape
                np_values, np_func_data = (np.asarray(values),
                                           np.asarray(func_data))
                assert np_values.shape == np_func_data.shape, \
                    "missmatch between shape of output and input"

            # testing exception raised when the input dimensions not
            # consistent with the numer of variables
            half_n: int = int(mfitter.num_variables / 2)
            half_data = data.iloc[:, half_n] if isinstance(
                data, pd.DataFrame) else data[:, half_n]
            with pytest.raises(ValueError):
                eval(f"mfitter.{func}(half_data)")
