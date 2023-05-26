import numpy as np
import pandas as pd

from sklarpy.copulas._copula import Copula
from sklarpy.misc import CorrelationMatrix
from sklarpy._utils import dataframe_or_array

__all__ = ['Elliptical']


class Elliptical(Copula):
    def __init__(self, marginals=None, name: str = None):
        Copula.__init__(self, marginals, name)
        self._corr: np.ndarray = None

    def _prepare_square_matrix(self, square_mat) -> np.ndarray:
        if self._type_keeper is not None:
            square_mat = self._type_keeper.match_square_matrix(square_mat)
        if isinstance(square_mat, pd.DataFrame):
            square_mat = square_mat.to_numpy()
        return square_mat

    def _fit_corr(self, method: str = 'laloux_pp_kendall', corr: dataframe_or_array = None, **kwargs):
        if corr is None:
            if self._marginals is None:
                self._no_data_or_params()
            else:
                corr = CorrelationMatrix(self._marginals).corr(method, **kwargs)
        self._corr = self._prepare_square_matrix(corr)
        Copula._check_num_variables(self, self._corr.shape[0])

    def _generate_multivariate_std_normal_rvs(self, size: int):
        A: np.ndarray = np.linalg.cholesky(self.corr).T
        rvs_std_normal: np.ndarray = np.random.normal(0, 1, size=(size, self._num_variables))
        return rvs_std_normal@A

    @property
    def corr(self) -> np.ndarray:
        self._fit_check()
        return self._corr.copy()
