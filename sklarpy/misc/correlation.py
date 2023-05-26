import numpy as np
import pandas as pd
import warnings


__all__ = ['CorrelationMatrix']


class CorrelationMatrix:
    IMPLEMENTED: tuple = (
        'pearson', 'spearman', 'kendall', 'pp_kendall',
        'rm_pearson', 'rm_spearman', 'rm_kendall', 'rm_pp_kendall',
        'laloux_pearson', 'laloux_spearman', 'laloux_kendall', 'laloux_pp_kendall',
    )

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array.")
        self._data: pd.DataFrame = pd.DataFrame(data)

    def _pearson_spearman_kendall(self, method: str, raise_error: bool):
        corr: np.ndarray = self._data.corr(method).to_numpy()
        self.check_correlation_matrix(corr, raise_error)
        return corr

    def pearson(self, raise_error: bool = False, **kwargs) -> np.ndarray:
        return self._pearson_spearman_kendall('pearson', raise_error)

    def spearman(self, raise_error: bool = False, **kwargs) -> np.ndarray:
        return self._pearson_spearman_kendall('spearman', raise_error)

    def kendall(self, raise_error: bool = False, **kwargs) -> np.ndarray:
        return self._pearson_spearman_kendall('kendall', raise_error)

    def pp_kendall(self, raise_error: bool = False, **kwargs) -> np.ndarray:
        corr: np.ndarray = np.sin(np.pi * 0.5 * self.kendall(False))
        self.check_correlation_matrix(corr, raise_error)
        return corr

    def _rm_corr(self, delta: float, renormalise: bool, method: str) -> np.ndarray:
        """Technique by Rousseeuw and Molenberghs to ensure a given correlation matrix is positive definite."""
        corr: np.ndarray = eval(f"self.{method}(False)")
        eigenvalues, eigenvectors = np.linalg.eig(corr)
        new_eigenvalues: np.ndarray = np.where(eigenvalues > 0, eigenvalues, delta)
        new_corr: np.ndarray = eigenvectors@np.diag(new_eigenvalues)@np.linalg.inv(eigenvectors)
        if renormalise:
            # setting the diagonal values to be exactly 1.0
            diagonal_indices = range(new_corr.shape[0])
            new_corr[diagonal_indices, diagonal_indices] = 1.0
        return new_corr

    def rm_pearson(self, delta: float = 10**-9, renormalise: bool = True, **kwargs) -> np.ndarray:
        return self._rm_corr(delta, renormalise, 'pearson')

    def rm_spearman(self, delta: float = 10**-9, renormalise: bool = True, **kwargs) -> np.ndarray:
        return self._rm_corr(delta, renormalise, 'spearman')

    def rm_kendall(self, delta: float = 10**-9, renormalise: bool = True, **kwargs) -> np.ndarray:
        return self._rm_corr(delta, renormalise, 'kendall')

    def rm_pp_kendall(self, delta: float = 10**-9, renormalise: bool = True, **kwargs) -> np.ndarray:
        return self._rm_corr(delta, renormalise, 'pp_kendall')

    def _laloux_corr(self, delta: float, method: str) -> np.ndarray:
        """Technique by Laloux et al which ensures a positive definite correlation matrix and denoising."""
        n, d = self._data.shape
        Q: float = n / d
        if Q < 1:
            raise ArithmeticError("laloux correlation matrices can only be calculated when number of data points "
                                  ">= number of variables.")

        # Performing the Rousseeuw and Molenberghs technique to get a positive definite correlation matrix
        corr: np.ndarray = eval(f"self.{method}(False)")
        eigenvalues, eigenvectors = np.linalg.eig(corr)
        rm_eigenvalues: np.ndarray = np.where(eigenvalues > 0, eigenvalues, delta)

        # substituting any eigenvalues in the bulk by their mean.
        new_eigenvalues: np.ndarray = rm_eigenvalues.copy()
        bulk: tuple = (1 - (Q ** -0.5)) ** 2, (1 + (Q ** -0.5)) ** 2
        eigenvalues_in_bulk: np.ndarray = np.where(new_eigenvalues < bulk[1])[0]
        if len(eigenvalues_in_bulk) > 0:
            new_eigenvalues[eigenvalues_in_bulk] = new_eigenvalues[eigenvalues_in_bulk].mean()

        new_corr: np.ndarray = eigenvectors @ np.diag(new_eigenvalues) @ np.linalg.inv(eigenvectors)

        # setting the diagonal values to be exactly 1.0
        diagonal_indices = range(new_corr.shape[0])
        new_corr[diagonal_indices, diagonal_indices] = 1.0
        return new_corr

    def laloux_pearson(self, delta: float = 10**-9, **kwargs) -> np.ndarray:
        return self._laloux_corr(delta, 'pearson')

    def laloux_spearman(self, delta: float = 10**-9, **kwargs) -> np.ndarray:
        return self._laloux_corr(delta, 'spearman')

    def laloux_kendall(self, delta: float = 10**-9, **kwargs) -> np.ndarray:
        return self._laloux_corr(delta, 'kendall')

    def laloux_pp_kendall(self, delta: float = 10**-9, **kwargs) -> np.ndarray:
        return self._laloux_corr(delta, 'pp_kendall')

    def corr(self, method: str, **kwargs) -> np.ndarray:
        # checking arguments
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        method: str = method.lower().strip()
        if method not in CorrelationMatrix.IMPLEMENTED:
            raise ValueError(f"{method} is not a valid argument. Specify from {CorrelationMatrix.IMPLEMENTED}")

        return eval(f"self.{method}(**kwargs)")

    @staticmethod
    def _is_2d(arr: np.ndarray) -> bool:
        """checks if an array is 2 dimensional."""
        shape: tuple = arr.shape
        if len(shape) != 2:
            return False
        return True

    @staticmethod
    def _is_square(arr: np.ndarray) -> bool:
        """checks if an array is square."""
        if not CorrelationMatrix._is_2d(arr):
            return False

        shape: tuple = arr.shape
        if shape[0] != shape[1]:
            return False
        return True

    @staticmethod
    def check_correlation_matrix(corr: np.ndarray, raise_error: bool = True, **kwargs) -> bool:
        """Performs checks on a given numpy array to see if it is a valid correlation matrix.

        Parameters
        ----------
        corr : np.ndarray
            the matrix to check
        raise_error : bool
            Whether to raise an error if corr is not a valid correlation matrix.
            Default is True

        Returns
        -------
        passes_checks: bool
            True if the array is a valid correlation matrix, False otherwise.
        """
        passes_checks: bool = True

        # checking numpy array passed
        if not isinstance(corr, np.ndarray):
            raise TypeError("corr must be a numpy array.")

        # checking correlation matrix is square
        if not CorrelationMatrix._is_square(corr):
            if raise_error:
                raise ValueError("Correlation matrix is not 2d and square")
            else:
                warnings.warn("Correlation matrix is not 2d and square")
            passes_checks = False

        # checking correlation matrix has all ones in diagonal
        if not np.all(corr.diagonal() == 1.0):
            if raise_error:
                raise ValueError("Correlation matrix does not have all ones in diagonal")
            else:
                warnings.warn("Correlation matrix does not have all ones in diagonal")
            passes_checks = False

        # checking correlation matrix is psd
        if not np.all(np.linalg.eigvals(corr) >= 0):
            if raise_error:
                raise ValueError("Correlation matrix is not positive semi-definite")
            else:
                warnings.warn("Correlation matrix is not positive semi-definite")
            passes_checks = False

        # checking correlation matrix is symmetric
        if not np.allclose(corr, corr.T):
            if raise_error:
                raise ValueError("Correlation matrix is not symmetric")
            else:
                warnings.warn("Correlation matrix is not symmetric")
            passes_checks = False

        return passes_checks

    @staticmethod
    def cov_from_corr(corr: np.ndarray, std: np.ndarray, raise_error: bool = True, **kwargs) -> np.ndarray:
        CorrelationMatrix.check_correlation_matrix(corr, raise_error)
        if not isinstance(std, np.ndarray):
            raise TypeError("std is not a np.ndarray")
        elif std.size != corr.shape[0]:
            raise ValueError("std length does not match the dimensions of corr.")

        std_diag: np.ndarray = np.diag(std.flatten())
        return std_diag @ corr @ std_diag

    def cov(self, method: str, **kwargs) -> np.ndarray:
        corr: np.ndarray = self.corr(method, **kwargs)
        std: np.ndarray = self._data.std(axis=0).to_numpy()
        return self.cov_from_corr(corr, std, **kwargs)

    @staticmethod
    def check_covariance_matrix(cov: np.ndarray, raise_error: bool = True, **kwargs):
        """Performs checks on a given numpy array to see if it is a valid covariance matrix.

        Parameters
        ----------
        cov : np.ndarray
            the matrix to check
        raise_error : bool
            Whether to raise an error if corr is not a valid covariance matrix.
            Default is True

        Returns
        -------
        passes_checks: bool
            True if the array is a valid covariance matrix, False otherwise.
        """
        passes_checks: bool = True

        # checking numpy array passed
        if not isinstance(cov, np.ndarray):
            raise TypeError("cov must be a numpy array.")

        # checking covariance matrix is square
        if not CorrelationMatrix._is_square(cov):
            if raise_error:
                raise ValueError("Covariance matrix is not 2d and square")
            else:
                warnings.warn("Covariance matrix is not 2d and square")
            passes_checks = False

        # checking covariance matrix is p.d
        if not np.all(np.linalg.eigvals(cov) > 0):
            if raise_error:
                raise ValueError("Covariance matrix is not positive definite")
            else:
                warnings.warn("Covariance matrix is not positive definite")
            passes_checks = False

        # checking covariance matrix is symmetric
        if not np.allclose(cov, cov.T):
            if raise_error:
                raise ValueError("Covariance matrix is not symmetric")
            else:
                warnings.warn("Covariance matrix is not symmetric")
            passes_checks = False

        return passes_checks

# def correlation_matrix(data: np.ndarray, corr: Union[np.ndarray, str], raise_error: bool = True) -> np.ndarray:
#     if isinstance(corr, str):
#         corr = corr.lower()
#
#         if corr in CorrelationMatrix.IMPLEMENTED:
#             return eval(f"CorrelationMatrix(data).{corr}()")
#         raise ValueError(f'{corr} is not a valid option for corr')
#     elif isinstance(corr, np.ndarray):
#         CorrelationMatrix.check_correlation_matrix(corr, raise_error=True)
#         return corr
#     raise TypeError("invalid type for correlation matrix")