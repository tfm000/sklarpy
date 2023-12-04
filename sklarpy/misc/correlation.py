# Contains code for fitting covariance and correlation matrices to data
import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Union

from sklarpy.utils._input_handlers import check_multivariate_data

__all__ = ['CorrelationMatrix']


class CorrelationMatrix:
    """Class for fitting covariance and correlation matrices to data."""
    IMPLEMENTED: tuple = (
        'pearson', 'spearman', 'kendall', 'pp_kendall',
        'rm_pearson', 'rm_spearman', 'rm_kendall', 'rm_pp_kendall',
        'laloux_pearson', 'laloux_spearman', 'laloux_kendall',
        'laloux_pp_kendall',
    )

    def __init__(self, data: Union[np.ndarray, pd.DataFrame]):
        """Object for fitting covariance and correlation matrices to data.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            Dataset of observations to use when fitting correlation /
            covariance matrices.
        """
        data_array: np.ndarray = \
            check_multivariate_data(data=data, allow_1d=False)
        self._data: pd.DataFrame = pd.DataFrame(data_array)

    def _pearson_spearman_kendall(self, method: str, raise_error: bool) \
            -> np.ndarray:
        """Utility function able to implement Pearson, Spearman and Kendall
        methods without duplicate code.

        Parameters
        ----------
        method : str
            The correlation fitting method to implement.
        raise_error: bool
            True to raise an error if the resultant matrix is not a valid
            correlation matrix. I.e. if the result is not square, symmetric,
            positive semi-definite and contains 1's in the diagonal.

        Returns
        -------
        corr: np.ndarray
            A correlation matrix estimator.
        """
        corr: np.ndarray = self._data.corr(method).to_numpy()
        self.check_correlation_matrix(corr, raise_error)
        return corr

    def pearson(self, raise_error: bool = False, **kwargs) -> np.ndarray:
        """Fits a Pearson correlation matrix estimator to the dataset.

        Parameters
        ----------
        raise_error: bool
            True to raise an error if the resultant matrix is not a valid
            correlation matrix. I.e. if the result is not square, symmetric,
            positive semi-definite and contains 1's in the diagonal.

        Returns
        -------
        pearson_corr np.ndarray
            A Pearson correlation matrix estimator.
        """
        return self._pearson_spearman_kendall('pearson', raise_error)

    def spearman(self, raise_error: bool = False, **kwargs) -> np.ndarray:
        """Fits a Spearman's rank correlation matrix estimator to the dataset.

        Parameters
        ----------
        raise_error: bool
            True to raise an error if the resultant matrix is not a valid
            correlation matrix. I.e. if the result is not square, symmetric,
            positive semi-definite and contains 1's in the diagonal.

        Returns
        -------
        spearman_corr np.ndarray
            A Spearman's rank correlation matrix estimator.
        """
        return self._pearson_spearman_kendall('spearman', raise_error)

    def kendall(self, raise_error: bool = False, **kwargs) -> np.ndarray:
        """Fits a Kendall rank correlation matrix estimator to the dataset.

        Parameters
        ----------
        raise_error: bool
            True to raise an error if the resultant matrix is not a valid
            correlation matrix. I.e. if the result is not square, symmetric,
            positive semi-definite and contains 1's in the diagonal.

        Returns
        -------
        pearson_corr np.ndarray
            A Kendall correlation matrix estimator.
        """
        return self._pearson_spearman_kendall('kendall', raise_error)

    def pp_kendall(self, raise_error: bool = False, **kwargs) -> np.ndarray:
        """Fits the robust Pseudo-Pearson Kendall (PP-Kendall) correlation
        matrix estimator.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        pp_kendall_corr: np.ndarray
            A PP-Kendall correlation matrix estimator.
        """
        corr: np.ndarray = np.sin(np.pi * 0.5 * self.kendall(False))
        self.check_correlation_matrix(corr, raise_error)
        return corr

    @staticmethod
    def _rm_pd(A: np.ndarray, delta: float) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Technique by Rousseeuw and Molenberghs to ensure a given matrix is
        transformed to be positive definite. Any negative eigenvalues are
        replaced with a positive delta value.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        res: Tuple[np.ndarray, np.ndarray, np.ndarray]
            positive definite matrix A, eigenvectors of A, eigenvalues of
            positive definite A
        """
        if not (isinstance(delta, float) or isinstance(delta, int)):
            raise TypeError("delta must be a scalar value.")
        elif delta <= 0:
            raise ValueError("delta must be a positive scalar value.")

        eigenvalues, eigenvectors = np.linalg.eig(A)
        if np.all(eigenvalues > 0):
            # no work to be done
            return A, eigenvectors, eigenvalues
        new_eigenvalues: np.ndarray = \
            np.where(eigenvalues > 0, eigenvalues, delta)
        new_A: np.ndarray = eigenvectors @ np.diag(new_eigenvalues) \
                            @ np.linalg.inv(eigenvectors)
        return new_A, eigenvectors, new_eigenvalues

    def _rm_corr(self, delta: float, renormalise: bool, method: str) \
            -> np.ndarray:
        """Utility function able to implement Rousseeuw and Molenberghs method
        to create positive definite Pearson, Spearman rank and Kendall rank
        correlation matrix estimators, without duplicate code.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.
        renormalise: bool
            True to set the diagonal elements of the resultant matrix to be
            exactly 1.0.
        method : str
            The correlation fitting method to implement.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        corr: np.ndarray
            A positive definite correlation matrix estimator.
        """
        corr: np.ndarray = eval(f"self.{method}(False)")
        new_corr, _, _ = self._rm_pd(corr, delta)
        if renormalise:
            # setting the diagonal values to be exactly 1.0
            diagonal_indices = range(new_corr.shape[0])
            new_corr[diagonal_indices, diagonal_indices] = 1.0
        return new_corr

    def rm_pearson(self, delta: float = 10 ** -9, renormalise: bool = True,
                   **kwargs) -> np.ndarray:
        """Fits a Pearson correlation matrix estimator to the dataset, in
        addition to applying Rousseeuw and Molenberghs' technique to ensure the
        matrix is transformed to be positive definite. Any negative eigenvalues
        are replaced with a positive delta value.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.
        renormalise: bool
            True to set the diagonal elements of the resultant matrix to be
            exactly 1.0.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        rm_pearson_corr np.ndarray
            A Pearson correlation matrix estimator, transformed to be positive
            definite.
        """
        return self._rm_corr(delta, renormalise, 'pearson')

    def rm_spearman(self, delta: float = 10 ** -9, renormalise: bool = True,
                    **kwargs) -> np.ndarray:
        """Fits a Spearman rank correlation matrix estimator to the dataset, in
        addition to applying Rousseeuw and Molenberghs' technique to ensure the
        matrix is transformed to be positive definite. Any negative eigenvalues
        are replaced with a positive delta value.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.
        renormalise: bool
            True to set the diagonal elements of the resultant matrix to be
            exactly 1.0.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        rm_spearman_corr np.ndarray
            A Spearman rank correlation matrix estimator, transformed to be
            positive definite.
        """
        return self._rm_corr(delta, renormalise, 'spearman')

    def rm_kendall(self, delta: float = 10 ** -9, renormalise: bool = True,
                   **kwargs) -> np.ndarray:
        """Fits a Kendall rank correlation matrix estimator to the dataset, in
        addition to applying Rousseeuw and Molenberghs' technique to ensure the
        matrix is transformed to be positive definite. Any negative eigenvalues
        are replaced with a positive delta value.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.
        renormalise: bool
            True to set the diagonal elements of the resultant matrix to be
            exactly 1.0.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        rm_kendall_corr np.ndarray
            A Kendall rank correlation matrix estimator, transformed to be
            positive definite.
        """
        return self._rm_corr(delta, renormalise, 'kendall')

    def rm_pp_kendall(self, delta: float = 10 ** -9, renormalise: bool = True,
                      **kwargs) -> np.ndarray:
        """Fits a Pseudo-Pearson Kendall (PP-Kendall) correlation matrix
        estimator to the dataset, in addition to applying Rousseeuw and
        Molenberghs' technique to ensure the matrix is transformed to be
        positive definite. Any negative eigenvalues are replaced with a
        positive delta value.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.
        renormalise: bool
            True to set the diagonal elements of the resultant matrix to be
            exactly 1.0.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        rm_pp_kendall_corr np.ndarray
            A PP-Kendall rank correlation matrix estimator, transformed to be
            positive definite.
        """
        return self._rm_corr(delta, renormalise, 'pp_kendall')

    def _laloux_corr(self, delta: float, method: str) -> np.ndarray:
        """Utility function able to implement Laloux et al.'s method
        to create positive definite, denoised Pearson, Spearman rank and
        Kendall rank correlation matrix estimators, without duplicate code.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.
        method : str
            The correlation fitting method to implement.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        corr: np.ndarray
            A positive definite, denoised correlation matrix estimator.
        """
        n, d = self._data.shape
        Q: float = n / d
        if Q < 1:
            raise ArithmeticError("laloux correlation matrices can only be "
                                  "calculated when number of data points >= "
                                  "number of variables.")

        # Performing the Rousseeuw and Molenberghs technique to get a positive
        # definite correlation matrix
        corr: np.ndarray = eval(f"self.{method}(False)")
        _, eigenvectors, rm_eigenvalues = self._rm_pd(corr, delta)

        # substituting any eigenvalues in the bulk by their mean.
        new_eigenvalues: np.ndarray = rm_eigenvalues.copy()
        bulk: tuple = ((1 - (Q ** -0.5)) ** 2, (1 + (Q ** -0.5)) ** 2)
        eigenvalues_in_bulk: np.ndarray = \
            np.where(new_eigenvalues < bulk[1])[0]
        if len(eigenvalues_in_bulk) > 0:
            new_eigenvalues[eigenvalues_in_bulk] = \
                new_eigenvalues[eigenvalues_in_bulk].mean()

        new_corr: np.ndarray = eigenvectors @ np.diag(
            new_eigenvalues) @ np.linalg.inv(eigenvectors)

        # setting the diagonal values to be exactly 1.0
        diagonal_indices = range(new_corr.shape[0])
        new_corr[diagonal_indices, diagonal_indices] = 1.0
        return new_corr

    def laloux_pearson(self, delta: float = 10 ** -9, **kwargs) -> np.ndarray:
        """Fits a Pearson correlation matrix estimator to the dataset, in
        addition to applying Laloux et al.'s technique to ensure the
        matrix is transformed to be positive definite and denoised.
        Any negative eigenvalues are replaced with a positive delta value. The
        bulk of the matrix is then calculated, with any eigenvalues below the
        bulk upper bound being replaced with their mean.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        laloux_pearson_corr np.ndarray
            A Pearson correlation matrix estimator, transformed to be positive
            definite and denoised.
        """
        return self._laloux_corr(delta, 'pearson')

    def laloux_spearman(self, delta: float = 10 ** -9, **kwargs) -> np.ndarray:
        """Fits a Spearman rank correlation matrix estimator to the dataset, in
        addition to applying Laloux et al.'s technique to ensure the
        matrix is transformed to be positive definite and denoised.
        Any negative eigenvalues are replaced with a positive delta value. The
        bulk of the matrix is then calculated, with any eigenvalues below the
        bulk upper bound being replaced with their mean.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        laloux_spearman_corr np.ndarray
            A Spearman rank correlation matrix estimator, transformed to be
            positive definite and denoised.
        """
        return self._laloux_corr(delta, 'spearman')

    def laloux_kendall(self, delta: float = 10 ** -9, **kwargs) -> np.ndarray:
        """Fits a Kendall rank correlation matrix estimator to the dataset, in
        addition to applying Laloux et al.'s technique to ensure the
        matrix is transformed to be positive definite and denoised.
        Any negative eigenvalues are replaced with a positive delta value. The
        bulk of the matrix is then calculated, with any eigenvalues below the
        bulk upper bound being replaced with their mean.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        laloux_kendall_corr np.ndarray
            A Kendall rank correlation matrix estimator, transformed to be
            positive definite and denoised.
        """
        return self._laloux_corr(delta, 'kendall')

    def laloux_pp_kendall(self, delta: float = 10 ** -9, **kwargs) \
            -> np.ndarray:
        """Fits a Pseudo-Pearson Kendall (PP-Kendall) correlation matrix
        estimator to the dataset, in addition to applying Laloux et al.'s
        technique to ensure the matrix is transformed to be positive definite
        and denoised. Any negative eigenvalues are replaced with a positive
        delta value. The bulk of the matrix is then calculated, with any
        eigenvalues below the bulk upper bound being replaced with their mean.

        Parameters
        ----------
        delta : float
            The value to replace any negative eigenvalues with.

        See also:
        ---------
        The benefit of using Random Matrix Theory to fit high-dimensional
        t-copulas by Jiali Xu and Loïc Brin.

        Returns
        -------
        laloux_pp_kendall_corr np.ndarray
            A PP-Kendall rank correlation matrix estimator, transformed to be
            positive definite and denoised.
        """
        return self._laloux_corr(delta, 'pp_kendall')

    def corr(self, method: str, **kwargs) -> np.ndarray:
        """Calculates correlation matrix estimators using a specified method.
        The user can use this method or the individual correlation methods
        directly (i.e. CorrelationMatrix.pearson), to produce the same results.

        Parameters
        -----------
        method: str
            The name of the method to use to estimate the correlation matrix.
            Can be 'pearson', 'spearman', 'kendall', 'pp_kendall',
            'rm_pearson', 'rm_spearman', 'rm_kendall', 'rm_pp_kendall',
            'laloux_pearson', 'laloux_spearman', 'laloux_kendall',
            'laloux_pp_kendall'. See individual method implementations for
            specifics.
        kwargs:
            See below

        Keyword arguments
        ------------------
        raise_error: bool
            For 'pearson', 'spearman', 'kendall' and 'pp_kendall' methods only.
            True to raise an error if the resultant matrix is not a valid
            correlation matrix. I.e. if the result is not square, symmetric,
            positive semi-definite and contains 1's in the diagonal.
            Default is False.
        renormalise: bool
            For 'rm_pearson', 'rm_spearman', 'rm_kendall', 'rm_pp_kendall'
            methods only.
            True to set the diagonal elements of the resultant matrix to be
            exactly 1.0.
            Default is True.
        delta : float
            For 'rm_pearson', 'rm_spearman', 'rm_kendall', 'rm_pp_kendall',
            'laloux_pearson', 'laloux_spearman', 'laloux_kendall',
            'laloux_pp_kendall' methods only.
            The value to replace any negative eigenvalues with.
            Default is 10 ** -9

        Returns
        -------
        corr: np.ndarray
            A correlation matrix estimator.
        """
        # checking arguments
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        method: str = method.lower().strip()\
            .replace('-', '_').replace(' ', '_')
        if method not in CorrelationMatrix.IMPLEMENTED:
            raise ValueError(
                f"{method} is not a valid argument. Specify from "
                f"{CorrelationMatrix.IMPLEMENTED}")

        return eval(f"self.{method}(**kwargs)")

    @staticmethod
    def _is_2d(arr: np.ndarray) -> bool:
        """checks if an array is 2 dimensional.

        Parameters
        ----------
        arr : np.ndarray
            The array to check.

        Returns
        -------
        is_2d: bool
            True if 2d, False otherwise.
        """
        shape: tuple = arr.shape
        if len(shape) != 2:
            return False
        return True

    @staticmethod
    def _is_square(arr: np.ndarray) -> bool:
        """checks if an array is square.

        Parameters
        ----------
        arr : np.ndarray
            The array to check.

        Returns
        -------
        is_square: bool
            True if square, False otherwise.
        """
        if not CorrelationMatrix._is_2d(arr):
            return False

        shape: tuple = arr.shape
        if shape[0] != shape[1]:
            return False
        return True

    @staticmethod
    def _check_matrix(name: str, definiteness: Union[str, None], ones: bool,
                      A: np.ndarray, raise_error: bool) -> bool:
        """Performs checks on a given matrix to determine if square and
        symmetric.

        Parameters
        ----------
        name: str
            The name of the matrix.
        definiteness: str
            'pd' to check if matrix is positive definite.
            'psd' to check if matrix is positive semi-definite.
             None to do no definitness checks.
        ones: bool
            True to check if matrix diagonal elements are all ones.
        A: np.ndarray
            The matrix to check.
        raise_error: bool
            True to raise an error if a test is failed.
            False to log warnings instead.

        Returns
        -------
        passes_checks: bool
            True if A passes all checks. False otherwise.
        """
        passes_checks: bool = True

        # checking numpy array passed
        if not isinstance(A, np.ndarray):
            raise TypeError(f"{name} matrix must be a numpy array.")

        # checking matrix is square
        if not CorrelationMatrix._is_square(A):
            square_msg: str = f"{name} matrix is not 2d and square"
            if raise_error:
                raise ValueError(square_msg)
            else:
                warnings.warn(square_msg)
            passes_checks = False

        # checking matrix is symmetric
        if not np.allclose(A, A.T):
            symmetric_msg: str = f"{name} matrix is not symmetric"
            if raise_error:
                raise ValueError(symmetric_msg)
            else:
                warnings.warn(symmetric_msg)
            passes_checks = False

        # checking matrix definiteness
        if definiteness is not None:
            psd: bool = definiteness == 'psd'
            eigenvalues: np.ndarray = np.linalg.eigvals(A)
            if (psd and not np.all(eigenvalues >= 0)) or (
            not np.all(eigenvalues > 0)):
                definiteness_msg: str = "semi-" if psd else ""
                definiteness_msg = f"{name} matrix is not positive " \
                                   f"{definiteness_msg}definite"
                if raise_error:
                    raise ValueError(definiteness_msg)
                else:
                    warnings.warn(definiteness_msg)
                passes_checks = False

        # checking matrix has all ones in diagonal
        if ones and not np.all(A.diagonal() == 1.0):
            ones_msg: str = f"{name} matrix does not have all ones in diagonal"
            if raise_error:
                raise ValueError(ones_msg)
            else:
                warnings.warn(ones_msg)
            passes_checks = False

        return passes_checks

    @staticmethod
    def check_correlation_matrix(corr: np.ndarray, raise_error: bool = True,
                                 **kwargs) -> bool:
        """Performs checks on a given numpy array to see if it is a valid
        correlation matrix. I.e. checks matrix is square, symmetric, positive
        semi-definite and has all elements equal to 1.

        Parameters
        ----------
        corr : np.ndarray
            the matrix to check
        raise_error : bool
            True to raise an error if corr is not a valid correlation matrix.
            Default is True

        Returns
        -------
        passes_checks: bool
            True if the array is a valid correlation matrix, False otherwise.
        """
        return CorrelationMatrix._check_matrix('Correlation', 'psd', True,
                                               corr, raise_error)

    @staticmethod
    def cov_from_corr(corr: np.ndarray, std: np.ndarray,
                      raise_error: bool = True, **kwargs) -> np.ndarray:
        """Calculates the covariance matrix from a given covariance matrix and
        standard deviations.

        Parameters
        ----------
        corr: np.ndarray
            A correlation matrix.
        std: np.ndarray
            A vector / numpy array containing the standard deviations of each
            variable. Note, the indexing of the correlation matrix and standard
            deviation arrays must match.
        raise_error: bool
            True to raise an error if corr is not a valid correlation matrix.
            Default is True

        Returns
        -------
        cov: np.ndarray
            The covariance matrix corresponding to corr and std.
        """
        # checking arguments
        CorrelationMatrix.check_correlation_matrix(corr, raise_error)
        if not isinstance(std, np.ndarray):
            raise TypeError("std is not a np.ndarray")
        elif std.size != corr.shape[0]:
            raise ValueError(
                "std length does not match the dimensions of corr.")

        std_diag: np.ndarray = np.diag(std.flatten())
        return std_diag @ corr @ std_diag

    def cov(self, method: str, **kwargs) -> np.ndarray:
        """Calculates covariance matrix estimators. First a correlation matrix
        estimator is calculated using a specified method. A covariance matrix
        estimator is then calculated using this and sample estimators for the
        standard deivation.

        The user can use this method, or calculate a correlation matrix
        estimator using one onf the implemented methods, before transforming
        this into a covariance matrix estimator using the cov_from_corr method.

        Parameters
        -----------
        method: str
            The name of the method to use to estimate the correlation matrix.
            Can be 'pearson', 'spearman', 'kendall', 'pp_kendall',
            'rm_pearson', 'rm_spearman', 'rm_kendall', 'rm_pp_kendall',
            'laloux_pearson', 'laloux_spearman', 'laloux_kendall',
            'laloux_pp_kendall'. See individual method implementations for
            specifics.
        kwargs:
            See below

        Keyword arguments
        ------------------
        raise_error: bool
            For 'pearson', 'spearman', 'kendall' and 'pp_kendall' methods only.
            True to raise an error if the resultant matrix is not a valid
            correlation matrix. I.e. if the result is not square, symmetric,
            positive semi-definite and contains 1's in the diagonal.
            Default is False.
        renormalise: bool
            For 'rm_pearson', 'rm_spearman', 'rm_kendall', 'rm_pp_kendall'
            methods only.
            True to set the diagonal elements of the correlation matrix to be
            exactly 1.0.
            Default is True.
        delta : float
            For 'rm_pearson', 'rm_spearman', 'rm_kendall', 'rm_pp_kendall',
            'laloux_pearson', 'laloux_spearman', 'laloux_kendall',
            'laloux_pp_kendall' methods only.
            The value to replace any negative eigenvalues of the correlation
            with during the rm step.
            Default is 10 ** -9

        Returns
        -------
        cov: np.ndarray
            A covariance matrix estimator.
        """
        corr: np.ndarray = self.corr(method, **kwargs)
        std: np.ndarray = self._data.std(axis=0).to_numpy()
        return self.cov_from_corr(corr, std, **kwargs)

    @staticmethod
    def check_covariance_matrix(cov: np.ndarray, raise_error: bool = True,
                                **kwargs):
        """Performs checks on a given numpy array to see if it is a valid
        covariance matrix. I.e. checks matrix is square, symmetric and positive
        definite.

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
        return CorrelationMatrix._check_matrix('Covariance', 'pd', False, cov,
                                               raise_error)
