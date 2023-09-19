# Contains code for creating numerical distributions
import numpy as np
import pandas as pd
from typing import Callable

__all__ = ['NumericalWrappers']


class NumericalWrappers:
    """Used when creating numerical distributions."""
    @staticmethod
    def numerical_pdf(x: np.ndarray, pdf_: Callable) -> np.ndarray:
        """A function used to ensure the outputs of a numerical/empirical pdf
        function are valid

        Parameters
        ----------
        x : np.ndarray
            The values to be parsed into a numerical pdf distribution function.
        pdf_: Callable
            A function interpolating between the data and its empirical pdf
            values.

        Returns
        -------
        pdf_values : np.ndarray
            An array of valid pdf values.
        """
        raw_pdf_values: np.ndarray = pdf_(x)
        pdf_values: np.ndarray = np.where(
            ~pd.isna(raw_pdf_values), raw_pdf_values, 0.0
        )
        return np.clip(pdf_values, 0.0, np.inf)

    @staticmethod
    def numerical_cdf(x, cdf_: Callable, xmin, xmax) -> np.ndarray:
        """A function used to ensure the outputs of a numerical/empirical cdf
        function are valid

        Parameters
        ----------
        x : np.ndarray
            The values to be parsed into a numerical cdf distribution function.
        cdf_: Callable
            A function interpolating between the data and its empirical cdf
            values.
        xmin:
            The minimum value of our dataset
        xmax:
            The maximum value of our dataset

        Returns
        -------
        cdf_values : np.ndarray
            An array of valid cdf values.
        """
        raw_cdf_values: np.ndarray = cdf_(x)
        cdf_values: np.ndarray = np.where(x >= xmin, raw_cdf_values, 0.0)
        cdf_values = np.where(x <= xmax, cdf_values, 1.0)
        return np.clip(cdf_values, 0.0, 1.0)

    @staticmethod
    def numerical_ppf(x, ppf_: Callable, xmin, xmax, F_xmin: float,
                      F_xmax: float) -> np.ndarray:
        """A function used to ensure the outputs of a numerical/empirical ppf
        function are valid

        Parameters
        ----------
        x : np.ndarray
            The values to be parsed into a numerical cdf distribution function.
        ppf_: Callable
            A function interpolating between the data and its empirical ppf
            values.
        xmin:
            The minimum value of our dataset
        xmax:
            The maximum value of our dataset
        F_xmin: float
            The value of the cdf function evaluated at xmin
        F_xmax: float
            The value of the cdf function evaluated at xmax

        Returns
        --------
        ppf_values : np.ndarray
            An array of valid ppf values.
        """
        raw_ppf_values: np.ndarray = ppf_(x)
        ppf_values: np.ndarray = np.where(x >= F_xmin, raw_ppf_values, xmin)
        ppf_values = np.where(x <= F_xmax, ppf_values, xmax)
        return ppf_values

    @staticmethod
    def numerical_support(xmin, xmax) -> tuple:
        """A function which returns the min and max values of the support of a
        numerical distribution.

        Parameters
        ----------
        xmin:
            The minimum value of our dataset
        xmax:
            The maximum value of our dataset

        Returns
        -------
        support: tuple
            xmin, xmax
        """
        return xmin, xmax
