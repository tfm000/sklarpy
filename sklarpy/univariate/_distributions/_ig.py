# Standard parametrization of the Inverse Gamma distribution
import numpy as np
import scipy.special
import scipy.optimize
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import _ShapeInfo


__all__ = ['_ig']


class ig_gen(rv_continuous):
    """The univariate Inverse Gaussian (IG) distribution,
    with the parametrization specified by McNeil et al."""
    def _argcheck(self, alpha: float, beta: float) -> bool:
        if (alpha > 0) and (beta > 0):
            return True
        return False

    def _shape_info(self) -> list:
        ialpha = _ShapeInfo("alpha", False, (0, np.inf), (False, False))
        ibeta = _ShapeInfo("beta", False, (0, np.inf), (False, False))
        return [ialpha, ibeta]

    def __pdf_single(self, xi: float, alpha: float, beta: float) -> float:
        return float(
            (beta ** alpha)
            * (xi ** - (alpha + 1))
            * np.exp(-beta/xi)
            / scipy.special.gamma(alpha)
        )

    def _pdf(self, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return np.vectorize(self.__pdf_single, otypes=[float])(x, alpha, beta)

    def fit(self, data: np.ndarray) -> tuple:
        def neg_loglikelihood(params: np.ndarray):
            alpha, beta = params
            pdf_vals: np.ndarray = self.pdf(data, alpha, beta)
            return -np.sum(np.log(pdf_vals))

        res = scipy.optimize.fmin(neg_loglikelihood, x0=(1.0, 1.0), disp=False)
        return tuple(res)


_ig = ig_gen(a=0.0, name="Ig")
