# Standard parametrization of the Generalized Inverse Gaussian distribution
import numpy as np
import scipy.optimize
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import _ShapeInfo
import scipy.special

from sklarpy.misc import kv

__all__ = ['_gig']


class gig_gen(rv_continuous):
    """The univariate Generalized Inverse Gaussian (GIG) distribution,
    with the parametrization specified by McNeil et al."""
    def _argcheck(self, lamb: float, chi: float, psi: float) -> bool:
        passes: bool = True

        if (lamb < 0) and ((chi <= 0) or (psi < 0)):
            passes = False
        elif (lamb == 0) and ((chi <= 0) or (psi <= 0)):
            passes = False
        elif (lamb > 0) and ((chi < 0) or (psi <= 0)):
            passes = False

        return passes

    def _shape_info(self) -> list:
        ilamb = _ShapeInfo("lamb", False, (-np.inf, np.inf), (False, False))
        ichi = _ShapeInfo("chi", False, (0, np.inf), (True, False))
        ipsi = _ShapeInfo("psi", False, (0, np.inf), (True, False))
        return [ilamb, ichi, ipsi]

    def __pdf_single(self, xi: float, lamb: float, chi: float, psi: float
                     ) -> float:
        # k: float = scipy.special.kv(lamb, (chi * psi) ** 0.5)
        k: float = kv.kv(lamb, (chi * psi) ** 0.5)
        return float(
            ((psi / chi) ** (lamb / 2))
            * (xi ** (lamb - 1))
            * np.exp(-0.5 * ((chi / xi) + (psi * xi)))
            / (2 * k)
        )

    def _pdf(self, x: np.ndarray, lamb: float, chi: float, psi: float
             ) -> np.ndarray:
        return np.vectorize(self.__pdf_single,
                            otypes=[float])(x, lamb, chi, psi)
    def fit(self, data: np.ndarray) -> tuple:
        def neg_loglikelihood(params: np.ndarray):
            pdf_vals: np.ndarray = self.pdf(data, *params)
            return -np.sum(np.log(pdf_vals))

        res = scipy.optimize.fmin(neg_loglikelihood, x0=(1.0, 1.0, 1.0),
                                  disp=False)
        return tuple(res)


_gig = gig_gen(a=0.0, name="GIG")
