# Standard parametrization of the Generalized Inverse Gaussian distribution
import numpy as np
# import scipy.integrate
import scipy.optimize
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import _ShapeInfo
import scipy.special

from sklarpy.misc import kv


__all__ = ['_gig']

# class GIG:
#     @staticmethod
#     def _pdf_single(xi: float, chi: float, psi: float, lambda_: float) -> float:
#         if (xi <= 0.0):
#             return np.nan
#         elif (lambda_ < 0) and ((chi <= 0) or (psi < 0)):
#             return np.nan
#         elif (lambda_ == 0) and ((chi <= 0) or (psi <= 0)):
#             return np.nan
#         elif (lambda_ > 0) and ((chi < 0) or (psi <= 0)):
#             return np.nan
#
#         k: float = scipy.special.kv(lambda_, (chi * psi) ** 0.5)
#         return ((psi / chi) ** (lambda_ / 2)) * (xi**(lambda_-1)) * np.exp(-0.5*((chi/xi) + (psi * xi))) / (2 * k)
#
#     @staticmethod
#     def _cdf_single(xi: float, chi: float, psi: float, lambda_: float) -> float:
#         return scipy.integrate.quad(GIG._pdf_single, a=0, b=xi, args=(chi, psi, lambda_))[0]
#
#     @staticmethod
#     def _ppf_single(qi: float, chi: float, psi: float, lambda_: float) -> float:
#         left, right = GIG.support(chi, psi, lambda_)
#
#         if qi == 1.0:
#             return right
#         elif qi == 0.0:
#             return left
#
#         if np.isinf(right):
#             small_int = 2.
#             right = small_int
#             while GIG._cdf_single(right, chi, psi, lambda_) - qi < 0.:
#                 right += small_int
#             # right is now such that cdf(right) >= q
#
#         to_solve = (lambda xi: abs(GIG._cdf_single(xi, chi, psi, lambda_) - qi))
#         res = scipy.optimize.differential_evolution(to_solve, bounds=[(0, right)])
#         return res['x'][0]
#
#     @staticmethod
#     def pdf(x: np.ndarray, chi: float, psi: float, lambda_: float) -> np.ndarray:
#         return np.vectorize(GIG._pdf_single, otypes=[float])(x, chi, psi, lambda_)
#
#     @staticmethod
#     def cdf(x: np.ndarray, chi: float, psi: float, lambda_: float) -> np.ndarray:
#         return np.vectorize(GIG._cdf_single, otypes=[float])(x, chi, psi, lambda_)
#
#     @staticmethod
#     def ppf(q: np.ndarray, chi: float, psi: float, lambda_: float) -> np.ndarray:
#         return np.vectorize(GIG._ppf_single, otypes=[float])(q, chi, psi, lambda_)
#
#     @staticmethod
#     def support(chi: float, psi: float, lambda_: float) -> tuple:
#         return 0, np.inf
#
#     @staticmethod
#     def fit(data: np.ndarray) -> tuple:
#         pass


# class gig_gen(rv_continuous):
#     def _argcheck(self, chi: float, psi: float, lambda_: float) -> bool:
#         passes: bool = True
#
#         if (lambda_ < 0) and ((chi <= 0) or (psi < 0)):
#             passes = False
#         elif (lambda_ == 0) and ((chi <= 0) or (psi <= 0)):
#             passes = False
#         elif (lambda_ > 0) and ((chi < 0) or (psi <= 0)):
#             passes = False
#
#         return passes
#
#     def _shape_info(self) -> list:
#         ichi = _ShapeInfo("chi", False, (0, np.inf), (True, False))
#         ipsi = _ShapeInfo("psi", False, (0, np.inf), (True, False))
#         ilambda = _ShapeInfo("lambda_", False, (-np.inf, np.inf), (False, False))
#         return [ichi, ipsi, ilambda]
#
#     def __pdf_single(self, xi: float, chi: float, psi: float, lambda_: float) -> float:
#         k: float = scipy.special.kv(lambda_, (chi * psi) ** 0.5)
#         return ((psi / chi) ** (lambda_ / 2)) * (xi ** (lambda_ - 1)) * np.exp(-0.5 * ((chi / xi) + (psi * xi))) / (2 * k)
#
#     def _pdf(self, x: np.ndarray, chi: float, psi: float, lambda_: float) -> np.ndarray:
#         return np.vectorize(self.__pdf_single, otypes=[float])(x, chi, psi, lambda_)
#
#     # def __cdf_single(self, xi: float, chi: float, psi: float, lambda_: float) -> float:
#     #     return scipy.integrate.quad(self.__pdf_single, a=0, b=xi, args=(chi, psi, lambda_))[0]
#
#     def fit(self, data: np.ndarray) -> tuple:
#         def neg_loglikelihood(params: np.ndarray):
#             chi, psi, lambda_ = params
#             pdf_vals: np.ndarray = self.pdf(data, chi, psi, lambda_)
#             return -np.sum(np.log(pdf_vals))
#
#         res = scipy.optimize.fmin(neg_loglikelihood, x0=(1.0, 1.0, 1.0), disp=False)
#         return tuple(res)


class gig_gen(rv_continuous):
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

    def __pdf_single(self, xi: float, lamb: float, chi: float, psi: float) -> float:
        # k: float = scipy.special.kv(lamb, (chi * psi) ** 0.5)
        k: float = kv.kv(lamb, (chi * psi) ** 0.5)
        return ((psi / chi) ** (lamb / 2)) * (xi ** (lamb - 1)) * np.exp(-0.5 * ((chi / xi) + (psi * xi))) / (2 * k)

    def _pdf(self, x: np.ndarray, lamb: float, chi: float, psi: float) -> np.ndarray:
        return np.vectorize(self.__pdf_single, otypes=[float])(x, lamb, chi, psi)

    # def __cdf_single(self, xi: float, chi: float, psi: float, lambda_: float) -> float:
    #     return scipy.integrate.quad(self.__pdf_single, a=0, b=xi, args=(chi, psi, lambda_))[0]

    def fit(self, data: np.ndarray) -> tuple:
        def neg_loglikelihood(params: np.ndarray):
            pdf_vals: np.ndarray = self.pdf(data, *params)
            return -np.sum(np.log(pdf_vals))

        res = scipy.optimize.fmin(neg_loglikelihood, x0=(1.0, 1.0, 1.0), disp=False)
        return tuple(res)


_gig = gig_gen(a=0.0, name="GIG")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # chi_ = 5
    # psi_ = 8
    # lam = -6
    chi_ = 1.7
    psi_ = 4.5
    lam = -0.5
    arr = np.linspace(0., 0.999, 50, dtype=float)
    # vals = GIG.ppf(arr, chi_, psi_, lam)
    # print(vals)
    vals2 = _gig.ppf(arr, lam, chi_, psi_)
    print(vals2)
    # breakpoint()
    rvs = _gig.rvs(lam, chi_, psi_, size=(1000,))

    fparams = _gig.fit(rvs)
    print(fparams)

    # breakpoint()
    # plt.plot(arr, vals)
    # plt.show()