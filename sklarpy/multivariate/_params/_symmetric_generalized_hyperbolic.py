# Contains code for holding Symmetric Generalized Hyperbolic parameters
import numpy as np

from sklarpy.multivariate._params._normal import MvtNormalParams
from sklarpy.multivariate._distributions._symmetric_generalized_hyperbolic \
    import multivariate_sym_gen_hyperbolic_gen

__all__ = ['MvtSGHParams']


class MvtSGHParams(MvtNormalParams):
    """Contains the fitted parameters of a Multivariate Symmetric Generalized
    Hyperbolic distribution."""
    _DIST_GENERATOR = multivariate_sym_gen_hyperbolic_gen

    @property
    def lamb(self) -> float:
        """The lambda/lamb parameter of the distribution and the underlying
        mixing variable W ~ GIG(lamb, chi, psi).

        Returns
        -------
        lamb: float
            The lambda/lamb parameter of the distribution.
        """
        return self.to_dict['lamb']

    @property
    def chi(self) -> float:
        """The chi parameter of the distribution and the underlying
        mixing variable W ~ GIG(lamb, chi, psi).

        Returns
        -------
        chi: float
            The chi parameter of the distribution.
        """
        return self.to_dict['chi']

    @property
    def psi(self) -> float:
        """The psi parameter of the distribution and the underlying
        mixing variable W ~ GIG(lamb, chi, psi).

        Returns
        -------
        psi: float
            The psi parameter of the distribution.
        """
        return self.to_dict['psi']

    @property
    def _gig_params_tuple(self) -> tuple:
        """The parameter values of the W ~ GIG distribution.

        Returns
        -------
        gig_params: tuple
            lamb, chi, psi
        """
        return self.lamb, self.chi, self.psi

    @property
    def exp_w(self) -> float:
        """The expected value, E[W], of the rv W ~ GIG(lamb, chi, psi).

        Returns
        -------
        exp_w: float
            E[W] of the rv W ~ GIG(lamb, chi, psi).
        """
        return self._DIST_GENERATOR._UNIVAR._exp_w(self._gig_params_tuple)

    @property
    def var_w(self) -> float:
        """The variance, var[W], of the rv W ~ GIG(lamb, chi, psi).

        Returns
        -------
        var_w: float
            var[W] of the rv W ~ GIG(lamb, chi, psi).
        """
        return self._DIST_GENERATOR._UNIVAR._var_w(self._gig_params_tuple)

    @property
    def cov(self) -> np.ndarray:
        return self.exp_w * self.shape
