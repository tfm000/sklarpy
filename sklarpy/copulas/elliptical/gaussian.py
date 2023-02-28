import numpy as np
import scipy.stats

from sklarpy.copulas.elliptical._elliptical import Elliptical
from sklarpy.copulas.elliptical._elliptical_params import GaussianCopulaParams
from sklarpy._utils import dataframe_or_array


__all__ = ['GaussianCopula']


class GaussianCopula(Elliptical):
    _OBJ_NAME = "GaussianCopula"
    _MAX_NUM_VARIABLES = np.inf
    _PARAMS_OBJ = GaussianCopulaParams

    def fit(self, method: str = 'laloux_pp_kendall', params: GaussianCopulaParams = None,
            corr: dataframe_or_array = None, **kwargs):
        if params is not None:
            self._params_check(params)
            corr = params.corr

        Elliptical._fit_corr(self, method, corr, **kwargs)

        self._params = {'corr': self._corr}
        self._fitted = True
        return self

    def _rvs(self, size: int) -> np.ndarray:
        X: np.ndarray = self._generate_multivariate_std_normal_rvs(size)
        return scipy.stats.norm.cdf(X)

    def _pdf(self, u: np.ndarray, **kwargs) -> np.ndarray:
        x: np.ndarray = scipy.stats.norm.ppf(u)
        return scipy.stats.multivariate_normal(cov=self.corr).pdf(x) / (scipy.stats.norm.pdf(x).prod(axis=1))

    def _cdf(self, u: np.ndarray, **kwargs) -> np.ndarray:
        return scipy.stats.multivariate_normal(cov=self.corr).cdf(scipy.stats.norm.ppf(u))


if __name__ == '__main__':
    from sklarpy.univariate import t, gamma
    from sklarpy.copulas import MarginalFitter
    import matplotlib.pyplot as plt
    import pandas as pd
    p = 0.5
    my_corr = np.array([[1, p], [p, 1]])
    # print(my_corr)
    # my_stds = np.array([2, 2])
    # my_cov = np.diag(my_stds)@my_corr@np.diag(my_stds)
    num_samples = 10000
    # my_normal_sample = multivariate_normal(cov=my_cov).rvs(size=num_samples)
    my_normal_sample = np.random.normal(0, 1, size=(num_samples, 2))@np.linalg.cholesky(my_corr).T
    my_cop_sample = scipy.stats.norm.cdf(my_normal_sample)
    # print(my_cop_sample)
    # plt.figure(figsize=(8,8))
    # plt.scatter(my_cop_sample[:, 0], my_cop_sample[:, 1], alpha=0.2)
    # plt.show()
    my_realistic_sample = my_cop_sample.copy()
    my_realistic_sample[:, 0] = t.ppf(my_realistic_sample[:, 0], (6,))
    my_realistic_sample[:, 1] = gamma.ppf(my_realistic_sample[:, 1], (4,3))
    # plt.scatter(my_realistic_sample[:, 0], my_realistic_sample[:, 1])
    # plt.show()

    # testing gauss
    # my_gauss: GaussianCopula = GaussianCopula(my_cop_sample)
    # print(my_gauss)
    # my_gauss.fit()
    # print(my_gauss.corr)
    # print(my_gauss.logpdf())

    # testing process
    my_realistic_sample = pd.DataFrame(my_realistic_sample, columns=['t', 'gamma'])
    mfitter: MarginalFitter = MarginalFitter(my_realistic_sample)
    mfitter.fit()

    my_gauss: GaussianCopula = GaussianCopula(mfitter)
    my_gauss.fit()
    print(my_gauss.pdf(np.array([[0.4, 0.5]])))
    my_gauss.pdf_plot()
    my_gauss.cdf_plot()
    my_gauss.marginal_pairplot()
    # mfitter.marginal_ppfs()


    # u = mfitter.marginal_cdfs(match_datatype=False)
    # q = mfitter.marginal_ppfs(u, match_datatype=False)
    # print(q.shape)
    # pdf_vals = multivariate_normal(cov=my_gauss.cov).pdf(q) * norm.pdf(q).prod(axis=1)
    import matplotlib.pyplot as plt

    # S = norm.ppf(u).T
    # top = S.T @ np.linalg.inv(my_gauss.corr) @ S /2
    # bottom = S.T @ np.eye(S.shape[0]) @ S / 2
    # pdf_vals = (np.exp(-top)/np.exp(-bottom)) * (np.linalg.norm(my_gauss.corr)**-0.5)
    # breakpoint()

    # fig = plt.figure(f"pdf Plot")
    # ax = plt.axes(projection='3d')
    # ax.plot_trisurf(u[:, 0], u[:, 1], pdf_vals)
    # plt.show()


    # print(mfitter.summary)

    #
    # ## generating rvs from copula
    # random_u = my_gauss.rvs(1000)
    # random_x = mfitter.marginal_ppfs(random_u)
    # # random_x.plot.scatter(*random_x.columns)
    # # plt.scatter(random_x[:, 0], random_x[:, 1])
    # # plt.show()
    # # breakpoint()
    #
    # my_gauss.pdf_plot()
    # my_gauss.cdf_plot()
