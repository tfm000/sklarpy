import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator, interpn, griddata, RegularGridInterpolator, RBFInterpolator
from sklarpy.univariate import gamma
from collections import deque
from scipy.misc import derivative
import matplotlib.pyplot as plt
from numdifftools import Gradient
from findiff import FinDiff
from scipy.stats import multivariate_normal

gamma_dist = gamma.fit(params=(4, 5,))
univar_data = gamma_dist.rvs((100, 1))

# emp_cdf_vals = deque()
# n = univar_data.size
# xmin, xmax = univar_data.min(), univar_data.max()
# for x in univar_data:
#     p = (univar_data <= x).sum() / n
#     emp_cdf_vals.append(p)
# emp_cdf_vals = np.asarray(emp_cdf_vals)
#
# emp_cdf_func_ = interp1d(univar_data.flatten(), emp_cdf_vals, bounds_error=False)
# def emp_cdf_func(x):
#     x = np.asarray(x)
#     raw_vals = emp_cdf_func_(x)
#     cdf_vals = np.where(x >= xmin, raw_vals, 0)
#     cdf_vals = np.where(x <= xmax, cdf_vals, 1)
#     return cdf_vals
#
# pdf_vals = np.vectorize(derivative)(emp_cdf_func, univar_data.flatten())
# emp_pdf_func_ = interp1d(univar_data.flatten(), pdf_vals, bounds_error=False)
# def emp_pdf_func(x):
#     x = np.asarray(x)
#     raw_vals = emp_pdf_func_(x)
#     pdf_vals = np.where(x >= xmin, raw_vals, 0)
#     pdf_vals = np.where(x <= xmax, pdf_vals, 0)
#     return pdf_vals
# emp_pdf_vals=emp_pdf_func(univar_data)


# xrange = np.linspace(xmin-1, xmax+2, 1000)
# real_pdf_vals = gamma_dist.pdf(xrange)
# real_cdf_vals = gamma_dist.cdf(xrange)
# emp_pdf_vals2 = emp_pdf_func(xrange)
# emp_cdf_vals2 = emp_cdf_func(xrange)
#
# plt.figure('pdf')
# plt.plot(xrange, real_pdf_vals, label='real')
# plt.plot(xrange, emp_pdf_vals2, label='emp')
# plt.legend()
#
# plt.figure('cdf')
# plt.plot(xrange, real_cdf_vals, label='real')
# plt.plot(xrange, emp_cdf_vals2, label='emp')
# plt.legend()
# plt.show()

multi_mean = [0, 0]
multi_cov = [[1, 0.5], [0.5, 1]]
multivar_data = np.random.multivariate_normal(multi_mean, multi_cov, 10000)
# cdf_vals = deque()
# n = multivar_data.size
# for row in multivar_data:
#     p: float = np.all(multivar_data <= row, axis=1).sum() / n
#     cdf_vals.append(p)
# cdf_vals: np.ndarray = np.asarray(cdf_vals)
# cdf_ = LinearNDInterpolator(multivar_data, cdf_vals)
# xmins, xmaxs = multivar_data.min(axis=0), multivar_data.max(axis=0)


xmins, xmaxs = multivar_data.min(axis=0), multivar_data.max(axis=0)
nvars = multivar_data.shape[1]
num_additional_points: int = 100
delta: float = 0.5
additional_points: np.ndarray = np.random.uniform(xmins - delta, xmaxs + delta, (num_additional_points, nvars))
empirical_range: np.ndarray = np.concatenate(
    [multivar_data, additional_points, (xmins - delta).reshape((1, nvars)), (xmaxs + delta).reshape((1, nvars))],
    axis=0)

eps = 10**-1
# breakpoint()
non_zero_points = deque()
pdf_vals = deque()
cdf_vals = deque()
n = multivar_data.shape[0]
for i, row in enumerate(empirical_range):
    p: float = np.all(multivar_data <= row, axis=1).sum() / n
    cdf_vals.append(p)

    pdf_val = np.all((multivar_data >= row - eps) & (multivar_data <= row + eps), axis=1).sum() / n
    pdf_vals.append(pdf_val)
    if pdf_val != 0: #p == 0 or 1-p == 0:
        non_zero_points.append(i)

cdf_vals: np.ndarray = np.asarray(cdf_vals)
cdf_ = LinearNDInterpolator(empirical_range, cdf_vals)

pdf_vals: np.ndarray = np.asarray(pdf_vals)
# pdf_ = LinearNDInterpolator(empirical_range[non_zero_points, :], pdf_vals[non_zero_points])
pdf_ = RBFInterpolator(empirical_range[non_zero_points, :], pdf_vals[non_zero_points], neighbors=100, smoothing=1)


# cdf_ = interpn(empirical_range, cdf_vals.reshape((cdf_vals.size, 1)), empirical_range, bounds_error=False)
# cdf_ = griddata(empirical_range, cdf_vals, empirical_range)
# cdf_ = RegularGridInterpolator(empirical_range, cdf_vals, bounds_error=False)

def pdf(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape((1, x.size))
    raw_vals = pdf_(x)
    vals = np.where(~np.isnan(raw_vals), raw_vals, 0.0)
    return np.clip(vals, 0.0, None)

def cdf(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape((1, x.size))
    # print(x.shape)
    for var in range(x.shape[1]):
        x[:, var] = np.where(x[:, var] >= xmins[var], x[:, var], xmins[var])
        x[:, var] = np.where(x[:, var] <= xmaxs[var], x[:, var], xmaxs[var])
    # print(x)
    # vals = cdf_(x)
    # if vals.size == 1:
    #     #needed for differientating
    #     return float(vals)
    return cdf_(x)
    # raw_vals = cdf_(x)
    # cdf_vals = np.where(np.all(x>=xmins, axis=1), raw_vals, 0.0)
    # cdf_vals = np.where(np.all(x<=xmaxs, axis=1), cdf_vals, 1.0)

    # TDO: doesnt handle cases like [a, inf] etc...
    
    # have it so that when one var is out of its range, its replaced by its min or max value. should fix the issue.
    # return cdf_vals


# breakpoint()
# cdf([100, 100])
# cdf([100, 1])
# cdf(multivar_data)
# plt.plot(empirical_range[:, 0], empirical_range[:, 1])
# fig = plt.figure(f"cdf Plot")
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(empirical_range[:, 0], empirical_range[:, 1], cdf(empirical_range))
# plt.show()

# derivative(cdf, row, n=nvars)
# pdf_vals: np.ndarray = np.vectorize(derivative)(cdf, empirical_range, 1, nvars)
# pdf_ = Gradient(cdf)
# # np.vectorize(Gradient)(cdf, empirical_range)
# # breakpoint()
# pdf_vals = deque()
# for row in empirical_range:
#     pdf_vals.append(pdf_(row))
# pdf_vals: np.ndarray = np.asarray(pdf_vals)


# spacing = 0.01
# pdf_ = FinDiff((0, spacing, 1), (1, spacing, 1))
# breakpoint()
# np.gradient()
# index0 = np.argsort(empirical_range[:, 0])
# index1 = np.argsort(empirical_range[:, 1])
# breakpoint()
# pdf_vals = np.gradient(np.gradient(cdf_vals[index0], empirical_range[index0, 0])[index1], empirical_range[index1, 1])
# fig = plt.figure(f"pdf Plot")
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(empirical_range[:, 0], empirical_range[:, 1], pdf_vals)
# plt.show()

# from typing import Callable
# def differentiate(func: Callable, data, eps: float = 10**-9):
#     for var in range(data.shape[1]):
#         # index1 = np.where(data[:, var] - eps >= xmins[var])[0]
#         # data = data[index1, :]
#         index2 = np.where(data[:, var] + eps <= xmaxs[var])[0]
#         data = data[index2, :]
#     return abs(func(data + eps) - func(data)) / (eps), data
#
# # def differentiate2(func: Callable, data, xmin, xmax, eps: float = 10**-9):
#
# eps = 10**-9
# er0p = empirical_range.copy()
# er0n = er0p.copy()
# er0p[:, 0] = er0p[:, 0] + eps
# er0n[:, 0] = er0n[:, 0] - eps
#
# first = (cdf(er0p) - cdf(er0n))/(2*eps)
# fr1p = first.copy()
# fr1n = first.copy()
#
# fr1p = fr1p + eps
# fr1n[:,1] = fr1n[:,1] - eps
# second = (cdf(fr1p) - cdf(fr1n))/(2*eps)

# pdf_vals, plot_data = differentiate(cdf, empirical_range)
# from csaps import csaps
#
# xdata = [np.linspace(-3, 3, 41), np.linspace(-3.5, 3.5, 31)]
# i, j = np.meshgrid(*xdata, indexing='ij')
# ydata = (3 * (1 - j)**2. * np.exp(-(j**2) - (i + 1)**2)
#          - 10 * (j / 5 - j**3 - i**5) * np.exp(-j**2 - i**2)
#          - 1 / 3 * np.exp(-(j + 1)**2 - i**2))
# ydata = ydata + (np.random.randn(*ydata.shape) * 0.75)
#
# ydata_s = csaps(xdata, ydata, xdata, smooth=0.988)

# breakpoint()
# ysmth = csaps(empirical_range.tolist(), pdf_vals.flatten(), empirical_range.tolist(), smooth=0.988)

fig = plt.figure(f"pdf Plot")
ax = plt.axes(projection='3d')
# ax.plot_trisurf(empirical_range[:,0], empirical_range[:,1], ysmth)
ax.plot_trisurf(empirical_range[:,0], empirical_range[:,1], pdf(empirical_range))
# ax.plot_trisurf(empirical_range[non_zero_points, 0], empirical_range[non_zero_points,1], pdf_vals[non_zero_points])
# ax.plot_trisurf(empirical_range[:,0], empirical_range[:,1], second)
# ax.plot_trisurf(plot_data[:, 0], plot_data[:, 1], pdf_vals)
# ax.plot_trisurf(plot_data[:, 0], plot_data[:, 1], multivariate_normal(multi_mean, multi_cov).cdf(plot_data))
plt.show()
