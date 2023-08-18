# Grouping univariate distributions

__all__ = ['distributions_map']


########################################################################################################################
# Continuous (Parametric)
########################################################################################################################
continuous_parametric_names: tuple = ('alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr',
                                      'burr12', 'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull',
                                      'erlang', 'expon', 'exponnorm', 'exponpow', 'exponweib', 'f', 'fatiguelife',
                                      'fisk', 'foldcauchy', 'foldnorm', 'gamma', 'gausshyper', 'genexpon', 'genextreme',
                                      'gengamma', 'genhalflogistic',
                                      'genhyperbolic',
                                      'geninvgauss', 'genlogistic',
                                      'gennorm', 'genpareto', 'gig',
                                      # 'gilbrat',
                                      'gompertz', 'gumbel_l', 'gumbel_r',
                                      'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm', 'hypsecant', 'ig', 'invgamma',
                                      'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa3', 'kappa4', 'ksone',
                                      'kstwo', 'kstwobign', 'laplace', 'laplace_asymmetric', 'levy', 'levy_l',
                                      'levy_stable', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'loguniform',
                                      'lomax', 'maxwell', 'mielke', 'moyal', 'nakagami', 'ncf', 'nct', 'ncx2', 'normal',
                                      'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm',
                                      'rayleigh', 'rdist', 'recipinvgauss', 'reciprocal', 'rice', 'semicircular',
                                      'skewcauchy', 'skewnorm', 'student_t', 'trapezoid', 'trapz', 'triang', 'truncexpon',
                                      'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald',
                                      'weibull_max', 'weibull_min', 'wrapcauchy')
common_continuous_parametric_names: tuple = ('cauchy', 'chi2', 'expon', 'gamma', 'lognorm', 'normal', 'powerlaw',
                                             'rayleigh', 'student_t', 'uniform')
continuous_multimodal_parametric_names: tuple = ('arcsine', 'beta')

########################################################################################################################
# Discrete (Parametric)
########################################################################################################################
discrete_parametric_names: tuple = ('discrete_laplace', 'discrete_uniform', 'geometric', 'planck', 'poisson')
common_discrete_parametric_names: tuple = ('discrete_laplace', 'discrete_uniform', 'geometric', 'poisson')
discrete_multimodal_parametric_names: tuple = ()


########################################################################################################################
# Numerical/Non-Parametric
########################################################################################################################
continuous_numerical_names: tuple = ('gaussian_kde', 'empirical')
discrete_numerical_names: tuple = ('discrete_empirical',)


########################################################################################################################
# Distribution Categories/Map
########################################################################################################################
all_continuous_names: tuple = (*continuous_parametric_names, *continuous_numerical_names)
all_discrete_names: tuple = (*discrete_parametric_names, *discrete_numerical_names)
all_common_names: tuple = (*common_continuous_parametric_names, *common_discrete_parametric_names)
all_multimodal_names: tuple = (*continuous_multimodal_parametric_names, *discrete_multimodal_parametric_names)
all_parametric_names: tuple = (*continuous_parametric_names, *discrete_parametric_names)
all_numerical_names: tuple = (*continuous_numerical_names, *discrete_numerical_names)
all_distributions: tuple = (*all_continuous_names, *all_discrete_names)

distributions_map: dict = {
    'all': all_distributions,
    'all continuous': all_continuous_names,
    'all discrete': all_discrete_names,
    'all common': all_common_names,
    'all multimodal': all_multimodal_names,
    'all parametric': all_parametric_names,
    'all numerical': all_numerical_names,
    'all continuous parametric': continuous_parametric_names,
    'all discrete parametric': discrete_parametric_names,
    'all continuous numerical': continuous_numerical_names,
    'all discrete numerical': discrete_numerical_names,
    'common continuous': common_continuous_parametric_names,
    'common discrete': common_discrete_parametric_names,
    'continuous multimodal': continuous_multimodal_parametric_names,
    'discrete multimodal': discrete_multimodal_parametric_names,
}
