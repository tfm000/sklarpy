from setuptools import setup, find_packages


setup(
    name='sklarpy',
    packages=find_packages(),
    version='1.0.0',
    license='MIT',
    description="A Python library for Copula, Multivariate and Univariate "
                "probability distribution fitting.",
    long_description="SklarPy (pronounced 'Sky-Lar-Pee' or 'Sky-La-Pie') is "
                     "an open-source software for probability distribution "
                     "fitting.  It contains useful tools for fitting Copula, "
                     "Multivariate and Univariate probability distributions. "
                     "In addition to over 100 univariate distributins, we "
                     "implement many multivariate normal mixture distributions"
                     " and their copulas, including Gaussian, Student-T, "
                     "Skewed-T and Generalized Hyperbolic distributions and "
                     "copulas. Named after Sklar's theorem and Abe Sklar, the "
                     "American mathematician who proved that multivariate "
                     "cumulative distribution functions can be expressed in "
                     "terms of copulas and their marginals.",
    author='Tyler Mitchell',
    author_email='sklarpy@gmail.com',
    url='https://github.com/sklarpy/sklarpy',
    download_url='https://github.com/sklarpy/sklarpy/archive/refs/tags/'
                 'v0.1.2.tar.gz',
    keywords=[
        'SklarPy',
        'Sklar',
        'Copula',
        'Copulas',
        'Probability',
        'Distribution',
        'Univariate',
        'Bivariate',
        'Multivariate',
        'Joint',
        'CDF',
        'PDF',
        'Modeling',
        'Quantitative',
        'Fitting'
        'SciPy',
        'Statistics',
        'Mathematics',
        'Science'
        'Scientific',
        'Finance',
        'Risk',
        'VaR',
    ],
    python_requires='>=3.9',
    install_requires=[
        'pandas>=1.4.3',
        'numpy>=1.23.0',
        'scipy>=1.11.0',
        'matplotlib>=3.5.2',
        'dill>=0.3.5.1',
        'tqdm>=4.64.1',
        'seaborn>=0.12.2',
    ],
    extras_require={
        "dev": [
            "pytest==7.1.2",
            "tox==3.25.1",
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
