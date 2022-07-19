from setuptools import setup, find_packages
import sys


if sys.version_info < (3, 9):
    raise ValueError('Versions of Python before 3.9 are not supported')


setup(
    name='sklarpy',
    packages=find_packages(),
    version='0.1.3',
    license='MIT',
    long_description="SklarPy (pronounced 'Sky-Lar-Pee' or 'Sky-La-Pie') is an open-source software for probability "
                     "distribution fitting.  It contains useful tools for fitting copula and univariate probability "
                     "distributions. Named after Sklar's theorem and Abe Sklar, the American mathematician who proved "
                     "that multivariate cumulative distribution functions can be expressed in terms of copulas and "
                     "their marginals.",
    author='Tyler Mitchell',
    author_email='sklarpy@gmail.com',
    url='https://github.com/sklarpy/sklarpy',
    download_url='https://github.com/sklarpy/sklarpy/archive/refs/tags/v0.1.2.tar.gz',
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
        'SciPy'
    ],
    install_requires=[
        'pandas>=1.4.3',
        'numpy>=1.23.0',
        'scipy~=1.8.1',
        'matplotlib~=3.5.2',
        'dill~=0.3.5.1'
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
    ],
)
