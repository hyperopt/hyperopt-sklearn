import logging
import os

try:
    import setuptools
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")


setup(
    name = "hpsklearn",
    version = '0.0.3',
    packages = find_packages(),
    scripts = [],
    url = 'http://hyperopt.github.com/hyperopt-sklearn/',
    download_url = 'https://github.com/hyperopt/hyperopt-sklearn/archive/0.0.3.tar.gz',
    author = 'James Bergstra',
    author_email = 'anon@anon.com',
    description = 'Hyperparameter Optimization for sklearn',
    long_description = open('README.md').read(),
    keywords = ['hyperopt', 'hyperparameter', 'sklearn'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    platforms = ['Linux', 'OS-X', 'Windows'],
    license = 'BSD',
    install_requires = [
        'hyperopt',
        'nose',
        'numpy',
        'scikit-learn',
        'scipy',
    ],
    extras_require = {
        'xgboost':  ['xgboost==0.6a2']
    }
)
