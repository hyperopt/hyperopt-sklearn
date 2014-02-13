import os
OMP_NUM_THREADS = os.environ.get('OMP_NUM_THREADS', None)
if OMP_NUM_THREADS != 1:
    print ('WARN: if you are using openblas set OMP_NUM_THREADS=1'
           ' or risk subprocess calls hanging indefinitely')

from estimator import hyperopt_estimator

# -- flake8
