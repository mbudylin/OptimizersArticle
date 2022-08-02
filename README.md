# OptimizersArticle
Обзор оптимизаторов, солверов.

Необходимо предварительно установить солверы, как описано ниже.

1) cbc для macOS: https://ampl.com/products/solvers/open-source/
прокинуть в .bash_profile путь в бинарнику.

2) cvxopt для macOS (отсюда https://cvxopt.org/install/):
```
brew install gsl fftw suite-sparse glpk
git clone https://github.com/cvxopt/cvxopt.git
cd cvxopt
git checkout `git describe --abbrev=0 --tags`
export CVXOPT_BUILD_FFTW=1    # optional
export CVXOPT_BUILD_GLPK=1    # optional
export CVXOPT_BUILD_GSL=1     # optional
python setup.py install
```

