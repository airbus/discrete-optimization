# Installation

## Prerequisites

### Minizinc 2.6+

You need to install [minizinc](https://www.minizinc.org/) (version greater than 2.6) and update the `PATH` environment variable
so that it can be found by Python.
See [minizinc documentation](https://www.minizinc.org/doc-latest/en/installation.html) for more details.

### Python 3.7+ environment

The use of a virtual environment is recommended, and you will need to ensure that the environment use a Python version
greater than 3.7.
This can be achieved for instance either by using [conda](https://docs.conda.io/en/latest/) or by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on windows)
and [venv](https://docs.python.org/fr/3/library/venv.html) module.

The following examples show how to create a virtual environment with Python version 3.8.13 with the mentioned methods.

#### With conda (all platforms)

```shell
conda create -n do-env python=3.8.13
conda activate do-env
```

#### With pyenv + venv (Linux/MacOS)

```shell
pyenv install 3.8.13
pyenv shell 3.8.13
python -m venv do-venv
source do-venv/bin/activate
```

#### With pyenv-win + venv (Windows)

```shell
pyenv install 3.8.13
pyenv shell 3.8.13
python -m venv do-venv
do-venv\Scripts\activate
```

### Gurobi [optional]


Optionally, install [gurobi](https://www.gurobi.com/) with its python binding (gurobipy)
and an appropriate license, if you want to try solvers that make use of gurobi.

> **NB**: If you just do `pip install gurobipy`, you get a minimal license which does not allow to use it on "real" models.


## Pip install discrete-optimization library

Install discrete-optimization from pip:

```shell
pip install discrete-optimization
```
