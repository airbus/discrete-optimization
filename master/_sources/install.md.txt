# Installation

## Python 3.10+ environment

The use of a virtual environment is recommended, and you will need to ensure that the environment use a Python version
greater than 3.10.
This can be achieved for instance either by using [conda](https://docs.conda.io/en/latest/), or by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on windows)
and [venv](https://docs.python.org/fr/3/library/venv.html) module, or by using [uv](https://docs.astral.sh/uv/).

The following examples show how to create a virtual environment with Python version 3.12 with the mentioned methods.

#### With conda

```shell
conda create -n do-env python=3.12
conda activate do-env
```

#### With pyenv + venv

```shell
pyenv install 3.12
pyenv shell 3.12
python -m venv do-venv
source do-venv/bin/activate  # do-venv\Scripts\activate on windows
```

#### With uv

```shell
uv venv do-venv --python 3.12
source do-venv/bin/activate  # do-venv\Scripts\activate on windows
```



## Pip install discrete-optimization library

Install discrete-optimization with pip:

```shell
pip install discrete-optimization
```

or via the faster `uv pip` if you already installed [uv](https://docs.astral.sh/uv/):

```shell
uv pip install discrete-optimization
```

You can also make full use of uv project management capabilities by using `uv add`.
Please refer to uv doc.


## Optional dependencies

### Package extras

Several extras are available:
- *gurobi*: to use [gurobi](https://www.gurobi.com/) based solvers.
  It will install the python api for gurobi with a minimal license which does not allow to use it on "real" models.
  You need to install an appropriate license if you want to make full use of it.
- *quantum*: to use [qiskit](https://www.ibm.com/quantum/qiskit) based solvers.
- *toulbar*: to use [pytoulbar2](https://toulbar2.github.io/toulbar2/) based solvers.
- *optuna*: to use our utility functions to generate and launch [optuna](https://optuna.org/) studies (hyperparameters optimization).
- *dashboard*: to use the d-o [dashboard](dashboard.md)

You can install them all via:
```shell
pip install discrete-optimization[gurobi, qunatum, toulbar, optuna, dashboard]
```

### Minizinc 2.8+

If you want to use discrete-optimization solvers based on [minizinc](https://www.minizinc.org/),
you need to install it (version greater than 2.8) and update the `PATH` environment variable
so that it can be found by Python.
See [minizinc documentation](https://www.minizinc.org/doc-latest/en/installation.html) for more details.

> **Tip:** You can easily install minizinc from the command line, which can be useful when on cloud.
> In order to make life easier to cloud users, we reproduce below the necessary lines. Please be careful that this
> is not an official documentation for minizinc and that the following lines can stop working without notice
> as we do not test them automatically.

#### Linux command line
On a Linux distribution, you can use the bundled [minizinc AppImage](https://www.minizinc.org/doc-latest/en/installation.html#appimage).

If [FUSE](https://en.wikipedia.org/wiki/Filesystem_in_Userspace) is available:
```
mkdir minizinc_install
curl -o minizinc_install/minizinc -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.5/MiniZincIDE-2.8.5-x86_64.AppImage
chmod +x minizinc_install/minizinc
export PATH="$(pwd)/minizinc_install/":$PATH
minizinc --version
```
Else, this is still possible by extracting the files:
```
mkdir minizinc_install
cd minizinc_install
curl -o minizinc.AppImage -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.5/MiniZincIDE-2.8.5-x86_64.AppImage
chmod +x minizinc.AppImage
./minizinc.AppImage --appimage-extract
cd ..
export LD_LIBRARY_PATH="$(pwd)/minizinc_install/squashfs-root/usr/lib/":$LD_LIBRARY_PATH
export PATH="$(pwd)/minizinc_install/squashfs-root/usr/bin/":$PATH
minizinc --version
```

#### MacOs command line
```
mkdir minizinc_install
curl -o minizinc.dmg -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.5/MiniZincIDE-2.8.5-bundled.dmg
hdiutil attach minizinc.dmg
cp -R /Volumes/MiniZinc*/MiniZincIDE.app minizinc_install/.
export PATH="$(pwd)/minizinc_install/MiniZincIDE.app/Contents/Resources":$PATH
minizinc --version
```

#### Windows command line
Works on Windows Server 2022 with bash shell:
```
mkdir minizinc_install
curl -o minizinc_setup.exe -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.5/MiniZincIDE-2.8.5-bundled-setup-win64.exe
cmd //c "minizinc_setup.exe /verysilent /currentuser /norestart /suppressmsgboxes /sp"
export PATH="~/AppData/Local/Programs/MiniZinc":$PATH
minizinc --version
```


### Gurobi

If you want to try solvers using gurobi, install [gurobi](https://www.gurobi.com/) with its python binding (gurobipy)
and an appropriate license.

> **NB**: If you just do `pip install gurobipy`, you get a minimal license which does not allow to use it on "real" models.
