# Installation

## Prerequisites

### Minizinc 2.6+

You need to install [minizinc](https://www.minizinc.org/) (version greater than 2.6) and update the `PATH` environment variable
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
curl -o minizinc_install/minizinc -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.6.3/MiniZincIDE-2.6.3-x86_64.AppImage
chmod +x minizinc_install/minizinc
export PATH="$(pwd)/minizinc_install/":$PATH
minizinc --version
```
Else, this is still possible by extracting the files:
```
mkdir minizinc_install
cd minizinc_install
curl -o minizinc.AppImage -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.6.3/MiniZincIDE-2.6.3-x86_64.AppImage
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
curl -o minizinc.dmg -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.6.3/MiniZincIDE-2.6.3-bundled.dmg
hdiutil attach minizinc.dmg
cp -R /Volumes/MiniZinc*/MiniZincIDE.app minizinc_install/.
export PATH="$(pwd)/minizinc_install/MiniZincIDE.app/Contents/Resources":$PATH
minizinc --version
```

#### Windows command line
Works on Windows Server 2022 with bash shell:
```
mkdir minizinc_install
curl -o minizinc_setup.exe -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.6.3/MiniZincIDE-2.6.3-bundled-setup-win64.exe
cmd //c "minizinc_setup.exe /verysilent /currentuser /norestart /suppressmsgboxes /sp"
export PATH="~/AppData/Local/Programs/MiniZinc":$PATH
minizinc --version
```

#### Skipping minizinc version check

It may happen that you need to use only a part of the library which is not relying on minizinc at all,
and that you do not want to install minzinc.
This can be troublesome as the minizinc binary version is checked at library import.
We provide a way to bypass this check by setting the environment variable DO_SKIP_MZN_CHECK:
```shell
export DO_SKIP_MZN_CHECK=1
```
Please note however that the library is never tested without minizinc (or minizinc versions < 2.6).
Most modules related to solvers will fail to be imported without minizinc as they are heavily relying on it.


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
