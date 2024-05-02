#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import importlib
import logging
import pkgutil

import discrete_optimization

logger = logging.getLogger(__name__)


def find_abs_modules(package):
    """Find names of all submodules in the package."""
    path_list = []
    for modinfo in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        if not modinfo.ispkg:
            path_list.append(modinfo.name)
    return path_list


def test_importing_all_submodules():
    modules_with_errors = []
    for m in find_abs_modules(discrete_optimization):
        try:
            importlib.import_module(m)
        except Exception as e:
            modules_with_errors.append(m)
            print(f"{m}: {e.__class__.__name__}: {e}")
    if len(modules_with_errors) > 0:
        raise ImportError(
            f"{len(modules_with_errors)} submodules of discrete_optimization cannot be imported\n"
            + f"{modules_with_errors}"
        )


if __name__ == "__main__":
    test_importing_all_submodules()
