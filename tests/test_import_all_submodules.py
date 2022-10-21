#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import importlib
import logging
import pkgutil
import sys

import discrete_optimization

logger = logging.getLogger(__name__)


def find_abs_modules(package):
    """Find names of all submodules in the package

    https://stackoverflow.com/questions/48879353/how-do-you-recursively-get-all-submodules-in-a-python-package

    """
    path_list = []
    spec_list = []
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__):
        if modname == "hub.__skdecide_hub_cpp":
            continue
        import_path = f"{package.__name__}.{modname}"
        if ispkg:
            spec = pkgutil._get_spec(importer, modname)
            try:
                importlib._bootstrap._load(spec)
                spec_list.append(spec)
            except Exception as e:
                logger.warning(
                    f"Could not load package {modname}, so it will be ignored ({e})."
                )
        else:
            path_list.append(import_path)
    for spec in spec_list:
        del sys.modules[spec.name]
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
