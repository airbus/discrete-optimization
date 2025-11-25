#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import glob
import os
import runpy

from pytest_cases import param_fixture

TUTO_DIRPATH = os.path.abspath(
    f"{os.path.dirname(__file__)}/../docs/source/howto_new_problem_implementation"
)


tuto_script = param_fixture(
    "tuto_script",
    [
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(f"{TUTO_DIRPATH}/*.py")
    ],
)


def test(monkeypatch, tuto_script):
    # Add temporarily the tuto directory to python path
    monkeypatch.syspath_prepend(TUTO_DIRPATH)
    # run the tuto script
    runpy.run_module(tuto_script, run_name="__main__")
