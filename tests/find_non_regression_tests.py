#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Find non-regression tests files.

Equivalent to `ls tests/**/test_*non_regression.py` but ** pattern is not always available.
(In particular the bash version on macos github runners is too low to allow **)

"""

import glob
import os

test_dir = os.path.dirname(__file__)


if __name__ == "__main__":
    print(" ".join(glob.glob(f"{test_dir}/**/test_*non_regression.py", recursive=True)))
