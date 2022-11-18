#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os

__version__ = "0.0.0"


# Check that minimal minizinc binary version is respected,
# except if environment variable DO_SKIP_MZN_CHECK is set to 1.
if ("DO_SKIP_MZN_CHECK" not in os.environ) or not (os.environ["DO_SKIP_MZN_CHECK"]):
    import minizinc

    _minizinc_minimal_parsed_version = (2, 6)
    _minizinc_minimal_str_version = ".".join(
        str(i) for i in _minizinc_minimal_parsed_version
    )

    if minizinc.default_driver is None:
        raise RuntimeError(
            "Minizinc binary has not been found.\n"
            "You need to install it and/or configure the PATH environment variable.\n"
            "See minizinc documentation for more details: https://www.minizinc.org/doc-latest/en/installation.html\n\n"
            "You can also bypass this check by setting the environment variable DO_SKIP_MZN_CHECK to 1, "
            "at your own risk."
        )
    if minizinc.default_driver.parsed_version < _minizinc_minimal_parsed_version:
        raise RuntimeError(
            f"Minizinc binary version must be at least {_minizinc_minimal_str_version}.\n"
            "Install an appropriate version of minizinc and/or configure the PATH environment variable.\n"
            "See minizinc documentation for more details: https://www.minizinc.org/doc-latest/en/installation.html\n\n"
            "You can also bypass this check by setting the environment variable DO_SKIP_MZN_CHECK to 1, "
            "at your own risk."
        )
