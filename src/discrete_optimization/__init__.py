#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("discrete-optimization")
except PackageNotFoundError:
    # package is not installed
    pass
