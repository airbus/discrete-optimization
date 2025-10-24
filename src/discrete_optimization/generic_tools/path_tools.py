#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os


def get_directory(file: str) -> str:
    return os.path.dirname(file)


def abspath_from_file(file: str, relative_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(file)), relative_path)
