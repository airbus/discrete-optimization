#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import sys
from typing import Any, Iterable, TypeVar, Union

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


def index_min(list_or_array: npt.ArrayLike) -> np.int_:
    """Argmin operator that can be used in gp.

    Args:
        list_or_array: any list or array

    Returns: index of minimum element of the array
    """
    return np.argmin(list_or_array)


def index_max(list_or_array: npt.ArrayLike) -> np.int_:
    """Argmax operator that can be used in gp.

    Args:
        list_or_array: any list or array

    Returns: index of maximum element of the array
    """
    return np.argmax(list_or_array)


def argsort(list_or_array: npt.ArrayLike) -> npt.NDArray[np.int_]:
    """Return the sorted array with indexes

    Args:
        list_or_array: any list or array

    Returns: indexes of array by increasing order.
    """
    return np.argsort(list_or_array)


def protected_div(left: float, right: float) -> float:
    if right != 0.0:
        return left / right
    else:
        return 1.0


class SupportsDunderLT(Protocol):
    def __lt__(self, __other: Any) -> Any:
        ...


class SupportsDunderGT(Protocol):
    def __gt__(self, __other: Any) -> Any:
        ...


SupportsRichComparison = Union[SupportsDunderLT, SupportsDunderGT]
SupportsRichComparisonT = TypeVar(
    "SupportsRichComparisonT", bound=SupportsRichComparison
)


def max_operator(
    left: SupportsRichComparisonT, right: SupportsRichComparisonT
) -> SupportsRichComparisonT:
    return max(left, right)


def min_operator(
    left: SupportsRichComparisonT, right: SupportsRichComparisonT
) -> SupportsRichComparisonT:
    return min(left, right)


def max_operator_list(
    list_: Iterable[SupportsRichComparisonT],
) -> SupportsRichComparisonT:
    return max(list_)


def min_operator_list(
    list_: Iterable[SupportsRichComparisonT],
) -> SupportsRichComparisonT:
    return min(list_)
