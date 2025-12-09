#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping


class AttributeType:
    length: int


@dataclass
class Permutation(AttributeType):
    """Attribute type for permutation."""

    range: Collection[int]
    """List of integers included in the permutation."""

    @property
    def length(self):
        """Nb of elements in the permutation."""
        return len(self.range)


@dataclass
class ListInteger(AttributeType):
    """Attribute type for list of integers with variable bounds."""

    length: int  # length of the list
    lows: list[int]  # lower bound for each element of the list
    ups: list[int]  # upper bound for each element of the list
    arities: list[int]  # nb of possible values for each element of the list

    def __init__(
        self,
        length: int,
        lows: list[int] | int = 0,
        ups: list[int] | int | None = None,
        arities: list[int] | int | None = None,
    ):
        """

        Args:
            length: length of the list
            ups: upper bound for each element of the list.
                If an integer, assumed to be the same for each element.
                If None, deduced from lows + arities
            lows: lower bounds for each element of the list.
                If an integer, assumed to be the same for each element.
            arities: used only if ups is None (cannot be None at the same time as it).
                Number of possible values for each element of the list.
                If an integer, assumed to be the same for each element.

        """
        self.length = length
        match lows:
            case int():
                self.lows = [lows] * length
            case _:
                self.lows = lows
        match ups:
            case None:
                match arities:
                    case None:
                        raise ValueError("ups and arities cannot be None together")
                    case int():
                        arities_list = [arities] * length
                    case _:
                        arities_list = arities
                self.ups = [
                    low + arity - 1 for low, arity in zip(self.lows, arities_list)
                ]
            case int():
                self.ups = [ups] * length
            case _:
                self.ups = ups

    @property
    def arities(self):
        return [up - low + 1 for low, up in zip(self.lows, self.ups)]


@dataclass
class ListBoolean(ListInteger):
    def __init__(self, length: int):
        super().__init__(length=length, lows=0, ups=1)


class EncodingRegister(Mapping[str, AttributeType]):
    """List the encoding attributes of a solution.

    Only the ones to be used by genetic algorithms and local search have to be specified.

    Attributes:
        dict_attribute_to_type (dict[str, AttributeType]): specifies the encodings of a solution object.
            Maps an attribute name to an attribute type.

    Each attribute name (key of the mapping) should correspond to
    - an actual attribute of the solution (solution.<attribute_name> should exist) with the proper type
    - a licit argument of __init__() method of the solution (at least to be used in genetic algorithms)

    You can bypass the latter hypothesis by overriding `build_solution_from_encoding()`

    """

    def __init__(self, dict_attribute_to_type: dict[str, AttributeType]):
        self.dict_attribute_to_type = dict_attribute_to_type

    def __getitem__(self, key, /):
        return self.dict_attribute_to_type[key]

    def __len__(self):
        return len(self.dict_attribute_to_type)

    def __iter__(self):
        return iter(self.dict_attribute_to_type)

    def get_first_attribute_of_type(
        self, attribute_type_cls: type[AttributeType]
    ) -> str:
        attributes = [k for k, t in self.items() if isinstance(t, attribute_type_cls)]
        if len(attributes) > 0:
            return attributes[0]
        else:
            raise ValueError(
                f"The encoding register must have at least one attribute of type {attribute_type_cls}."
            )

    def __str__(self) -> str:
        return "Encoding : " + str(self.dict_attribute_to_type)
