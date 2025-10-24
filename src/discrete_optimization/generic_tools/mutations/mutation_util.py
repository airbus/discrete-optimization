#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    TypeAttribute,
)


def get_attribute_for_type(
    register: EncodingRegister, type_attribute: TypeAttribute
) -> str:
    attributes = [
        k
        for k in register.dict_attribute_to_type
        for t in register.dict_attribute_to_type[k].get("type", [])
        if t == type_attribute
    ]
    if len(attributes) > 0:
        return attributes[0]
    else:
        raise ValueError(
            f"The encoding register must have at least one attribute of type {type_attribute}."
        )
