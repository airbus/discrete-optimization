#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from discrete_optimization.generic_tools.do_problem import ListInteger, Permutation


class ListIntegerRcpsp(ListInteger):
    """Attribute type permutation specific to RcpspSolution.

    Useful to make mutation catalog map RcpspSolution attribute to specific mutations.

    """

    ...


class PermutationRcpsp(Permutation):
    """Attribute type permutation specific to RcpspSolution.

    Useful to make mutation catalog map RcpspSolution attribute to specific mutations.

    """

    ...
