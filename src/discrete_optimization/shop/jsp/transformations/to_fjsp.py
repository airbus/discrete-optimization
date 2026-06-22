#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from JobShop to FlexibleJobShop.
With new common API for shop-problems, there is no really transformation needed.
"""

from copy import deepcopy
from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.shop.fjsp.problem import FJobShopProblem, FJobShopSolution
from discrete_optimization.shop.jsp.problem import JobShopProblem, JobShopSolution


class JspToFjspTransformation(
    ProblemTransformation[
        JobShopProblem, JobShopSolution, FJobShopProblem, FJobShopSolution
    ]
):
    """Transform JobShop to FlexibleJobShop.

    Mapping:
    - Each subjob with fixed machine → subjob with 1 machine option
    - Job structure preserved
    - Precedence within jobs preserved

    JobShop is a special case of FlexibleJobShop where each operation
    can only be processed on exactly one machine.
    """

    def transform_problem(self, source_problem: JobShopProblem) -> FJobShopProblem:
        """Transform JobShop to FlexibleJobShop.

        Args:
            source_problem: JobShop problem instance

        Returns:
            Equivalent FlexibleJobShop problem

        """
        return FJobShopProblem(
            list_jobs=deepcopy(source_problem.list_jobs),
            n_jobs=source_problem.n_jobs,
            n_machines=source_problem.n_machines,
            horizon=source_problem.horizon,
        )

    def back_transform_solution(
        self, solution: FJobShopSolution, source_problem: JobShopProblem
    ) -> JobShopSolution:
        """Transform FlexibleJobShop solution back to JobShop solution.

        Args:
            solution: FlexibleJobShop solution
            source_problem: Original JobShop problem

        Returns:
            Equivalent JobShop solution

        """
        return solution

    def forward_transform_solution(
        self, solution: JobShopSolution, target_problem: FJobShopProblem
    ) -> Optional[FJobShopSolution]:
        """Transform JobShop solution to FlexibleJobShop solution (for warmstart).

        Args:
            solution: JobShop solution
            target_problem: Target FlexibleJobShop problem

        Returns:
            Equivalent FlexibleJobShop solution for warmstart

        """
        return solution
