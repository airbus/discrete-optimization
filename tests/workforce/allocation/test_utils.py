#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.utils import (
    create_subproblem_allocation,
)


def test_create_subproblem_allocation():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance, multiobjective=True)
    subset_tasks = list(allocation_problem.graph_activity.graph_nx)[:2]
    create_subproblem_allocation(allocation_problem, subset_tasks=subset_tasks)
