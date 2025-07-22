from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.problem import TeamAllocationProblem


def test_problem():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    assert isinstance(allocation_problem, TeamAllocationProblem)
