import random

import didppy as dp
import numpy as np
import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.dp import DpAllocationSolver


@pytest.fixture
def random_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.mark.parametrize(
    "multiobjective, symmbreak_on_used, force_allocation_when_possible",
    [
        (False, True, False),
        (False, False, False),
        # (False, False, True),  # randomly do not find solutions (1 / 10 times )
        (True, True, False),
        (True, False, False),
        (True, False, True),
    ],
)
@pytest.mark.parametrize("solver_cls", [dp.CABS, dp.LNBS, dp.DDLNS])
def test_dp_params(
    random_seed,
    solver_cls,
    multiobjective,
    symmbreak_on_used,
    force_allocation_when_possible,
):
    import logging

    from discrete_optimization.generic_tools.dyn_prog_tools import logger

    logger.setLevel(logging.DEBUG)
    instance = [p for p in get_data_available() if "instance_0.json" in p][0]
    allocation_problem = parse_to_allocation_problem(
        instance, multiobjective=multiobjective
    )
    kwargs = dict(
        symmbreak_on_used=symmbreak_on_used,
        force_allocation_when_possible=force_allocation_when_possible,
    )
    if solver_cls in [dp.CABS, dp.DDLNS]:
        kwargs["seed"] = random_seed
    solver = DpAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    # check solve
    res = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        solver=solver_cls,
    )
    assert len(res) == 1
    sol = res.get_best_solution()
    allocation_problem.evaluate(sol)
    # assert allocation_problem.satisfy(sol)

    # warm start
    sol1 = allocation_problem.get_dummy_solution()
    assert not (sol.allocation == sol1.allocation)
    solver.set_warm_start(sol1)
    res = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    assert len(res) == 1
    # assert sol.allocation == sol1.allocation  # dummy solution is not an ok solution so not the one found
