#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Test OptalCP solver for single batch processing."""

import pytest

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.singlebatch.solvers.optal import (
    OptalSingleBatchSolver,
    optalcp_available,
)


@pytest.mark.skipif(not optalcp_available, reason="OptalCP not available")
class TestOptalSolver:
    """Test BpmOptalSolver."""

    @pytest.fixture
    def solver_and_result(self, tiny_problem):
        """Solve once and reuse result for multiple tests."""
        solver = OptalSingleBatchSolver(problem=tiny_problem)
        solver.init_model()
        result = solver.solve(time_limit=10)
        return solver, result, tiny_problem

    def test_inheritance(self):
        """Test that solver inherits from SolverDO."""
        assert issubclass(OptalSingleBatchSolver, SolverDO)

    def test_initialization(self, tiny_problem):
        """Test solver initialization."""
        solver = OptalSingleBatchSolver(problem=tiny_problem)

        assert solver.problem == tiny_problem

    def test_init_model(self, tiny_problem):
        """Test init_model method."""
        solver = OptalSingleBatchSolver(problem=tiny_problem)
        solver.init_model()

        # Check that model was created
        assert solver.cp_model is not None
        assert hasattr(solver, "batch_itvs")
        assert hasattr(solver, "job_itvs")
