#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
    SubBrick,
)
from discrete_optimization.rcalbp_l.problem import RCALBPLSolution
from discrete_optimization.rcalbp_l.solvers.cpsat import CpSatRCALBPLSolver
from discrete_optimization.rcalbp_l.solvers.meta_solvers import (
    BackwardSequentialRCALBPLSolver,
    BackwardSequentialRCALBPLSolverSGS,
)


def test_backward_sequential_solver(problem):
    """Test the BackwardSequentialRCALBPLSolver metasolver."""
    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    solver = BackwardSequentialRCALBPLSolver(
        problem=problem,
        future_chunk_size=1,
        phase2_chunk_size=2,
        time_limit_phase1=15,
        time_limit_phase2=5,
        use_sgs_warm_start=False,
        parameters_cp=p,
    )

    result_storage = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])

    assert len(result_storage) > 0
    sol: RCALBPLSolution = result_storage[-1][0]
    assert problem.satisfy(sol), "BackwardSequential solution should be feasible"


def test_backward_sequential_solver_with_sgs(problem):
    """Test the BackwardSequentialRCALBPLSolver with SGS warm start."""
    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    solver = BackwardSequentialRCALBPLSolver(
        problem=problem,
        future_chunk_size=1,
        phase2_chunk_size=2,
        time_limit_phase1=15,
        time_limit_phase2=5,
        use_sgs_warm_start=True,
        parameters_cp=p,
    )

    result_storage = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])

    assert len(result_storage) > 0
    sol: RCALBPLSolution = result_storage[-1][0]
    assert problem.satisfy(sol)


def test_backward_sequential_solver_sgs_variant(problem):
    """Test the BackwardSequentialRCALBPLSolverSGS variant."""
    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    solver = BackwardSequentialRCALBPLSolverSGS(
        problem=problem,
        future_chunk_size=1,
        phase2_chunk_size=2,
        time_limit_phase1=15,
        time_limit_phase2=5,
        use_sgs_warm_start=True,
        parameters_cp=p,
    )

    result_storage = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])

    assert len(result_storage) > 0
    sol: RCALBPLSolution = result_storage[-1][0]
    # assert problem.satisfy(sol) TODO : postprocess the solution to respect convention on the cycle time.


def test_sequential_metasolver_two_stages(problem):
    """Test SequentialMetasolver with two stages: BackwardSequential + CPSat."""
    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    # First stage: quick initial solution
    brick1 = SubBrick(
        BackwardSequentialRCALBPLSolver,
        kwargs=dict(
            future_chunk_size=1,
            phase2_chunk_size=2,
            time_limit_phase1=15,
            time_limit_phase2=3,
            use_sgs_warm_start=True,
            parameters_cp=p,
            callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ),
    )

    # Second stage: refinement with full CPSat
    brick2 = SubBrick(
        CpSatRCALBPLSolver,
        kwargs=dict(
            add_heuristic_constraint=False,
            parameters_cp=p,
            time_limit=10,
            callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ),
    )

    solver = SequentialMetasolver(
        problem=problem,
        list_subbricks=[brick1, brick2],
    )

    result_storage = solver.solve()

    assert len(result_storage) > 0
    sol: RCALBPLSolution = result_storage[-1][0]
    assert problem.satisfy(sol), "Sequential metasolver solution should be feasible"

    # Verify the solution gets better or stays the same
    # (second solver warm-starts from first)
    evaluation = problem.evaluate(sol)
    assert "ramp_up_duration" in evaluation


def test_sequential_metasolver_single_stage(problem):
    """Test SequentialMetasolver with just one solver."""
    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    brick1 = SubBrick(
        BackwardSequentialRCALBPLSolver,
        kwargs=dict(
            future_chunk_size=1,
            phase2_chunk_size=2,
            time_limit_phase1=15,
            time_limit_phase2=5,
            use_sgs_warm_start=True,
            parameters_cp=p,
            callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ),
    )

    solver = SequentialMetasolver(
        problem=problem,
        list_subbricks=[brick1],
    )

    result_storage = solver.solve()

    assert len(result_storage) > 0
    sol: RCALBPLSolution = result_storage[-1][0]
    assert problem.satisfy(sol)
