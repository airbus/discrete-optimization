from cpmpy.expressions.core import BoolVal

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.solvers.cpmpy import CpmpyColoringSolver
from discrete_optimization.generic_tools.cpmpy_tools import (
    MetaCpmpyConstraint,
    is_trivially_false,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver


def test_cpmpy_solver():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = CpmpyColoringSolver(color_problem)
    solver.init_model(nb_colors=20)
    result_store = solver.solve()
    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)
    assert solver.status_solver == StatusSolver.OPTIMAL
    nb_colors_optimal = solution.nb_color

    # test get_others_meta_constraint()
    # with default soft and hard metas
    assert len(solver.get_others_meta_constraint()) == 0
    # with custom metas
    meta_constraints = [
        MetaCpmpyConstraint(
            "toto", solver.get_hard_meta_constraints()[0].constraints[0][:4]
        )
    ] + solver.get_soft_meta_constraints()[1:2]
    meta_others = solver.get_others_meta_constraint(meta_constraints)
    assert meta_others.name == "others"
    constraints_strings = [str(c) for c in meta_others]
    assert "(x[0]) != (x[16])" in constraints_strings
    assert "(x[4]) <= (nb_colors)" in constraints_strings
    assert "(x[1]) <= (nb_colors)" not in constraints_strings
    assert not any("x[1]" in c for c in constraints_strings)

    # add impossible constraint (nb_colors < optimal)
    solver = CpmpyColoringSolver(color_problem)
    solver.init_model(nb_colors=nb_colors_optimal - 1)
    solver.solve()
    assert solver.status_solver == StatusSolver.UNSATISFIABLE

    # explain (fine and meta)
    assert len(solver.explain_unsat_fine()) < len(solver.get_soft_constraints())
    metaconstraints_mus = solver.explain_unsat_meta()
    assert len(metaconstraints_mus) < len(solver.get_soft_meta_constraints())
    for meta in metaconstraints_mus:
        assert isinstance(meta, MetaCpmpyConstraint)
    metaconstraints_deduced_mus = solver.explain_unsat_deduced_meta()
    assert len(metaconstraints_deduced_mus) < len(solver.get_soft_meta_constraints())
    for meta in metaconstraints_deduced_mus:
        assert isinstance(meta, MetaCpmpyConstraint)

    # correct meta
    meta_mcs = solver.correct_unsat_meta()
    assert 0 < len(meta_mcs) < len(solver.get_soft_meta_constraints())
    subconstraints_mcs_ids = set()
    for meta in meta_mcs:
        subconstraints_mcs_ids.update({id(c) for c in meta.constraints})
    solver_model_constraints_bak = solver.model.constraints
    solver.model.constraints = [
        c for c in solver_model_constraints_bak if id(c) not in subconstraints_mcs_ids
    ]  # NB: solver.model.constraints.remove(cstr) not working as expected
    solver.reset_cpm_solver()
    solver.solve()
    assert solver.status_solver == StatusSolver.OPTIMAL
    # correct with deduced meta
    solver.status_solver = (
        StatusSolver.UNSATISFIABLE
    )  # reset status solver to allow next method
    solver.model.constraints = solver_model_constraints_bak
    meta_mcs = solver.correct_unsat_deduced_meta()
    assert 0 < len(meta_mcs) < len(solver.get_soft_meta_constraints())
    subconstraints_mcs_ids = set()
    for meta in meta_mcs:
        subconstraints_mcs_ids.update({id(c) for c in meta.constraints})
    solver_model_constraints_bak = solver.model.constraints
    solver.model.constraints = [
        c for c in solver_model_constraints_bak if id(c) not in subconstraints_mcs_ids
    ]  # NB: solver.model.constraints.remove(cstr) not working as expected
    solver.reset_cpm_solver()
    solver.solve()
    assert solver.status_solver == StatusSolver.OPTIMAL

    # tranform meta-constraints into nodes
    meta_mcs_nodes = solver.convert_metaconstraints2nodes(meta_mcs)
    assert len(meta_mcs_nodes) == len(meta_mcs)
    for node in meta_mcs_nodes:
        assert node in color_problem.graph.nodes_name

    # tranform meta-constraints into edges
    constraints = meta_mcs[0]
    edges = solver.convert_constraints2edges(constraints)
    assert len(constraints) == len(edges)
    for edge in edges:
        assert edge[0] in color_problem.graph.nodes_name
        assert edge[1] in color_problem.graph.nodes_name


def test_explainability_with_False_constraints():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = CpmpyColoringSolver(color_problem)
    solver.init_model(nb_colors=20)
    result_store = solver.solve()
    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)
    assert solver.status_solver == StatusSolver.OPTIMAL
    nb_colors_optimal = solution.nb_color

    # add False constraint and unsatifiable condition on nb colors
    solver = CpmpyColoringSolver(color_problem)
    solver.init_model(nb_colors=nb_colors_optimal - 1)
    solver.model.constraints.extend([False, False])
    solver.reset_cpm_solver()
    solver.solve()
    assert solver.status_solver == StatusSolver.UNSATISFIABLE
    # fine
    mus = solver.explain_unsat_fine(soft=solver.model.constraints)
    assert mus == [BoolVal(False)]
    mcs = solver.correct_unsat_fine(soft=solver.model.constraints)
    assert any(is_trivially_false(cstr) for cstr in mcs)
    assert len(mcs) > 1
    # meta
    softs_meta = solver.get_soft_meta_constraints()
    hards_meta = solver.get_hard_meta_constraints()
    softs_meta[0].append(False)
    softs_meta[-1].append(False)
    mus_meta = solver.explain_unsat_meta(soft=softs_meta, hard=hards_meta)
    assert len(mus_meta) == 1
    assert any(is_trivially_false(cstr) for cstr in mus_meta[0])
    mcs_meta = solver.correct_unsat_meta(soft=softs_meta, hard=hards_meta)
    assert softs_meta[0] in mcs_meta
    assert softs_meta[1] in mcs_meta
    assert len(mcs_meta) > 2
