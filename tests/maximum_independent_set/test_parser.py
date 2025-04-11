import os

from discrete_optimization.maximum_independent_set.parser import dimacs_parser

test_dir = os.path.dirname(__file__)


def test_dimacs_parser():
    filename = f"{test_dir}/myciel3.col"
    mis_problem = dimacs_parser(filename)
    assert 11 in mis_problem.graph_nx
    assert 0 not in mis_problem.graph_nx


def test_dimacs_parser_isolated_node():
    filename = f"{test_dir}/myciel3.mod.col"
    mis_problem = dimacs_parser(filename)
    assert 12 in mis_problem.graph_nx
    assert 0 not in mis_problem.graph_nx
    assert len(mis_problem.graph_nx[11]) > 0
    assert len(mis_problem.graph_nx[12]) == 0
