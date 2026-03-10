#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.rcalbp_l.parser import get_data_available, parse_rcalbpl_json


def test_parser():
    """Test that we can parse an RC-ALBP/L JSON file."""
    files = get_data_available()
    assert len(files) > 0, "No data files found"

    # Use the smallest instance for testing
    file = [f for f in files if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)

    # Check basic attributes
    assert problem.nb_tasks > 0
    assert problem.nb_stations > 0
    assert problem.nb_periods > 0
    assert problem.c_target > 0
    assert problem.c_max >= problem.c_target
    assert len(problem.tasks) == problem.nb_tasks
    assert len(problem.stations) == problem.nb_stations
    assert len(problem.periods) == problem.nb_periods


def test_problem_objectives():
    """Test that the problem has properly configured objectives."""
    files = get_data_available()
    file = [f for f in files if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)

    # Check objective register
    obj_register = problem.get_objective_register()
    assert obj_register is not None
    assert obj_register.objective_sense is not None
    # The problem should have at least one objective
    assert hasattr(obj_register, "dict_objective_to_doc")


def test_no_dataset(fake_data_home):
    """Test that proper error is raised when dataset is missing."""
    with pytest.raises(
        FileNotFoundError, match="python -m discrete_optimization.datasets"
    ):
        get_data_available()


if __name__ == "__main__":
    test_parser()
    test_problem_objectives()
