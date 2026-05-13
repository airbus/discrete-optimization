#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.alb.rcalbp.utils import create_from_rcpsp, load_rcpsp_as_albp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file


def test_create_from_rcpsp():
    """Test creating RC-ALBP from RCPSP problem."""
    # Load a small RCPSP instance
    files = get_data_available()
    rcpsp_problem = parse_file(files[0])

    # Convert to RC-ALBP
    albp_problem = create_from_rcpsp(rcpsp_problem, nb_stations=3, seed=42)

    assert albp_problem.nb_tasks == rcpsp_problem.n_jobs
    assert albp_problem.nb_stations == 3
    assert len(albp_problem.resources) == len(rcpsp_problem.resources_list)


def test_load_rcpsp_as_albp():
    """Test loading RCPSP instance and converting to RC-ALBP."""
    albp_problem = load_rcpsp_as_albp(instance_name="j301_1", nb_stations=3, seed=42)

    assert albp_problem.nb_tasks > 0
    assert albp_problem.nb_stations == 3
    assert len(albp_problem.resources) > 0
    assert len(albp_problem.stations) == 3
