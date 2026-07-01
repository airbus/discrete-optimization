#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.lotsizing.parser import parse_input_data
from discrete_optimization.lotsizing.problem import LotSizingProblem


def test_parse_simple_instance():
    """Test parsing a simple lot sizing instance."""
    input_data = """15
5
0 0 0 0 0 0 0 1 0 0 0 0 0 1 0
0 0 0 0 1 0 0 0 0 0 0 1 0 0 1
0 0 0 0 0 0 1 0 0 0 0 1 0 1 0
0 0 0 0 0 0 0 1 0 0 1 0 0 0 1
0 0 0 0 0 0 0 0 1 0 0 1 0 0 1
10

0 105 154 130 100
146 0 135 139 167
101 183 0 193 113
188 112 111 0 103
179 117 161 124 0

1195
"""

    problem = parse_input_data(input_data)

    # Check basic properties
    assert isinstance(problem, LotSizingProblem)
    assert problem.horizon == 15
    assert problem.nb_items_type == 5
    assert problem.capacity_machine == 1
    assert problem.is_binary

    # Check demands
    assert len(problem.demands) == 5
    assert all(len(d) == 15 for d in problem.demands)
    assert problem.demands[0] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    assert problem.total_demands_per_item == {0: 2, 1: 3, 2: 3, 3: 3, 4: 3}

    # Check stock cost
    assert all(c == 10 for c in problem.stock_cost_per_type_per_time_per_unit)

    # Check changeover costs
    assert len(problem.changeover_costs) == 5
    assert problem.changeover_costs[0][1] == 105
    assert problem.changeover_costs[1][0] == 146
    assert problem.changeover_costs[0][0] == 0  # Diagonal is 0

    # Check known bound
    assert problem.known_bound == 1195


def test_parse_tiny_instance():
    """Test parsing a tiny instance."""
    input_data = """5
2
0 1 0 0 1
1 0 0 0 1
2

0 5
3 0

10
"""

    problem = parse_input_data(input_data)

    assert problem.horizon == 5
    assert problem.nb_items_type == 2
    assert problem.demands[0] == [0, 1, 0, 0, 1]
    assert problem.demands[1] == [1, 0, 0, 0, 1]
    assert all(c == 2 for c in problem.stock_cost_per_type_per_time_per_unit)
    assert problem.changeover_costs == [[0, 5], [3, 0]]
    assert problem.known_bound == 10


def test_parse_without_known_bound():
    """Test parsing without known bound in the file."""
    input_data = """3
2
1 0 1
0 1 0
1

0 2
2 0
"""

    problem = parse_input_data(input_data)

    assert problem.horizon == 3
    assert problem.nb_items_type == 2
    assert problem.known_bound is None
