# #  Copyright (c) 2026 AIRBUS and its affiliates.
# #  This source code is licensed under the MIT license found in the
# #  LICENSE file in the root directory of this source tree.
#
# import pytest
# from pytest_cases import fixture
#
# from discrete_optimization.ovensched.parser import get_data_available, parse_dat_file
# from discrete_optimization.ovensched.problem import (
#     MachineData,
#     OvenSchedulingProblem,
#     TaskData,
# )
#
#
# @fixture
# def tiny_problem():
#     """Create a tiny test problem instance (3 jobs, 2 machines)."""
#     tasks_data = [
#         TaskData(
#             attribute=0,
#             min_duration=10,
#             max_duration=15,
#             earliest_start=0,
#             latest_end=100,
#             eligible_machines={0, 1},
#             size=5,
#         ),
#         TaskData(
#             attribute=1,
#             min_duration=8,
#             max_duration=12,
#             earliest_start=0,
#             latest_end=100,
#             eligible_machines={0, 1},
#             size=3,
#         ),
#         TaskData(
#             attribute=0,
#             min_duration=12,
#             max_duration=18,
#             earliest_start=0,
#             latest_end=100,
#             eligible_machines={0, 1},
#             size=4,
#         ),
#     ]
#
#     machines_data = [
#         MachineData(
#             capacity=10,
#             initial_attribute=0,
#             availability=[(0, 100)],
#         ),
#         MachineData(
#             capacity=10,
#             initial_attribute=1,
#             availability=[(0, 100)],
#         ),
#     ]
#
#     setup_costs = [
#         [0, 5],
#         [3, 0],
#     ]
#
#     setup_times = [
#         [0, 2],
#         [1, 0],
#     ]
#
#     return OvenSchedulingProblem(
#         n_jobs=3,
#         n_machines=2,
#         tasks_data=tasks_data,
#         machines_data=machines_data,
#         setup_costs=setup_costs,
#         setup_times=setup_times,
#     )
