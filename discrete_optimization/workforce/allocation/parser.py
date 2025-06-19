#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import json
import os
from typing import Optional

from discrete_optimization.datasets import get_data_home
from discrete_optimization.workforce.allocation.allocation_problem_utils import (
    cut_number_of_team,
)
from discrete_optimization.workforce.allocation.problem import TeamAllocationProblem
from discrete_optimization.workforce.scheduling.alloc_scheduling_utils import (
    build_allocation_problem_from_scheduling,
)
from discrete_optimization.workforce.scheduling.parser import parse_json_to_problem


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for tsp.

    Params:
        data_folder: folder where datasets for tsp whould be find.
            If None, we look in "workforce" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/workforce"

    try:
        files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    except FileNotFoundError:
        files = []
    return [os.path.abspath(os.path.join(data_folder, f)) for f in files]


def parse_to_allocation_problem(
    json_path: str, multiobjective: bool = True
) -> TeamAllocationProblem:
    return build_allocation_problem_from_scheduling(
        parse_json_to_problem(json_path), solution=None, multiobjective=multiobjective
    )


def parse_to_allocation_problem_additional_constraint(
    json_path: str, multiobjective: bool = True
) -> TeamAllocationProblem:
    allocation_pb = build_allocation_problem_from_scheduling(
        parse_json_to_problem(json_path), solution=None, multiobjective=multiobjective
    )
    d = json.load(open(json_path, "r"))
    if "additional_constraint" in d:
        if "nb_teams_bounds" in d["additional_constraint"]:
            allocation_pb.allocation_additional_constraint.nb_max_teams = d[
                "additional_constraint"
            ]["nb_teams_bounds"][1]
        if "team_used_constraint" in d["additional_constraint"]:
            subset_teams = [
                int(t)
                for t in d["additional_constraint"]["team_used_constraint"]
                if d["additional_constraint"]["team_used_constraint"][t] != False
            ]
            if len(subset_teams) > 0:
                subset_teams = [allocation_pb.teams_name[i] for i in subset_teams]
            print(subset_teams)
            return cut_number_of_team(
                team_allocation=allocation_pb,
                nb_teams_keep=None,
                subset_teams_keep=subset_teams,
            )
    return allocation_pb
