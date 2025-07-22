#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import json
import logging

from discrete_optimization.workforce.allocation.problem import TeamAllocationProblem
from discrete_optimization.workforce.allocation.utils import cut_number_of_team
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.utils import (
    build_allocation_problem_from_scheduling,
)

logger = logging.getLogger(__name__)


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
            logger.debug(subset_teams)
            return cut_number_of_team(
                team_allocation=allocation_pb,
                nb_teams_keep=None,
                subset_teams_keep=subset_teams,
            )
    return allocation_pb
