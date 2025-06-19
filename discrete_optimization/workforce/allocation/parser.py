import json

from discrete_optimization.workforce.allocation.allocation_problem_utils import (
    cut_number_of_team,
)
from discrete_optimization.workforce.allocation.problem import TeamAllocationProblem
from discrete_optimization.workforce.scheduling.alloc_scheduling_utils import (
    build_allocation_problem_from_scheduling,
)
from discrete_optimization.workforce.scheduling.parser import parse_json_to_problem


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
