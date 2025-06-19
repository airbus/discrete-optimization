import json
import os
from typing import Hashable

import numpy as np

from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    SpecialConstraintsDescription,
    TasksDescription,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    AdditionalCPConstraints,
)

this_folder = os.path.abspath(os.path.dirname(__file__))
data_folder = os.path.join(this_folder, "../data")


def update_calendars_with_disruptions(
    list_drop_resource,
    teams: list[Hashable],
    calendar_team: dict[Hashable, list[tuple[int, int]]],
):
    for from_time, to_time, index_team_name in list_drop_resource:
        nc = []
        team_name = teams[index_team_name]
        for st, end in calendar_team[team_name]:
            if st <= from_time and end >= to_time:
                nc.append((st, from_time))
                nc.append((to_time, end))
            elif st <= from_time and end <= from_time:
                nc.append((st, end))
            elif from_time <= st <= to_time:
                # nc.append((st, min(end, to_time)))
                if end > to_time:
                    nc.append((to_time, end))
            else:
                nc.append((st, end))
        calendar_team[team_name] = nc
    return calendar_team


def parse_json_to_problem(json_path: str) -> AllocSchedulingProblem:
    d = json.load(open(json_path, "r"))
    # Keys are teams, tasks, calendar, teams_to_index,
    # tasks_data, same_allocation, compatible_teams, start_window, end_window
    horizon = int(max(d["end_window"][task][1] for task in d["end_window"]) + 100)
    if "disruption" in d:
        if d["disruption"]["type"] == "resource":
            d["calendar"] = update_calendars_with_disruptions(
                list_drop_resource=d["disruption"]["disruptions"][:10],
                teams=d["teams"],
                calendar_team=d["calendar"],
            )
    pb = AllocSchedulingProblem(
        team_names=d["teams"],
        calendar_team=d["calendar"],
        tasks_list=d["tasks"],
        tasks_data={
            t: TasksDescription(duration_task=d["tasks_data"][t]["duration"])
            for t in d["tasks_data"]
        },
        same_allocation=[set(x) for x in d["same_allocation"]],
        precedence_constraints=d["successors"],
        available_team_for_activity=d["compatible_teams"],
        start_window=d["start_window"],
        end_window=d["end_window"],
        original_start=d["original_start"],
        original_end=d["original_end"],
        horizon_start_shift=d["horizon_shift"],
        resources_list=[],
        resources_capacity=None,
        horizon=horizon,
    )
    if "base_solution" in d:
        sol = AllocSchedulingSolution(
            problem=pb,
            schedule=np.array(d["base_solution"]["schedule"]),
            allocation=np.array(d["base_solution"]["allocation"]),
        )
        pb.base_solution = sol

    if "additional_constraint" in d:
        if "team_used_constraint" in d["additional_constraint"]:
            d["additional_constraint"]["team_used_constraint"] = {
                int(x): d["additional_constraint"]["team_used_constraint"][x]
                for x in d["additional_constraint"]["team_used_constraint"]
            }
        additional_constraint = AdditionalCPConstraints(**d["additional_constraint"])
        pb.additional_constraint = additional_constraint
    return pb


if __name__ == "__main__":
    pb = parse_json_to_problem(
        os.path.join(
            data_folder, "jsons/2023-01-01 05:00:00-2023-01-02 05:00:00-problem.json"
        )
    )
    print(pb)
