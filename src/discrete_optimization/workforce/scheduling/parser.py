#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import json
import logging
import os
from typing import Hashable, Optional

import numpy as np

from discrete_optimization.datasets import get_data_home
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    TasksDescription,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    AdditionalCPConstraints,
)

logger = logging.getLogger(__name__)


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for tsp.

    Params:
        data_folder: folder where datasets for tsp whould be find.
            If None, we look in "tsp" subdirectory of `data_home`.
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
