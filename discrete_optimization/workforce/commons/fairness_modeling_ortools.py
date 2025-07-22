#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Optional, Union

from ortools.sat.python.cp_model import CpModel, IntVar

from discrete_optimization.workforce.commons.fairness_modeling import (
    ModelisationDispersion,
)


def cumulate_value_per_teams(
    used_team: Union[dict[int, IntVar], list[IntVar]],
    allocation_variables: Union[list[list[IntVar]], list[dict[int, IntVar]]],
    value_per_task: list[int],
    cp_model: CpModel,
    number_teams: Optional[int] = None,
    name_value: Optional[str] = "",
):
    """
    - used_team :
        - either a dict : {index_team: boolean_var} for index_team in range(nb_teams)
        - or simply an array of size (nb_teams)
    - allocation_variables :
        - either a 2D array of vars : [index_activity, index_team] or a list of size (nb_activities) of dict : {index_team: boolean} for index_team in "available_teams for a given task"

    - value per task :
        - list of values linked to task, to aggregate for a given team.
    """
    if number_teams is None:
        number_teams = len(used_team)
    upper_bound_values = int(sum(value_per_task))
    workload_per_team = [
        cp_model.NewIntVar(
            lb=0, ub=upper_bound_values, name=f"cumulated_value_{name_value}_{i}"
        )
        for i in range(number_teams)
    ]
    for index_team in range(number_teams):
        team_load = 0
        if isinstance(allocation_variables[0], dict):
            team_load = sum(
                [
                    allocation_variables[i][index_team] * value_per_task[i]
                    for i in range(len(allocation_variables))
                    if index_team in allocation_variables[i]
                ]
            )
        if isinstance(allocation_variables[0], list):
            team_load = sum(
                [
                    allocation_variables[i][index_team] * value_per_task[i]
                    for i in range(len(allocation_variables))
                ]
            )
        cp_model.Add(workload_per_team[index_team] == team_load).OnlyEnforceIf(
            used_team[index_team]
        )
    return {"workload_per_team": workload_per_team}


def cumulate_value_per_teams_version_2(
    used_team: Union[dict[int, IntVar], list[IntVar]],
    allocation_variables: Union[list[list[IntVar]], list[dict[int, IntVar]]],
    value_per_task: list[int],
    cp_model: CpModel,
    number_teams: Optional[int] = None,
    name_value: Optional[str] = "",
):
    """
    - used_team :
        - either a dict : {index_team: boolean_var} for index_team in range(nb_teams)
        - or simply an array of size (nb_teams)
    - allocation_variables :
        - either a 2D array of vars : [index_activity, index_team] or a list of size (nb_activities) of dict : {index_team: boolean} for index_team in "available_teams for a given task"

    - value per task :
        - list of values linked to task, to aggregate for a given team.
    """
    if number_teams is None:
        number_teams = len(used_team)
    upper_bound_values = int(sum(value_per_task))
    workload_per_team = [
        cp_model.NewIntVar(
            lb=0, ub=upper_bound_values, name=f"cumulated_value_{name_value}_{i}"
        )
        for i in range(number_teams)
    ]
    workload_per_team_non_zeros = [
        cp_model.NewIntVar(
            lb=0, ub=upper_bound_values, name=f"cumulated_value_nz_{name_value}_{i}"
        )
        for i in range(number_teams)
    ]
    for index_team in range(number_teams):
        team_load = 0
        if isinstance(allocation_variables[0], dict):
            team_load = sum(
                [
                    allocation_variables[i][index_team] * value_per_task[i]
                    for i in range(len(allocation_variables))
                    if index_team in allocation_variables[i]
                ]
            )
        if isinstance(allocation_variables[0], list):
            team_load = sum(
                [
                    allocation_variables[i][index_team] * value_per_task[i]
                    for i in range(len(allocation_variables))
                ]
            )
        cp_model.Add(team_load == workload_per_team[index_team])
        cp_model.Add(
            workload_per_team_non_zeros[index_team] == team_load
        ).OnlyEnforceIf(used_team[index_team])
        cp_model.Add(
            workload_per_team_non_zeros[index_team] == upper_bound_values
        ).OnlyEnforceIf(used_team[index_team].Not())
    return {
        "workload_per_team": workload_per_team,
        "workload_per_team_nz": workload_per_team_non_zeros,
    }


def define_fairness_criteria_from_cumulated_value(
    used_team: Union[dict[int, IntVar], list[IntVar]],
    cumulated_value_per_team: list[IntVar],
    value_per_task: list[int],
    modelisation_dispersion: ModelisationDispersion,
    cp_model: CpModel,
    cumulated_value_per_team_nz: Optional[list[IntVar]] = None,
    name_value: Optional[str] = "",
):
    if (
        modelisation_dispersion
        == ModelisationDispersion.EXACT_MODELING_WITH_IMPLICATION
    ):
        upper_bound = sum(value_per_task)
        max_value = cp_model.NewIntVar(
            lb=0,  # upper_bound//len(used_team),
            ub=upper_bound,
            name=f"max_value_{name_value}",
        )
        min_value = cp_model.NewIntVar(
            lb=0,  # upper_bound//len(used_team),
            ub=upper_bound,
            name=f"min_value_{name_value}",
        )
        cp_model.AddMaxEquality(max_value, cumulated_value_per_team)
        cp_model.AddMinEquality(min_value, cumulated_value_per_team)
        return {
            "obj": max_value - min_value,
            f"max_value_{name_value}": max_value,
            f"min_value_{name_value}": min_value,
        }
    elif (
        modelisation_dispersion == ModelisationDispersion.EXACT_MODELING_DUPLICATED_VARS
    ):
        upper_bound = sum(value_per_task)
        max_value = cp_model.NewIntVar(
            lb=0,  # upper_bound // len(used_team),
            ub=upper_bound,
            name=f"max_value_{name_value}",
        )
        min_value = cp_model.NewIntVar(
            lb=0,  # upper_bound // len(used_team),
            ub=upper_bound,
            name=f"min_value_{name_value}",
        )
        cp_model.AddMaxEquality(max_value, cumulated_value_per_team)
        cp_model.AddMinEquality(min_value, cumulated_value_per_team_nz)
        return {
            "obj": max_value - min_value,
            f"max_value_{name_value}": max_value,
            f"min_value_{name_value}": min_value,
        }
    elif modelisation_dispersion == ModelisationDispersion.MAX_DIFF:
        upper_bound = sum(value_per_task)
        max_diff = cp_model.NewIntVar(
            lb=0, ub=upper_bound, name=f"max_diff_{name_value}"
        )
        cp_model.AddMaxEquality(
            max_diff,
            [x - y for x in cumulated_value_per_team for y in cumulated_value_per_team],
        )
        return {"obj": max_diff, f"max_diff_{name_value}": max_diff, "constraints": []}
    elif modelisation_dispersion == ModelisationDispersion.PROXY_MAX_MIN:
        upper_bound = sum(value_per_task)
        max_value = cp_model.NewIntVar(
            lb=0,  # upper_bound // len(used_team),
            ub=upper_bound,
            name=f"max_value_{name_value}",
        )
        cp_model.AddMaxEquality(max_value, cumulated_value_per_team)
        return {"obj": max_value, f"max_value_{name_value}": max_value}
    elif modelisation_dispersion == ModelisationDispersion.PROXY_MIN_MAX:
        upper_bound = sum(value_per_task)
        min_value = cp_model.NewIntVar(
            lb=0,  # upper_bound // len(used_team),
            ub=upper_bound,
            name=f"min_value_{name_value}",
        )
        cp_model.AddMinEquality(min_value, cumulated_value_per_team)
        return {"obj": -min_value, f"min_value_{name_value}": min_value}
    elif modelisation_dispersion == ModelisationDispersion.PROXY_SUM:
        upper_bound = sum(value_per_task)
        abs_deltas = [
            {
                j: cp_model.NewIntVar(lb=0, ub=upper_bound, name=f"delta_{i}_{j}")
                for j in range(i + 1, len(cumulated_value_per_team))
            }
            for i in range(len(cumulated_value_per_team))
        ]
        for i in range(len(abs_deltas)):
            for j in abs_deltas[i]:
                cp_model.AddAbsEquality(
                    abs_deltas[i][j],
                    cumulated_value_per_team[i] - cumulated_value_per_team[j],
                )

        return {
            "obj": sum(
                [
                    abs_deltas[i][j]
                    for i in range(len(abs_deltas))
                    for j in abs_deltas[i]
                ]
            )
        }
    elif modelisation_dispersion == ModelisationDispersion.PROXY_SLACK:
        some_expected_value = cp_model.NewIntVar(
            lb=0, ub=sum(value_per_task), name=f"expected_value_{name_value}"
        )
        slack = cp_model.NewIntVar(
            lb=0, ub=sum(value_per_task), name=f"slack_{name_value}"
        )
        constraints = []
        for index_team in range(len(cumulated_value_per_team)):
            (
                cp_model.Add(
                    cumulated_value_per_team[index_team] <= some_expected_value + slack
                ).OnlyEnforceIf(used_team[index_team])
            )
            (
                cp_model.Add(
                    cumulated_value_per_team[index_team] >= some_expected_value - slack
                ).OnlyEnforceIf(used_team[index_team])
            )
        return {"obj": slack}
    else:
        raise NotImplementedError(f"Method {modelisation_dispersion} unknown")


def model_fairness(
    used_team: Union[dict[int, IntVar], list[IntVar]],
    allocation_variables: Union[list[list[IntVar]], list[dict[int, IntVar]]],
    value_per_task: list[int],
    modelisation_dispersion: ModelisationDispersion,
    cp_model: CpModel,
    number_teams: Optional[int] = None,
    name_value: Optional[str] = "",
):
    if modelisation_dispersion == ModelisationDispersion.EXACT_MODELING_DUPLICATED_VARS:
        dict_cumulated_variable = cumulate_value_per_teams_version_2(
            used_team=used_team,
            allocation_variables=allocation_variables,
            value_per_task=value_per_task,
            number_teams=number_teams,
            cp_model=cp_model,
            name_value=name_value,
        )
        cumulated_value_per_team = dict_cumulated_variable["workload_per_team"]
        fairness_dict = define_fairness_criteria_from_cumulated_value(
            used_team=used_team,
            cumulated_value_per_team=cumulated_value_per_team,
            value_per_task=value_per_task,
            cumulated_value_per_team_nz=dict_cumulated_variable["workload_per_team_nz"],
            modelisation_dispersion=modelisation_dispersion,
            cp_model=cp_model,
            name_value=name_value,
        )
    else:
        dict_cumulated_variable = cumulate_value_per_teams(
            used_team=used_team,
            allocation_variables=allocation_variables,
            value_per_task=value_per_task,
            number_teams=number_teams,
            cp_model=cp_model,
            name_value=name_value,
        )
        cumulated_value_per_team = dict_cumulated_variable["workload_per_team"]
        fairness_dict = define_fairness_criteria_from_cumulated_value(
            used_team=used_team,
            cumulated_value_per_team=cumulated_value_per_team,
            value_per_task=value_per_task,
            modelisation_dispersion=modelisation_dispersion,
            cp_model=cp_model,
            name_value=name_value,
        )
    d = fairness_dict
    d["cumulated_value"] = cumulated_value_per_team
    return d
