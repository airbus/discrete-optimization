#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Optional, Union

import cpmpy as cp
from cpmpy.expressions.variables import NDVarArray

from discrete_optimization.workforce.commons.fairness_modeling import (
    ModelisationDispersion,
)


def cumulate_value_per_teams(
    used_team: Union[dict[int, cp.model.Expression], NDVarArray],
    allocation_variables: Union[NDVarArray, list[dict[int, cp.model.Expression]]],
    value_per_task: list[int],
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
    workload_per_team = cp.intvar(
        lb=0,
        ub=upper_bound_values,
        shape=(number_teams,),
        name=f"cumulated_value_{name_value}",
    )
    constraints = []
    for index_team in range(number_teams):
        team_load = 0
        if isinstance(allocation_variables, NDVarArray):
            team_load = cp.sum(
                [
                    allocation_variables[i, index_team] * value_per_task[i]
                    for i in range(allocation_variables.shape[0])
                ]
            )
        if isinstance(allocation_variables, list):
            team_load = cp.sum(
                [
                    allocation_variables[i][index_team] * value_per_task[i]
                    for i in range(len(allocation_variables))
                    if index_team in allocation_variables[i]
                ]
            )
        constraints.append(
            used_team[index_team].implies(team_load == workload_per_team[index_team])
        )
    return {"workload_per_team": workload_per_team, "constraints": constraints}


def cumulate_value_per_teams_version_2(
    used_team: Union[dict[int, cp.model.Expression], NDVarArray],
    allocation_variables: Union[NDVarArray, list[dict[int, cp.model.Expression]]],
    value_per_task: list[int],
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
    workload_per_team = cp.intvar(
        lb=0,
        ub=upper_bound_values,
        shape=(number_teams,),
        name=f"cumulated_value_{name_value}",
    )
    workload_per_team_non_zeros = cp.intvar(
        lb=0,
        ub=upper_bound_values,
        shape=(number_teams,),
        name=f"cumulated_value_non_zeros_{name_value}",
    )
    constraints = []
    for index_team in range(number_teams):
        team_load = 0
        if isinstance(allocation_variables, NDVarArray):
            team_load = cp.sum(
                [
                    allocation_variables[i, index_team] * value_per_task[i]
                    for i in range(allocation_variables.shape[0])
                ]
            )
        if isinstance(allocation_variables, list):
            team_load = cp.sum(
                [
                    allocation_variables[i][index_team] * value_per_task[i]
                    for i in range(len(allocation_variables))
                    if index_team in allocation_variables[i]
                ]
            )
        constraints.append(team_load == workload_per_team[index_team])
        constraints.append(
            used_team[index_team].implies(
                team_load == workload_per_team_non_zeros[index_team]
            )
        )
        constraints.append(
            (~used_team[index_team]).implies(
                upper_bound_values == workload_per_team_non_zeros[index_team]
            )
        )
    return {
        "workload_per_team": workload_per_team,
        "workload_per_team_nz": workload_per_team_non_zeros,
        "constraints": constraints,
    }


def define_fairness_criteria_from_cumulated_value(
    used_team: Union[dict[int, cp.model.Expression], NDVarArray],
    cumulated_value_per_team: NDVarArray,
    value_per_task: list[int],
    modelisation_dispersion: ModelisationDispersion,
    cumulated_value_per_team_nz: NDVarArray = None,
    name_value: Optional[str] = "",
):
    if (
        modelisation_dispersion
        == ModelisationDispersion.EXACT_MODELING_WITH_IMPLICATION
    ):
        return {
            "obj": cp.max(cumulated_value_per_team) - cp.min(cumulated_value_per_team),
            "constraints": [],
        }
    elif (
        modelisation_dispersion == ModelisationDispersion.EXACT_MODELING_DUPLICATED_VARS
    ):
        return {
            "obj": cp.max(cumulated_value_per_team)
            - cp.min(cumulated_value_per_team_nz),
            "constraints": [],
        }
    elif modelisation_dispersion == ModelisationDispersion.MAX_DIFF:
        return {
            "obj": cp.max(
                [
                    x - y
                    for x in cumulated_value_per_team
                    for y in cumulated_value_per_team
                ]
            ),
            "constraints": [],
        }
    elif modelisation_dispersion == ModelisationDispersion.PROXY_MIN_MAX:
        return {"obj": cp.max(cumulated_value_per_team), "constraints": []}
    elif modelisation_dispersion == ModelisationDispersion.PROXY_MAX_MIN:
        return {"obj": -cp.min(cumulated_value_per_team), "constraints": []}
    elif modelisation_dispersion == ModelisationDispersion.PROXY_SUM:
        return {
            "obj": cp.sum(
                [
                    abs(x - y)
                    for x in cumulated_value_per_team
                    for y in cumulated_value_per_team
                ]
            ),
            "constraints": [],
        }
    elif modelisation_dispersion == ModelisationDispersion.PROXY_SLACK:
        some_expected_value = cp.intvar(
            lb=0, ub=sum(value_per_task), name=f"expected_value_{name_value}"
        )
        slack = cp.intvar(lb=0, ub=sum(value_per_task), name=f"slack_{name_value}")
        constraints = []
        for index_team in range(len(cumulated_value_per_team)):
            constraints.append(
                used_team[index_team].implies(
                    cumulated_value_per_team[index_team] <= some_expected_value + slack
                )
            )
            constraints.append(
                used_team[index_team].implies(
                    cumulated_value_per_team[index_team] >= some_expected_value - slack
                )
            )
        return {"obj": slack, "constraints": constraints}

    else:
        raise NotImplementedError(f"Method {modelisation_dispersion} unknown")


def model_fairness(
    used_team: Union[dict[int, cp.model.Expression], NDVarArray],
    allocation_variables: Union[NDVarArray, list[dict[int, cp.model.Expression]]],
    value_per_task: list[int],
    modelisation_dispersion: ModelisationDispersion,
    number_teams: Optional[int] = None,
    name_value: Optional[str] = "",
):
    constraints_to_add = []
    if modelisation_dispersion == ModelisationDispersion.EXACT_MODELING_DUPLICATED_VARS:
        dict_cumulated_variable = cumulate_value_per_teams_version_2(
            used_team=used_team,
            allocation_variables=allocation_variables,
            value_per_task=value_per_task,
            number_teams=number_teams,
            name_value=name_value,
        )
        constraints_to_add.extend(dict_cumulated_variable["constraints"])
        cumulated_value_per_team = dict_cumulated_variable["workload_per_team"]
        fairness_dict = define_fairness_criteria_from_cumulated_value(
            used_team=used_team,
            cumulated_value_per_team=cumulated_value_per_team,
            value_per_task=value_per_task,
            cumulated_value_per_team_nz=dict_cumulated_variable["workload_per_team_nz"],
            modelisation_dispersion=modelisation_dispersion,
            name_value=name_value,
        )
    else:
        dict_cumulated_variable = cumulate_value_per_teams(
            used_team=used_team,
            allocation_variables=allocation_variables,
            value_per_task=value_per_task,
            number_teams=number_teams,
            name_value=name_value,
        )
        constraints_to_add.extend(dict_cumulated_variable["constraints"])
        cumulated_value_per_team = dict_cumulated_variable["workload_per_team"]
        fairness_dict = define_fairness_criteria_from_cumulated_value(
            used_team=used_team,
            cumulated_value_per_team=cumulated_value_per_team,
            value_per_task=value_per_task,
            modelisation_dispersion=modelisation_dispersion,
            name_value=name_value,
        )
    constraints_to_add.extend(fairness_dict["constraints"])
    return {
        "obj": fairness_dict["obj"],
        "constraints": constraints_to_add,
        "cumulated_value": cumulated_value_per_team,
    }
