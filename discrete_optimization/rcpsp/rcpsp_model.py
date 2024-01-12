#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TupleFitness,
    TypeAttribute,
    TypeObjective,
)
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.rcpsp.fast_function_rcpsp import (
    compute_mean_ressource,
    sgs_fast,
    sgs_fast_partial_schedule_incomplete_permutation_tasks,
)
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import intersect
from discrete_optimization.rcpsp.special_constraints import (
    PairModeConstraint,
    SpecialConstraintsDescription,
)

logger = logging.getLogger(__name__)


class ScheduleGenerationScheme(Enum):
    SERIAL_SGS = 0
    PARALLEL_SGS = 1


class RCPSPModel(Problem):
    """

    Attributes:
        resources:
        non_renewable_resources:
        mode_details:
        successors:
        horizon:
        horizon_multiplier:
        tasks_list:
        source_task:
        sink_task:
        name_task:
        n_jobs (int):
        n_jobs_non_dummy (int):   excluding dummy activities Start (0) and End (n)
        special_constraints:
        do_special_constraints (bool):
        relax_the_start_at_end (bool): relax some conditions only if do_special_constraints
        fixed_permutation (Optional[List[int]]):
        fixed_modes (Optional[List[int]]):

    Args:
        resources: {resource_name: number_of_resource}
        non_renewable_resources: [resource_name3, resource_name4]
        mode_details:  {job_id: {mode_id: {resource_name1: number_of_resources_needed, resource_name2: ...}}   one key being "duration"
        successors:
        horizon:
        horizon_multiplier:
        tasks_list:  {task_id: list of successor task ids}
        source_task:
        sink_task:
        name_task:
        special_constraints:
        relax_the_start_at_end:
        fixed_permutation:
        fixed_modes:
        **kwargs:

    """

    sgs: ScheduleGenerationScheme

    def __init__(
        self,
        resources: Dict[str, Union[int, List[int]]],
        non_renewable_resources: List[str],
        mode_details: Dict[Hashable, Dict[int, Dict[str, int]]],
        successors: Dict[Hashable, List[Hashable]],
        horizon: int,
        horizon_multiplier: int = 1,
        tasks_list: Optional[List[Hashable]] = None,
        source_task: Optional[Hashable] = None,
        sink_task: Optional[Hashable] = None,
        name_task: Optional[Dict[Hashable, str]] = None,
        calendar_details: Optional[Dict[str, List[List[int]]]] = None,
        special_constraints: Optional[SpecialConstraintsDescription] = None,
        relax_the_start_at_end: bool = True,
        fixed_permutation: Optional[List[int]] = None,
        fixed_modes: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        self.resources = resources
        self.resources_list = list(self.resources.keys())
        self.non_renewable_resources = non_renewable_resources
        self.mode_details = mode_details
        self.successors = successors
        self.horizon = horizon
        self.horizon_multiplier = horizon_multiplier
        self.calendar_details = calendar_details
        if name_task is None:
            self.name_task = {x: str(x) for x in self.mode_details}
        else:
            self.name_task = name_task
        if tasks_list is None:
            self.tasks_list = list(self.mode_details.keys())
        else:
            self.tasks_list = tasks_list
        self.n_jobs = len(self.mode_details.keys())
        self.n_jobs_non_dummy = self.n_jobs - 2
        self.index_task = {self.tasks_list[i]: i for i in range(self.n_jobs)}
        if source_task is None:
            if all((isinstance(t, int) for t in self.tasks_list)):
                self.source_task = min(self.tasks_list)  # type: ignore
            else:
                raise ValueError(
                    "source_task cannot be None if tasks id given in tasks_list are not all integers."
                )
        else:
            self.source_task = source_task
        if sink_task is None:
            if all((isinstance(t, int) for t in self.tasks_list)):
                self.sink_task = max(self.tasks_list)  # type: ignore
            else:
                raise ValueError(
                    "sink_task cannot be None if tasks id given in tasks_list are not all integers."
                )
        else:
            self.sink_task = sink_task

        self.tasks_list_non_dummy = [
            t for t in self.tasks_list if t not in {self.source_task, self.sink_task}
        ]
        self.index_task_non_dummy = {
            self.tasks_list_non_dummy[i]: i for i in range(self.n_jobs_non_dummy)
        }
        self.max_number_of_mode = max(
            [len(self.mode_details[key1].keys()) for key1 in self.mode_details.keys()]
        )
        self.is_multimode = self.max_number_of_mode > 1
        self.is_calendar = False
        if any(isinstance(self.resources[res], Iterable) for res in self.resources):
            self.is_calendar = (
                max(
                    (
                        len(
                            {self.resources[res]}
                            if isinstance(self.resources[res], int)
                            else set(self.resources[res])  # type: ignore
                        )
                        for res in self.resources
                    )
                )
                > 1
            )
            if not self.is_calendar:
                self.resources = {
                    r: self.resources[r]
                    if isinstance(self.resources[r], int)
                    else self.resources[r][0]  # type: ignore
                    for r in self.resources
                }
        (
            self.func_sgs,
            self.func_sgs_2,
            self.compute_mean_resource,
        ) = create_np_data_and_jit_functions(self)
        self.costs: Dict[str, bool] = {
            "makespan": True,
            "mean_resource_reserve": kwargs.get("mean_resource_reserve", False),
        }

        if special_constraints is None:
            self.do_special_constraints = False
            self.special_constraints = SpecialConstraintsDescription()
        else:
            self.do_special_constraints = True
            self.special_constraints = special_constraints
            predecessors_dict: Dict[Hashable, List[Hashable]] = {
                task: [] for task in self.successors
            }
            for task in self.successors:
                for stask in self.successors[task]:
                    predecessors_dict[stask] += [task]
            for t1, t2 in self.special_constraints.start_at_end:
                if t2 not in self.successors[t1]:
                    self.successors[t1].append(t2)
            for t1, t2, off in self.special_constraints.start_at_end_plus_offset:
                if t2 not in self.successors[t1]:
                    self.successors[t1].append(t2)
            for t1, t2 in self.special_constraints.start_together:
                for predt1 in predecessors_dict[t1]:
                    if t2 not in self.successors[predt1]:
                        self.successors[predt1] += [t2]
                for predt2 in predecessors_dict[t2]:
                    if t1 not in self.successors[predt2]:
                        self.successors[predt2] += [t1]
        self.graph = self.compute_graph(
            compute_predecessors=self.do_special_constraints
        )
        if self.do_special_constraints:
            self.predecessors = self.graph.predecessors_dict
        self.relax_the_start_at_end = relax_the_start_at_end
        self.fixed_permutation = fixed_permutation
        self.fixed_modes = fixed_modes

    def update_functions(self) -> None:
        (
            self.func_sgs,
            self.func_sgs_2,
            self.compute_mean_resource,
        ) = create_np_data_and_jit_functions(rcpsp_problem=self)

    def is_rcpsp_multimode(self) -> bool:
        return self.is_multimode

    def is_varying_resource(self) -> bool:
        return self.is_calendar

    def is_preemptive(self) -> bool:
        return False

    def is_multiskill(self) -> bool:
        return False

    def includes_special_constraint(self) -> bool:
        return self.do_special_constraints

    def get_resource_names(self) -> List[str]:
        return self.resources_list

    def get_tasks_list(self) -> List[Hashable]:
        return self.tasks_list

    def get_resource_availability_array(self, res: str) -> List[int]:
        if self.is_varying_resource() and not isinstance(self.resources[res], int):
            return self.resources[res]  # type: ignore
        else:
            return self.horizon * [self.resources[res]]  # type: ignore

    def compute_graph(self, compute_predecessors: bool = False) -> Graph:
        nodes: List[Tuple[Hashable, Dict[str, Any]]] = [
            (
                n,
                {
                    str(mode): self.mode_details[n][mode]["duration"]
                    for mode in self.mode_details[n]
                },
            )
            for n in self.tasks_list
        ]
        edges: List[Tuple[Hashable, Hashable, Dict[str, Any]]] = []
        for n in self.successors:
            for succ in self.successors[n]:
                edges += [(n, succ, {})]
        return Graph(
            nodes, edges, compute_predecessors=compute_predecessors, undirected=False
        )

    def evaluate_function(self, rcpsp_sol: RCPSPSolution) -> Tuple[int, float, int]:
        if rcpsp_sol._schedule_to_recompute:
            rcpsp_sol.generate_schedule_from_permutation_serial_sgs()
        makespan = rcpsp_sol.rcpsp_schedule[self.sink_task]["end_time"]
        if self.costs["mean_resource_reserve"]:
            obj_mean_resource_reserve = rcpsp_sol.compute_mean_resource_reserve()
        else:
            obj_mean_resource_reserve = 0.0
        if self.do_special_constraints:
            penalty = evaluate_constraints(
                solution=rcpsp_sol, constraints=self.special_constraints
            )
        else:
            penalty = 0

        return makespan, obj_mean_resource_reserve, penalty

    def evaluate_from_encoding(
        self, int_vector: List[int], encoding_name: str
    ) -> Dict[str, float]:
        if encoding_name == "rcpsp_permutation":
            if self.fixed_modes is None:
                rcpsp_modes = [1 for i in range(self.n_jobs_non_dummy)]
            else:
                rcpsp_modes = self.fixed_modes
            rcpsp_sol = RCPSPSolution(
                problem=self, rcpsp_permutation=int_vector, rcpsp_modes=rcpsp_modes
            )
        elif encoding_name == "rcpsp_modes":
            if self.fixed_permutation is not None:
                rcpsp_sol = RCPSPSolution(
                    problem=self,
                    rcpsp_permutation=self.fixed_permutation,
                    rcpsp_modes=int_vector,
                )
            else:
                raise RuntimeError(
                    "Encoding rcpsp_modes possible "
                    "only if self.fixed_permutation is not None"
                )
        else:
            raise NotImplementedError(f"Encoding {encoding_name} not implemented")
        objectives = self.evaluate(rcpsp_sol)
        return objectives

    def evaluate(self, rcpsp_sol: RCPSPSolution) -> Dict[str, float]:  # type: ignore
        obj_makespan, obj_mean_resource_reserve, penalty = self.evaluate_function(
            rcpsp_sol
        )
        return {
            "makespan": float(obj_makespan),
            "mean_resource_reserve": obj_mean_resource_reserve,
            "constraint_penalty": float(penalty),
        }

    def evaluate_mobj(self, rcpsp_sol: RCPSPSolution) -> TupleFitness:  # type: ignore
        return self.evaluate_mobj_from_dict(self.evaluate(rcpsp_sol))

    def evaluate_mobj_from_dict(self, dict_values: Dict[str, float]) -> TupleFitness:
        return TupleFitness(
            np.array([-dict_values["makespan"], dict_values["mean_resource_reserve"]]),
            2,
        )

    def build_mode_dict(
        self, rcpsp_modes_from_solution: List[int]
    ) -> Dict[Hashable, int]:
        modes_dict = {
            self.tasks_list_non_dummy[i]: rcpsp_modes_from_solution[i]
            for i in range(self.n_jobs_non_dummy)
        }
        modes_dict[self.source_task] = 1
        modes_dict[self.sink_task] = 1
        return modes_dict

    def build_mode_array(self, rcpsp_modes_from_solution: List[int]) -> List[int]:
        modes_dict = self.build_mode_dict(
            rcpsp_modes_from_solution=rcpsp_modes_from_solution
        )
        return [modes_dict[t] for t in self.tasks_list]

    def return_index_task(self, task: Hashable, offset: int = 0) -> int:
        return self.index_task[task] + offset

    def satisfy(self, rcpsp_sol: RCPSPSolution) -> bool:  # type: ignore
        if rcpsp_sol.rcpsp_schedule_feasible is False:
            logger.debug("Schedule flagged as infeasible when generated")
            return False

        if self.do_special_constraints:
            if not check_solution_with_special_constraints(
                problem=self,
                solution=rcpsp_sol,
                relax_the_start_at_end=self.relax_the_start_at_end,
            ):
                return False

        modes_dict = self.build_mode_dict(
            rcpsp_modes_from_solution=rcpsp_sol.rcpsp_modes
        )
        start_times = [
            rcpsp_sol.rcpsp_schedule[t]["start_time"] for t in rcpsp_sol.rcpsp_schedule
        ]
        for t in start_times:
            resource_usage = {}
            for res in self.resources_list:
                resource_usage[res] = 0
            for act_id in rcpsp_sol.rcpsp_schedule:
                start = rcpsp_sol.rcpsp_schedule[act_id]["start_time"]
                end = rcpsp_sol.rcpsp_schedule[act_id]["end_time"]
                mode = modes_dict[act_id]
                for res in self.resources_list:
                    if start <= t < end:
                        resource_usage[res] += self.mode_details[act_id][mode].get(
                            res, 0
                        )
            for res in self.resources.keys():
                if resource_usage[res] > self.get_resource_available(res, t):
                    logger.debug(
                        [
                            act
                            for act in rcpsp_sol.rcpsp_schedule
                            if rcpsp_sol.rcpsp_schedule[act]["start_time"]
                            <= t
                            < rcpsp_sol.rcpsp_schedule[act]["end_time"]
                        ]
                    )
                    logger.debug(
                        f"Time step resource violation: time: {t} "
                        f"res {res} res_usage: {resource_usage[res]}"
                        f"res_avail: {self.resources[res]}"
                    )
                    return False

        # Check for non-renewable resource violation
        for res in self.non_renewable_resources:
            usage = 0
            for act_id in rcpsp_sol.rcpsp_schedule:
                mode = modes_dict[act_id]
                usage += self.mode_details[act_id][mode][res]
                if usage > self.get_max_resource_capacity(res):
                    logger.debug(
                        f"Non-renewable resource violation: act_id: {act_id}"
                        f"res {res} res_usage: {usage} res_avail: {self.resources[res]}"
                    )
                    return False
        # Check precedences / successors
        for act_id in list(self.successors.keys()):
            for succ_id in self.successors[act_id]:
                start_succ = rcpsp_sol.rcpsp_schedule[succ_id]["start_time"]
                end_pred = rcpsp_sol.rcpsp_schedule[act_id]["end_time"]
                if start_succ < end_pred:
                    logger.debug(
                        f"Precedence relationship broken: {act_id} end at {end_pred} "
                        f"while {succ_id} start at {start_succ}"
                    )
                    return False

        return True

    def __str__(self) -> str:
        val = (
            "I'm a RCPSP model with "
            + str(self.n_jobs)
            + " tasks.."
            + " and ressources ="
            + str(self.resources_list)
        )
        return val

    def get_solution_type(self) -> Type[Solution]:
        return RCPSPSolution

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {
            "rcpsp_permutation": {
                "name": "rcpsp_permutation",
                "type": [TypeAttribute.PERMUTATION, TypeAttribute.PERMUTATION_RCPSP],
                "range": range(self.n_jobs_non_dummy),
                "n": self.n_jobs_non_dummy,
            }
        }
        max_number_modes = max([len(self.mode_details[x]) for x in self.mode_details])
        dict_register["rcpsp_modes"] = {
            "name": "rcpsp_modes",
            "type": [TypeAttribute.LIST_INTEGER],
            "n": self.n_jobs_non_dummy,
            "low": 1,  # integer.
            "up": max_number_modes,  # integer.
            "arity": max_number_modes,
        }
        mode_arity = [
            len(self.mode_details[task]) for task in self.tasks_list_non_dummy
        ]
        dict_register["rcpsp_modes_arity_fix"] = {
            "name": "rcpsp_modes",
            "type": [TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY],
            "n": self.n_jobs_non_dummy,
            "low": 1,
            "up": mode_arity,
            "arities": mode_arity,
        }

        return EncodingRegister(dict_register)

    def get_objective_register(self) -> ObjectiveRegister:
        objective_handling = ObjectiveHandling.SINGLE
        dict_objective = {
            "makespan": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1.0)
        }
        # "mean_resource_reserve": {"type": TypeObjective.OBJECTIVE, "default_weight": 1}}
        if self.do_special_constraints:
            objective_handling = ObjectiveHandling.AGGREGATE
            dict_objective["constraint_penalty"] = ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100.0
            )
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=objective_handling,
            dict_objective_to_doc=dict_objective,
        )

    def compute_resource_consumption(
        self, rcpsp_sol: RCPSPSolution
    ) -> npt.NDArray[np.int_]:
        modes_dict = self.build_mode_dict(rcpsp_sol.rcpsp_modes)
        makespan = rcpsp_sol.rcpsp_schedule[self.sink_task]["end_time"]
        consumptions = np.zeros((len(self.resources), makespan + 1), dtype=np.int_)
        for act_id in rcpsp_sol.rcpsp_schedule:
            for ir in range(len(self.resources)):
                consumptions[
                    ir,
                    rcpsp_sol.rcpsp_schedule[act_id]["start_time"]
                    + 1 : rcpsp_sol.rcpsp_schedule[act_id]["end_time"]
                    + 1,
                ] += self.mode_details[act_id][modes_dict[act_id]][
                    self.resources_list[ir]
                ]
        return consumptions

    def plot_ressource_view(self, rcpsp_sol: RCPSPSolution) -> None:
        consumption = self.compute_resource_consumption(rcpsp_sol=rcpsp_sol)
        fig, ax = plt.subplots(nrows=len(self.resources_list), sharex=True)
        for i in range(len(self.resources_list)):
            ax[i].axhline(
                y=self.resources[self.resources_list[i]], label=self.resources_list[i]
            )
            ax[i].plot(consumption[i, :])
            ax[i].legend()

    def copy(self) -> "RCPSPModel":
        model = RCPSPModel(
            resources=self.resources,
            tasks_list=self.tasks_list,
            source_task=self.source_task,
            sink_task=self.sink_task,
            non_renewable_resources=self.non_renewable_resources,
            mode_details=deepcopy(self.mode_details),
            successors=deepcopy(self.successors),
            horizon=self.horizon,
            horizon_multiplier=self.horizon_multiplier,
            name_task=self.name_task,
            mean_resource_reserve=self.costs.get("mean_resource_reserve", False),
            fixed_modes=self.fixed_modes,
            fixed_permutation=self.fixed_permutation,
        )

        return model

    def get_dummy_solution(self) -> RCPSPSolution:
        sol = RCPSPSolution(
            problem=self,
            rcpsp_permutation=list(range(self.n_jobs_non_dummy)),
            rcpsp_modes=[1 for i in range(self.n_jobs_non_dummy)],
        )
        return sol

    def get_resource_available(self, res: str, time: int) -> int:
        if self.is_calendar:
            return self.resources.get(res, [0])[time]  # type: ignore
        return self.resources.get(res, 0)  # type: ignore

    def get_max_resource_capacity(self, res: str) -> int:
        if self.is_calendar:
            return max(self.resources.get(res, [0]))  # type: ignore
        return self.resources.get(res, 0)  # type: ignore

    def set_fixed_attributes(self, encoding_str: str, sol: RCPSPSolution) -> None:
        att = self.get_attribute_register().dict_attribute_to_type[encoding_str]["name"]
        if att == "rcpsp_modes":
            self.set_fixed_modes(sol.rcpsp_modes)
        elif att == "rcpsp_permutation":
            self.set_fixed_permutation(sol.rcpsp_permutation)

    def set_fixed_modes(self, fixed_modes: List[int]) -> None:
        self.fixed_modes = fixed_modes

    def set_fixed_permutation(self, fixed_permutation: List[int]) -> None:
        self.fixed_permutation = fixed_permutation


def create_np_data_and_jit_functions(
    rcpsp_problem: RCPSPModel,
) -> Tuple[
    Callable[
        ...,
        Tuple[Dict[int, Tuple[int, int]], bool],
    ],
    Callable[
        ...,
        Tuple[Dict[int, Tuple[int, int]], bool],
    ],
    Callable[
        ...,
        float,
    ],
]:
    consumption_array = np.zeros(
        (
            rcpsp_problem.n_jobs,
            rcpsp_problem.max_number_of_mode,
            len(rcpsp_problem.resources_list),
        ),
        dtype=np.int_,
    )
    duration_array = np.zeros(
        (rcpsp_problem.n_jobs, rcpsp_problem.max_number_of_mode), dtype=np.int_
    )
    predecessors = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int_)
    successors = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int_)
    horizon = rcpsp_problem.horizon
    ressource_available = np.zeros(
        (len(rcpsp_problem.resources_list), horizon), dtype=np.int_
    )
    ressource_renewable = np.ones((len(rcpsp_problem.resources_list)), dtype=bool)
    minimum_starting_time_array = np.zeros(rcpsp_problem.n_jobs, dtype=np.int_)

    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        index_mode = 0
        for mode in sorted(
            rcpsp_problem.mode_details[rcpsp_problem.tasks_list[i]].keys()
        ):
            for k in range(len(rcpsp_problem.resources_list)):
                consumption_array[i, index_mode, k] = rcpsp_problem.mode_details[task][
                    mode
                ].get(rcpsp_problem.resources_list[k], 0)
            duration_array[i, index_mode] = rcpsp_problem.mode_details[task][mode][
                "duration"
            ]
            index_mode += 1

    task_index = {rcpsp_problem.tasks_list[i]: i for i in range(rcpsp_problem.n_jobs)}
    for k in range(len(rcpsp_problem.resources_list)):
        if rcpsp_problem.is_varying_resource():
            ressource_available[k, :] = rcpsp_problem.resources[  # type: ignore
                rcpsp_problem.resources_list[k]
            ][: ressource_available.shape[1]]
        else:
            ressource_available[k, :] = np.full(
                ressource_available.shape[1],
                rcpsp_problem.resources[rcpsp_problem.resources_list[k]],
                dtype=np.int_,
            )
        if rcpsp_problem.resources_list[k] in rcpsp_problem.non_renewable_resources:
            ressource_renewable[k] = False

    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        for s in rcpsp_problem.successors[task]:
            index_s = task_index[s]
            predecessors[index_s, i] = 1
            successors[i, index_s] = 1

    if "special_constraints" in rcpsp_problem.__dict__.keys():
        for t in rcpsp_problem.special_constraints.start_times_window:
            if rcpsp_problem.special_constraints.start_times_window[t][0] is not None:
                minimum_starting_time_array[
                    rcpsp_problem.index_task[t]
                ] = rcpsp_problem.special_constraints.start_times_window[t][0]
    func_sgs = partial(
        sgs_fast,
        consumption_array=consumption_array,
        duration_array=duration_array,
        predecessors=predecessors,
        successors=successors,
        horizon=horizon,
        ressource_available=ressource_available,
        ressource_renewable=ressource_renewable,
        minimum_starting_time_array=minimum_starting_time_array,
    )
    func_sgs_2 = partial(
        sgs_fast_partial_schedule_incomplete_permutation_tasks,
        consumption_array=consumption_array,
        duration_array=duration_array,
        predecessors=predecessors,
        successors=successors,
        horizon=horizon,
        ressource_available=ressource_available,
        ressource_renewable=ressource_renewable,
        minimum_starting_time_array=minimum_starting_time_array,
    )
    func_compute_mean_resource = partial(
        compute_mean_ressource,
        consumption_array=consumption_array,
        ressource_available=ressource_available,
        ressource_renewable=ressource_renewable,
    )
    return func_sgs, func_sgs_2, func_compute_mean_resource


def evaluate_constraints(
    solution: RCPSPSolution,
    constraints: SpecialConstraintsDescription,
) -> int:
    list_constraints_not_respected = compute_constraints_details(solution, constraints)
    return sum([x[-1] for x in list_constraints_not_respected])


def compute_constraints_details(
    solution: RCPSPSolution,
    constraints: SpecialConstraintsDescription,
) -> List[Tuple[str, Hashable, Hashable, Optional[int], Optional[int], int]]:
    if not solution.rcpsp_schedule_feasible:
        return []
    start_together = constraints.start_together
    start_at_end = constraints.start_at_end
    start_at_end_plus_offset = constraints.start_at_end_plus_offset
    start_after_nunit = constraints.start_after_nunit
    disjunctive = constraints.disjunctive_tasks
    list_constraints_not_respected: List[
        Tuple[str, Hashable, Hashable, Optional[int], Optional[int], int]
    ] = []
    for (t1, t2) in start_together:
        time1 = solution.get_start_time(t1)
        time2 = solution.get_start_time(t2)
        b = time1 == time2
        if not b:
            list_constraints_not_respected += [
                ("start_together", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for (t1, t2) in start_at_end:
        time1 = solution.get_end_time(t1)
        time2 = solution.get_start_time(t2)
        b = time1 == time2
        if not b:
            list_constraints_not_respected += [
                ("start_at_end", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for (t1, t2, off) in start_at_end_plus_offset:
        time1 = solution.get_end_time(t1) + off
        time2 = solution.get_start_time(t2)
        b = time2 >= time1
        if not b:
            list_constraints_not_respected += [
                ("start_at_end_plus_offset", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for (t1, t2, off) in start_after_nunit:
        time1 = solution.get_start_time(t1) + off
        time2 = solution.get_start_time(t2)
        b = time2 >= time1
        if not b:
            list_constraints_not_respected += [
                ("start_after_nunit", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for t1, t2 in disjunctive:
        segt = intersect(
            [solution.get_start_time(t1), solution.get_end_time(t1)],
            [solution.get_start_time(t2), solution.get_end_time(t2)],
        )
        if segt is not None:
            list_constraints_not_respected += [
                ("disjunctive", t1, t2, None, None, segt[1] - segt[0])
            ]
    for t in constraints.start_times_window:
        if constraints.start_times_window[t][0] is not None:
            if solution.get_start_time(t) < constraints.start_times_window[t][0]:  # type: ignore
                list_constraints_not_respected += [
                    (
                        "start_window_0",
                        t,
                        t,
                        None,
                        None,
                        constraints.start_times_window[t][0]  # type: ignore
                        - solution.get_start_time(t),
                    )
                ]

        if constraints.start_times_window[t][1] is not None:
            if solution.get_start_time(t) > constraints.start_times_window[t][1]:  # type: ignore
                list_constraints_not_respected += [
                    (
                        "start_window_1",
                        t,
                        t,
                        None,
                        None,
                        -constraints.start_times_window[t][1]  # type: ignore
                        + solution.get_start_time(t),
                    )
                ]

    for t in constraints.end_times_window:
        if constraints.end_times_window[t][0] is not None:
            if solution.get_end_time(t) < constraints.end_times_window[t][0]:  # type: ignore
                list_constraints_not_respected += [
                    (
                        "end_window_0",
                        t,
                        t,
                        None,
                        None,
                        constraints.end_times_window[t][0] - solution.get_end_time(t),  # type: ignore
                    )
                ]

        if constraints.end_times_window[t][1] is not None:
            if solution.get_end_time(t) > constraints.end_times_window[t][1]:  # type: ignore
                list_constraints_not_respected += [
                    (
                        "end_window_1",
                        t,
                        t,
                        None,
                        None,
                        -constraints.end_times_window[t][1] + solution.get_end_time(t),  # type: ignore
                    )
                ]
    if constraints.pair_mode_constraint is not None:
        list_constraints_not_respected += compute_details_mode_constraint(
            solution=solution, pair_mode_constraint=constraints.pair_mode_constraint
        )
    return list_constraints_not_respected


def check_solution_with_special_constraints(
    problem: RCPSPModel,
    solution: RCPSPSolution,
    relax_the_start_at_end: bool = True,
) -> bool:
    if not solution.rcpsp_schedule_feasible:
        return False
    start_together = problem.special_constraints.start_together
    start_at_end = problem.special_constraints.start_at_end
    start_at_end_plus_offset = problem.special_constraints.start_at_end_plus_offset
    start_after_nunit = problem.special_constraints.start_after_nunit
    disjunctive = problem.special_constraints.disjunctive_tasks
    for (t1, t2) in start_together:
        if not relax_the_start_at_end:
            b = solution.get_start_time(t1) == solution.get_start_time(t2)
            if not b:
                return False
    for (t1, t2) in start_at_end:
        if relax_the_start_at_end:
            b = solution.get_start_time(t2) >= solution.get_end_time(t1)
        else:
            b = solution.get_start_time(t2) == solution.get_end_time(t1)
        if not b:
            return False
    for (t1, t2, off) in start_at_end_plus_offset:
        b = solution.get_start_time(t2) >= solution.get_end_time(t1) + off
        if not b:
            logger.debug(("start_at_end_plus_offset NOT respected: ", t1, t2, off))
            logger.debug(
                (
                    solution.get_start_time(t2),
                    " >= ",
                    solution.get_end_time(t1),
                    "+",
                    off,
                )
            )
            return False
    for (t1, t2, off) in start_after_nunit:
        b = solution.get_start_time(t2) >= solution.get_start_time(t1) + off
        if not b:
            logger.debug(("start_after_nunit NOT respected: ", t1, t2, off))
            return False
    for t1, t2 in disjunctive:
        if (
            intersect(
                [solution.get_start_time(t1), solution.get_end_time(t1)],
                [solution.get_start_time(t2), solution.get_end_time(t2)],
            )
            is not None
        ):
            return False
    for t in problem.special_constraints.start_times_window:
        if problem.special_constraints.start_times_window[t][0] is not None:
            if (
                solution.get_start_time(t)  # type: ignore
                < problem.special_constraints.start_times_window[t][0]
            ):
                logger.debug(
                    (
                        "start time 0, ",
                        t,
                        solution.get_start_time(t),
                        problem.special_constraints.start_times_window[t][0],
                    )
                )
                return False
        if problem.special_constraints.start_times_window[t][1] is not None:
            if (
                solution.get_start_time(t)  # type: ignore
                > problem.special_constraints.start_times_window[t][1]
            ):
                logger.debug(
                    (
                        "start time 1, ",
                        t,
                        solution.get_start_time(t),
                        problem.special_constraints.start_times_window[t][1],
                    )
                )
                return False
    for t in problem.special_constraints.end_times_window:
        if problem.special_constraints.end_times_window[t][0] is not None:
            if (
                solution.get_end_time(t)  # type: ignore
                < problem.special_constraints.end_times_window[t][0]
            ):
                logger.debug(
                    (
                        "end time 0, ",
                        t,
                        solution.get_end_time(t),
                        problem.special_constraints.end_times_window[t][0],
                    )
                )
                return False
        if problem.special_constraints.end_times_window[t][1] is not None:
            if (
                solution.get_end_time(t)  # type: ignore
                > problem.special_constraints.end_times_window[t][1]
            ):
                logger.debug(
                    (
                        "end time 1, ",
                        t,
                        solution.get_end_time(t),
                        problem.special_constraints.end_times_window[t][1],
                    )
                )
                return False
    if problem.special_constraints.pair_mode_constraint is not None:
        b = check_pair_mode_constraint(
            solution=solution,
            pair_mode_constraint=problem.special_constraints.pair_mode_constraint,
        )
        if not b:
            return False
    return True


def check_pair_mode_constraint(
    solution: RCPSPSolution, pair_mode_constraint: PairModeConstraint
):
    if pair_mode_constraint.allowed_mode_assignment is not None:
        for ac1, ac2 in pair_mode_constraint.allowed_mode_assignment:
            mode_ac1 = solution.get_mode(ac1)
            mode_ac2 = solution.get_mode(ac2)
            if (mode_ac1, mode_ac2) not in pair_mode_constraint.allowed_mode_assignment[
                ac1, ac2
            ]:
                return False
        return True
    if pair_mode_constraint.same_score_mode is not None:
        for ac1, ac2 in pair_mode_constraint.same_score_mode:
            score_ac1 = pair_mode_constraint.score_mode[ac1, solution.get_mode(ac1)]
            score_ac2 = pair_mode_constraint.score_mode[ac2, solution.get_mode(ac2)]
            if score_ac1 != score_ac2:
                return False
        return True


def compute_details_mode_constraint(
    solution: RCPSPSolution, pair_mode_constraint: PairModeConstraint
):
    list_constraints_not_respected: List[
        Tuple[str, Hashable, Hashable, Optional[int], Optional[int], int]
    ] = []
    if pair_mode_constraint.allowed_mode_assignment is not None:
        for ac1, ac2 in pair_mode_constraint.allowed_mode_assignment:
            mode_ac1 = solution.get_mode(ac1)
            mode_ac2 = solution.get_mode(ac2)
            if (mode_ac1, mode_ac2) not in pair_mode_constraint.allowed_mode_assignment[
                ac1, ac2
            ]:
                list_constraints_not_respected.append(
                    ("pair_mode_assignment", ac1, ac2, mode_ac1, mode_ac2, 100)
                )
        return list_constraints_not_respected
    if pair_mode_constraint.same_score_mode is not None:
        for ac1, ac2 in pair_mode_constraint.same_score_mode:
            score_ac1 = pair_mode_constraint.score_mode[ac1, solution.get_mode(ac1)]
            score_ac2 = pair_mode_constraint.score_mode[ac2, solution.get_mode(ac2)]
            if score_ac1 != score_ac2:
                list_constraints_not_respected.append(
                    ("pair_mode_score", ac1, ac2, score_ac1, score_ac2, 100)
                )
        return list_constraints_not_respected
