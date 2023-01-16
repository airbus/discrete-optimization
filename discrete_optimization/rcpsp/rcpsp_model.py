#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, randint, rv_discrete
from sortedcontainers import SortedDict

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    MethodAggregating,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    RobustProblem,
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

logger = logging.getLogger(__name__)


def tree():
    return defaultdict(tree)


class ScheduleGenerationScheme(Enum):
    SERIAL_SGS = 0
    PARALLEL_SGS = 1


class TaskDetails:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class RCPSPSolution(Solution):
    rcpsp_permutation: Union[List[int], np.array]
    rcpsp_schedule: Dict[Hashable, Dict]
    rcpsp_modes: List[int]
    standardised_permutation: Union[List[int], np.array]

    def __init__(
        self,
        problem,
        rcpsp_permutation=None,
        rcpsp_schedule=None,
        rcpsp_modes=None,
        rcpsp_schedule_feasible=None,
        standardised_permutation=None,
        fast=True,
    ):
        self.problem = problem
        self.rcpsp_permutation = rcpsp_permutation
        self.rcpsp_schedule = rcpsp_schedule
        self._schedule_to_recompute = rcpsp_schedule is None
        self.rcpsp_modes = rcpsp_modes
        self.rcpsp_schedule_feasible = rcpsp_schedule_feasible
        self.standardised_permutation = standardised_permutation

        if self.rcpsp_modes is None:
            if not self.problem.is_rcpsp_multimode():
                self.rcpsp_modes = [1 for i in range(self.problem.n_jobs_non_dummy)]
            else:
                self.rcpsp_modes = self.problem.fixed_modes
        if self.rcpsp_permutation is None:
            if isinstance(self.problem, MultiModeRCPSPModel):
                self.rcpsp_permutation = self.problem.fixed_permutation
            if self.rcpsp_schedule is not None:
                self.standardised_permutation = (
                    self.generate_permutation_from_schedule()
                )
                self.rcpsp_permutation = deepcopy(self.standardised_permutation)
                self._schedule_to_recompute = False
        if rcpsp_schedule is None:
            if not isinstance(problem, Aggreg_RCPSPModel):
                self.generate_schedule_from_permutation_serial_sgs(do_fast=fast)
        if self.standardised_permutation is None:
            if not isinstance(problem, Aggreg_RCPSPModel):
                self.standardised_permutation = (
                    self.generate_permutation_from_schedule()
                )
        self.fast = fast

    def change_problem(self, new_problem: Problem):
        self.__init__(
            problem=new_problem,
            rcpsp_permutation=self.rcpsp_permutation,
            rcpsp_modes=self.rcpsp_modes,
        )

    def __setattr__(self, key, value):
        super.__setattr__(self, key, value)
        if key == "rcpsp_permutation":
            self._schedule_to_recompute = True

    def copy(self):
        return RCPSPSolution(
            problem=self.problem,
            rcpsp_permutation=deepcopy(self.rcpsp_permutation),
            rcpsp_modes=deepcopy(self.rcpsp_modes),
            rcpsp_schedule=deepcopy(self.rcpsp_schedule),
            rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
            standardised_permutation=self.standardised_permutation,
            fast=self.fast,
        )

    def lazy_copy(self):
        return RCPSPSolution(
            problem=self.problem,
            rcpsp_permutation=self.rcpsp_permutation,
            rcpsp_modes=self.rcpsp_modes,
            rcpsp_schedule=self.rcpsp_schedule,
            rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
            standardised_permutation=self.standardised_permutation,
            fast=self.fast,
        )

    def __str__(self):
        if self.rcpsp_schedule is None:
            sched_str = "None"
        else:
            sched_str = str(self.rcpsp_schedule)
        val = "RCPSP solution (rcpsp_schedule): " + sched_str
        return val

    def generate_permutation_from_schedule(self):
        sorted_task = [
            self.problem.index_task_non_dummy[i]
            for i in sorted(
                self.rcpsp_schedule, key=lambda x: self.rcpsp_schedule[x]["start_time"]
            )
            if i in self.problem.index_task_non_dummy
        ]
        return sorted_task

    def compute_mean_resource_reserve(self, fast=True):
        if not fast:
            return compute_mean_resource_reserve(
                solution=self, rcpsp_problem=self.problem
            )
        else:
            if not self.rcpsp_schedule_feasible:
                return 0.0
            last_activity = self.problem.sink_task
            makespan = self.rcpsp_schedule[last_activity]["end_time"]
            if max(self.rcpsp_modes) > self.problem.max_number_of_mode:
                # non existing modes
                return 0.0
            else:
                return self.problem.compute_mean_resource(
                    horizon=makespan,
                    modes_array=np.array(
                        self.problem.build_mode_array(self.rcpsp_modes)
                    )
                    - 1,  # permutation_task=array(task)->task index
                    start_array=np.array(
                        [
                            self.rcpsp_schedule[t]["start_time"]
                            for t in self.problem.tasks_list
                        ]
                    ),
                    end_array=np.array(
                        [
                            self.rcpsp_schedule[t]["end_time"]
                            for t in self.problem.tasks_list
                        ]
                    ),
                )

    def generate_schedule_from_permutation_serial_sgs(self, do_fast=True):
        if do_fast:
            if max(self.rcpsp_modes) > self.problem.max_number_of_mode:
                # non existing modes
                schedule, unfeasible = {}, True
            else:
                schedule, unfeasible = self.problem.func_sgs(
                    permutation_task=permutation_do_to_permutation_sgs_fast(
                        self.problem, self.rcpsp_permutation
                    ),
                    modes_array=np.array(
                        self.problem.build_mode_array(self.rcpsp_modes)
                    )
                    - 1,
                )
            self.rcpsp_schedule_feasible = not unfeasible
            self.rcpsp_schedule = {}
            for k in schedule:
                self.rcpsp_schedule[self.problem.tasks_list[k]] = {
                    "start_time": schedule[k][0],
                    "end_time": schedule[k][1],
                }
            if self.problem.sink_task not in self.rcpsp_schedule:
                self.rcpsp_schedule[self.problem.sink_task] = {
                    "start_time": 99999999,
                    "end_time": 99999999,
                }
            self._schedule_to_recompute = False
        else:
            schedule, feasible = generate_schedule_from_permutation_serial_sgs(
                solution=self, rcpsp_problem=self.problem
            )
            self.rcpsp_schedule = schedule
            self.rcpsp_schedule_feasible = feasible
            self._schedule_to_recompute = False

    def generate_schedule_from_permutation_serial_sgs_2(
        self,
        current_t=0,
        completed_tasks: Dict[Hashable, TaskDetails] = None,
        scheduled_tasks_start_times: Dict[Hashable, int] = None,
        do_fast=True,
    ):
        if completed_tasks is None:
            completed_tasks = {}
        if scheduled_tasks_start_times is None:
            scheduled_tasks_start_times = None
        if do_fast:
            if max(self.rcpsp_modes) > self.problem.max_number_of_mode:
                # non existing modes
                schedule, unfeasible = {}, True
            else:
                schedule, unfeasible = self.problem.func_sgs_2(
                    current_time=current_t,
                    completed_task_indicator=np.array(
                        [
                            1 if self.problem.tasks_list[i] in completed_tasks else 0
                            for i in range(self.problem.n_jobs)
                        ]
                    ),
                    completed_task_times=np.array(
                        [
                            completed_tasks[self.problem.tasks_list[i]].end
                            if self.problem.tasks_list[i] in completed_tasks
                            else 0
                            for i in range(self.problem.n_jobs)
                        ]
                    ),
                    scheduled_task=np.array(
                        [
                            scheduled_tasks_start_times[self.problem.tasks_list[i]]
                            if self.problem.tasks_list[i] in scheduled_tasks_start_times
                            else -1
                            for i in range(self.problem.n_jobs)
                        ]
                    ),
                    permutation_task=permutation_do_to_permutation_sgs_fast(
                        self.problem, self.rcpsp_permutation
                    ),
                    modes_array=np.array(
                        self.problem.build_mode_array(self.rcpsp_modes)
                    )
                    - 1,
                )
            self.rcpsp_schedule = {}
            for k in schedule:
                self.rcpsp_schedule[self.problem.tasks_list[k]] = {
                    "start_time": schedule[k][0],
                    "end_time": schedule[k][1],
                }
            if self.problem.sink_task not in self.rcpsp_schedule:
                self.rcpsp_schedule[self.problem.sink_task] = {
                    "start_time": 999999999,
                    "end_time": 999999999,
                }
            self.rcpsp_schedule_feasible = not unfeasible
            self._schedule_to_recompute = False
        else:
            (
                schedule,
                feasible,
            ) = generate_schedule_from_permutation_serial_sgs_partial_schedule(
                solution=self,
                current_t=current_t,
                completed_tasks=completed_tasks,
                scheduled_tasks_start_times=scheduled_tasks_start_times,
                rcpsp_problem=self.problem,
            )
            self.rcpsp_schedule = schedule
            self.rcpsp_schedule_feasible = not feasible
            self._schedule_to_recompute = False

    def get_max_end_time(self):
        return max([self.get_end_time(x) for x in self.rcpsp_schedule])

    def get_start_time(self, task):
        return self.rcpsp_schedule.get(task, {"start_time": None})["start_time"]

    def get_end_time(self, task):
        return self.rcpsp_schedule.get(task, {"end_time": None})["end_time"]

    def get_start_times_list(self, task):
        return [self.get_start_time(task)]

    def get_end_times_list(self, task):
        return [self.get_end_time(task)]

    def get_active_time(self, task):
        return list(range(self.get_start_time(task), self.get_end_time(task)))

    def __hash__(self):
        return hash((tuple(self.rcpsp_permutation), tuple(self.rcpsp_modes)))

    def __eq__(self, other):
        return (
            self.rcpsp_permutation == other.rcpsp_permutation
            and self.rcpsp_modes == other.rcpsp_modes
        )


class PartialSolution:
    def __init__(
        self,
        task_mode: Dict[int, int] = None,
        start_times: Dict[int, int] = None,
        end_times: Dict[int, int] = None,
        partial_permutation: List[int] = None,
        list_partial_order: List[List[int]] = None,
        start_together: List[Tuple[int, int]] = None,
        start_at_end: List[Tuple[int, int]] = None,
        start_at_end_plus_offset: List[Tuple[int, int, int]] = None,
        start_after_nunit: List[Tuple[int, int, int]] = None,
        disjunctive_tasks: List[Tuple[int, int]] = None,
        start_times_window: Dict[Hashable, Tuple[int, int]] = None,
        end_times_window: Dict[Hashable, Tuple[int, int]] = None,
    ):
        self.task_mode = task_mode
        self.start_times = start_times
        self.end_times = end_times
        self.partial_permutation = partial_permutation
        self.list_partial_order = list_partial_order
        self.start_together = start_together
        self.start_at_end = start_at_end
        self.start_after_nunit = start_after_nunit
        self.start_at_end_plus_offset = start_at_end_plus_offset
        self.disjunctive_tasks = disjunctive_tasks
        self.start_times_window = start_times_window
        self.end_times_window = end_times_window
        # one element in self.list_partial_order is a list [l1, l2, l3]
        # indicating that l1 should be started before l1, and  l2 before l3 for example


class RCPSPModel(Problem):
    sgs: ScheduleGenerationScheme
    resources: Union[
        Dict[str, int], Dict[str, List[int]]
    ]  # {resource_name: number_of_resource}
    non_renewable_resources: List[str]  # e.g. [resource_name3, resource_name4]
    n_jobs: int
    n_jobs_non_dummy: int  # excluding dummy activities Start (0) and End (n)
    mode_details: Dict[Hashable, Dict[int, Dict[str, int]]]
    # e.g. {job_id: {mode_id: {resource_name1: number_of_resources_needed, resource_name2: ...}}
    # one key being "duration"
    successors: Dict[int, List[int]]  # {task_id: list of successor task ids}

    def __init__(
        self,
        resources: Union[Dict[str, int], Dict[str, List[int]]],
        non_renewable_resources: List[str],
        mode_details: Dict[Hashable, Dict[Union[str, int], Dict[str, int]]],
        successors: Dict[Union[int, str], List[Union[str, int]]],
        horizon,
        horizon_multiplier=1,
        tasks_list: List[Union[int, str]] = None,
        source_task=None,
        sink_task=None,
        name_task: Dict[int, str] = None,
        **args,
    ):
        self.resources = resources
        self.resources_list = list(self.resources.keys())
        self.non_renewable_resources = non_renewable_resources
        self.mode_details = mode_details
        self.successors = successors
        self.horizon = horizon
        self.horizon_multiplier = horizon_multiplier
        self.name_task = name_task
        if name_task is None:
            self.name_task = {x: str(x) for x in self.mode_details}
        self.tasks_list = tasks_list
        if tasks_list is None:
            self.tasks_list = sorted(self.mode_details.keys())
        self.n_jobs = len(self.mode_details.keys())
        self.n_jobs_non_dummy = self.n_jobs - 2
        self.index_task = {self.tasks_list[i]: i for i in range(self.n_jobs)}
        self.source_task = source_task
        if source_task is None:
            self.source_task = min(self.tasks_list)
        self.sink_task = sink_task
        if sink_task is None:
            self.sink_task = max(self.tasks_list)
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
                    [
                        len(
                            set(self.resources[res])
                            if isinstance(self.resources[res], Iterable)
                            else {self.resources[res]}
                        )
                        for res in self.resources
                    ]
                )
                > 1
            )
            if not self.is_calendar:
                self.resources = {r: int(self.resources[r][0]) for r in self.resources}
        (
            self.func_sgs,
            self.func_sgs_2,
            self.compute_mean_resource,
        ) = create_np_data_and_jit_functions(self)
        self.costs = {
            "makespan": True,
            "mean_resource_reserve": args.get("mean_resource_reserve", False),
        }
        self.graph = self.compute_graph()

    def update_functions(self):
        (
            self.func_sgs,
            self.func_sgs_2,
            self.compute_mean_resource,
        ) = create_np_data_and_jit_functions(rcpsp_problem=self)

    def is_rcpsp_multimode(self):
        return self.is_multimode

    def is_varying_resource(self):
        return self.is_calendar

    def is_preemptive(self):
        return False

    def is_multiskill(self):
        return False

    def get_resource_names(self):
        return self.resources_list

    def get_tasks_list(self):
        return self.tasks_list

    def get_resource_availability_array(self, res):
        if self.is_varying_resource():
            return self.resources[res]
        else:
            return np.full(self.horizon, self.resources[res])

    def compute_graph(self, compute_predecessors: bool = False) -> Graph:
        nodes = [
            (
                n,
                {
                    mode: self.mode_details[n][mode]["duration"]
                    for mode in self.mode_details[n]
                },
            )
            for n in self.tasks_list
        ]
        edges = []
        for n in self.successors:
            for succ in self.successors[n]:
                edges += [(n, succ, {})]
        return Graph(
            nodes, edges, compute_predecessors=compute_predecessors, undirected=False
        )

    def evaluate_function(self, rcpsp_sol: RCPSPSolution):
        if rcpsp_sol._schedule_to_recompute:
            rcpsp_sol.generate_schedule_from_permutation_serial_sgs()
        makespan = rcpsp_sol.rcpsp_schedule[self.sink_task]["end_time"]
        if self.costs["mean_resource_reserve"]:
            obj_mean_resource_reserve = rcpsp_sol.compute_mean_resource_reserve()
            return makespan, obj_mean_resource_reserve
        return makespan, 0

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == "rcpsp_permutation":
            single_mode_list = [1 for i in range(self.n_jobs_non_dummy)]
            rcpsp_sol = RCPSPSolution(
                problem=self, rcpsp_permutation=int_vector, rcpsp_modes=single_mode_list
            )
            objectives = self.evaluate(rcpsp_sol)
            return objectives
        return None

    def evaluate(self, rcpsp_sol: RCPSPSolution) -> Dict[str, float]:
        obj_makespan, obj_mean_resource_reserve = self.evaluate_function(rcpsp_sol)
        return {
            "makespan": obj_makespan,
            "mean_resource_reserve": obj_mean_resource_reserve,
        }

    def evaluate_mobj(self, rcpsp_sol: RCPSPSolution):
        return self.evaluate_mobj_from_dict(self.evaluate(rcpsp_sol))

    def evaluate_mobj_from_dict(self, dict_values: Dict[str, float]) -> TupleFitness:
        return TupleFitness(
            np.array([-dict_values["makespan"], dict_values["mean_resource_reserve"]]),
            2,
        )

    def build_mode_dict(self, rcpsp_modes_from_solution):
        modes_dict = {
            self.tasks_list_non_dummy[i]: rcpsp_modes_from_solution[i]
            for i in range(self.n_jobs_non_dummy)
        }
        modes_dict[self.source_task] = 1
        modes_dict[self.sink_task] = 1
        return modes_dict

    def build_mode_array(self, rcpsp_modes_from_solution):
        modes_dict = {
            self.tasks_list_non_dummy[i]: rcpsp_modes_from_solution[i]
            for i in range(self.n_jobs_non_dummy)
        }
        modes_dict[self.source_task] = 1
        modes_dict[self.sink_task] = 1
        return [modes_dict[t] for t in self.tasks_list]

    def return_index_task(self, task, offset=0):
        return self.index_task[task] + offset

    def satisfy(self, rcpsp_sol: RCPSPSolution) -> bool:
        if rcpsp_sol.rcpsp_schedule_feasible is False:
            logger.debug("Schedule flagged as infeasible when generated")
            return False
        else:
            modes_dict = self.build_mode_dict(
                rcpsp_modes_from_solution=rcpsp_sol.rcpsp_modes
            )
            start_times = [
                rcpsp_sol.rcpsp_schedule[t]["start_time"]
                for t in rcpsp_sol.rcpsp_schedule
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
                    if usage > self.resources[res]:
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

    def __str__(self):
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
        dict_objective = {
            "makespan": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1.0)
        }
        # "mean_resource_reserve": {"type": TypeObjective.OBJECTIVE, "default_weight": 1}}
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.SINGLE,
            dict_objective_to_doc=dict_objective,
        )

    def compute_resource_consumption(self, rcpsp_sol: RCPSPSolution):
        modes_dict = self.build_mode_dict(rcpsp_sol.rcpsp_modes)
        last_activity = max(rcpsp_sol.rcpsp_schedule)
        makespan = rcpsp_sol.rcpsp_schedule[last_activity]["end_time"]
        consumptions = np.zeros((len(self.resources), makespan + 1))
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

    def plot_ressource_view(self, rcpsp_sol: RCPSPSolution):
        consumption = self.compute_resource_consumption(rcpsp_sol=rcpsp_sol)
        fig, ax = plt.subplots(nrows=len(self.resources_list), sharex=True)
        for i in range(len(self.resources_list)):
            ax[i].axhline(
                y=self.resources[self.resources_list[i]], label=self.resources_list[i]
            )
            ax[i].plot(consumption[i, :])
            ax[i].legend()

    def copy(self):
        return RCPSPModel(
            resources=self.resources,
            tasks_list=self.tasks_list,
            source_task=self.source_task,
            sink_task=self.sink_task,
            non_renewable_resources=self.non_renewable_resources,
            mode_details=deepcopy(self.mode_details),
            successors=deepcopy(self.successors),
            horizon=self.horizon,
            horizon_multiplier=self.horizon_multiplier,
        )

    def get_dummy_solution(self):
        sol = RCPSPSolution(
            problem=self,
            rcpsp_permutation=list(range(self.n_jobs_non_dummy)),
            rcpsp_modes=[1 for i in range(self.n_jobs_non_dummy)],
        )
        return sol

    def get_resource_available(self, res, time):
        if self.is_calendar:
            return self.resources.get(res, [0])[time]
        return self.resources.get(res, 0)

    def get_max_resource_capacity(self, res):
        if self.is_calendar:
            return max(self.resources.get(res, [0]))
        return self.resources.get(res, 0)


class RCPSPModelCalendar(RCPSPModel):
    resources: Dict[str, List[int]]

    def __init__(
        self,
        resources: Dict[str, List[int]],
        non_renewable_resources: List[str],
        mode_details: Dict[Hashable, Dict[int, Dict[str, int]]],
        successors: Dict[int, List[int]],
        horizon,
        horizon_multiplier=1,
        tasks_list: List[Hashable] = None,
        source_task=None,
        sink_task=None,
        name_task: Dict[int, str] = None,
        calendar_details: Dict[str, List[List[int]]] = None,
        name_ressource_to_index: Dict[str, int] = None,
    ):
        super().__init__(
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            horizon_multiplier=horizon_multiplier,
            tasks_list=tasks_list,
            source_task=source_task,
            sink_task=sink_task,
            name_task=name_task,
        )
        self.calendar_details = calendar_details
        self.name_ressource_to_index = name_ressource_to_index

    def get_resource_available(self, res, time):
        return self.resources.get(res, [0])[time]

    def copy(self):
        return RCPSPModelCalendar(
            resources={w: deepcopy(self.resources[w]) for w in self.resources},
            non_renewable_resources=self.non_renewable_resources,
            mode_details=deepcopy(self.mode_details),
            successors=deepcopy(self.successors),
            horizon=self.horizon,
            horizon_multiplier=self.horizon_multiplier,
            calendar_details=deepcopy(self.calendar_details),
            tasks_list=self.tasks_list,
            source_task=self.source_task,
            sink_task=self.sink_task,
            name_task=self.name_task,
            name_ressource_to_index=self.name_ressource_to_index,
        )


def create_np_data_and_jit_functions(
    rcpsp_problem: Union[RCPSPModel, RCPSPModelCalendar]
):
    consumption_array = np.zeros(
        (
            rcpsp_problem.n_jobs,
            rcpsp_problem.max_number_of_mode,
            len(rcpsp_problem.resources_list),
        ),
        dtype=np.int32,
    )
    duration_array = np.zeros(
        (rcpsp_problem.n_jobs, rcpsp_problem.max_number_of_mode), dtype=np.int32
    )
    predecessors = np.zeros(
        (rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int32
    )
    successors = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int32)
    horizon = rcpsp_problem.horizon
    ressource_available = np.zeros(
        (len(rcpsp_problem.resources_list), horizon), dtype=np.int32
    )
    ressource_renewable = np.ones((len(rcpsp_problem.resources_list)), dtype=bool)

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
            ressource_available[k, :] = rcpsp_problem.resources[
                rcpsp_problem.resources_list[k]
            ][: ressource_available.shape[1]]
        else:
            ressource_available[k, :] = np.full(
                ressource_available.shape[1],
                rcpsp_problem.resources[rcpsp_problem.resources_list[k]],
                dtype=int,
            )
        if rcpsp_problem.resources_list[k] in rcpsp_problem.non_renewable_resources:
            ressource_renewable[k] = False

    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        for s in rcpsp_problem.successors[task]:
            index_s = task_index[s]
            predecessors[index_s, i] = 1
            successors[i, index_s] = 1
    minimum_starting_time_array = np.zeros(rcpsp_problem.n_jobs, dtype=int)
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


def permutation_do_to_permutation_sgs_fast(rcpsp_problem: RCPSPModel, permutation_do):
    perm_extended = [
        rcpsp_problem.index_task[rcpsp_problem.tasks_list_non_dummy[x]]
        for x in permutation_do
    ]
    perm_extended.insert(0, rcpsp_problem.index_task[rcpsp_problem.source_task])
    perm_extended.append(rcpsp_problem.index_task[rcpsp_problem.sink_task])
    return np.array(perm_extended, dtype=np.int32)


class SingleModeRCPSPModel(RCPSPModel):
    def copy(self):
        return SingleModeRCPSPModel(
            resources=self.resources,
            non_renewable_resources=self.non_renewable_resources,
            mode_details=deepcopy(self.mode_details),
            successors=deepcopy(self.successors),
            horizon=self.horizon,
            horizon_multiplier=self.horizon_multiplier,
        )


class MultiModeRCPSPModel(RCPSPModel):
    fixed_modes: List[int]
    fixed_permutation: Union[List[int], np.array]

    def __init__(
        self,
        resources,
        non_renewable_resources,
        mode_details,
        successors,
        horizon,
        horizon_multiplier=1,
    ):
        RCPSPModel.__init__(
            self,
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            horizon_multiplier=horizon_multiplier,
        )
        self.fixed_modes = None
        self.fixed_permutation = None

    def set_fixed_attributes(self, encoding_str: str, sol: RCPSPSolution):
        att = self.get_attribute_register().dict_attribute_to_type[encoding_str]["name"]
        if att == "rcpsp_modes":
            self.set_fixed_modes(sol.rcpsp_modes)
        elif att == "rcpsp_permutation":
            self.set_fixed_permutation(sol.rcpsp_permutation)

    def set_fixed_modes(self, fixed_modes):
        self.fixed_modes = fixed_modes

    def set_fixed_permutation(self, fixed_permutation):
        self.fixed_permutation = fixed_permutation

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == "rcpsp_permutation":
            # change the permutation in the solution with int_vector and set the modes with self.fixed_modes
            rcpsp_sol = RCPSPSolution(
                problem=self, rcpsp_permutation=int_vector, rcpsp_modes=self.fixed_modes
            )
        elif encoding_name == "rcpsp_modes":
            rcpsp_sol = RCPSPSolution(
                problem=self,
                rcpsp_permutation=self.fixed_permutation,
                rcpsp_modes=int_vector,
            )
        objectives = self.evaluate(rcpsp_sol)
        return objectives

    def copy(self):
        mm = MultiModeRCPSPModel(
            resources=self.resources,
            non_renewable_resources=self.non_renewable_resources,
            mode_details=deepcopy(self.mode_details),
            successors=deepcopy(self.successors),
            horizon=self.horizon,
            horizon_multiplier=self.horizon_multiplier,
        )
        mm.fixed_permutation = self.fixed_permutation
        mm.fixed_modes = self.fixed_modes
        return mm

    def get_dummy_solution(self):
        sol = RCPSPSolution(
            problem=self,
            rcpsp_permutation=list(range(self.n_jobs_non_dummy)),
            rcpsp_modes=[1 for i in range(self.n_jobs_non_dummy)],
        )
        return sol


class Aggreg_RCPSPModel(RobustProblem, RCPSPModel):
    def __init__(
        self, list_problem: Sequence[RCPSPModel], method_aggregating: MethodAggregating
    ):
        RobustProblem.__init__(
            self, list_problem=list_problem, method_aggregating=method_aggregating
        )
        self.horizon = list_problem[0].horizon
        self.horizon_multiplier = list_problem[0].horizon_multiplier
        self.resources = list_problem[0].resources
        self.successors = list_problem[0].successors
        self.n_jobs = list_problem[0].n_jobs
        self.mode_details = list_problem[0].mode_details
        self.resources_list = list_problem[0].resources_list

    def get_dummy_solution(self):
        a: RCPSPSolution = self.list_problem[0].get_dummy_solution()
        a._schedule_to_recompute = True
        return a

    def get_unique_rcpsp_model(self) -> RCPSPModel:
        # Create a unique rcpsp instance coherent with the aggregating method.
        model = self.list_problem[0].copy()
        for job in model.mode_details:
            for mode in model.mode_details[job]:
                for res in model.mode_details[job][mode]:
                    rs = np.array(
                        [
                            self.list_problem[i].mode_details[job][mode][res]
                            for i in range(self.nb_problem)
                        ]
                    )
                    agg = int(self.agg_vec(rs))
                    model.mode_details[job][mode][res] = agg
        return model

    def evaluate_from_encoding(self, int_vector, encoding_name):
        fits = [
            self.list_problem[i].evaluate_from_encoding(int_vector, encoding_name)
            for i in range(self.nb_problem)
        ]
        keys = fits[0].keys()
        aggreg = {}
        for k in keys:
            vals = np.array([fit[k] for fit in fits])
            aggreg[k] = self.agg_vec(vals)
        return aggreg

    def evaluate(self, variable: Solution):
        fits = []
        for i in range(self.nb_problem):
            var: RCPSPSolution = variable.lazy_copy()
            var.rcpsp_schedule = None
            var._schedule_to_recompute = True
            var.problem = self.list_problem[i]
            fit = self.list_problem[i].evaluate(var)
            fits += [fit]
        keys = fits[0].keys()
        aggreg = {}
        for k in keys:
            vals = np.array([fit[k] for fit in fits])
            aggreg[k] = self.agg_vec(vals)
        return aggreg


class MethodBaseRobustification(Enum):
    AVERAGE = 0
    WORST_CASE = 1
    BEST_CASE = 2
    PERCENTILE = 3
    SAMPLE = 4


class MethodRobustification:
    method_base: MethodBaseRobustification
    percentile: float

    def __init__(self, method_base: MethodBaseRobustification, percentile: float = 50):
        self.method_base = method_base
        self.percentile = percentile


def create_poisson_laws_duration(rcpsp_model: RCPSPModel, range_around_mean=3):
    poisson_dict = {}
    source = rcpsp_model.source_task
    sink = rcpsp_model.sink_task
    for job in rcpsp_model.mode_details:
        poisson_dict[job] = {}
        for mode in rcpsp_model.mode_details[job]:
            poisson_dict[job][mode] = {}
            duration = rcpsp_model.mode_details[job][mode]["duration"]
            if job in {source, sink}:
                poisson_dict[job][mode]["duration"] = (duration, duration, duration)
            else:
                min_duration = max(1, duration - range_around_mean)
                max_duration = duration + range_around_mean
                poisson_dict[job][mode]["duration"] = (
                    min_duration,
                    duration,
                    max_duration,
                )
    return poisson_dict


def create_poisson_laws_resource(rcpsp_model: RCPSPModel, range_around_mean=1):
    poisson_dict = {}
    source = rcpsp_model.source_task
    sink = rcpsp_model.sink_task
    limit_resource = rcpsp_model.resources
    resources = rcpsp_model.resources_list
    resources_non_renewable = rcpsp_model.non_renewable_resources
    for job in rcpsp_model.mode_details:
        poisson_dict[job] = {}
        for mode in rcpsp_model.mode_details[job]:
            poisson_dict[job][mode] = {}
            for resource in rcpsp_model.mode_details[job][mode]:
                if resource == "duration":
                    continue
                if resource in resources_non_renewable:
                    continue
                resource_consumption = rcpsp_model.mode_details[job][mode][resource]
                if job in {source, sink}:
                    poisson_dict[job][mode][resource] = (
                        resource_consumption,
                        resource_consumption,
                        resource_consumption,
                    )
                else:
                    min_rc = max(0, resource_consumption - range_around_mean)
                    max_rc = min(
                        resource_consumption + range_around_mean,
                        limit_resource[resource],
                    )
                    poisson_dict[job][mode][resource] = (
                        min_rc,
                        resource_consumption,
                        max_rc,
                    )
    return poisson_dict


def create_poisson_laws(
    base_rcpsp_model: RCPSPModel,
    range_around_mean_resource: int = 1,
    range_around_mean_duration: int = 3,
    do_uncertain_resource: bool = True,
    do_uncertain_duration: bool = True,
):
    poisson_laws = tree()
    if do_uncertain_duration:
        poisson_laws_duration = create_poisson_laws_duration(
            base_rcpsp_model, range_around_mean=range_around_mean_duration
        )
        for job in poisson_laws_duration:
            for mode in poisson_laws_duration[job]:
                for res in poisson_laws_duration[job][mode]:
                    poisson_laws[job][mode][res] = poisson_laws_duration[job][mode][res]
    if do_uncertain_resource:
        poisson_laws_resource = create_poisson_laws_resource(
            base_rcpsp_model, range_around_mean=range_around_mean_resource
        )
        for job in poisson_laws_resource:
            for mode in poisson_laws_resource[job]:
                for res in poisson_laws_resource[job][mode]:
                    poisson_laws[job][mode][res] = poisson_laws_resource[job][mode][res]
    return poisson_laws


class UncertainRCPSPModel:
    def __init__(
        self,
        base_rcpsp_model: RCPSPModel,
        poisson_laws: Dict[int, Dict[int, Dict[str, Tuple[int, int, int]]]],
        uniform_law=True,
    ):
        self.base_rcpsp_model = base_rcpsp_model
        self.poisson_laws = poisson_laws
        self.probas = {}
        for activity in poisson_laws:
            self.probas[activity] = {}
            for mode in poisson_laws[activity]:
                self.probas[activity][mode] = {}
                for detail in poisson_laws[activity][mode]:
                    min_, mean_, max_ = poisson_laws[activity][mode][detail]
                    if uniform_law:
                        rv = randint(low=min_, high=max_ + 1)
                    else:
                        rv = poisson(mean_)
                    self.probas[activity][mode][detail] = {
                        "value": np.arange(min_, max_ + 1, 1),
                        "proba": np.zeros((max_ - min_ + 1)),
                    }
                    for k in range(len(self.probas[activity][mode][detail]["value"])):
                        self.probas[activity][mode][detail]["proba"][k] = rv.pmf(
                            self.probas[activity][mode][detail]["value"][k]
                        )
                    self.probas[activity][mode][detail]["proba"] /= np.sum(
                        self.probas[activity][mode][detail]["proba"]
                    )
                    self.probas[activity][mode][detail][
                        "prob-distribution"
                    ] = rv_discrete(
                        name=str(activity) + "-" + str(mode) + "-" + str(detail),
                        values=(
                            self.probas[activity][mode][detail]["value"],
                            self.probas[activity][mode][detail]["proba"],
                        ),
                    )

    def create_rcpsp_model(self, method_robustification: MethodRobustification):
        model = self.base_rcpsp_model.copy()
        for activity in self.probas:
            if activity in {
                self.base_rcpsp_model.source_task,
                self.base_rcpsp_model.sink_task,
            }:
                continue
            for mode in self.probas[activity]:
                for detail in self.probas[activity][mode]:
                    if (
                        method_robustification.method_base
                        == MethodBaseRobustification.AVERAGE
                    ):
                        model.mode_details[activity][mode][detail] = int(
                            self.probas[activity][mode][detail][
                                "prob-distribution"
                            ].mean()
                        )
                    if (
                        method_robustification.method_base
                        == MethodBaseRobustification.WORST_CASE
                    ):
                        model.mode_details[activity][mode][detail] = self.probas[
                            activity
                        ][mode][detail]["prob-distribution"].support()[1]
                    if (
                        method_robustification.method_base
                        == MethodBaseRobustification.BEST_CASE
                    ):
                        model.mode_details[activity][mode][detail] = self.probas[
                            activity
                        ][mode][detail]["prob-distribution"].support()[0]
                    if (
                        method_robustification.method_base
                        == MethodBaseRobustification.PERCENTILE
                    ):
                        model.mode_details[activity][mode][detail] = max(
                            int(
                                self.probas[activity][mode][detail][
                                    "prob-distribution"
                                ].isf(q=1 - method_robustification.percentile / 100)
                            ),
                            1,
                        )
                    if (
                        method_robustification.method_base
                        == MethodBaseRobustification.SAMPLE
                    ):
                        model.mode_details[activity][mode][detail] = self.probas[
                            activity
                        ][mode][detail]["prob-distribution"].rvs(size=1)[0]
        return model


def generate_schedule_from_permutation_serial_sgs(
    solution: RCPSPSolution, rcpsp_problem: RCPSPModel
):
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon

    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][
                : new_horizon + 1
            ]
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, rcpsp_problem.resources[res], dtype=int
            ).tolist()
    minimum_starting_time = {}
    for act in rcpsp_problem.tasks_list:
        minimum_starting_time[act] = 0
    perm_extended = [
        rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
    ]
    perm_extended.insert(0, rcpsp_problem.source_task)
    perm_extended.append(rcpsp_problem.sink_task)
    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)
    for k in modes_dict:
        if modes_dict[k] not in rcpsp_problem.mode_details[k]:
            modes_dict[k] = 1
    while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
        for id_successor in perm_extended:
            respected = True
            for pred in rcpsp_problem.successors:
                if (
                    id_successor in rcpsp_problem.successors[pred]
                    and pred in perm_extended
                ):
                    respected = False
                    break
            if respected:
                act_id = id_successor
                break
        # for act_id in perm_extended:
        current_min_time = minimum_starting_time[act_id]
        valid = False
        while not valid:
            valid = True
            for t in range(
                current_min_time,
                current_min_time
                + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"],
            ):
                for res in rcpsp_problem.resources_list:
                    if (
                        rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                            res, 0
                        )
                        == 0
                    ):
                        continue
                    if t < new_horizon:
                        if (
                            resource_avail_in_time[res][t]
                            < rcpsp_problem.mode_details[act_id][modes_dict[act_id]][
                                res
                            ]
                        ):
                            valid = False
                    else:
                        unfeasible_non_renewable_resources = True
            if not valid:
                current_min_time += 1
        if not unfeasible_non_renewable_resources:
            end_t = (
                current_min_time
                + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
            )
            for t in range(current_min_time, end_t):
                for res in resource_avail_in_time:
                    if (
                        rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                            res, 0
                        )
                        == 0
                    ):
                        continue
                    resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[
                        act_id
                    ][modes_dict[act_id]][res]
                    if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                        for tt in range(end_t, new_horizon):
                            resource_avail_in_time[res][
                                tt
                            ] -= rcpsp_problem.mode_details[act_id][modes_dict[act_id]][
                                res
                            ]
                            if resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True
            activity_end_times[act_id] = end_t
            perm_extended.remove(act_id)
            for s in rcpsp_problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = (
            activity_end_times[act_id]
            - rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
    if unfeasible_non_renewable_resources:
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["start_time"] = 99999999
            rcpsp_schedule[last_act_id]["end_time"] = 9999999
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def generate_schedule_from_permutation_serial_sgs_partial_schedule(
    solution: RCPSPSolution,
    rcpsp_problem: RCPSPModel,
    current_t,
    completed_tasks,
    scheduled_tasks_start_times,
):
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon
    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][
                : new_horizon + 1
            ]
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, rcpsp_problem.resources[res], dtype=int
            ).tolist()
    minimum_starting_time = {}
    for act in rcpsp_problem.tasks_list:
        if act in list(scheduled_tasks_start_times.keys()):
            minimum_starting_time[act] = scheduled_tasks_start_times[act]
        else:
            minimum_starting_time[act] = current_t
    perm_extended = [
        rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
    ]
    perm_extended.insert(0, rcpsp_problem.source_task)
    perm_extended.append(rcpsp_problem.sink_task)
    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)

    # Update current resource usage by the scheduled task (ongoing task, in practice)
    for act_id in scheduled_tasks_start_times:
        current_min_time = scheduled_tasks_start_times[act_id]
        end_t = (
            current_min_time
            + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        for t in range(current_min_time, end_t):
            for res in resource_avail_in_time:
                resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[act_id][
                    modes_dict[act_id]
                ].get(res, 0)
                if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                    for tt in range(end_t, new_horizon):
                        resource_avail_in_time[res][tt] -= rcpsp_problem.mode_details[
                            act_id
                        ][modes_dict[act_id]].get(res, 0)
                        if resource_avail_in_time[res][tt] < 0:
                            unfeasible_non_renewable_resources = True
        activity_end_times[act_id] = end_t
        perm_extended.remove(act_id)
        for s in rcpsp_problem.successors[act_id]:
            minimum_starting_time[s] = max(
                minimum_starting_time[s], activity_end_times[act_id]
            )

    perm_extended = [x for x in perm_extended if x not in list(completed_tasks)]
    # fix modes in case specified mode not in mode details for the activites
    for ac in modes_dict:
        if modes_dict[ac] not in rcpsp_problem.mode_details[ac]:
            modes_dict[ac] = 1
    while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
        # get first activity in perm with precedences respected
        for id_successor in perm_extended:
            respected = True
            for pred in rcpsp_problem.successors.keys():
                if (
                    id_successor in rcpsp_problem.successors[pred]
                    and pred in perm_extended
                ):
                    respected = False
                    break
            if respected:
                act_id = id_successor
                break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        while not valid:
            valid = True
            for t in range(
                current_min_time,
                current_min_time
                + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"],
            ):
                for res in resource_avail_in_time:
                    if t < new_horizon:
                        if resource_avail_in_time[res][t] < rcpsp_problem.mode_details[
                            act_id
                        ][modes_dict[act_id]].get(res, 0):
                            valid = False
                    else:
                        unfeasible_non_renewable_resources = True
            if not valid:
                current_min_time += 1
        if not unfeasible_non_renewable_resources:
            end_t = (
                current_min_time
                + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
            )
            for t in range(current_min_time, end_t):
                for res in resource_avail_in_time:
                    resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[
                        act_id
                    ][modes_dict[act_id]].get(res, 0)
                    if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                        for tt in range(end_t + 1, new_horizon):
                            resource_avail_in_time[res][
                                tt
                            ] -= rcpsp_problem.mode_details[act_id][
                                modes_dict[act_id]
                            ].get(
                                res, 0
                            )
                            if resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True
            activity_end_times[act_id] = end_t
            perm_extended.remove(act_id)
            for s in rcpsp_problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = (
            activity_end_times[act_id]
            - rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
    for act_id in completed_tasks:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = completed_tasks[act_id].start
        rcpsp_schedule[act_id]["end_time"] = completed_tasks[act_id].end
    if unfeasible_non_renewable_resources:
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["start_time"] = 99999999
            rcpsp_schedule[last_act_id]["end_time"] = 9999999
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def compute_mean_resource_reserve(solution: RCPSPSolution, rcpsp_problem: RCPSPModel):
    if not solution.rcpsp_schedule_feasible:
        return 0.0
    last_activity = rcpsp_problem.sink_task
    makespan = solution.rcpsp_schedule[last_activity]["end_time"]
    resource_avail_in_time = {}
    modes = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][: makespan + 1]
        else:
            resource_avail_in_time[res] = np.full(
                makespan, rcpsp_problem.resources[res], dtype=int
            ).tolist()
    for act_id in rcpsp_problem.tasks_list:
        start_time = solution.rcpsp_schedule[act_id]["start_time"]
        end_time = solution.rcpsp_schedule[act_id]["end_time"]
        mode = modes[act_id]
        for t in range(start_time, end_time):
            for res in resource_avail_in_time:
                if rcpsp_problem.mode_details[act_id][mode].get(res, 0) == 0:
                    continue
                resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[act_id][
                    mode
                ][res]
                if res in rcpsp_problem.non_renewable_resources and t == end_time:
                    for tt in range(end_time, makespan):
                        resource_avail_in_time[res][tt] -= rcpsp_problem.mode_details[
                            act_id
                        ][mode][res]
    mean_avail = {}
    for res in resource_avail_in_time:
        mean_avail[res] = np.mean(resource_avail_in_time[res])
    mean_resource_reserve = np.mean(
        [
            mean_avail[res] / max(rcpsp_problem.resources[res])
            if rcpsp_problem.is_varying_resource()
            else mean_avail[res] / rcpsp_problem.resources[res]
            for res in rcpsp_problem.resources_list
        ]
    )
    return mean_resource_reserve


class SGSWithoutArray:
    def __init__(self, rcpsp_model: RCPSPModel):
        self.rcpsp_model = rcpsp_model
        self.resource_avail_in_time = {}
        for res in self.rcpsp_model.resources_list:
            if self.rcpsp_model.is_varying_resource():
                self.resource_avail_in_time[res] = np.array(
                    self.rcpsp_model.resources[res][: self.rcpsp_model.horizon + 1]
                )
            else:
                self.resource_avail_in_time[res] = np.full(
                    self.rcpsp_model.horizon, self.rcpsp_model.resources[res], dtype=int
                )
        self.dict_step_ressource = {}
        for res in self.resource_avail_in_time:
            self.dict_step_ressource[res] = SGSWithoutArray.create_absolute_dict(
                self.resource_avail_in_time[res]
            )

    @staticmethod
    def get_available_from_delta(sdict: SortedDict, time):
        index = sdict.bisect_left(time)
        if time in sdict:
            r = range(index + 1)
        else:
            r = range(index)
        s = sum(sdict.peekitem(j)[1] for j in r)
        return s

    @staticmethod
    def get_available_from_absolute(sdict: SortedDict, time):
        index = sdict.bisect_right(time)
        s = sdict.peekitem(index - 1)[1]
        return s

    @staticmethod
    def add_event_delta(sdict: SortedDict, time_start, delta, time_end):
        if time_start in sdict:
            sdict[time_start] += delta
        else:
            sdict.update({time_start: delta})
        if time_end in sdict:
            sdict[time_end] -= delta
        else:
            sdict.update({time_end: -delta})

    @staticmethod
    def add_event_delta_in_absolute(
        sdict: SortedDict,
        time_start,
        delta,  # Negative usually
        time_end,
        liberate=True,
    ):

        for t in [k for k in sdict if time_start <= k < time_end]:
            sdict[t] += delta
        i = sdict.bisect_right(time_start)
        if time_start not in sdict:
            sdict[time_start] = sdict.peekitem(i - 1)[1] + delta
        i = sdict.bisect_right(time_end)
        if time_end not in sdict:
            sdict[time_end] = sdict.peekitem(i - 1)[1] - delta

    @staticmethod
    def create_delta_dict(vector):
        v = np.array(vector)
        delta = v[:-1] - v[1:]
        index_non_zero = np.nonzero(delta)[0]
        l = {0: v[0]}
        for j in range(len(index_non_zero)):
            ind = index_non_zero[j]
            l[ind + 1] = -delta[ind]
        l = SortedDict(l)
        return l

    @staticmethod
    def create_absolute_dict(vector):
        v = np.array(vector)
        delta = v[:-1] - v[1:]
        index_non_zero = np.nonzero(delta)[0]
        l = {0: v[0]}
        for j in range(len(index_non_zero)):
            ind = index_non_zero[j]
            l[ind + 1] = v[ind + 1]
        l = SortedDict(l)
        return l

    def generate_schedule_from_permutation_serial_sgs(
        self, solution: RCPSPSolution, rcpsp_problem: RCPSPModel
    ):
        activity_end_times = {}
        unfeasible_non_renewable_resources = False
        new_horizon = rcpsp_problem.horizon
        resource_avail_in_time = deepcopy(self.dict_step_ressource)
        minimum_starting_time = {}
        for act in rcpsp_problem.tasks_list:
            minimum_starting_time[act] = 0
        perm_extended = [
            rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
        ]
        perm_extended.insert(0, rcpsp_problem.source_task)
        perm_extended.append(rcpsp_problem.sink_task)
        modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)
        for k in modes_dict:
            if modes_dict[k] not in rcpsp_problem.mode_details[k]:
                modes_dict[k] = 1
        while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
            for id_successor in perm_extended:
                respected = True
                for pred in rcpsp_problem.successors:
                    if (
                        id_successor in rcpsp_problem.successors[pred]
                        and pred in perm_extended
                    ):
                        respected = False
                        break
                if respected:
                    act_id = id_successor
                    break
            current_min_time = minimum_starting_time[act_id]
            valid = False
            while not valid:
                valid = True
                if current_min_time is None:
                    unfeasible_non_renewable_resources = True
                    break
                start_time = current_min_time
                end_time = (
                    current_min_time
                    + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                )
                for res in rcpsp_problem.resources_list:
                    need = rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                        res, 0
                    )
                    if need == 0:
                        continue
                    else:
                        if (
                            self.get_available_from_absolute(
                                sdict=resource_avail_in_time[res], time=current_min_time
                            )
                            < need
                        ):
                            current_min_time = next(
                                (
                                    k
                                    for k in resource_avail_in_time[res]
                                    if k > start_time
                                    and resource_avail_in_time[res][k] >= need
                                ),
                                None,
                            )
                            valid = False
                            break
                        keys = [
                            k
                            for k in resource_avail_in_time[res]
                            if start_time <= k < end_time
                        ]
                        for k in keys:
                            if resource_avail_in_time[res][k] < need:
                                current_min_time = next(
                                    (
                                        ki
                                        for ki in resource_avail_in_time[res]
                                        if ki > k
                                        and resource_avail_in_time[res][ki] >= need
                                    ),
                                    None,
                                )
                                valid = False
                                break
            if not unfeasible_non_renewable_resources:
                end_t = (
                    current_min_time
                    + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                )
                for res in resource_avail_in_time:
                    need = rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                        res, 0
                    )
                    if need == 0:
                        continue
                    if res not in rcpsp_problem.non_renewable_resources:
                        self.add_event_delta_in_absolute(
                            resource_avail_in_time[res],
                            time_start=current_min_time,
                            delta=-need,
                            time_end=end_t,
                        )
                    else:
                        self.add_event_delta_in_absolute(
                            resource_avail_in_time[res],
                            time_start=new_horizon,
                            delta=-need,
                            time_end=new_horizon + 2,
                        )
                        if resource_avail_in_time[res][new_horizon] < 0:
                            unfeasible_non_renewable_resources = True
                if unfeasible_non_renewable_resources:
                    break
                activity_end_times[act_id] = end_t
                perm_extended.remove(act_id)
                for s in rcpsp_problem.successors[act_id]:
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s], activity_end_times[act_id]
                    )
        rcpsp_schedule = {}
        for act_id in activity_end_times:
            rcpsp_schedule[act_id] = {}
            rcpsp_schedule[act_id]["start_time"] = (
                activity_end_times[act_id]
                - rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
            )
            rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
        if unfeasible_non_renewable_resources:
            rcpsp_schedule_feasible = False
            last_act_id = rcpsp_problem.sink_task
            if last_act_id not in rcpsp_schedule:
                rcpsp_schedule[last_act_id] = {}
                rcpsp_schedule[last_act_id]["start_time"] = 99999999
                rcpsp_schedule[last_act_id]["end_time"] = 9999999
        else:
            rcpsp_schedule_feasible = True
        return rcpsp_schedule, rcpsp_schedule_feasible, resource_avail_in_time
