#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Hashable, List, Sequence, Tuple

import numpy as np
from scipy.stats import poisson, randint, rv_discrete

from discrete_optimization.generic_tools.do_problem import (
    MethodAggregating,
    RobustProblem,
    Solution,
)
from discrete_optimization.rcpsp import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution


def tree() -> Dict[Any, Any]:
    return defaultdict(tree)


class AggregRCPSPModel(RobustProblem, RCPSPModel):
    list_problem: Sequence[RCPSPModel]

    def __init__(
        self, list_problem: Sequence[RCPSPModel], method_aggregating: MethodAggregating
    ):
        RobustProblem.__init__(
            self, list_problem=list_problem, method_aggregating=method_aggregating
        )
        RCPSPModel.__init__(
            self,
            resources=list_problem[0].resources,
            non_renewable_resources=list_problem[0].non_renewable_resources,
            mode_details=list_problem[0].mode_details,
            successors=list_problem[0].successors,
            horizon=list_problem[0].horizon,
            horizon_multiplier=list_problem[0].horizon_multiplier,
            tasks_list=list_problem[0].tasks_list,
            source_task=list_problem[0].source_task,
            sink_task=list_problem[0].sink_task,
            name_task=list_problem[0].name_task,
            calendar_details=list_problem[0].calendar_details,
            special_constraints=list_problem[0].special_constraints
            if list_problem[0].do_special_constraints
            else None,
            relax_the_start_at_end=list_problem[0].relax_the_start_at_end,
            fixed_permutation=list_problem[0].fixed_permutation,
            fixed_modes=list_problem[0].fixed_modes,
        )

    def get_dummy_solution(self) -> RCPSPSolution:
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

    def evaluate_from_encoding(
        self, int_vector: List[int], encoding_name: str
    ) -> Dict[str, float]:
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

    def evaluate(self, variable: RCPSPSolution) -> Dict[str, float]:  # type: ignore
        fits = []
        for i in range(self.nb_problem):
            var: RCPSPSolution = variable.lazy_copy()
            var.rcpsp_schedule = {}
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


def create_poisson_laws_duration(
    rcpsp_model: RCPSPModel, range_around_mean: int = 3
) -> Dict[Hashable, Dict[int, Dict[str, Tuple[int, int, int]]]]:
    poisson_dict: Dict[Hashable, Dict[int, Dict[str, Tuple[int, int, int]]]] = {}
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


def create_poisson_laws_resource(
    rcpsp_model: RCPSPModel, range_around_mean: int = 1
) -> Dict[Hashable, Dict[int, Dict[str, Tuple[int, int, int]]]]:
    poisson_dict: Dict[Hashable, Dict[int, Dict[str, Tuple[int, int, int]]]] = {}
    source = rcpsp_model.source_task
    sink = rcpsp_model.sink_task
    limit_resource = rcpsp_model.resources
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
                        rcpsp_model.get_max_resource_capacity(resource),
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
) -> Dict[Hashable, Dict[int, Dict[str, Tuple[int, int, int]]]]:
    poisson_laws: Dict[Hashable, Dict[int, Dict[str, Tuple[int, int, int]]]] = tree()
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
        poisson_laws: Dict[Hashable, Dict[int, Dict[str, Tuple[int, int, int]]]],
        uniform_law: bool = True,
    ):
        self.base_rcpsp_model = base_rcpsp_model
        self.poisson_laws = poisson_laws
        self.probas: Dict[Hashable, Dict[int, Dict[str, Dict[str, Any]]]] = {}
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

    def create_rcpsp_model(
        self, method_robustification: MethodRobustification
    ) -> RCPSPModel:
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
