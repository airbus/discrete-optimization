#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Optional

import discrete_optimization.rcpsp.solver.rcpsp_cp_lns_solver as rcpsp_lns
from discrete_optimization.generic_rcpsp_tools.generic_rcpsp_solver import (
    SolverGenericRCPSP,
)
from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    build_graph_rcpsp_object,
)
from discrete_optimization.generic_rcpsp_tools.ls_solver import (
    LS_SOLVER,
    LS_RCPSP_Solver,
)
from discrete_optimization.generic_rcpsp_tools.neighbor_builder import (
    OptionNeighborRandom,
    build_neighbor_mixing_cut_parts,
    build_neighbor_mixing_methods,
    build_neighbor_random,
    mix_both,
)
from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import (
    BasicConstraintBuilder,
    ConstraintHandlerScheduling,
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborConstraintBreaks,
    NeighborRandomAndNeighborGraph,
    ParamsConstraintBuilder,
)
from discrete_optimization.generic_rcpsp_tools.solution_repair import (
    NeighborRepairProblems,
)
from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP
from discrete_optimization.generic_tools.cp_tools import CPSolverName, MinizincCPSolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import LNS_CP, ConstraintHandler
from discrete_optimization.generic_tools.lns_mip import (
    InitialSolution,
    InitialSolutionFromSolver,
    PostProcessSolution,
)
from discrete_optimization.rcpsp.solver.cp_lns_methods_preemptive import (
    PostProLeftShift,
)
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CP_MRCPSP_MZN_PREEMPTIVE,
    CP_RCPSP_MZN,
    CP_RCPSP_MZN_PREEMPTIVE,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN,
    CP_MS_MRCPSP_MZN_PREEMPTIVE,
)
from discrete_optimization.rcpsp_multiskill.solvers.lns_post_process_rcpsp import (
    PostProMSRCPSP,
)


class ConstraintHandlerType(Enum):
    MIX_SUBPROBLEMS = 0
    SOLUTION_REPAIR = 1


def build_default_cp_model(rcpsp_problem: ANY_RCPSP, partial_solution=None, **kwargs):
    if rcpsp_problem.is_preemptive():
        if rcpsp_problem.is_multiskill():
            solver = CP_MS_MRCPSP_MZN_PREEMPTIVE(
                problem=rcpsp_problem,
                cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
            )
            solver.init_model(
                output_type=True,
                model_type="ms_rcpsp_preemptive",
                nb_preemptive=kwargs.get("nb_preemptive", 13),
                max_preempted=kwargs.get("max_preempted", 100),
                partial_solution=partial_solution,
                possibly_preemptive=[
                    rcpsp_problem.preemptive_indicator[t]
                    for t in rcpsp_problem.tasks_list
                ],
                unit_usage_preemptive=kwargs.get("unit_usage_preemptive", True),
                exact_skills_need=False,
                add_calendar_constraint_unit=False,
                add_partial_solution_hard_constraint=kwargs.get(
                    "add_partial_solution_hard_constraint", False
                ),
                **{
                    x: kwargs[x]
                    for x in kwargs
                    if x
                    not in [
                        "nb_preemptive",
                        "max_preempted",
                        "fake_tasks",
                        "add_partial_solution_hard_constraint",
                    ]
                },
            )
            return solver
    if rcpsp_problem.is_preemptive():
        if not rcpsp_problem.is_multiskill():
            if rcpsp_problem.is_rcpsp_multimode():
                solver = CP_MRCPSP_MZN_PREEMPTIVE(
                    problem=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                )
                solver.init_model(
                    output_type=True,
                    model_type="multi-preemptive",
                    nb_preemptive=kwargs.get("nb_preemptive", 13),
                    max_preempted=kwargs.get("max_preempted", 100),
                    fake_tasks=kwargs.get("fake_tasks", True),
                    partial_solution=partial_solution,
                    possibly_preemptive=[
                        rcpsp_problem.preemptive_indicator[t]
                        for t in rcpsp_problem.tasks_list
                    ],
                    add_partial_solution_hard_constraint=kwargs.get(
                        "add_partial_solution_hard_constraint", False
                    ),
                    **{
                        x: kwargs[x]
                        for x in kwargs
                        if x
                        not in [
                            "nb_preemptive",
                            "max_preempted",
                            "fake_tasks",
                            "add_partial_solution_hard_constraint",
                        ]
                    },
                )
                return solver
            if not rcpsp_problem.is_rcpsp_multimode():
                solver = CP_RCPSP_MZN_PREEMPTIVE(
                    problem=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                )
                solver.init_model(
                    output_type=True,
                    model_type="single-preemptive",
                    nb_preemptive=kwargs.get("nb_preemptive", 13),
                    max_preempted=kwargs.get("max_preempted", 100),
                    fake_tasks=kwargs.get("fake_tasks", True),
                    partial_solution=partial_solution,
                    possibly_preemptive=[
                        rcpsp_problem.preemptive_indicator[t]
                        for t in rcpsp_problem.tasks_list
                    ],
                    add_partial_solution_hard_constraint=kwargs.get(
                        "add_partial_solution_hard_constraint", False
                    ),
                    **{
                        x: kwargs[x]
                        for x in kwargs
                        if x
                        not in [
                            "nb_preemptive",
                            "max_preempted",
                            "fake_tasks",
                            "add_partial_solution_hard_constraint",
                        ]
                    },
                )
                return solver
    if not rcpsp_problem.is_preemptive():
        if rcpsp_problem.is_multiskill():
            solver = CP_MS_MRCPSP_MZN(
                problem=rcpsp_problem,
                cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                one_ressource_per_task=kwargs.get(
                    "one_ressource_per_task", rcpsp_problem.one_unit_per_task_max
                ),
            )
            solver.init_model(
                output_type=True,
                exact_skills_need=False,
                partial_solution=partial_solution,
                add_partial_solution_hard_constraint=kwargs.get(
                    "add_partial_solution_hard_constraint", False
                ),
                **{
                    x: kwargs[x]
                    for x in kwargs
                    if x not in ["add_partial_solution_hard_constraint"]
                },
            )
            return solver
        if not rcpsp_problem.is_multiskill():
            if rcpsp_problem.is_rcpsp_multimode():
                solver = CP_MRCPSP_MZN(
                    problem=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                )
                solver.init_model(
                    output_type=True,
                    partial_solution=partial_solution,
                    add_partial_solution_hard_constraint=kwargs.get(
                        "add_partial_solution_hard_constraint", False
                    ),
                    **{
                        x: kwargs[x]
                        for x in kwargs
                        if x not in ["add_partial_solution_hard_constraint"]
                    },
                )
                return solver
            if not rcpsp_problem.is_rcpsp_multimode():
                solver = CP_RCPSP_MZN(
                    problem=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                )
                solver.init_model(
                    output_type=True,
                    partial_solution=partial_solution,
                    add_partial_solution_hard_constraint=kwargs.get(
                        "add_partial_solution_hard_constraint", False
                    ),
                    **{
                        x: kwargs[x]
                        for x in kwargs
                        if x not in ["add_partial_solution_hard_constraint"]
                    },
                )
                return solver


def build_default_postpro(rcpsp_problem: ANY_RCPSP, partial_solution=None, **kwargs):
    if not rcpsp_problem.is_multiskill():
        if rcpsp_problem.is_preemptive():
            post_process_solution = PostProLeftShift(
                problem=rcpsp_problem,
                params_objective_function=None,
                do_ls=kwargs.get("do_ls", False),
            )
            return post_process_solution
        if partial_solution is not None:
            return None
        if not rcpsp_problem.is_preemptive():
            post_process_solution = rcpsp_lns.PostProcessLeftShift(
                rcpsp_problem=rcpsp_problem, partial_solution=partial_solution
            )
            return post_process_solution
    if rcpsp_problem.is_multiskill():
        if rcpsp_problem.is_preemptive():
            return None
        else:
            post_process_solution = PostProMSRCPSP(
                problem=rcpsp_problem, params_objective_function=None
            )
            return post_process_solution


def build_default_initial_solution(rcpsp_problem: ANY_RCPSP, **kwargs):
    if rcpsp_problem.is_multiskill():
        initial_solution_provider = InitialSolutionFromSolver(
            LS_RCPSP_Solver(problem=rcpsp_problem, ls_solver=LS_SOLVER.SA),
            nb_iteration_max=500,
        )
        return initial_solution_provider
    if not rcpsp_problem.is_multiskill():
        initial_solution_provider = InitialSolutionFromSolver(
            LS_RCPSP_Solver(problem=rcpsp_problem, ls_solver=LS_SOLVER.SA),
            nb_iteration_max=200,
        )
        return initial_solution_provider


def build_constraint_handler(rcpsp_problem: ANY_RCPSP, graph, **kwargs):
    constraint_handler_type = kwargs.get(
        "constraint_handler_type", ConstraintHandlerType.MIX_SUBPROBLEMS
    )
    constraint_handler = None
    if constraint_handler_type == ConstraintHandlerType.MIX_SUBPROBLEMS:
        n1 = NeighborBuilderSubPart(
            problem=rcpsp_problem,
            graph=graph,
            nb_cut_part=kwargs.get("nb_cut_part", 10),
        )
        n2 = NeighborRandomAndNeighborGraph(
            problem=rcpsp_problem,
            graph=graph,
            fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
        )
        if rcpsp_problem.includes_special_constraint():
            n3 = NeighborConstraintBreaks(
                problem=rcpsp_problem,
                graph=graph,
                fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
                other_constraint_handler=n1,
            )
            n_mix = NeighborBuilderMix(
                list_neighbor=[n1, n2, n3], weight_neighbor=[0.2, 0.5, 0.3]
            )
        else:
            n_mix = NeighborBuilderMix(
                list_neighbor=[n1, n2], weight_neighbor=[0.5, 0.5]
            )
        basic_constraint_builder = BasicConstraintBuilder(
            neighbor_builder=n_mix,
            preemptive=kwargs.get("preemptive", False),
            multiskill=kwargs.get("multiskill", False),
        )
        if "params_list" in kwargs:
            params_list = kwargs["params_list"]
        else:
            if kwargs["params_0_kwargs"] is None:
                params_0 = ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=400,
                    plus_delta_secondary=400,
                    constraint_max_time_to_current_solution=False,
                )
            else:
                params_0_cls = kwargs["params_0_cls"]
                params_0_kwargs = kwargs["params_0_kwargs"]
                params_0 = params_0_cls(**params_0_kwargs)
            if kwargs["params_1_kwargs"] is None:
                params_1 = ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=0,
                    plus_delta_secondary=0,
                    constraint_max_time_to_current_solution=False,
                )
            else:
                params_1_cls = kwargs["params_1_cls"]
                params_1_kwargs = kwargs["params_1_kwargs"]
                params_1 = params_1_cls(**params_1_kwargs)
            params_list = [params_0, params_1]
        constraint_handler = ConstraintHandlerScheduling(
            problem=rcpsp_problem,
            basic_constraint_builder=basic_constraint_builder,
            params_list=params_list,
            use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", False),
        )
    elif constraint_handler_type == ConstraintHandlerType.SOLUTION_REPAIR:
        if "params_list" in kwargs:
            params_list = kwargs["params_list"]
        else:
            if kwargs["params_0_kwargs"] is None:
                params_0 = ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=400,
                    plus_delta_secondary=400,
                    constraint_max_time_to_current_solution=False,
                )
            else:
                params_0_cls = kwargs["params_0_cls"]
                params_0_kwargs = kwargs["params_0_kwargs"]
                params_0 = params_0_cls(**params_0_kwargs)
            if kwargs["params_1_kwargs"] is None:
                params_1 = ParamsConstraintBuilder(
                    minus_delta_primary=5000,
                    plus_delta_primary=5000,
                    minus_delta_secondary=2,
                    plus_delta_secondary=2,
                    constraint_max_time_to_current_solution=False,
                )
            else:
                params_1_cls = kwargs["params_1_cls"]
                params_1_kwargs = kwargs["params_1_kwargs"]
                params_1 = params_1_cls(**params_1_kwargs)
            params_list = [params_0, params_1]
        constraint_handler = NeighborRepairProblems(
            problem=rcpsp_problem, params_list=params_list
        )
    return constraint_handler


def build_constraint_handler_helper(rcpsp_problem: ANY_RCPSP, graph, **kwargs):
    option = kwargs.get("option_neighbor_operator", 2)
    constraint_handler = None
    if option == 0:
        constraint_handler = build_neighbor_random(
            option_neighbor=kwargs.get(
                "option_neighbor_random", OptionNeighborRandom.MIX_ALL
            ),
            rcpsp_model=rcpsp_problem,
        )
    if option == 1:
        constraint_handler = mix_both(
            rcpsp_model=rcpsp_problem,
            option_neighbor_random=kwargs.get(
                "option_neighbor_random", OptionNeighborRandom.MIX_ALL
            ),
            graph=graph,
            fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
            cut_part=kwargs.get("cut_part", 10),
            params_list=kwargs.get(
                "params_list",
                [
                    ParamsConstraintBuilder(
                        minus_delta_primary=6000,
                        plus_delta_primary=6000,
                        minus_delta_secondary=400,
                        plus_delta_secondary=400,
                        constraint_max_time_to_current_solution=False,
                    ),
                    ParamsConstraintBuilder(
                        minus_delta_primary=6000,
                        plus_delta_primary=6000,
                        minus_delta_secondary=0,
                        plus_delta_secondary=0,
                        constraint_max_time_to_current_solution=False,
                    ),
                ],
            ),
        )
    if option == 2:

        constraint_handler = build_neighbor_mixing_methods(
            rcpsp_model=rcpsp_problem,
            graph=graph,
            fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
            cut_part=kwargs.get("cut_part", 10),
            params_list=kwargs.get(
                "params_list",
                [
                    ParamsConstraintBuilder(
                        minus_delta_primary=6000,
                        plus_delta_primary=6000,
                        minus_delta_secondary=400,
                        plus_delta_secondary=400,
                        constraint_max_time_to_current_solution=False,
                    ),
                    ParamsConstraintBuilder(
                        minus_delta_primary=6000,
                        plus_delta_primary=6000,
                        minus_delta_secondary=0,
                        plus_delta_secondary=0,
                        constraint_max_time_to_current_solution=False,
                    ),
                ],
            ),
        )
    if option == 3:
        constraint_handler = build_neighbor_mixing_cut_parts(
            rcpsp_model=rcpsp_problem,
            graph=graph,
            params_list=kwargs.get(
                "params_list",
                [
                    ParamsConstraintBuilder(
                        minus_delta_primary=6000,
                        plus_delta_primary=6000,
                        minus_delta_secondary=400,
                        plus_delta_secondary=400,
                        constraint_max_time_to_current_solution=False,
                    ),
                    ParamsConstraintBuilder(
                        minus_delta_primary=6000,
                        plus_delta_primary=6000,
                        minus_delta_secondary=0,
                        plus_delta_secondary=0,
                        constraint_max_time_to_current_solution=False,
                    ),
                ],
            ),
        )
    return constraint_handler


class LargeNeighborhoodSearchScheduling(LNS_CP, SolverGenericRCPSP):
    hyperparameters = [
        EnumHyperparameter(
            name="cp_solver_name", enum=CPSolverName, default=CPSolverName.CHUFFED
        ),
        CategoricalHyperparameter(name="do_ls", choices=[True, False], default=False),
        EnumHyperparameter(
            name="constraint_handler_type",
            enum=ConstraintHandlerType,
            default=ConstraintHandlerType.MIX_SUBPROBLEMS,
        ),
        FloatHyperparameter(
            name="fraction_subproblem", default=0.05, low=0.0, high=1.0
        ),
        IntegerHyperparameter(name="nb_cut_part", default=10, low=0, high=100),
        CategoricalHyperparameter(
            name="use_makespan_of_subtasks", choices=[True, False], default=False
        ),
        SubBrickHyperparameter(
            name="params_0_cls",
            choices=[ParamsConstraintBuilder],
            default=ParamsConstraintBuilder,
        ),
        SubBrickKwargsHyperparameter(
            name="params_0_kwargs", subbrick_hyperparameter="params_0_cls"
        ),
        SubBrickHyperparameter(
            name="params_1_cls",
            choices=[ParamsConstraintBuilder],
            default=ParamsConstraintBuilder,
        ),
        SubBrickKwargsHyperparameter(
            name="params_1_kwargs", subbrick_hyperparameter="params_1_cls"
        ),
    ]

    def __init__(
        self,
        problem: ANY_RCPSP,
        partial_solution=None,
        cp_solver: Optional[MinizincCPSolver] = None,
        initial_solution_provider: Optional[InitialSolution] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        SolverGenericRCPSP.__init__(
            self, problem=problem, params_objective_function=params_objective_function
        )
        graph = build_graph_rcpsp_object(self.problem)
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        if cp_solver is None:
            if "cp_solver_kwargs" not in kwargs or kwargs["cp_solver_kwargs"] is None:
                cp_solver_kwargs = kwargs
            else:
                cp_solver_kwargs = kwargs["cp_solver_kwargs"]
            if "cp_solver_cls" not in kwargs or kwargs["cp_solver_cls"] is None:
                cp_solver = build_default_cp_model(
                    rcpsp_problem=problem,
                    partial_solution=partial_solution,
                    **cp_solver_kwargs,
                )
            else:
                cp_solver_cls = kwargs["cp_solver_cls"]
                cp_solver = cp_solver_cls(problem=self.problem, **cp_solver_kwargs)
                cp_solver.init_model(**cp_solver_kwargs)
        self.cp_solver = cp_solver

        if constraint_handler is None:
            if (
                "constraint_handler_kwargs" not in kwargs
                or kwargs["constraint_handler_kwargs"] is None
            ):
                constraint_handler_kwargs = kwargs
            else:
                constraint_handler_kwargs = kwargs["constraint_handler_kwargs"]
            if (
                "constraint_handler_cls" not in kwargs
                or kwargs["constraint_handler_cls"] is None
            ):
                constraint_handler = build_constraint_handler(
                    rcpsp_problem=self.problem,
                    graph=graph,
                    multiskill=self.problem.is_multiskill(),
                    preemptive=self.problem.is_preemptive(),
                    **constraint_handler_kwargs,
                )
            else:
                constraint_handler_cls = kwargs["constraint_handler_cls"]
                constraint_handler = constraint_handler_cls(
                    problem=self.problem, **constraint_handler_kwargs
                )
        self.constraint_handler = constraint_handler

        if post_process_solution is None:
            if (
                "post_process_solution_kwargs" not in kwargs
                or kwargs["post_process_solution_kwargs"] is None
            ):
                post_process_solution_kwargs = kwargs
            else:
                post_process_solution_kwargs = kwargs["post_process_solution_kwargs"]
            if (
                "post_process_solution_cls" not in kwargs
                or kwargs["post_process_solution_cls"] is None
            ):
                post_process_solution = build_default_postpro(
                    rcpsp_problem=self.problem,
                    partial_solution=partial_solution,
                    **post_process_solution_kwargs,
                )
            else:
                post_process_solution_cls = kwargs["post_process_solution_cls"]
                post_process_solution = post_process_solution_cls(
                    problem=self.problem,
                    params_objective_function=self.params_objective_function,
                    **post_process_solution_kwargs,
                )
        self.post_process_solution = post_process_solution

        if initial_solution_provider is None:
            if (
                "initial_solution_provider_kwargs" not in kwargs
                or kwargs["initial_solution_provider_kwargs"] is None
            ):
                initial_solution_provider_kwargs = kwargs
            else:
                initial_solution_provider_kwargs = kwargs[
                    "initial_solution_provider_kwargs"
                ]
            if (
                "initial_solution_provider_cls" not in kwargs
                or kwargs["initial_solution_provider_cls"] is None
            ):
                initial_solution_provider = build_default_initial_solution(
                    rcpsp_problem=self.problem,
                    **initial_solution_provider_kwargs,
                )
            else:
                initial_solution_provider_cls = kwargs["initial_solution_provider_cls"]
                initial_solution_provider = initial_solution_provider_cls(
                    problem=self.problem,
                    params_objective_function=self.params_objective_function,
                    **initial_solution_provider_kwargs,
                )
        self.initial_solution_provider = initial_solution_provider
