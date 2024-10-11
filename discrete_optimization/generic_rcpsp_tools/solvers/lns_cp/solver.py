#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Optional

from discrete_optimization.generic_rcpsp_tools.graph_tools import (
    build_graph_rcpsp_object,
)
from discrete_optimization.generic_rcpsp_tools.solvers import GenericRcpspSolver
from discrete_optimization.generic_rcpsp_tools.solvers.lns_cp.neighbor_builder import (
    OptionNeighborRandom,
    build_neighbor_mixing_cut_parts,
    build_neighbor_mixing_methods,
    build_neighbor_random,
    mix_both,
)
from discrete_optimization.generic_rcpsp_tools.solvers.lns_cp.neighbor_tools import (
    BasicConstraintBuilder,
    ConstraintHandlerScheduling,
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborConstraintBreaks,
    NeighborRandomAndNeighborGraph,
    ParamsConstraintBuilder,
)
from discrete_optimization.generic_rcpsp_tools.solvers.lns_cp.solution_repair import (
    NeighborRepairProblems,
)
from discrete_optimization.generic_rcpsp_tools.solvers.ls import (
    LsGenericRcpspSolver,
    LsSolverType,
)
from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP
from discrete_optimization.generic_tools.cp_tools import CpSolverName, MinizincCpSolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    SubBrickKwargsHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import LnsCpMzn, MznConstraintHandler
from discrete_optimization.generic_tools.lns_mip import (
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.lns_tools import InitialSolutionFromSolver
from discrete_optimization.rcpsp.solvers.cp_mzn import (
    CpMultimodePreemptiveRcpspSolver,
    CpMultimodeRcpspSolver,
    CpPreemptiveRcpspSolver,
    CpRcpspSolver,
)
from discrete_optimization.rcpsp.solvers.lns_cp import PostProcessLeftShift
from discrete_optimization.rcpsp.solvers.lns_cp_preemptive import PostProLeftShift
from discrete_optimization.rcpsp_multiskill.solvers.cp_mzn import (
    CpMultiskillRcpspSolver,
    CpPreemptiveMultiskillRcpspSolver,
)
from discrete_optimization.rcpsp_multiskill.solvers.lns_post_process import (
    MultiskillRcpspPostProcessSolution,
)


class ConstraintHandlerType(Enum):
    MIX_SUBPROBLEMS = 0
    SOLUTION_REPAIR = 1


def build_default_cp_model(rcpsp_problem: ANY_RCPSP, partial_solution=None, **kwargs):
    if rcpsp_problem.is_preemptive():
        if rcpsp_problem.is_multiskill():
            solver = CpPreemptiveMultiskillRcpspSolver(
                problem=rcpsp_problem,
                cp_solver_name=kwargs.get("cp_solver_name", CpSolverName.CHUFFED),
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
                solver = CpMultimodePreemptiveRcpspSolver(
                    problem=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CpSolverName.CHUFFED),
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
                solver = CpPreemptiveRcpspSolver(
                    problem=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CpSolverName.CHUFFED),
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
            solver = CpMultiskillRcpspSolver(
                problem=rcpsp_problem,
                cp_solver_name=kwargs.get("cp_solver_name", CpSolverName.CHUFFED),
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
                solver = CpMultimodeRcpspSolver(
                    problem=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CpSolverName.CHUFFED),
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
                solver = CpRcpspSolver(
                    problem=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CpSolverName.CHUFFED),
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
            post_process_solution = PostProcessLeftShift(
                rcpsp_problem=rcpsp_problem, partial_solution=partial_solution
            )
            return post_process_solution
    if rcpsp_problem.is_multiskill():
        if rcpsp_problem.is_preemptive():
            return None
        else:
            post_process_solution = MultiskillRcpspPostProcessSolution(
                problem=rcpsp_problem, params_objective_function=None
            )
            return post_process_solution


def build_default_initial_solution(rcpsp_problem: ANY_RCPSP, **kwargs):
    if rcpsp_problem.is_multiskill():
        initial_solution_provider = InitialSolutionFromSolver(
            LsGenericRcpspSolver(problem=rcpsp_problem, ls_solver=LsSolverType.SA),
            nb_iteration_max=500,
        )
        return initial_solution_provider
    if not rcpsp_problem.is_multiskill():
        initial_solution_provider = InitialSolutionFromSolver(
            LsGenericRcpspSolver(problem=rcpsp_problem, ls_solver=LsSolverType.SA),
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
                params_0_kwargs = kwargs["params_0_kwargs"]
                params_0 = ParamsConstraintBuilder(**params_0_kwargs)
            if kwargs["params_1_kwargs"] is None:
                params_1 = ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=0,
                    plus_delta_secondary=0,
                    constraint_max_time_to_current_solution=False,
                )
            else:
                params_1_kwargs = kwargs["params_1_kwargs"]
                params_1 = ParamsConstraintBuilder(**params_1_kwargs)
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
                params_0_kwargs = kwargs["params_0_kwargs"]
                params_0 = ParamsConstraintBuilder(**params_0_kwargs)
            if kwargs["params_1_kwargs"] is None:
                params_1 = ParamsConstraintBuilder(
                    minus_delta_primary=5000,
                    plus_delta_primary=5000,
                    minus_delta_secondary=2,
                    plus_delta_secondary=2,
                    constraint_max_time_to_current_solution=False,
                )
            else:
                params_1_kwargs = kwargs["params_1_kwargs"]
                params_1 = ParamsConstraintBuilder(**params_1_kwargs)
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
            rcpsp_problem=rcpsp_problem,
        )
    if option == 1:
        constraint_handler = mix_both(
            rcpsp_problem=rcpsp_problem,
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
            rcpsp_problem=rcpsp_problem,
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
            rcpsp_problem=rcpsp_problem,
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


class LnsCpMznGenericRcpspSolver(LnsCpMzn, GenericRcpspSolver):
    hyperparameters = [
        EnumHyperparameter(
            name="cp_solver_name", enum=CpSolverName, default=CpSolverName.CHUFFED
        ),
        CategoricalHyperparameter(name="do_ls", choices=[True, False], default=False),
        EnumHyperparameter(
            name="constraint_handler_type",
            enum=ConstraintHandlerType,
            default=ConstraintHandlerType.MIX_SUBPROBLEMS,
        ),
        FloatHyperparameter(
            name="fraction_subproblem",
            default=0.05,
            low=0.0,
            high=1.0,
            depends_on=(
                "constraint_handler_type",
                [ConstraintHandlerType.MIX_SUBPROBLEMS],
            ),
        ),
        IntegerHyperparameter(
            name="nb_cut_part",
            default=10,
            low=0,
            high=100,
            depends_on=(
                "constraint_handler_type",
                [ConstraintHandlerType.MIX_SUBPROBLEMS],
            ),
        ),
        CategoricalHyperparameter(
            name="use_makespan_of_subtasks",
            choices=[True, False],
            default=False,
            depends_on=(
                "constraint_handler_type",
                [ConstraintHandlerType.MIX_SUBPROBLEMS],
            ),
        ),
        SubBrickKwargsHyperparameter(
            name="params_0_kwargs", subbrick_cls=ParamsConstraintBuilder
        ),
        SubBrickKwargsHyperparameter(
            name="params_1_kwargs", subbrick_cls=ParamsConstraintBuilder
        ),
    ]

    def __init__(
        self,
        problem: ANY_RCPSP,
        partial_solution=None,
        subsolver: Optional[MinizincCpSolver] = None,
        initial_solution_provider: Optional[InitialSolution] = None,
        constraint_handler: Optional[MznConstraintHandler] = None,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        GenericRcpspSolver.__init__(
            self, problem=problem, params_objective_function=params_objective_function
        )
        graph = build_graph_rcpsp_object(self.problem)
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        if subsolver is None:
            subsolver = build_default_cp_model(
                rcpsp_problem=problem,
                partial_solution=partial_solution,
                **kwargs,
            )
        self.subsolver = subsolver

        if constraint_handler is None:
            constraint_handler = build_constraint_handler(
                rcpsp_problem=self.problem,
                graph=graph,
                multiskill=self.problem.is_multiskill(),
                preemptive=self.problem.is_preemptive(),
                **kwargs,
            )
        self.constraint_handler = constraint_handler

        if post_process_solution is None:
            post_process_solution = build_default_postpro(
                rcpsp_problem=self.problem,
                partial_solution=partial_solution,
                **kwargs,
            )
        self.post_process_solution = post_process_solution

        if initial_solution_provider is None:
            initial_solution_provider = build_default_initial_solution(
                rcpsp_problem=self.problem,
                **kwargs,
            )
        self.initial_solution_provider = initial_solution_provider
