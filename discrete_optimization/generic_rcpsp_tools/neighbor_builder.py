#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from enum import Enum

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    GraphRCPSP,
    build_graph_rcpsp_object,
)
from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import (
    BasicConstraintBuilder,
    ConstraintHandlerMultiskillAllocation,
    ConstraintHandlerScheduling,
    EquilibrateMultiskillAllocation,
    EquilibrateMultiskillAllocationNonPreemptive,
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborBuilderTimeWindow,
    NeighborConstraintBreaks,
    NeighborRandom,
    NeighborRandomAndNeighborGraph,
    ObjectiveSubproblem,
    ParamsConstraintBuilder,
)
from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP
from discrete_optimization.generic_tools.lns_cp import (
    ConstraintHandler,
    ConstraintHandlerMix,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
)


class OptionNeighborRandom(Enum):
    MIX_ALL = 0
    MIX_FAST = 1
    MIX_LARGE_NEIGH = 2
    LARGE = 4
    DEBUG = 5
    NO_CONSTRAINT = 6


def return_random_basic_constraint_handler(
    rcpsp_model: ANY_RCPSP,
    graph=None,
    fraction_to_fix=0.9,
    minus_delta=100,
    plus_delta=100,
    minus_delta_2=4000,
    plus_delta_2=4000,
):
    return BasicConstraintBuilder(
        params_constraint_builder=ParamsConstraintBuilder(
            minus_delta_primary=minus_delta,
            plus_delta_primary=plus_delta,
            minus_delta_secondary=minus_delta_2,
            plus_delta_secondary=plus_delta_2,
        ),
        neighbor_builder=NeighborRandom(
            problem=rcpsp_model, graph=graph, fraction_subproblem=fraction_to_fix
        ),
    )


class Params:
    def __init__(self, fraction_to_fix, minus_delta, plus_delta):
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta


def build_neighbor_random(
    option_neighbor: OptionNeighborRandom, rcpsp_model: ANY_RCPSP
) -> ConstraintHandler:
    graph = build_graph_rcpsp_object(rcpsp_problem=rcpsp_model)
    params_use_case = [
        ConstraintHandlerScheduling(
            problem=rcpsp_model,
            basic_constraint_builder=BasicConstraintBuilder(
                params_constraint_builder=ParamsConstraintBuilder(
                    minus_delta_primary=100,
                    plus_delta_primary=100,
                    minus_delta_secondary=4000,
                    plus_delta_secondary=4000,
                ),
                neighbor_builder=NeighborRandom(
                    problem=rcpsp_model, graph=graph, fraction_subproblem=0.75
                ),
            ),
            params_list=[
                ParamsConstraintBuilder(
                    minus_delta_primary=100,
                    plus_delta_primary=100,
                    minus_delta_secondary=4000,
                    plus_delta_secondary=4000,
                )
            ],
            use_makespan_of_subtasks=False,
        )
    ]
    params_all = [
        ConstraintHandlerScheduling(
            problem=rcpsp_model,
            basic_constraint_builder=return_random_basic_constraint_handler(
                rcpsp_model,
                graph,
                fraction_to_fix=p.fraction_to_fix,
                minus_delta=p.minus_delta,
                plus_delta=p.plus_delta,
                minus_delta_2=4000,
                plus_delta_2=4000,
            ),
            params_list=[
                ParamsConstraintBuilder(
                    p.minus_delta,
                    p.plus_delta,
                    4000,
                    4000,
                    constraint_max_time_to_current_solution=True,
                )
            ],
            use_makespan_of_subtasks=False,
        )
        for p in [
            Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
            Params(fraction_to_fix=0.85, minus_delta=3, plus_delta=3),
            Params(fraction_to_fix=0.9, minus_delta=4, plus_delta=4),
            Params(fraction_to_fix=0.9, minus_delta=4, plus_delta=4),
            Params(fraction_to_fix=0.92, minus_delta=10, plus_delta=0),
            Params(fraction_to_fix=0.88, minus_delta=0, plus_delta=10),
            Params(fraction_to_fix=0.9, minus_delta=10, plus_delta=0),
            Params(fraction_to_fix=0.8, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.85, minus_delta=15, plus_delta=15),
            Params(fraction_to_fix=0.9, minus_delta=3, plus_delta=3),
            Params(fraction_to_fix=1.0, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.85, minus_delta=1, plus_delta=1),
            Params(fraction_to_fix=0.8, minus_delta=2, plus_delta=2),
            Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.95, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.95, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
            Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
            Params(fraction_to_fix=0.8, minus_delta=2, plus_delta=2),
            Params(fraction_to_fix=0.98, minus_delta=2, plus_delta=2),
            Params(fraction_to_fix=0.9, minus_delta=3, plus_delta=3),
            Params(fraction_to_fix=0.98, minus_delta=3, plus_delta=3),
            Params(fraction_to_fix=0.98, minus_delta=8, plus_delta=8),
            Params(fraction_to_fix=0.98, minus_delta=10, plus_delta=10),
        ]
    ]
    params_fast = [
        ConstraintHandlerScheduling(
            problem=rcpsp_model,
            basic_constraint_builder=return_random_basic_constraint_handler(
                rcpsp_model,
                graph,
                fraction_to_fix=p.fraction_to_fix,
                minus_delta=p.minus_delta,
                plus_delta=p.plus_delta,
                minus_delta_2=4000,
                plus_delta_2=4000,
            ),
            params_list=[
                ParamsConstraintBuilder(
                    p.minus_delta,
                    p.plus_delta,
                    4000,
                    4000,
                    constraint_max_time_to_current_solution=False,
                )
            ],
            use_makespan_of_subtasks=False,
        )
        for p in [
            Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
            Params(fraction_to_fix=0.8, minus_delta=1, plus_delta=1),
            Params(fraction_to_fix=0.8, minus_delta=2, plus_delta=2),
            Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
            Params(fraction_to_fix=0.92, minus_delta=3, plus_delta=3),
            Params(fraction_to_fix=0.98, minus_delta=7, plus_delta=7),
            Params(fraction_to_fix=0.95, minus_delta=5, plus_delta=5),
        ]
    ]
    params_debug = [
        ConstraintHandlerScheduling(
            problem=rcpsp_model,
            basic_constraint_builder=return_random_basic_constraint_handler(
                rcpsp_model,
                graph,
                fraction_to_fix=p.fraction_to_fix,
                minus_delta=p.minus_delta,
                plus_delta=p.plus_delta,
                minus_delta_2=4000,
                plus_delta_2=4000,
            ),
            params_list=[
                ParamsConstraintBuilder(
                    p.minus_delta,
                    p.plus_delta,
                    4000,
                    4000,
                    constraint_max_time_to_current_solution=False,
                )
            ],
            use_makespan_of_subtasks=False,
        )
        for p in [Params(fraction_to_fix=1.0, minus_delta=0, plus_delta=0)]
    ]
    params_large = [
        ConstraintHandlerScheduling(
            problem=rcpsp_model,
            basic_constraint_builder=return_random_basic_constraint_handler(
                rcpsp_model,
                graph,
                fraction_to_fix=p.fraction_to_fix,
                minus_delta=p.minus_delta,
                plus_delta=p.plus_delta,
                minus_delta_2=4000,
                plus_delta_2=4000,
            ),
            params_list=[
                ParamsConstraintBuilder(
                    minus_delta_primary=p.minus_delta,
                    plus_delta_primary=p.plus_delta,
                    minus_delta_secondary=4000,
                    plus_delta_secondary=4000,
                    constraint_max_time_to_current_solution=False,
                )
            ],
            use_makespan_of_subtasks=False,
        )
        for p in [
            Params(fraction_to_fix=0.9, minus_delta=12, plus_delta=12),
            Params(fraction_to_fix=0.8, minus_delta=3, plus_delta=3),
            Params(fraction_to_fix=0.7, minus_delta=12, plus_delta=12),
            Params(fraction_to_fix=0.7, minus_delta=5, plus_delta=5),
            Params(fraction_to_fix=0.6, minus_delta=3, plus_delta=3),
            Params(fraction_to_fix=0.4, minus_delta=2, plus_delta=2),
            Params(fraction_to_fix=0.9, minus_delta=4, plus_delta=4),
            Params(fraction_to_fix=0.7, minus_delta=4, plus_delta=4),
            Params(fraction_to_fix=0.8, minus_delta=5, plus_delta=5),
        ]
    ]
    params_no_constraint = [
        ConstraintHandlerScheduling(
            problem=rcpsp_model,
            basic_constraint_builder=return_random_basic_constraint_handler(
                rcpsp_model,
                graph,
                fraction_to_fix=p.fraction_to_fix,
                minus_delta=p.minus_delta,
                plus_delta=p.plus_delta,
                minus_delta_2=10000,
                plus_delta_2=10000,
            ),
            params_list=[
                ParamsConstraintBuilder(
                    p.minus_delta,
                    p.plus_delta,
                    4000,
                    4000,
                    constraint_max_time_to_current_solution=False,
                )
            ],
            use_makespan_of_subtasks=False,
        )
        for p in [Params(0.0, 4000, 4000)]
    ]
    constraints_handler = None
    if option_neighbor.name == OptionNeighborRandom.MIX_ALL.name:
        constraints_handler = params_all
    if option_neighbor.name == OptionNeighborRandom.MIX_FAST.name:
        constraints_handler = params_fast
    if option_neighbor.name == OptionNeighborRandom.MIX_LARGE_NEIGH.name:
        constraints_handler = params_large
    if option_neighbor.name == OptionNeighborRandom.DEBUG.name:
        constraints_handler = params_debug
    if option_neighbor.name == OptionNeighborRandom.LARGE.name:
        constraints_handler = params_use_case
    if option_neighbor.name == OptionNeighborRandom.NO_CONSTRAINT.name:
        constraints_handler = params_no_constraint
    probas = [1 / len(constraints_handler)] * len(constraints_handler)
    constraint_handler = ConstraintHandlerMix(
        problem=rcpsp_model,
        list_constraints_handler=constraints_handler,
        list_proba=probas,
    )
    return constraint_handler


def build_constraint_handler_cut_part(rcpsp_problem: ANY_RCPSP, graph=None, **kwargs):
    n1 = NeighborBuilderSubPart(
        problem=rcpsp_problem, graph=graph, nb_cut_part=kwargs.get("nb_cut_part", 10)
    )
    basic_constraint_builder = BasicConstraintBuilder(
        neighbor_builder=n1,
        preemptive=kwargs.get("preemptive", False),
        multiskill=kwargs.get("multiskill", False),
    )
    params_list = kwargs.get(
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
    )
    constraint_handler = ConstraintHandlerScheduling(
        problem=rcpsp_problem,
        basic_constraint_builder=basic_constraint_builder,
        params_list=params_list,
        use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", False),
        objective_subproblem=kwargs.get(
            "objective_subproblem", ObjectiveSubproblem.GLOBAL_MAKESPAN
        ),
    )
    return constraint_handler


def build_basic_random_constraint_handler(rcpsp_problem: ANY_RCPSP, graph, **kwargs):
    return ConstraintHandlerScheduling(
        problem=rcpsp_problem,
        basic_constraint_builder=BasicConstraintBuilder(
            neighbor_builder=NeighborRandom(
                problem=rcpsp_problem,
                graph=graph,
                fraction_subproblem=kwargs.get("fraction_subproblem", 0.1),
                delta_abs_time_from_makespan_to_not_fix=kwargs.get(
                    "delta_abs_time_from_makespan_to_not_fix", 5
                ),
                delta_rel_time_from_makespan_to_not_fix=kwargs.get(
                    "delta_rel_time_from_makespan_to_not_fix", 0.1
                ),
            ),
            multiskill=kwargs.get("multiskill", False),
            preemptive=kwargs.get("preemptive", False),
        ),
        params_list=kwargs.get(
            "params_list",
            [
                ParamsConstraintBuilder(
                    minus_delta_primary=100,
                    plus_delta_primary=100,
                    minus_delta_secondary=0,
                    plus_delta_secondary=0,
                )
            ],
        ),
        use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", True),
        objective_subproblem=kwargs.get(
            "objective_subproblem", ObjectiveSubproblem.GLOBAL_MAKESPAN
        ),
    )


def build_basic_time_window_constraint_handler(
    rcpsp_problem: ANY_RCPSP, graph, **kwargs
):
    return ConstraintHandlerScheduling(
        problem=rcpsp_problem,
        basic_constraint_builder=BasicConstraintBuilder(
            neighbor_builder=NeighborBuilderTimeWindow(
                problem=rcpsp_problem,
                graph=graph,
                time_window_length=kwargs.get("time_window_length", 10),
            ),
            multiskill=kwargs.get("multiskill", False),
            preemptive=kwargs.get("preemptive", False),
        ),
        params_list=kwargs.get(
            "params_list",
            [
                ParamsConstraintBuilder(
                    minus_delta_primary=100,
                    plus_delta_primary=100,
                    minus_delta_secondary=0,
                    plus_delta_secondary=0,
                )
            ],
        ),
        use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", True),
        objective_subproblem=kwargs.get(
            "objective_subproblem", ObjectiveSubproblem.GLOBAL_MAKESPAN
        ),
    )


def build_basic_random_and_neighbor(rcpsp_problem: ANY_RCPSP, graph, **kwargs):
    return ConstraintHandlerScheduling(
        problem=rcpsp_problem,
        basic_constraint_builder=BasicConstraintBuilder(
            neighbor_builder=NeighborRandomAndNeighborGraph(
                problem=rcpsp_problem,
                graph=graph,
                fraction_subproblem=kwargs.get("fraction_subproblem", 0.1),
            ),
            multiskill=kwargs.get("multiskill", False),
            preemptive=kwargs.get("preemptive", False),
        ),
        params_list=kwargs.get(
            "params_list",
            [
                ParamsConstraintBuilder(
                    minus_delta_primary=100,
                    plus_delta_primary=100,
                    minus_delta_secondary=0,
                    plus_delta_secondary=0,
                )
            ],
        ),
        use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", True),
        objective_subproblem=kwargs.get(
            "objective_subproblem", ObjectiveSubproblem.GLOBAL_MAKESPAN
        ),
    )


def build_neighbor_mixing_methods(
    rcpsp_model: ANY_RCPSP, graph: GraphRCPSP = None, **kwargs
):
    if graph is None:
        graph = build_graph_rcpsp_object(rcpsp_problem=rcpsp_model)
    n1 = NeighborBuilderSubPart(
        problem=rcpsp_model, graph=graph, nb_cut_part=kwargs.get("cut_part", 10)
    )
    n2 = NeighborRandomAndNeighborGraph(
        problem=rcpsp_model,
        graph=graph,
        fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
    )
    n3 = NeighborConstraintBreaks(
        problem=rcpsp_model,
        graph=graph,
        fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
        other_constraint_handler=n1,
    )
    n_mix = NeighborBuilderMix(
        list_neighbor=[n1, n2, n3], weight_neighbor=[0.3, 0.3, 0.4]
    )
    basic_constraint_builder = BasicConstraintBuilder(
        params_constraint_builder=ParamsConstraintBuilder(
            minus_delta_primary=6000,
            plus_delta_primary=6000,
            minus_delta_secondary=400,
            plus_delta_secondary=400,
            constraint_max_time_to_current_solution=False,
        ),
        neighbor_builder=n_mix,
        preemptive=rcpsp_model.is_preemptive(),
        multiskill=isinstance(rcpsp_model, (MS_RCPSPModel_Variant, MS_RCPSPModel)),
    )
    params_list = kwargs.get(
        "params_list",
        [
            ParamsConstraintBuilder(
                minus_delta_primary=6000,
                plus_delta_primary=6000,
                minus_delta_secondary=10,
                plus_delta_secondary=10,
                constraint_max_time_to_current_solution=False,
            )
        ],
    )
    constraint_handler = ConstraintHandlerScheduling(
        problem=rcpsp_model,
        basic_constraint_builder=basic_constraint_builder,
        params_list=params_list,
    )
    return constraint_handler


def build_neighbor_mixing_cut_parts(
    rcpsp_model: ANY_RCPSP, graph: GraphRCPSP = None, **kwargs
):
    if graph is None:
        graph = build_graph_rcpsp_object(rcpsp_problem=rcpsp_model)

    n1_s = [
        NeighborBuilderSubPart(problem=rcpsp_model, graph=graph, nb_cut_part=c_part)
        for c_part in [4, 5, 6, 7]
    ]
    n2_s = []
    n_mix = NeighborBuilderMix(
        list_neighbor=n1_s + n2_s,
        weight_neighbor=[1 / (len(n1_s) + len(n2_s))] * (len(n1_s) + len(n2_s)),
    )
    basic_constraint_builder = BasicConstraintBuilder(
        params_constraint_builder=ParamsConstraintBuilder(
            minus_delta_primary=6000,
            plus_delta_primary=6000,
            minus_delta_secondary=400,
            plus_delta_secondary=400,
            constraint_max_time_to_current_solution=False,
        ),
        neighbor_builder=n_mix,
        preemptive=rcpsp_model.is_preemptive(),
        multiskill=isinstance(rcpsp_model, (MS_RCPSPModel_Variant, MS_RCPSPModel)),
    )
    params_list = kwargs.get(
        "params_list",
        [
            ParamsConstraintBuilder(
                minus_delta_primary=6000,
                plus_delta_primary=6000,
                minus_delta_secondary=10,
                plus_delta_secondary=10,
                constraint_max_time_to_current_solution=False,
            )
        ],
    )
    constraint_handler = ConstraintHandlerScheduling(
        problem=rcpsp_model,
        basic_constraint_builder=basic_constraint_builder,
        params_list=params_list,
    )
    return constraint_handler


def mix_both(
    rcpsp_model: ANY_RCPSP,
    option_neighbor_random: OptionNeighborRandom,
    graph: GraphRCPSP = None,
    **kwargs
):
    a_s = [
        build_neighbor_random(
            option_neighbor=option_neighbor_random, rcpsp_model=rcpsp_model
        )
    ]
    b = build_neighbor_mixing_methods(rcpsp_model, graph=graph, **kwargs)
    list_c = a_s + [b]
    return ConstraintHandlerMix(
        problem=rcpsp_model,
        list_constraints_handler=list_c,
        list_proba=[1 / len(list_c)] * len(list_c),
        update_proba=False,
    )


def random_neigh(rcpsp_model, fraction_subproblem: float = 0.35, **kwargs):
    return build_basic_random_constraint_handler(
        rcpsp_problem=rcpsp_model,
        graph=build_graph_rcpsp_object(rcpsp_problem=rcpsp_model),
        params_list=kwargs.get(
            "params_list",
            [
                ParamsConstraintBuilder(
                    minus_delta_primary=10,
                    plus_delta_primary=10,
                    minus_delta_secondary=2,
                    plus_delta_secondary=2,
                    constraint_max_time_to_current_solution=True,
                )
            ],
        ),
        preemptive=kwargs.get("preemptive", rcpsp_model.is_preemptive()),
        multiskill=kwargs.get("multiskill", rcpsp_model.is_multiskill()),
        use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", False),
        objective_subproblem=kwargs.get(
            "objective_subproblem", ObjectiveSubproblem.GLOBAL_MAKESPAN
        ),
        fraction_subproblem=fraction_subproblem,
        delta_rel_time_from_makespan_to_not_fix=0.1,
        delta_abs_time_from_makespan_to_not_fix=5,
    )


def time_window_neigh(rcpsp_model, **kwargs):
    return build_basic_time_window_constraint_handler(
        rcpsp_problem=rcpsp_model,
        graph=build_graph_rcpsp_object(rcpsp_problem=rcpsp_model),
        params_list=kwargs.get(
            "params_list",
            [
                ParamsConstraintBuilder(
                    minus_delta_primary=10,
                    plus_delta_primary=10,
                    minus_delta_secondary=2,
                    plus_delta_secondary=2,
                    constraint_max_time_to_current_solution=True,
                )
            ],
        ),
        preemptive=kwargs.get("preemptive", rcpsp_model.is_preemptive()),
        multiskill=kwargs.get("multiskill", rcpsp_model.is_multiskill()),
        use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", False),
        objective_subproblem=kwargs.get(
            "objective_subproblem", ObjectiveSubproblem.GLOBAL_MAKESPAN
        ),
        time_window_length=kwargs.get("time_window_length", 10),
    )


def constraint_neigh(rcpsp_model, fraction_subproblem: float = 0.35, **kwargs):
    gr = build_graph_rcpsp_object(rcpsp_problem=rcpsp_model)
    n1 = NeighborBuilderSubPart(
        problem=rcpsp_model,
        graph=build_graph_rcpsp_object(rcpsp_problem=rcpsp_model),
        nb_cut_part=kwargs.get("nb_cut_part", 3),
    )
    n2 = NeighborConstraintBreaks(
        problem=rcpsp_model,
        graph=gr,
        fraction_subproblem=fraction_subproblem,
        other_constraint_handler=n1,
    )
    return ConstraintHandlerScheduling(
        problem=rcpsp_model,
        basic_constraint_builder=BasicConstraintBuilder(
            neighbor_builder=n2,
            preemptive=kwargs.get("preemptive", rcpsp_model.is_preemptive()),
            multiskill=kwargs.get("multiskill", rcpsp_model.is_multiskill()),
        ),
        params_list=kwargs.get(
            "params_list",
            [
                ParamsConstraintBuilder(
                    minus_delta_primary=100,
                    plus_delta_primary=100,
                    minus_delta_secondary=0,
                    plus_delta_secondary=0,
                    fraction_of_task_assigned_multiskill=0.8,
                    constraint_max_time_to_current_solution=False,
                )
            ],
        ),
        use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", False),
        objective_subproblem=kwargs.get(
            "objective_subproblem", ObjectiveSubproblem.GLOBAL_MAKESPAN
        ),
    )


def cut_parts(rcpsp_model, nb_cut_part=4, **kwargs):
    constraint_handler = build_constraint_handler_cut_part(
        rcpsp_problem=rcpsp_model,
        nb_cut_part=nb_cut_part,
        preemptive=kwargs.get("preemptive", rcpsp_model.is_preemptive()),
        multiskill=kwargs.get("multiskill", rcpsp_model.is_multiskill()),
        graph=build_graph_rcpsp_object(rcpsp_problem=rcpsp_model),
        params_list=kwargs.get(
            "params_list",
            [
                ParamsConstraintBuilder(
                    minus_delta_primary=10,
                    plus_delta_primary=10,
                    minus_delta_secondary=1,
                    plus_delta_secondary=1,
                    constraint_max_time_to_current_solution=True,
                )
            ],
        ),
        use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", False),
        objective_subproblem=kwargs.get(
            "objective_subproblem", ObjectiveSubproblem.GLOBAL_MAKESPAN
        ),
        verbose=True,
    )
    return constraint_handler


def mix(rcpsp_model, nb_cut_part=3, fraction_subproblem=0.25):
    c1 = cut_parts(rcpsp_model, nb_cut_part)
    c2 = random_neigh(rcpsp_model, fraction_subproblem=fraction_subproblem)
    return ConstraintHandlerMix(
        problem=rcpsp_model,
        list_constraints_handler=[c1, c2],
        list_proba=[0.5, 0.5],
        update_proba=True,
        tag_constraint_handler=["cut_parts", "random"],
    )


def mix_lot(rcpsp_model, nb_cut_parts, fraction_subproblems, **kwargs):
    c1 = [cut_parts(rcpsp_model, nb_cut_part, **kwargs) for nb_cut_part in nb_cut_parts]
    c2 = [
        random_neigh(rcpsp_model, fraction_subproblem=fraction_subproblem, **kwargs)
        for fraction_subproblem in fraction_subproblems
    ]
    tags = ["cut_parts_" + str(c) for c in nb_cut_parts]
    tags += ["random_" + str(f) for f in fraction_subproblems]
    if (
        "generalized_precedence_constraint" in kwargs
        and kwargs["generalized_precedence_constraint"]
    ):
        c3 = [constraint_neigh(rcpsp_model, fraction_subproblem=0.25, **kwargs)]
        tags += ["generalized_precedence_constraint"]
    else:
        c3 = []
    c4 = []
    if "time_windows" in kwargs and kwargs["time_windows"]:
        c4 = [time_window_neigh(rcpsp_model=rcpsp_model, **kwargs)]
        tags += ["time_windows"]

    c5 = []
    if kwargs.get("equilibrate_multiskill", False):
        if rcpsp_model.is_preemptive():
            c5 = [
                EquilibrateMultiskillAllocation(problem=rcpsp_model)
                for i in range(max(1, int((len(c1) + len(c2) + len(c3)) / 2)))
            ]
            tags += ["equilibrate_" + str(i) for i in range(len(c5))]
        else:
            c5 = [
                EquilibrateMultiskillAllocationNonPreemptive(problem=rcpsp_model)
                for i in range(max(1, int((len(c1) + len(c2) + len(c3)) / 2)))
            ]
            tags += ["equilibrate_" + str(i) for i in range(len(c5))]
    if kwargs.get("equilibrate_multiskill_v2", False):
        c5 = [
            ConstraintHandlerMultiskillAllocation(problem=rcpsp_model)
            for i in range(max(1, int((len(c1) + len(c2) + len(c3)) / 2)))
        ]
        tags += ["equilibrate_" + str(i) for i in range(len(c5))]

    return ConstraintHandlerMix(
        problem=rcpsp_model,
        list_constraints_handler=c1 + c2 + c3 + c4 + c5,
        list_proba=[1 / (len(c1) + len(c2) + len(c3) + len(c4) + len(c5))]
        * (len(c1) + len(c2) + len(c3) + len(c4) + len(c5)),
        update_proba=False,
        tag_constraint_handler=tags,
    )
