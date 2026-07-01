#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Uncapacitated single item lot sizing problem
from discrete_optimization.generic_tools.do_problem import ObjectiveRegister
from discrete_optimization.lotsizing.problem import LotSizingProblem


class UncapacitatedLotSizingProblem(LotSizingProblem):
    def __init__(
        self,
        demands: list[int],
        stock_capacity: int,
        stock_cost_per_time_per_unit: list[int],
        delay_cost_per_time_per_unit: list[int],
        allow_delays: bool = False,
    ):
        super().__init__(
            nb_items_type=1,
            capacity_machine=None,
            changeover_costs=[[0]],
            demands=[demands],
            stock_capacity=stock_capacity,
            stock_cost_per_type_per_time_per_unit=[stock_cost_per_time_per_unit],
            delay_cost_per_type_per_time_per_unit=[delay_cost_per_time_per_unit],
            allow_delays=allow_delays,
            known_bound=None,
        )

    def get_objective_register(self) -> ObjectiveRegister:
        obj = super().get_objective_register()
        obj.dict_objective_to_doc.pop("changeover")
        return obj
