# TODO : do it..
import os

from minizinc import Instance

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective

from discrete_optimization.generic_tools.cp_tools import CPSolver, CPSolverName, ParametersCP
from discrete_optimization.vrp.vrp_model import VrpProblem, VrpProblem2D

this_path = os.path.dirname(os.path.abspath(__file__))

files_mzn = {"vrp": os.path.join(this_path, "../minizinc/vrp.mzn"),
             "vrp-wip": os.path.join(this_path, "../minizinc/vrp-wip.mzn")}


def init_model_vpr_wip(vrp_model: VrpProblem):
    pass


class VRP_CP_SOLVER(CPSolver):
    def __init__(self, vrp_model: VrpProblem,
                 cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction = None, **kwargs):
        self.vrp_model = vrp_model
        self.instance: Instance = None
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["s"]  # For now, I've put the var name of the CP model (not the rcpsp_model)
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.vrp_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **args):
        model_type = args.get("model_type", "vrp")




