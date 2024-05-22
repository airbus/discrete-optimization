import os
import subprocess
import warnings

from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.mis_model import MisSolution
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver

this_folder = os.path.abspath(os.path.dirname(__file__))
folder_ = os.environ["KAMIS_DEPLOY"]
redumis = os.path.join(folder_, "redumis")
online_mis = os.path.join(folder_, "online_mis")
weighted = os.path.join(folder_, "weighted_branch_reduce")
weighted_local_search = os.path.join(folder_, "weighted_local_search")
methods = {
    "redumis": redumis,
    "weighted": weighted,
    "weighted_local_search": weighted_local_search,
    "online_mis": online_mis,
}
tmp_dir = os.path.join(this_folder, "tmp/")
import logging
import time

logger = logging.getLogger(__file__)


class MisKamisSolver(MisSolver):
    """
    To use Kamis Solver you need to define an environment variable "KAMIS_DEPLOY" who is the path
    to the folder where are Kamis executable.
    """

    def init_model(self):

        t = time.time_ns()
        file = os.path.join(tmp_dir, f"file_{t}.graph")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        lines_to_write = []
        lines_to_write += [
            f"{self.problem.number_nodes} {len(self.problem.edges)} {11} \n"
        ]
        nb_edge = 0
        for i in range(self.problem.number_nodes):
            node = self.problem.index_to_nodes[i]
            neighs = sorted(
                self.problem.graph.neighbors(node),
                key=lambda x: self.problem.nodes_to_index[x],
            )
            s = " ".join(
                [
                    f"{int(self.problem.attr_list[i])} "
                    + f"{self.problem.nodes_to_index[x] + 1}"
                    for x in neighs
                ]
            )
            nb_edge += len(neighs)
            lines_to_write += [s + "\n"]

        with open(file, "w") as f:
            f.writelines(lines_to_write)

        self.current_tag = t
        self.current_file = file

    def solve(self, **kwargs) -> ResultStorage:

        if folder_ is None:
            warnings.warn(
                'The environment variable "KAMIS_DEPLOY" is not define, you need to define it as the path to the folder where are Kamis executable'
            )

        time_limit = kwargs.get("time_limit", 60)
        method = kwargs.get("method", "weighted")

        if method not in methods:
            logger.warning(
                f"{method} is not available, please use one of : {list(methods.keys())}"
            )
            logger.warning("setting to default value weighted_local_search")
            method = "weighted_local_search"

        output = os.path.join(tmp_dir, f"res_{self.current_tag}.txt")
        command = (
            f"{methods[method]} {self.current_file} "
            f"--output {output} --time_limit={time_limit} --console_log"
        )

        # Launch the command and wait for it to finish
        print("Launching command")
        try:
            result = subprocess.run(
                command, shell=True, check=True, timeout=time_limit + 50
            )
            chosen = []
            with open(output, "r") as file:
                for line in file:
                    # Convert each line to an integer and append to the list
                    number = int(
                        line.strip()
                    )  # Remove leading/trailing whitespaces and convert to int
                    chosen.append(number)
            os.remove(self.current_file)
            sol = MisSolution(problem=self.problem, chosen=chosen)
            fit = self.aggreg_from_sol(sol)
            return ResultStorage(
                [(sol, fit)], mode_optim=self.params_objective_function.sense_function
            )
        except:
            print(
                "A problem has been encountered, exceptionally Kamis ignored time-out"
            )
            sol = MisSolution(problem=self.problem, chosen=[])
            fit = self.aggreg_from_sol(sol)
            return ResultStorage(
                [(sol, fit)], mode_optim=self.params_objective_function.sense_function
            )
