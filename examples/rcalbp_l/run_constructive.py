import logging
import random

from discrete_optimization.rcalbp_l.parser import parse_rcalbpl_json
from discrete_optimization.rcalbp_l.problem import (
    RCALBPLVectorSolution,
)

logging.basicConfig(level=logging.INFO)


def main():
    problem = parse_rcalbpl_json("instances/187_2_26_2880.json")
    sol = RCALBPLVectorSolution(
        problem=problem,
        allocation_task=[
            random.randint(0, problem.nb_stations - 1) for t in problem.tasks
        ],
        permutation_task=problem.tasks,
        resource=[
            problem.capa_resources[r] // 2
            for w in problem.stations
            for r in problem.resources
        ],
    )
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    main()
