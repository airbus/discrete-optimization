import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))
from copy import deepcopy

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    Problem,
    Solution,
    TypeAttribute,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)


class Permutation(Solution):
    def __init__(self, permutation):
        self.permutation = permutation

    def copy(self):
        return Permutation(deepcopy(self.permutation))

    def lazy_copy(self):
        return Permutation(self.permutation)

    def get_attribute_register(self, problem):
        return EncodingRegister(
            {
                "permutation": {
                    "type": [TypeAttribute.PERMUTATION],
                    "range": list(range(problem.length)),
                }
            }
        )

    def __str__(self):
        return "Perm : " + str(self.permutation)


# TODO : have default solution builders like done here for the dummy solution
class PermutationProblem(Problem):
    def __init__(self, length: int):
        self.length = length

    def evaluate(self, permutation: Permutation):
        s = 0
        for i in range(self.length - 1):
            s += 1 / abs(permutation.permutation[i] - permutation.permutation[i + 1])
        return s

    def satisfy(self, permutation: Permutation):
        return len(set(permutation.permutation)) == self.length

    def get_dummy_solution(self):
        return Permutation(list(range(self.length)))


from discrete_optimization.generic_tools.ls.simulated_annealing import (
    ModeMutation,
    RestartHandler,
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)

if __name__ == "__main__":
    problem = PermutationProblem(40)
    solution = problem.get_dummy_solution()
    _, list_mutation = get_available_mutations(problem, solution)
    print(list_mutation)
    for mutate in list_mutation:
        mutation = mutate[0](problem, solution, **mutate[1])
        sol, move = mutation.mutate(solution)
        print(mutate[0].__name__, sol.permutation)
        move.backtrack_local_move(sol)
        print(mutate[0].__name__, sol.permutation)

    mutate = list_mutation[1]
    print("Mutation used : ", mutate[0].__name__)
    res = RestartHandler()
    sa = SimulatedAnnealing(
        evaluator=problem,
        mutator=mutate[0](problem, solution, **mutate[1]),
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(0.1, res, 1000, 0.999),
        mode_mutation=ModeMutation.MUTATE,
    )
    sa.solve(solution, 1000000, ModeOptim.MINIMIZATION, False)

    # mutation_partial_shuffle = PermutationPartialShuffleMutation(problem, solution, 0.3)
    # mutation_shuffle = PermutationShuffleMutation(problem, solution)

    # print("Prev : ", solution.permutation)
    # sol, move = mutation_partial_shuffle.mutate(solution)
    # print("New : ", sol.permutation)
    # sol = move.backtrack_local_move(sol)
    # print("Back : ", sol.permutation)

    # for j in range(100):
    #     sol, move = mutation_partial_shuffle.mutate(solution)
    #     value = problem.evaluate(sol)
    #     print("value", value)
    #     sat = problem.satisfy(sol)
    #     print("sat ", sat)
    #     print(sol)
    #     print(move.permutation)
    #     print(move.prev_permutation)

    # sol, move = mutation_shuffle.mutate(solution)
    # print(sol)
    # print(move.permutation)
    # print(move.prev_permutation)
