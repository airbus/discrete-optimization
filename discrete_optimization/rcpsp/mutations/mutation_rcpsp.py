import numpy as np

from discrete_optimization.generic_tools.do_mutation import LocalMove
from discrete_optimization.generic_tools.do_problem import EncodingRegister, TypeAttribute
from discrete_optimization.generic_tools.mutations.permutation_mutations import PermutationShuffleMutation, \
    PermutationPartialShuffleMutation, PermutationSwap, TwoOptMutation, Mutation, Problem, Solution, SwapsLocalMove, \
    ShuffleMove
from typing import Union, Tuple, Dict
from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution, RCPSPModel
import random

class PermutationMutationRCPSP(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs):
        other_mutation = kwargs.get('other_mutation', PermutationShuffleMutation)
        other_mutation = other_mutation.build(problem, solution, **kwargs)
        return PermutationMutationRCPSP(problem, solution, other_mutation=other_mutation)

    def __init__(self, problem: Problem,
                 solution: Solution,
                 other_mutation: Mutation):
        self.problem = problem
        self.solution = solution
        self.other_mutation = other_mutation

    def mutate(self, solution: RCPSPSolution) -> Tuple[Solution, LocalMove]:
        s, lm = self.other_mutation.mutate(solution)
        try:
            s.standardised_permutation = s.generate_permutation_from_schedule()
            s._schedule_to_recompute = True
        except:
            s._schedule_to_recompute = True
            pass
        return s, lm

    def mutate_and_compute_obj(self, solution: Solution) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        s, lm, fit = self.other_mutation.mutate_and_compute_obj(solution)
        try:
            s._schedule_to_recompute = True
            s.standardised_permutation = s.generate_permutation_from_schedule()
        except:
            s._schedule_to_recompute = True
            pass
        return s, lm, fit


class DeadlineMutationRCPSP(Mutation):
    def __init__(self,
                 problem: Problem,
                 solution: Solution,
                 attribute: str = None,
                 nb_swap: int = 1):
        self.problem = problem
        self.register: EncodingRegister = solution.get_attribute_register(problem)
        self.nb_swap = nb_swap
        self.attribute = attribute
        if self.attribute is None:
            attributes = [k
                          for k in self.register.dict_attribute_to_type
                          for t in self.register.dict_attribute_to_type[k]["type"]
                          if t == TypeAttribute.PERMUTATION]
            if len(attributes) > 0:
                self.attribute = attributes[0]
        self.length = len(self.register.dict_attribute_to_type[self.attribute]["range"])
        self.full_predecessors = self.problem.graph.ancestors_map()

    def mutate(self, solution: Solution) -> Tuple[Solution, LocalMove]:
        if "special_constraints" in self.problem.__dict__.keys():
            ls = [(t, solution.get_end_time(t)-self.problem.special_constraints.end_times_window[t][1])
                  for t in self.problem.special_constraints.end_times_window
                  if self.problem.special_constraints.end_times_window[t][1] is not None
                  and solution.get_end_time(t) > self.problem.special_constraints.end_times_window[t][1]]
            if len(ls) > 0:
                x = random.choice(ls)
                t = x[0]
                if True:
                    pred = [tt for tt in self.full_predecessors[t]]+[t]
                    previous = list(getattr(solution, self.attribute))
                    new = [self.problem.index_task_non_dummy[tt] for tt in pred if tt in self.problem.index_task_non_dummy]
                    for x in previous:
                        if x not in new:
                            new += [x]
                    sol = solution.lazy_copy()
                    setattr(sol, self.attribute, new)
                    return (sol, ShuffleMove(self.attribute,
                                             new_permutation=new,
                                             prev_permutation=previous))
                index_in = getattr(solution, self.attribute).index(self.problem.index_task_non_dummy[x[0]])
                move = SwapsLocalMove(self.attribute, [(2, index_in)])
                next_sol = move.apply_local_move(solution)
                return (next_sol, move)
        swaps = np.random.randint(low=0, high=self.length - 1, size=(1, 2))
        move = SwapsLocalMove(self.attribute, [(swaps[i, 0], swaps[i, 1]) for i in range(1)])
        next_sol = move.apply_local_move(solution)
        return (next_sol, move)

    def mutate_and_compute_obj(self, solution: Solution) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return (sol, move, obj)

    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs):
        return DeadlineMutationRCPSP(problem, solution)


