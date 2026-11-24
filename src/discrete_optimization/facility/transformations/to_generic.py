#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Transformation from Facility to GenSchedulingImpl
"""

from discrete_optimization.facility.problem import (
    Facility,
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.generic_tasks_tools.generic_allocation_impl import (
    GenericAllocationImplProblem,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.objectives.allocation_cost import (
    AllocationCostComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
    UnaryResourcesUsedComputer,
)
from discrete_optimization.generic_tasks_tools.transformations.generic_allocation_impl import (
    FromPureToGenericAllocationImpl,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)


class FacilityToGenericSchedulingTransformation(
    ProblemTransformation[
        FacilityProblem,
        FacilitySolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    """
    Transform Facility to GenericSchedulingImpl.
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (Facility → GenSchedulingImpl).

        This direction is EXACT: all constraints can be represented in GenSchedulingImpl.
        """
        return exact_transformation(
            use_cases=[
                "Exact encoding of Facility as scheduling problem",
            ]
        )

    def transform_problem(
        self, source_problem: FacilityProblem
    ) -> GenericSchedulingImplProblem:
        """Transform FacilityProblem to GenericSchedulingImpl problem.

        Args:
            source_problem: FacilityProblem problem instance

        Returns:
            Equivalent GenericSchedulingImpl problem

        """
        tasks = source_problem.tasks_list
        unary_resource = {f for f in source_problem.unary_resources_list}
        non_renewable_resource = {
            f"N{f.index}": int(f.capacity) for f in source_problem.facilities
        }
        resource_consumptions = {t: {0: {"is_a_facility": 1}} for t in tasks}
        unary_resource_consumptions_dependent = {}
        for t in tasks:
            unary_resource_consumptions_dependent[t] = {
                0: {n: {} for n in non_renewable_resource}
            }
            for f in source_problem.facilities:
                corresp_capa = f"N{f.index}"
                unary_resource_consumptions_dependent[t][0][corresp_capa][f] = int(
                    t.demand
                )
        computer_used_facilities = UnaryResourcesUsedComputer(
            weight_objective=1,
            weight_per_unary_resource={
                f: int(f.setup_cost) for f in source_problem.facilities
            },
        )
        allocation_computer = AllocationCostComputer(
            weight_objective=1,
            cost_allocation_resource_to_task={
                t: {
                    f: int(
                        source_problem.evaluate_customer_facility(
                            customer=t, facility=f
                        )
                    )
                    for f in unary_resource
                }
                for t in tasks
            },
        )

        return GenericSchedulingImplProblem(
            horizon=1,
            durations_per_mode={n: {0: 0} for n in source_problem.customers},
            resource_consumptions=resource_consumptions,
            unary_resources=unary_resource,
            skills={"is_a_facility"},
            non_renewable_resources=non_renewable_resource,
            unary_resource_consumptions_dependent=unary_resource_consumptions_dependent,
            unary_resources_skills={f: {"is_a_facility": 1} for f in unary_resource},
            list_objective_computer=[allocation_computer, computer_used_facilities],
        )

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: FacilityProblem
    ) -> FacilitySolution:
        """Transform GenericSchedulingImpl solution back to FacilityLocation solution.

        Returns:
            Equivalent FacilityLocation solution

        """
        # Extract bin assignment from start times
        # Tasks scheduled at time t are assigned to bin t
        facility_for_customer = [-1] * len(source_problem.tasks_list)
        for i, item in enumerate(source_problem.tasks_list):
            if solution.is_present(item):
                allocated = solution.get_task_allocation(item)
                if len(allocated) >= 1:
                    one = list(allocated)[0]
                    facility_for_customer[i] = (
                        source_problem.get_index_from_unary_resource(one)
                    )
        return FacilitySolution(
            problem=source_problem, facility_for_customers=facility_for_customer
        )


class FacilityToGenericAllocationTransformation(
    FromPureToGenericAllocationImpl[
        FacilityProblem,
        FacilitySolution,
    ]
):
    """
    Transform Facility to GenericSchedulingImpl.
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (Facility → GenAllocationImpl).

        This direction is EXACT: all constraints can be represented in GenAllocationImpl.
        """
        return exact_transformation(
            use_cases=[
                "Exact encoding of Facility as generic allocation problem",
            ]
        )

    def transform_problem(
        self, source_problem: FacilityProblem
    ) -> GenericAllocationImplProblem:
        """Transform FacilityProblem to GenericAllocationImplProblem problem.

        Args:
            source_problem: FacilityProblem problem instance

        Returns:
            Equivalent GenericAllocationImplProblem problem

        """
        generic_problem = super().transform_problem(source_problem)
        # facility capacities constraint
        facility_to_nr_resource: dict[Facility, str] = {
            facility: f"N{facility.index}" for facility in source_problem.facilities
        }
        generic_problem.non_renewable_resources = {
            facility_to_nr_resource[facility]: int(facility.capacity)
            for facility in source_problem.facilities
        }
        generic_problem.unary_resource_consumptions_dependent = {
            # task: customer
            customer: {
                # mode
                0: {
                    # non-renewable resource (facility capacity)
                    facility_to_nr_resource[facility]: {
                        # unary resource: facility -> nr conso = customer demand
                        facility: int(customer.demand)
                    }
                    for facility in source_problem.facilities
                }
            }
            for customer in source_problem.customers
        }
        # objective
        computer_used_facilities = UnaryResourcesUsedComputer(
            weight_objective=1,
            weight_per_unary_resource={
                facility: int(facility.setup_cost)
                for facility in source_problem.facilities
            },
        )
        allocation_computer = AllocationCostComputer(
            weight_objective=1,
            cost_allocation_resource_to_task={
                customer: {
                    facility: int(
                        source_problem.evaluate_customer_facility(
                            customer=customer, facility=facility
                        )
                    )
                    for facility in source_problem.facilities
                }
                for customer in source_problem.customers
            },
        )
        generic_problem.list_objective_computer = [
            computer_used_facilities,
            allocation_computer,
        ]
        # take into account changes
        generic_problem.update_problem()

        return generic_problem

    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: FacilityProblem,
    ) -> FacilitySolution:
        facility_for_customers = [-1] * source_problem.customer_count
        for i_customer, customer in enumerate(source_problem.tasks_list):
            task_variable = raw_sol.task_variables[customer]
            assert len(task_variable.allocated) == 1, (
                "One facility and one only must be allocated to the customer."
            )
            facility: Facility = next(iter(task_variable.allocated))
            facility_for_customers[i_customer] = facility.index
        return FacilitySolution(
            problem=source_problem, facility_for_customers=facility_for_customers
        )
