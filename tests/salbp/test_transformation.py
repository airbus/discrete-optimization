from discrete_optimization.alb.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.alb.salbp.transformations.to_binpack import (
    SalbpToBinpackTransformation,
)
from discrete_optimization.alb.salbp.transformations.to_facility import (
    SalbpToFacilityTransformation,
)
from discrete_optimization.alb.salbp.transformations.to_rcalbp_l import (
    SalbpToRcalbpLTransformation,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver


def test_transformation_rcalbp_l():
    from discrete_optimization.alb.rcalbp_l.solvers.cpsat import CpSatRCALBPLSolver

    files = get_data_available()
    file = [f for f in files if "instance_n=20_10" in f][0]
    problem = parse_alb_file(file)
    solver = TransformationSolver(
        transformation=SalbpToRcalbpLTransformation(20),
        source_problem=problem,
        solver_brick=SubBrick(cls=CpSatRCALBPLSolver, kwargs={"time_limit": 10}),
    )
    result = solver.solve()
    # This is not working


def test_transformation_facility():
    from discrete_optimization.facility.solvers.greedy import GreedyFacilitySolver

    files = get_data_available()
    file = [f for f in files if "instance_n=20_10" in f][0]
    problem = parse_alb_file(file)
    solver = TransformationSolver(
        transformation=SalbpToFacilityTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=GreedyFacilitySolver, kwargs={}),
    )
    result = solver.solve()
    sol = result[-1][0]
    # Precedence not respected by this solver...


def test_transformation_binpack():
    from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver

    files = get_data_available()
    file = [f for f in files if "instance_n=20_10" in f][0]
    problem = parse_alb_file(file)
    solver = TransformationSolver(
        transformation=SalbpToBinpackTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=GreedyBinPackSolver, kwargs={}),
    )
    result = solver.solve()
    sol = result[-1][0]
    print(problem.satisfy(sol))
    print(problem.evaluate(sol))
    # Precedence not respected by this solver...
