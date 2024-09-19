from enum import Enum
from typing import Any, Optional

import optuna
import pytest

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    CategoricalListWithoutReplacementHyperparameter,
    EnumHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    ListHyperparameter,
    SubBrick,
    SubBrickClsHyperparameter,
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
    SubBrickListWithoutReplacementHyperparameter,
    TrialDropped,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class Method(Enum):
    DUMMY = 0
    GREEDY = 1


class BigMethod(Enum):
    DUMMY = 0
    GREEDY = 1
    OTHER = 2


class DummySolver(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=2, default=1),
        FloatHyperparameter("coeff", low=-1.0, high=1.0, default=1.0, step=0.25),
        CategoricalHyperparameter("use_it", choices=[True, False], default=True),
        EnumHyperparameter("method", enum=Method, default=Method.GREEDY),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


class DummySolverWithList(SolverDO):
    hyperparameters = [
        ListHyperparameter(
            name="list_nb",
            hyperparameter_template=IntegerHyperparameter("nb", low=0, high=2),
            length_low=2,
            length_high=3,
        ),
        CategoricalListWithoutReplacementHyperparameter(
            name="list_method",
            hyperparameter_template=EnumHyperparameter("method", enum=Method),
            length_low=1,
            length_high=5,
        ),
    ]


class DummySolverWithFloatSuggestingBound(SolverDO):
    hyperparameters = [
        FloatHyperparameter("coeff_no", low=0.0, high=1.0),
        FloatHyperparameter("coeff_low", low=0.0, high=1.0, suggest_low=True),
        FloatHyperparameter("coeff_high", low=0.0, high=1.0, suggest_high=True),
        FloatHyperparameter(
            "coeff_both", low=0.0, high=1.0, suggest_low=True, suggest_high=True
        ),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


def bee1_impl(x):
    return x + 1


def bee2_impl(x):
    return x + 2


class DummySolverWithCallableHyperparameter(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=2, default=1),
        FloatHyperparameter("coeff", low=-1.0, high=1.0, default=1.0),
        CategoricalHyperparameter(
            "heuristic",
            choices={
                "bee1": bee1_impl,
                "bee2": bee2_impl,
            },
            default=bee1_impl,
        ),
        EnumHyperparameter("method", enum=Method, default=Method.GREEDY),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


class DummySolverWithEnumSubset(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=2, default=1),
        FloatHyperparameter("coeff", low=-1.0, high=1.0, default=1.0),
        CategoricalHyperparameter("use_it", choices=[True, False], default=True),
        EnumHyperparameter(
            "method",
            enum=BigMethod,
            default=BigMethod.GREEDY,
            choices=[BigMethod.DUMMY, BigMethod.GREEDY],
        ),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


class DummySolverWithDependencies(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=2, default=1),
        FloatHyperparameter(
            "coeff",
            low=-1.0,
            high=1.0,
            default=1.0,
            depends_on=("method", [Method.GREEDY]),
        ),
        CategoricalHyperparameter("use_it", choices=[True, False], default=True),
        EnumHyperparameter(
            "method", enum=Method, default=Method.GREEDY, depends_on=("use_it", [True])
        ),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


class DummySolverWithNameInKwargs(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=2, default=1),
        FloatHyperparameter(
            "coeff_greedy",
            name_in_kwargs="coeff",
            low=2.0,
            high=3.0,
            default=2.0,
            depends_on=("method", [Method.GREEDY]),
        ),
        FloatHyperparameter(
            "coeff_dummy",
            name_in_kwargs="coeff",
            low=-1.0,
            high=1.0,
            default=1.0,
            depends_on=("method", [Method.DUMMY]),
        ),
        CategoricalHyperparameter("use_it", choices=[True, False], default=True),
        EnumHyperparameter(
            "method", enum=Method, default=Method.GREEDY, depends_on=("use_it", [True])
        ),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


class DummySolver2(SolverDO):
    hyperparameters = [
        CategoricalHyperparameter("nb", choices=[0, 1, 2, 3], default=1),
        FloatHyperparameter("coeff2", low=-1.0, high=1.0, default=1.0),
        FloatHyperparameter("coeff3", low=-1.0, high=1.0, default=1.0),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


class BaseMetaSolver(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=1, default=1),
        SubBrickHyperparameter("subsolver", choices=[DummySolver]),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


class MetaSolver(BaseMetaSolver):
    # we check that copy_and_update_hyperparameters create a hyperparameter subsolver
    # whose attribute choices_str2cls is working properly
    hyperparameters = BaseMetaSolver.copy_and_update_hyperparameters(
        subsolver=dict(choices=[DummySolver, DummySolver2]),
    )


class MetaSolverBis(SolverDO):
    # we check that copy_and_update_hyperparameters create a hyperparameter subsolver
    # whose attribute choices_str2cls is working properly
    hyperparameters = [
        SubBrickHyperparameter("subsolver", choices=[DummySolver, DummySolver2]),
    ]


class MetaSolverFixedSubsolver(SolverDO):
    # subbrickkwargs with fixed subbrick class
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=2, step=2, default=2),
        SubBrickKwargsHyperparameter("kwargs_subsolver", subbrick_cls=DummySolver),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


class MetaMetaSolver(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=1, high=1, default=1),
        SubBrickClsHyperparameter("subsolver", choices=[MetaSolver]),
        SubBrickKwargsHyperparameter(
            "kwargs_subsolver", subbrick_hyperparameter="subsolver"
        ),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


class MetaSolverWithListWithoutReplacement(SolverDO):
    hyperparameters = [
        SubBrickListWithoutReplacementHyperparameter(
            "subsolvers",
            length_low=1,
            length_high=2,
            hyperparameter_template=SubBrickHyperparameter(
                name="subsolver",
                choices=[DummySolver, DummySolver2, MetaSolverBis],
            ),
        ),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.create_result_storage()


def test_get_hyperparameters_and_co():
    assert DummySolver.get_hyperparameters_names() == [
        "nb",
        "coeff",
        "use_it",
        "method",
    ]
    assert isinstance(DummySolver.get_hyperparameter("nb"), IntegerHyperparameter)


def test_get_default_hyperparameters():
    kwargs = DummySolver.get_default_hyperparameters(["coeff", "use_it"])
    assert len(kwargs) == 2
    assert "coeff" in kwargs
    assert "use_it" in kwargs

    kwargs = DummySolver.get_default_hyperparameters()
    assert kwargs["coeff"] == 1.0
    assert kwargs["nb"] == 1
    assert kwargs["use_it"]
    assert kwargs["method"] == Method.GREEDY

    kwargs = DummySolverWithNameInKwargs.get_default_hyperparameters()
    assert kwargs["coeff"] == 1.0
    assert kwargs["nb"] == 1
    assert kwargs["use_it"]
    assert kwargs["method"] == Method.GREEDY


def test_complete_with_default_hyperparameters():
    kwargs = {"coeff": 0.5, "toto": "youpi"}
    kwargs = DummySolver.complete_with_default_hyperparameters(kwargs)

    assert kwargs["toto"] == "youpi"
    assert kwargs["coeff"] == 0.5
    assert kwargs["nb"] == 1
    assert kwargs["use_it"]
    assert kwargs["method"] == Method.GREEDY


def test_complete_with_default_hyperparameters_specific_names():
    kwargs = {"coeff": 0.5, "toto": "youpi"}
    kwargs = DummySolver.complete_with_default_hyperparameters(
        kwargs, names=["nb", "coeff"]
    )

    assert kwargs["toto"] == "youpi"
    assert kwargs["coeff"] == 0.5
    assert kwargs["nb"] == 1
    assert "use_it" not in kwargs


def test_suggest_with_optuna():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            DummySolver.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name={
                    "coeff": dict(step=0.5),
                    "nb": dict(high=1),
                    "use_it": dict(choices=[True]),
                },
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 4
        assert isinstance(suggested_hyperparameters_kwargs["method"], Method)
        assert 0 <= suggested_hyperparameters_kwargs["nb"]
        assert 1 >= suggested_hyperparameters_kwargs["nb"]
        assert -1.0 <= suggested_hyperparameters_kwargs["coeff"]
        assert 1.0 >= suggested_hyperparameters_kwargs["coeff"]
        assert suggested_hyperparameters_kwargs["use_it"] is True

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * 2 * 5 * 1


def test_suggest_with_optuna_with_float_bound_suggestion():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            DummySolverWithFloatSuggestingBound.suggest_hyperparameters_with_optuna(
                trial=trial
            )
        )
        assert 0.0 < suggested_hyperparameters_kwargs["coeff_no"]
        assert 0.0 < suggested_hyperparameters_kwargs["coeff_high"]
        assert 1.0 > suggested_hyperparameters_kwargs["coeff_no"]
        assert 1.0 > suggested_hyperparameters_kwargs["coeff_low"]

        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=10)

    coeff_high_values = [trial.params["coeff_high"] for trial in study.trials]
    coeff_low_values = [trial.params["coeff_low"] for trial in study.trials]
    coeff_both_values = [trial.params["coeff_both"] for trial in study.trials]
    assert 1.0 in coeff_high_values
    assert 1.0 in coeff_both_values
    assert 0.0 in coeff_low_values
    assert 0.0 in coeff_both_values
    assert len([v for v in coeff_high_values if v == 1.0]) < len(coeff_high_values)
    assert len([v for v in coeff_both_values if v == 1.0]) < len(coeff_both_values)
    assert len([v for v in coeff_both_values if v == 0.0]) < len(coeff_both_values)
    assert len([v for v in coeff_low_values if v == 0.0]) < len(coeff_low_values)


def test_suggest_with_optuna_default_float_step():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            DummySolver.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name={
                    "nb": dict(high=1),
                    "use_it": dict(choices=[True]),
                },
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 4
        assert isinstance(suggested_hyperparameters_kwargs["method"], Method)
        assert 0 <= suggested_hyperparameters_kwargs["nb"]
        assert 1 >= suggested_hyperparameters_kwargs["nb"]
        assert -1.0 <= suggested_hyperparameters_kwargs["coeff"]
        assert 1.0 >= suggested_hyperparameters_kwargs["coeff"]
        assert suggested_hyperparameters_kwargs["use_it"] is True

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * 2 * 9 * 1


def test_suggest_with_optuna_with_choices_dict():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            DummySolverWithCallableHyperparameter.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name={
                    "coeff": dict(step=0.5),
                    "nb": dict(high=1),
                },
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 4
        assert isinstance(suggested_hyperparameters_kwargs["method"], Method)
        assert 0 <= suggested_hyperparameters_kwargs["nb"]
        assert 1 >= suggested_hyperparameters_kwargs["nb"]
        assert -1.0 <= suggested_hyperparameters_kwargs["coeff"]
        assert 1.0 >= suggested_hyperparameters_kwargs["coeff"]
        assert callable(suggested_hyperparameters_kwargs["heuristic"])

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * 2 * 5 * 2
    assert study.best_trial.params["heuristic"] in ["bee1", "bee2"]


def test_suggest_with_optuna_with_enum_subset():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            DummySolverWithEnumSubset.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name={
                    "coeff": dict(step=0.5),
                    "nb": dict(high=1),
                    "use_it": dict(choices=[True]),
                },
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 4
        assert isinstance(suggested_hyperparameters_kwargs["method"], BigMethod)
        assert 0 <= suggested_hyperparameters_kwargs["nb"]
        assert 1 >= suggested_hyperparameters_kwargs["nb"]
        assert -1.0 <= suggested_hyperparameters_kwargs["coeff"]
        assert 1.0 >= suggested_hyperparameters_kwargs["coeff"]
        assert suggested_hyperparameters_kwargs["use_it"] is True

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * 2 * 5 * 1


def test_suggest_with_optuna_with_dependencies():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            DummySolverWithDependencies.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name={
                    "coeff": dict(step=0.5),
                    "nb": dict(high=1),
                },
            )
        )
        if suggested_hyperparameters_kwargs["use_it"]:
            if suggested_hyperparameters_kwargs["method"] == Method.GREEDY:
                assert len(suggested_hyperparameters_kwargs) == 4
            else:
                assert len(suggested_hyperparameters_kwargs) == 3
        else:
            assert len(suggested_hyperparameters_kwargs) == 2

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * (1 + (1 + 5))


def test_suggest_with_optuna_with_name_in_kwargs():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            DummySolverWithNameInKwargs.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name={
                    "coeff_greedy": dict(step=0.5),
                    "coeff_dummy": dict(step=1.0),
                    "nb": dict(high=1),
                },
            )
        )
        if suggested_hyperparameters_kwargs["use_it"]:
            assert len(suggested_hyperparameters_kwargs) == 4
            assert "coeff" in suggested_hyperparameters_kwargs
            if suggested_hyperparameters_kwargs["method"] == Method.GREEDY:
                assert suggested_hyperparameters_kwargs["coeff"] >= 2.0
            else:
                assert suggested_hyperparameters_kwargs["coeff"] <= 1.0
        else:
            assert len(suggested_hyperparameters_kwargs) == 2

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * (1 + (3 + 3))


def test_suggest_with_optuna_with_dependencies_and_fixed_hyperparameters():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            DummySolverWithDependencies.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name={
                    "coeff": dict(step=0.5),
                    "nb": dict(high=1),
                },
                fixed_hyperparameters={"method": Method.GREEDY},
            )
        )
        if suggested_hyperparameters_kwargs["use_it"]:
            assert len(suggested_hyperparameters_kwargs) == 3 + 1
        else:
            assert len(suggested_hyperparameters_kwargs) == 2 + 1

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * (1 + 5)


def test_suggest_with_optuna_meta_solver():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            MetaSolver.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name=dict(
                    subsolver=dict(
                        names=["nb", "coeff", "coeff2"],
                        kwargs_by_name=dict(
                            coeff=dict(step=0.5), coeff2=dict(step=0.5)
                        ),
                    )
                ),
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 2
        print(suggested_hyperparameters_kwargs)
        assert "nb" in suggested_hyperparameters_kwargs
        assert "nb" in suggested_hyperparameters_kwargs["subsolver"].kwargs
        assert "nb" in trial.params
        assert (
            "subsolver.DummySolver.nb" in trial.params
            or "subsolver.DummySolver2.nb" in trial.params
        )
        if "subsolver.DummySolver.nb" in trial.params:
            param_name = "subsolver.DummySolver.nb"
        else:
            param_name = "subsolver.DummySolver2.nb"

        assert (
            trial.params[param_name]
            == suggested_hyperparameters_kwargs["subsolver"].kwargs["nb"]
        )

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * (4 * 5 + 3 * 5)


def test_suggest_with_optuna_meta_solver_customized_by_subsolver():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            MetaSolver.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name=dict(
                    subsolver=dict(
                        names=["nb"],
                        names_by_subbrick={
                            DummySolver: ["coeff"],
                            DummySolver2: ["coeff2"],
                        },
                        kwargs_by_name=dict(
                            coeff=dict(step=0.5), coeff2=dict(step=0.5)
                        ),
                        kwargs_by_name_by_subbrick={
                            DummySolver: dict(nb=dict(high=1)),
                            DummySolver2: dict(nb=dict(choices=[2, 3])),
                        },
                        fixed_hyperparameters_by_subbrick={
                            DummySolver: dict(use_it=False)
                        },
                    )
                ),
            )
        )

        assert len(suggested_hyperparameters_kwargs) == 2

        print(suggested_hyperparameters_kwargs)
        assert "nb" in suggested_hyperparameters_kwargs
        assert "nb" in suggested_hyperparameters_kwargs["subsolver"].kwargs
        assert "nb" in trial.params
        assert (
            "subsolver.DummySolver.nb" in trial.params
            or "subsolver.DummySolver2.nb" in trial.params
        )
        if "subsolver.DummySolver.nb" in trial.params:
            param_name = "subsolver.DummySolver.nb"
            assert suggested_hyperparameters_kwargs["subsolver"].kwargs["nb"] <= 1
            assert (
                suggested_hyperparameters_kwargs["subsolver"].kwargs["use_it"] is False
            )
            assert "coeff" in suggested_hyperparameters_kwargs["subsolver"].kwargs
            assert len(suggested_hyperparameters_kwargs["subsolver"].kwargs) == 3
        else:
            param_name = "subsolver.DummySolver2.nb"
            assert suggested_hyperparameters_kwargs["subsolver"].kwargs["nb"] >= 2
            assert "coeff2" in suggested_hyperparameters_kwargs["subsolver"].kwargs
            assert len(suggested_hyperparameters_kwargs["subsolver"].kwargs) == 2

        assert (
            trial.params[param_name]
            == suggested_hyperparameters_kwargs["subsolver"].kwargs["nb"]
        )

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * 2 * 2 * 5


def test_suggest_with_optuna_meta_solver_level2():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            MetaMetaSolver.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name=dict(
                    kwargs_subsolver=dict(
                        kwargs_by_name=dict(
                            subsolver=dict(
                                names=["nb", "coeff", "coeff2"],
                                kwargs_by_name=dict(
                                    coeff=dict(step=0.5), coeff2=dict(step=0.5)
                                ),
                            )
                        )
                    )
                ),
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 3
        print(suggested_hyperparameters_kwargs)
        assert "nb" in suggested_hyperparameters_kwargs
        assert "nb" in suggested_hyperparameters_kwargs["kwargs_subsolver"]
        assert (
            "nb"
            in suggested_hyperparameters_kwargs["kwargs_subsolver"]["subsolver"].kwargs
        )
        assert "nb" in trial.params
        assert "subsolver.MetaSolver.nb" in trial.params
        assert (
            "subsolver.MetaSolver.subsolver.DummySolver.nb" in trial.params
            or "subsolver.MetaSolver.subsolver.DummySolver2.nb" in trial.params
        )
        assert (
            trial.params["subsolver.MetaSolver.nb"]
            == suggested_hyperparameters_kwargs["kwargs_subsolver"]["nb"]
        )
        param_name = "subsolver.MetaSolver.subsolver.DummySolver.nb"
        if param_name not in trial.params:
            param_name = "subsolver.MetaSolver.subsolver.DummySolver2.nb"
        assert (
            trial.params[param_name]
            == suggested_hyperparameters_kwargs["kwargs_subsolver"]["subsolver"].kwargs[
                "nb"
            ]
        )

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * (4 * 5 + 3 * 5)


def test_suggest_with_optuna_meta_solver_nok_fixed_subsolver_not_given():
    def objective(trial: optuna.Trial) -> float:
        MetaMetaSolver.suggest_hyperparameters_with_optuna(
            trial=trial,
            names=["nb", "kwargs_subsolver"],
            kwargs_by_name=dict(
                kwargs_subsolver=dict(
                    names=["nb", "coeff", "coeff2"],
                    kwargs_by_name=dict(coeff=dict(step=0.5), coeff2=dict(step=0.5)),
                )
            ),
        )

        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler())
    with pytest.raises(ValueError, match="choice of 'subsolver'"):
        study.optimize(objective, n_trials=1)


def test_suggest_with_optuna_meta_solver_fixed_subsolver():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            MetaMetaSolver.suggest_hyperparameters_with_optuna(
                trial=trial,
                fixed_hyperparameters={"subsolver": DummySolver},
                names=["nb", "kwargs_subsolver"],
                kwargs_by_name=dict(
                    kwargs_subsolver=dict(
                        names=["nb", "coeff", "coeff2"],
                        kwargs_by_name=dict(coeff=dict(step=0.5)),
                    )
                ),
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 2 + 1
        print(suggested_hyperparameters_kwargs)
        assert "nb" in suggested_hyperparameters_kwargs
        assert "nb" in suggested_hyperparameters_kwargs["kwargs_subsolver"]
        assert "nb" in trial.params
        assert "subsolver.DummySolver.nb" in trial.params
        assert (
            trial.params["subsolver.DummySolver.nb"]
            == suggested_hyperparameters_kwargs["kwargs_subsolver"]["nb"]
        )

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 1 * (3 * 5)


def test_suggest_with_optuna_meta_solver_subsolver_kwargs_only():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            MetaSolverFixedSubsolver.suggest_hyperparameters_with_optuna(
                trial=trial,
                names=["nb", "kwargs_subsolver"],
                kwargs_by_name=dict(
                    kwargs_subsolver=dict(
                        names=["nb", "coeff", "coeff2"],
                        kwargs_by_name=dict(coeff=dict(step=0.5)),
                    )
                ),
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 2
        print(suggested_hyperparameters_kwargs)
        assert "nb" in suggested_hyperparameters_kwargs
        assert "nb" in suggested_hyperparameters_kwargs["kwargs_subsolver"]
        assert "nb" in trial.params
        assert "kwargs_subsolver.nb" in trial.params
        assert (
            trial.params["kwargs_subsolver.nb"]
            == suggested_hyperparameters_kwargs["kwargs_subsolver"]["nb"]
        )

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * (3 * 5)


def test_copy_and_update_hyperparameters():
    hyperparameters = DummySolver.copy_and_update_hyperparameters(
        nb=dict(high=5), use_it=dict(choices=[False])
    )
    original_hyperparameters = DummySolver.hyperparameters
    assert len(hyperparameters) == len(original_hyperparameters)
    for h, ho in zip(hyperparameters, original_hyperparameters):
        assert h.name == ho.name
        assert h is not ho
        if h.name == "nb":
            assert h.high == 5
            assert ho.high == 2
        elif h.name == "use_it":
            assert len(h.choices) == 1
            assert len(ho.choices) == 2


def test_suggest_with_optuna_with_list():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            DummySolverWithList.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name={
                    "list_nb": dict(high=1),
                },
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 2
        assert len(suggested_hyperparameters_kwargs["list_nb"]) >= 2
        assert len(suggested_hyperparameters_kwargs["list_nb"]) <= 3
        for nb in suggested_hyperparameters_kwargs["list_nb"]:
            assert nb in [0, 1]
        assert trial.params["nb_1"] == suggested_hyperparameters_kwargs["list_nb"][1]
        assert len(suggested_hyperparameters_kwargs["list_method"]) >= 1
        assert len(suggested_hyperparameters_kwargs["list_method"]) <= 2
        for method in suggested_hyperparameters_kwargs["list_method"]:
            assert isinstance(method, Method)
        assert len(set(suggested_hyperparameters_kwargs["list_method"])) == len(
            suggested_hyperparameters_kwargs["list_method"]
        )

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective, catch=TrialDropped)

    completed_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    assert len(completed_trials) == (2**2 + 2**3) * (2 + 2)


def test_suggest_with_optuna_with_list_subsolvers_wo_replacement():
    def objective(trial: optuna.Trial) -> float:

        # restrict some hyperparameters
        kwargs_by_name_dummysolver = {
            "coeff": dict(low=1),
            "use_it": dict(choices=[True]),
            "method": dict(choices=[Method.GREEDY]),
        }
        kwargs_by_name_dummysolver2 = {
            "coeff2": dict(step=1, low=1),
            "coeff3": dict(step=1, low=1),
        }
        kwargs_by_name_both_dummysolvers = dict(kwargs_by_name_dummysolver)
        kwargs_by_name_both_dummysolvers.update(kwargs_by_name_dummysolver2)
        kwargs_by_name_metasolver = {
            "subsolver": dict(kwargs_by_name=kwargs_by_name_both_dummysolvers)
        }
        kwargs_by_name_all_subsolvers = dict(kwargs_by_name_metasolver)
        kwargs_by_name_all_subsolvers.update(kwargs_by_name_both_dummysolvers)

        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            MetaSolverWithListWithoutReplacement.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name={
                    "subsolvers": dict(kwargs_by_name=kwargs_by_name_all_subsolvers)
                },
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 1
        assert len(suggested_hyperparameters_kwargs["subsolvers"]) <= 2
        assert len(suggested_hyperparameters_kwargs["subsolvers"]) >= 1
        for subsolver in suggested_hyperparameters_kwargs["subsolvers"]:
            assert isinstance(subsolver, SubBrick)

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective, catch=TrialDropped)

    completed_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    n_dummysolver = 3
    n_dummysolver2 = 4
    n_metasolver = n_dummysolver + n_dummysolver2
    n_list_of_size_1 = n_dummysolver + n_dummysolver2 + n_metasolver
    n_list_of_size_2_wo_duplicates = n_list_of_size_1**2 - n_list_of_size_1
    expected_n_trials = n_list_of_size_1 + n_list_of_size_2_wo_duplicates
    assert len(completed_trials) == expected_n_trials
