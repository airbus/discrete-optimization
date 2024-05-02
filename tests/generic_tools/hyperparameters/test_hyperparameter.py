from enum import Enum
from typing import Any, List, Optional

import optuna
import pytest

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class Method(Enum):
    DUMMY = 0
    GREEDY = 1


class DummySolver(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=2, default=1),
        FloatHyperparameter("coeff", low=-1.0, high=1.0, default=1.0),
        CategoricalHyperparameter("use_it", choices=[True, False], default=True),
        EnumHyperparameter("method", enum=Method, default=Method.GREEDY),
    ]

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return ResultStorage([])


class DummySolver2(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=3, default=1),
        FloatHyperparameter("coeff2", low=-1.0, high=1.0, default=1.0),
        FloatHyperparameter("coeff3", low=-1.0, high=1.0, default=1.0),
    ]

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return ResultStorage([])


class BaseMetaSolver(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=0, high=1, default=1),
        SubBrickHyperparameter("subsolver", choices=[DummySolver]),
        SubBrickKwargsHyperparameter(
            "kwargs_subsolver", subbrick_hyperparameter="subsolver"
        ),
    ]

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return ResultStorage([])


class MetaSolver(BaseMetaSolver):
    # we check that copy_and_update_hyperparameters create a hyperparameter subsolver
    # whose attribute choices_str2cls is working properly
    hyperparameters = BaseMetaSolver.copy_and_update_hyperparameters(
        subsolver=dict(choices=[DummySolver, DummySolver2]),
    )


class MetaMetaSolver(SolverDO):
    hyperparameters = [
        IntegerHyperparameter("nb", low=1, high=1, default=1),
        SubBrickHyperparameter("subsolver", choices=[MetaSolver]),
        SubBrickKwargsHyperparameter(
            "kwargs_subsolver", subbrick_hyperparameter="subsolver"
        ),
    ]

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return ResultStorage([])


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


def test_suggest_with_optuna_meta_solver():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            MetaSolver.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name=dict(
                    kwargs_subsolver=dict(
                        names=["nb", "coeff", "coeff2"],
                        kwargs_by_name=dict(
                            coeff=dict(step=0.5), coeff2=dict(step=0.5)
                        ),
                    )
                ),
            )
        )
        assert len(suggested_hyperparameters_kwargs) == 3
        print(suggested_hyperparameters_kwargs)
        assert "nb" in suggested_hyperparameters_kwargs
        assert "nb" in suggested_hyperparameters_kwargs["kwargs_subsolver"]
        assert "nb" in trial.params
        assert "subsolver.nb" in trial.params
        assert (
            trial.params["subsolver.nb"]
            == suggested_hyperparameters_kwargs["kwargs_subsolver"]["nb"]
        )

        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective)

    assert len(study.trials) == 2 * (4 * 5 + 3 * 5)


def test_suggest_with_optuna_meta_solver_level2():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            MetaMetaSolver.suggest_hyperparameters_with_optuna(
                trial=trial,
                kwargs_by_name=dict(
                    kwargs_subsolver=dict(
                        kwargs_by_name=dict(
                            kwargs_subsolver=dict(
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
            in suggested_hyperparameters_kwargs["kwargs_subsolver"]["kwargs_subsolver"]
        )
        assert "nb" in trial.params
        assert "subsolver.nb" in trial.params
        assert "subsolver.subsolver.nb" in trial.params
        assert (
            trial.params["subsolver.nb"]
            == suggested_hyperparameters_kwargs["kwargs_subsolver"]["nb"]
        )
        assert (
            trial.params["subsolver.subsolver.nb"]
            == suggested_hyperparameters_kwargs["kwargs_subsolver"]["kwargs_subsolver"][
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
        MetaSolver.suggest_hyperparameters_with_optuna(
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
    with pytest.raises(ValueError, match="The choice of 'subsolver'"):
        study.optimize(objective, n_trials=1)


def test_suggest_with_optuna_meta_solver_fixed_subsolver():
    def objective(trial: optuna.Trial) -> float:
        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            MetaSolver.suggest_hyperparameters_with_optuna(
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
        assert len(suggested_hyperparameters_kwargs) == 2
        print(suggested_hyperparameters_kwargs)
        assert "nb" in suggested_hyperparameters_kwargs
        assert "nb" in suggested_hyperparameters_kwargs["kwargs_subsolver"]
        assert "nb" in trial.params
        assert "subsolver.nb" in trial.params
        assert (
            trial.params["subsolver.nb"]
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
