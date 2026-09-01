#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""
Parser for loading FlexProblem instances from JSON files.
This allows loading problem instances exported from external sources.
"""

import json
import os
from dataclasses import fields
from typing import Any, Dict, Optional

import numpy as np

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.flex_scheduling.problem import (
    ConstraintsTask,
    FlexProblem,
    GroupType,
    ObjectiveParamEarliness,
    ObjectiveParamResource,
    ObjectiveParams,
    ObjectiveParamTardiness,
    ObjectiveParamWIP,
    ObjectivesEnum,
    ResourceData,
    TaskData,
    TaskGroupAbstraction,
    TaskObject,
    TasksGroups,
)


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for jobshop."""
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/flex_scheduling/datasets/"

    try:
        files = [
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.endswith(".json")
        ]
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)
    return files


def decode_flex_problem_json(dct: Dict[str, Any]) -> Any:
    """Custom JSON decoder for FlexProblem objects."""

    # Decode numpy arrays
    if "__ndarray__" in dct:
        return np.array(dct["__ndarray__"], dtype=dct["dtype"])

    # Decode sets
    if "__set__" in dct:
        return set(dct["__set__"])

    # Decode enums
    if "__enum__" in dct:
        if dct["__enum__"] == "GroupType":
            return GroupType(dct["value"])
        elif dct["__enum__"] == "ObjectivesEnum":
            return ObjectivesEnum(dct["value"])

    # Decode dataclasses
    if "__dataclass__" in dct:
        class_name = dct["__dataclass__"]
        data = dct["data"]

        # Map class names to actual classes
        class_map = {
            "TaskData": TaskData,
            "TaskObject": TaskObject,
            "ResourceData": ResourceData,
            "TasksGroups": TasksGroups,
            "ConstraintsTask": ConstraintsTask,
            "ObjectiveParams": ObjectiveParams,
            "TaskGroupAbstraction": TaskGroupAbstraction,
            "ObjectiveParamWIP": ObjectiveParamWIP,
            "ObjectiveParamResource": ObjectiveParamResource,
            "ObjectiveParamTardiness": ObjectiveParamTardiness,
            "ObjectiveParamEarliness": ObjectiveParamEarliness,
        }

        if class_name in class_map:
            cls = class_map[class_name]
            # Recursively decode nested structures
            decoded_data = {k: decode_value(v) for k, v in data.items()}

            # Filter out fields with init=False
            init_fields = {f.name for f in fields(cls) if f.init}
            filtered_data = {k: v for k, v in decoded_data.items() if k in init_fields}

            return cls(**filtered_data)

    # Decode enum keys in dictionaries and try to convert string keys to ints
    result = {}
    enum_map = {
        "ObjectivesEnum": ObjectivesEnum,
        "GroupType": GroupType,
    }

    for key, value in dct.items():
        # Check if key is an encoded enum
        if isinstance(key, str) and key.startswith("__ENUM_KEY__"):
            # Format: __ENUM_KEY__ClassName__VALUE__value
            parts = key.split("__VALUE__")
            if len(parts) == 2:
                enum_class_name = parts[0].replace("__ENUM_KEY__", "")
                enum_value = int(parts[1])

                if enum_class_name in enum_map:
                    decoded_key = enum_map[enum_class_name](enum_value)
                else:
                    decoded_key = key
            else:
                decoded_key = key
        elif isinstance(key, str):
            # Try to convert string keys to integers (for dict keys like modes)
            try:
                decoded_key = int(key)
            except ValueError:
                decoded_key = key
        else:
            decoded_key = key

        result[decoded_key] = value

    return result


def decode_value(value: Any) -> Any:
    """Recursively decode values that may contain encoded objects."""
    if isinstance(value, dict):
        decoded = decode_flex_problem_json(value)
        return decoded
    elif isinstance(value, list):
        return [decode_value(item) for item in value]
    return value


def dict_to_problem(data: Dict[str, Any]) -> FlexProblem:
    """Convert dictionary back to FlexProblem."""

    # Decode all components
    resources = [decode_value(r) for r in data["resources"]]
    tasks = [decode_value(t) for t in data["tasks"]]
    tasks_group = [decode_value(g) for g in data["tasks_group"]]
    constraints = decode_value(data["constraints"])
    objective_params = decode_value(data["objective_params"])
    horizon = data["horizon"]

    # Create FlexProblem instance
    problem = FlexProblem(
        resources=resources,
        tasks=tasks,
        tasks_group=tasks_group,
        constraints=constraints,
        objective_params=objective_params,
        horizon=horizon,
    )

    return problem


def load_problem_from_json(filepath: str) -> FlexProblem:
    """
    Load a FlexProblem instance from a JSON file.

    Args:
        filepath: Path to input JSON file

    Returns:
        FlexProblem instance

    Example:
        >>> from discrete_optimization.flex_scheduling.parser import load_problem_from_json
        >>> problem = load_problem_from_json("my_problem.json") # doctest: +SKIP
    """
    with open(filepath, "r") as f:
        data = json.load(f, object_hook=decode_flex_problem_json)

    problem = dict_to_problem(data)

    print(f"Problem loaded from {filepath}")
    print(f"  Tasks: {len(problem.tasks)}")
    print(f"  Resources: {len(problem.resources)}")
    print(f"  Horizon: {problem.horizon}")

    return problem
