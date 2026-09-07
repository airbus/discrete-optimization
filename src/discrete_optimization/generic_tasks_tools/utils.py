#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from discrete_optimization.generic_tasks_tools.base import TasksProblem


def optional_override(funcobj):
    """A decorator indicating primitive methods likely to be overriden when creating a new task problem class.

    Many tasks mixins are defining abstract methods to be implemented. To reduce the burden when implementing
    a new tasks problem class, some other primitive methods have a default implementation.
    Other methods of the mixins are then derived from these primitive methods.

    This decorator helps the problem developper to identify such primitive methods with default implementation that he
    could be interested in overriding.

    Note: This does not modify the function behaviour, and is merely used as documentation.

    """
    funcobj.__is_optional_override__ = True
    return funcobj


def get_mandatory_methods_to_implement(
    cls: type[TasksProblem],
) -> dict[str, Callable[[...], Any]]:
    """Find methods mandatory to implement when deriving from the given class.

    Args:
        cls: class to inspect

    Returns:
        A dictionary mapping abstract method names to their class methods.

    Example:
        >>> from discrete_optimization.generic_tasks_tools.generic_scheduling import GenericSchedulingProblem
        >>> get_mandatory_methods_to_implement(GenericSchedulingProblem)  # doctest:+ELLIPSIS
        {...}


    """
    return {name: getattr(cls, name) for name in cls.__abstractmethods__}


def get_optional_methods_to_override(
    cls: type[TasksProblem] | TasksProblem, include_subclass_overrides: bool = False
) -> dict[str, Callable[[...], Any]]:
    """Find methods likely to be overriden when deriving from the given task problem class.

    It finds methods decorated with @optional_override.
    If the option is activated, it also finds such methods already overriden (including abstract methods).

    Args:

        cls: The class (or instance) to inspect.
        include_subclass_overrides: If True, also includes
          - methods that overrode an @optional_override method from a parent mixin without re-applying
          the decorator
          - methods that overrode an @abstractmethod method from a parent mixin

    Returns:

        A dictionary mapping method names to their class methods.

    Example:
        >>> from discrete_optimization.generic_tasks_tools.generic_scheduling import GenericSchedulingProblem
        >>> optional_methods = get_optional_methods_to_override(GenericSchedulingProblem)
        >>> optional_methods_including_subclasses = get_optional_methods_to_override(GenericSchedulingProblem, include_subclass_overrides=True)
        >>> assert len(optional_methods_including_subclasses) > len(optional_methods)


    """
    # use the class if an instance was given:
    if not isinstance(cls, type):
        cls = type(cls)

    optional_methods = {}
    # loop over members
    for name in dir(cls):
        # Safely resolve the raw attribute without triggering properties or descriptors
        raw_attr = inspect.getattr_static(cls, name)
        # Unwrap @classmethod or @staticmethod descriptors if present
        unwrapped = getattr(raw_attr, "__func__", raw_attr)
        # Direct match: The current class method has the decorator
        if getattr(unwrapped, "__is_optional_override__", False):
            optional_methods[name] = getattr(cls, name)
            continue
        # Inherited match: Walk the MRO to check if a base class decorated it
        if include_subclass_overrides:
            for base in cls.__mro__:
                base_raw = inspect.getattr_static(base, name, None)
                if base_raw:
                    base_unwrapped = getattr(base_raw, "__func__", base_raw)

                    if getattr(
                        base_unwrapped, "__is_optional_override__", False
                    ) or getattr(base_unwrapped, "__isabstractmethod__", False):
                        optional_methods[name] = getattr(cls, name)
                        break

    return optional_methods
