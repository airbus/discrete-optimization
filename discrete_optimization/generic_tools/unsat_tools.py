"""Tools for explaining unsatisfiability."""
from abc import abstractmethod
from collections.abc import MutableSequence
from typing import Any, Iterable, Optional, overload

Constraint = Any


class MetaConstraint(MutableSequence[Constraint]):
    """Meta constraint

    Constraint gathering several finer constraints.
    For instance, in coloring problems, it could be
    "colors of neigbours of the given node must be different of its color"

    """

    def __init__(
        self, name: str, constraints: Optional[list[Constraint]] = None, metadata=None
    ):
        self.name = name
        if constraints is None:
            constraints = []
        self.constraints = constraints
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __len__(self) -> int:
        return len(self.constraints)

    def insert(self, index: int, value: Constraint) -> None:
        self.constraints.insert(index, value)

    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> Constraint:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> MutableSequence[Constraint]:
        ...

    def __getitem__(self, index: int) -> Constraint:
        return self.constraints[index]

    @overload
    @abstractmethod
    def __setitem__(self, index: int, value: Constraint) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, index: slice, value: Iterable[Constraint]) -> None:
        ...

    def __setitem__(self, index: int, value: Constraint) -> None:
        self.constraints[index] = value

    @overload
    @abstractmethod
    def __delitem__(self, index: int) -> None:
        ...

    @overload
    @abstractmethod
    def __delitem__(self, index: slice) -> None:
        ...

    def __delitem__(self, index: int) -> None:
        del self.constraints[index]
