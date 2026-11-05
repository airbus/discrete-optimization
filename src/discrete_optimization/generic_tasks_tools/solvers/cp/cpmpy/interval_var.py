#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from cpmpy.expressions.core import Expression


@dataclass
class IntervalVar:
    start: Expression  # start of the interval
    name: str
    duration: Expression | None = None  # if None, deduced from start and end
    end: Expression | None = None  # if None, deduced from start and duration
    presence: Expression = (
        True  # if optional, the bool variable telling if the interval is present or not
    )

    def __post_init__(self):
        if self.duration is None and self.end is None:
            raise ValueError("duration and end cannot be None together.")
        elif self.duration is None:
            self.duration = self.end - self.start
        elif self.end is None:
            self.end = self.start + self.duration
