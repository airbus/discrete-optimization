from enum import Enum


class StartOrEnd(Enum):
    START = "start"
    END = "end"

    def __invert__(self):
        if self == StartOrEnd.START:
            return StartOrEnd.END
        else:
            return StartOrEnd.START


class MinOrMax(Enum):
    MIN = "min"
    MAX = "max"

    def __invert__(self):
        if self == MinOrMax.MIN:
            return MinOrMax.MAX
        else:
            return MinOrMax.MIN
