#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tasks_tools.renewable_resource import (
    convert_availability_intervals_to_calendar,
    convert_calendar_to_availability_intervals,
    merge_resources_availability_intervals,
)


def test_convert_availability_intervals_to_calendar():
    assert convert_availability_intervals_to_calendar(
        intervals=[(0, 2, 0), (2, 3, 1), (3, 5, 0), (5, 7, 4), (7, 8, 0)], horizon=8
    ) == [0, 0, 1, 0, 0, 4, 4, 0]
    assert convert_availability_intervals_to_calendar(
        intervals=[(2, 3, 1), (5, 7, 4)], horizon=8
    ) == [0, 0, 1, 0, 0, 4, 4, 0]

    calendar = [0, 0, 1, 0, 0, 4, 4, 0]
    intervals = [(0, 2, 0), (2, 3, 1), (3, 5, 0), (5, 7, 4), (7, 8, 0)]
    horizon = 8
    assert (
        convert_availability_intervals_to_calendar(
            convert_calendar_to_availability_intervals(
                calendar=calendar, horizon=horizon
            ),
            horizon=horizon,
        )
        == calendar
    )
    assert (
        convert_calendar_to_availability_intervals(
            convert_availability_intervals_to_calendar(
                intervals=intervals, horizon=horizon
            ),
            horizon=horizon,
        )
        == intervals
    )


def test_merge_resources_availability_intervals():
    assert merge_resources_availability_intervals(
        [[(2, 3, 1), (5, 7, 4)], [(1, 3, 3), (3, 5, 2)]], horizon=8
    ) == [(0, 1, 0), (1, 2, 3), (2, 3, 4), (3, 5, 2), (5, 7, 4), (7, 8, 0)]
