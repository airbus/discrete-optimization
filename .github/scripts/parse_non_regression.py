#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import re
import sys
from typing import TextIO

sys.path.insert(0, os.path.dirname(__file__))

from parse_pytest_output_utils import (
    TEST_SUMMARY_HEADER,
    get_log_url,
    get_workflow_summary_url,
)

REGRESSIONS_CSVFILE = "regressions.csv"
PR_COMMENT_FILE = "regressions_pr_comment.md"


def parse_pytest_output(input: TextIO | None = None):
    if input is None:
        input = sys.stdin

    # Fetch environment variables supplied by GitHub Actions
    summary_file = os.getenv("GITHUB_STEP_SUMMARY", "summary.md")

    regression_failures = []
    other_failures = []
    collection_failures = []
    log_line_number = 0
    in_test_summary = False
    for line in input:
        log_line_number += 1
        print(line, end="")  # Preserve standard log output

        if not in_test_summary:
            match = re.search(r"^E\s*CSV_LINE:(.*?)$", line)
            if match:
                csv_line = match.group(1)
                test_id = csv_line.split(",")[0]
                regression_failures.append(
                    {
                        "test_id": test_id,
                        "csv_line": csv_line,
                        "log_url": get_log_url(log_line_number=log_line_number),
                    }
                )
            if TEST_SUMMARY_HEADER in line:
                in_test_summary = True
                regression_test_ids = {fail["test_id"] for fail in regression_failures}
        else:
            match = re.search(r"^ERROR\s+([^\s]+)\s(.*)$", line)
            if match:
                test_id = match.group(1)
                collection_failures.append(
                    {
                        "test_id": test_id,
                        "log_url": get_log_url(log_line_number=log_line_number),
                    }
                )
            match = re.search(r"^FAILED\s+([^\s]+)\s(.*)$", line)
            if match:
                test_id = match.group(1)
                if test_id not in regression_test_ids:
                    other_failures.append(
                        {
                            "test_id": test_id,
                            "log_url": get_log_url(log_line_number=log_line_number),
                        }
                    )

    if regression_failures:
        # summary of failures with proper links
        with open(summary_file, "at", encoding="utf-8") as f:
            f.write(f"\n### ⚠️ Regression found in non-regression tests\n\n")
            f.write("| Test id | Jump to Execution Logs |\n")
            f.write("| --- | --- |\n")
            for fail in regression_failures:
                f.write(
                    f"| `{fail['test_id']}` | [Go to Log Line]({fail['log_url']}) |\n"
                )
    if other_failures:
        with open(summary_file, "at", encoding="utf-8") as f:
            f.write(f"\n### ⚠️ Error found in non-regression tests\n\n")
            f.write("| Test id | Jump to Execution Logs |\n")
            f.write("| --- | --- |\n")
            for fail in other_failures:
                f.write(
                    f"| `{fail['test_id']}` | [Go to Log Line]({fail['log_url']}) |\n"
                )
    if collection_failures:
        with open(summary_file, "at", encoding="utf-8") as f:
            f.write(f"\n### ⚠️ Error found collecting non-regression tests\n\n")
            f.write("| Test id | Jump to Execution Logs |\n")
            f.write("| --- | --- |\n")
            for fail in collection_failures:
                f.write(
                    f"| `{fail['test_id']}` | [Go to Log Line]({fail['log_url']}) |\n"
                )

    # regressions.csv
    if regression_failures:
        with open(REGRESSIONS_CSVFILE, "wt") as f:
            for fail in regression_failures:
                f.write(fail["csv_line"] + "\n")

    if os.getenv("GITHUB_EVENT_NAME", "") == "pull_request":
        failure_types_found_list = []
        if collection_failures:
            failure_types_found_list.append("collecting errors")
        if other_failures:
            failure_types_found_list.append("errors")
        if regression_failures:
            failure_types_found_list.append("regressions")
        if failure_types_found_list:
            if len(failure_types_found_list) > 2:
                first_types = ", ".join(failure_types_found_list[:-1])
                failure_types_found = ", and ".join(
                    [first_types, failure_types_found_list[-1]]
                )
            else:
                failure_types_found = " and ".join(failure_types_found_list)
            worker_id = os.getenv("WORKER_ID", "")
            PR_comment = (
                f"### ⚠️ Non-regression tests{' for ' + worker_id if worker_id else ''}\n\n"
                f"{failure_types_found.capitalize()} found. See [workflow logs]({get_workflow_summary_url()}) for more details.\n"
            )
            with open(PR_COMMENT_FILE, "at", encoding="utf-8") as f:
                f.write(PR_comment)


if __name__ == "__main__":
    parse_pytest_output()
