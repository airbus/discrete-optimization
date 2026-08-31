#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import re
import sys
from typing import TextIO

sys.path.insert(0, os.path.dirname(__file__))

from parse_pytest_output_utils import TEST_SUMMARY_HEADER, get_code_url, get_log_url


def parse_pytest_output(input: TextIO | None = None):
    if input is None:
        input = sys.stdin

    # Fetch environment variables supplied by GitHub Actions
    summary_file = os.getenv("GITHUB_STEP_SUMMARY", "summary.md")

    doctest_failures = []
    log_line_number = 0
    in_test_summary = False
    collection_failures = []

    for line in input:
        log_line_number += 1
        print(line, end="")  # Preserve standard log output

        if not in_test_summary:
            if TEST_SUMMARY_HEADER in line:
                in_test_summary = True
            elif any(
                exception_to_catch in line
                for exception_to_catch in ["DocTestFailure", "UnexpectedException"]
            ):
                match = re.search(r"([\w./-]+):(\d+):", line)
                if match:
                    raw_file = match.group(1)
                    line_number = match.group(2)

                    # Normalize path to be relative to repo root
                    clean_file = raw_file
                    if "src/" in raw_file:
                        clean_file = "src/" + raw_file.split("src/", 1)[1]
                    clean_file = clean_file.lstrip("./")

                    doctest_failures.append(
                        {
                            "file": clean_file,
                            "line": line_number,
                            "code_url": get_code_url(
                                filename=clean_file, line_number=line_number
                            ),
                            "log_url": get_log_url(log_line_number=log_line_number),
                        }
                    )

        else:
            match = re.search(r"^ERROR\s+([^\s]+)\s(.*)$", line)
            if match:
                file = match.group(1)
                collection_failures.append(
                    {
                        "file": file,
                        "code_url": get_code_url(filename=file),
                        "line": "-",
                        "log_url": get_log_url(log_line_number=log_line_number),
                    }
                )

    failures = doctest_failures + collection_failures

    # summary of failures with proper links
    if failures:
        with open(summary_file, "a") as f:
            f.write(f"\n### ⚠️ Failing examples found in docstrings\n\n")
            f.write("| Target File | Line | Jump to Execution Logs | Link to code |\n")
            f.write("| --- | --- | --- | --- |\n")
            for fail in failures:
                f.write(
                    f"| `{fail['file']}` | {fail['line']} | [Go to Log Line]({fail['log_url']}) | [View Code]({fail['code_url']}) |\n"
                )


if __name__ == "__main__":
    parse_pytest_output()
