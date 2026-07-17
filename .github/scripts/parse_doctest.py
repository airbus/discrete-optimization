#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import re
import sys


def parse_pytest_output():
    # Matches patterns like: ---- FAILURE/ERROR TYPE: /path/to/file.py:line_num: docstring ----
    # Or common verbose outputs: UNEXPECTED EXCEPTION / FAILURE at src/app.py:12
    # Standard pytest doctest error blocks look like:
    # ________________ [doctest] src.app.func ________________
    # 010
    # 011 text
    # UNEXPECTED EXCEPTION or FAILURE

    current_file = None
    current_line = 1

    # Read from standard input line by line
    for line in sys.stdin:
        # Print the original output so it still shows up normally in the logs
        print(line, end="")

        # Check for doctest block headers to extract file path
        # Pytest doctest blocks often look like: /home/runner/work/.../src/app.py:10: DocTestFailure
        if "DocTestFailure" in line or "UnexpectedException" in line:
            match = re.search(r"([\w./-]+):(\d+):", line)
            if match:
                current_file = match.group(1)
                current_line = match.group(2)

                # Strip absolute runner paths if present to make inline warnings clean
                if "src/" in current_file:
                    current_file = "src/" + current_file.split("src/", 1)[1]

                print(
                    f"::warning file={current_file},line={current_line}::Doctest failure detected at {current_file}:{current_line}"
                )


if __name__ == "__main__":
    parse_pytest_output()
