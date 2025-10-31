#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import glob
import importlib
import importlib.machinery
import importlib.util
import logging
import os
import sys
import traceback

logger = logging.getLogger(__name__)

EXAMPLES_DIR = os.path.abspath(f"{os.path.dirname(__file__)}/../examples")


def find_python_filepaths(root_dir: str) -> list[str]:
    return glob.glob(f"{root_dir}/**/*.py", recursive=True)


def import_script(filepath) -> None:
    working_dir_bak = os.getcwd()
    working_dir, filename = os.path.split(filepath)
    module_name, _ = os.path.splitext(filename)
    os.chdir(working_dir)  # to have same import available as from the script
    sys.path.insert(0, working_dir)  # pytest does not add working_dir to python path
    try:
        importlib.import_module(module_name)
    finally:
        os.chdir(working_dir_bak)
        sys.path.pop(0)


def print_script_traceback(e: Exception, filepath: str) -> None:
    frames = traceback.extract_tb(e.__traceback__)
    relevant_frames = []
    relevant_frame_reached = False
    filepath = os.path.abspath(filepath)
    for frame in frames:
        if frame.filename == filepath:
            relevant_frame_reached = True
        if relevant_frame_reached:
            relevant_frames.append(frame)

    print(f"{filepath}: {e.__class__.__name__}", file=sys.stderr)
    traceback.print_list(relevant_frames)
    print(f"{e.__class__.__name__}: {e}", file=sys.stderr)


def test_importing_all_examples():
    scripts_with_errors = []

    for filepath in find_python_filepaths(EXAMPLES_DIR):
        try:
            import_script(filepath)
        except Exception as e:
            mini_filepath = filepath[len(EXAMPLES_DIR) + 1 :]
            scripts_with_errors.append(mini_filepath)
            print_script_traceback(e, filepath)
    if len(scripts_with_errors) > 0:
        raise ImportError(
            f"{len(scripts_with_errors)} scripts from examples/ cannot be imported\n"
            + f"{scripts_with_errors}"
        )


if __name__ == "__main__":
    test_importing_all_examples()
