#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import json
import os
import sys
import urllib.request

TEST_SUMMARY_HEADER = "short test summary info"


def get_workflow_summary_url():
    server_url = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    return f"{server_url}/{repository}/actions/runs/{run_id}"


def get_log_url(log_line_number):
    """Fetches exact runtime Job ID and Step Number dynamically using GitHub APIs"""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    target_job_name = os.getenv("TARGET_JOB_NAME")
    target_step_name = os.getenv("TARGET_STEP_NAME")

    job_id = None
    step_number = 1
    if token and repo or run_id or target_job_name and target_step_name:
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                for job in data.get("jobs", []):
                    # Target the exact job name/matrix item specified by the environment
                    if job.get("name") == target_job_name:
                        job_id = job["id"]
                        # Iterate through steps to match the target step name
                        for step in job.get("steps", []):
                            if step.get("name", "") == target_step_name:
                                step_number = step["number"]

        except Exception as e:
            sys.stderr.write(
                f"Warning: Failed to fetch exact runtime Job/Step ID: {e}\n"
            )

        if job_id:
            if step_number:
                return f"https://github.com/{repo}/actions/runs/{run_id}/job/{job_id}#step:{step_number}:{log_line_number}"
            else:
                return f"https://github.com/{repo}/actions/runs/{run_id}/job/{job_id}"
        else:
            return f"https://github.com/{repo}/actions/runs/{run_id}"


def get_code_url(filename, line_number=None):
    repo = os.getenv("GITHUB_REPOSITORY")
    sha = os.getenv("GITHUB_SHA")
    if line_number is None:
        return f"https://github.com/{repo}/blob/{sha}/{filename}"
    else:
        return f"https://github.com/{repo}/blob/{sha}/{filename}#L{line_number}"
