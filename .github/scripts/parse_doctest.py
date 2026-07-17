import json
import os
import re
import sys
import urllib.request


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
        return _construct_log_url(repo, run_id, job_id, step_number, log_line_number)


def _construct_log_url(repo, run_id, job_id, step_number, log_line_number):
    if job_id:
        if step_number:
            return f"https://github.com/{repo}/actions/runs/{run_id}/job/{job_id}#step:{step_number}:{log_line_number}"
        else:
            return f"https://github.com/{repo}/actions/runs/{run_id}/job/{job_id}"
    else:
        return f"https://github.com/{repo}/actions/runs/{run_id}"


def get_code_url(filename, line_number):
    repo = os.getenv("GITHUB_REPOSITORY")
    sha = os.getenv("GITHUB_SHA")
    return f"https://github.com/{repo}/blob/{sha}/{filename}#L{line_number}"


def parse_pytest_output():
    # Fetch environment variables supplied by GitHub Actions
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")

    failures = []
    log_line_number = 0

    for line in sys.stdin:
        log_line_number += 1
        print(line, end="")  # Preserve standard log output

        if "DocTestFailure" in line or "UnexpectedException" in line:
            match = re.search(r"([\w./-]+):(\d+):", line)
            if match:
                raw_file = match.group(1)
                line_number = match.group(2)

                # Normalize path to be relative to repo root
                clean_file = raw_file
                if "src/" in raw_file:
                    clean_file = "src/" + raw_file.split("src/", 1)[1]
                clean_file = clean_file.lstrip("./")

                failures.append(
                    {
                        "file": clean_file,
                        "line": line_number,
                        "code_url": get_code_url(
                            filename=clean_file, line_number=line_number
                        ),
                        "log_url": get_log_url(log_line_number=log_line_number),
                        "warning": f"::warning file={clean_file},line={line_number}::Doctest failure at {clean_file}:{line_number}",
                    }
                )
    # warning workflow commands
    for fail in failures:
        print(fail["warning"])

    # summary of failures with proper links
    if failures and summary_file:
        with open(summary_file, "a") as f:
            f.write("\n### ⚠️ Failing examples found in docstrings\n\n")
            f.write("| Target File | Line | Jump to Execution Logs | Link to code |\n")
            f.write("| --- | --- | --- | --- |\n")
            for fail in failures:
                f.write(
                    f"| `{fail['file']}` | {fail['line']} | [Go to Log Line]({fail['log_url']}) | [View Code]({fail['code_url']}) |\n"
                )


if __name__ == "__main__":
    parse_pytest_output()
