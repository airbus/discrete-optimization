import os
from typing import Optional

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.salbp.problem import SalbpProblem


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for tsp.

    Params:
        data_folder: folder where datasets for weighted tardiness problem should be found.
            If None, we look in "wt" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/salpb"

    try:
        subfolders = [
            f
            for f in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, f))
        ]
        files = []
        print(subfolders)
        for subfolder in subfolders:
            sf = os.path.join(data_folder, subfolder)
            files.extend([os.path.join(sf, f) for f in os.listdir(sf) if "alb" in f])
        return files
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)


def remove_artifacts(text: str) -> str:
    cleaned = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "[":
            # Check if this is a source tag
            # We look ahead to see if it closes with ]
            # If so, we skip until ]
            j = text.find("]", i)
            if j != -1:
                # check if it looks like a source tag to be safe
                # (optional, but good practice)
                snippet = text[i : j + 1]
                if "source" in snippet:
                    i = j + 1
                    continue

        cleaned.append(text[i])
        i += 1
    return "".join(cleaned)


def parse_alb_file(file_path: str) -> SalbpProblem:
    """
    Parses a .alb file using string splitting and tokenization.
    No Regular Expressions used.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # 1. Clean artifacts
    content = remove_artifacts(raw_content)

    data = {"number_of_tasks": 0, "cycle_time": 0, "task_times": {}, "precedence": []}

    # 2. Split by lines to find section headers, but treat data as tokens
    # We will identify the index in the 'tokens' list where sections start.

    # Normalize newlines and split into tokens (words/numbers)
    # This handles the case where "id" is on one line and "time" on the next.
    tokens = content.split()
    # State machine parsing over tokens
    current_section = None

    iterator = iter(tokens)

    try:
        while True:
            token = next(iterator)

            # Detect Tags
            if token.startswith("<"):
                # Reconstruct full tag if it has spaces (e.g., <number of tasks>)
                tag_acc = [token]
                while not token.endswith(">"):
                    token = next(iterator)
                    tag_acc.append(token)
                full_tag = " ".join(tag_acc).replace("<", "").replace(">", "")

                current_section = full_tag
                continue

            # Process Data based on current section
            if current_section == "number of tasks":
                data["number_of_tasks"] = int(token)
                current_section = None  # Done with this section

            elif current_section == "cycle time":
                data["cycle_time"] = int(token)
                current_section = None  # Done with this section

            elif current_section == "task times":
                # Expecting pairs: ID Duration
                t_id = int(token)
                t_time = int(next(iterator))
                data["task_times"][t_id] = t_time

            elif current_section == "precedence relations":
                # Expecting: pred,succ (possibly with spaces if file is messy, but usually no spaces around comma)
                # If the parser split "1,2" into "1,2", we good.
                # If it split "1, 2", we need to handle comma.

                if "," in token:
                    parts = token.split(",")
                    p = int(parts[0])
                    s = int(parts[1])
                    data["precedence"].append((p, s))
                else:
                    # Maybe format is "1 2"? Standard is comma.
                    # Let's assume standard "1,9" format based on file provided.
                    pass

    except StopIteration:
        pass

    return SalbpProblem(
        data["number_of_tasks"],
        data["cycle_time"],
        data["task_times"],
        data["precedence"],
    )
