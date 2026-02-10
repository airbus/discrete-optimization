from typing import Any

from discrete_optimization.salbp.problem import SalbpProblem


def remove_artifacts(text: str) -> str:
    """
    Removes artifacts without using Regex.
    It iterates through the string and skips content between '[' and ']'.
    """
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


def parse_alb_file(file_path: str) -> dict[str, Any]:
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

    # Helper to find data between tags
    def get_tokens_between(start_tag_tokens, end_tag_token_start):
        # Find start
        start_idx = -1
        # Naive search for the tag sequence in the token list
        # e.g. ["<", "number", "of", "tasks", ">"]

        # Easier approach: Since tags are unique strings like "<number of tasks>",
        # let's map known tags to simpler keys in the token stream if possible,
        # OR just iterate the tokens statefully.
        pass

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


if __name__ == "__main__":
    file = "/Users/poveda_g/Downloads/SALBP_benchmark/very large data set_n=1000/instance_n=1000_1.alb"
    p: SalbpProblem = parse_alb_file(file)
    print(p.number_of_tasks, p.tasks, p.task_times, p.cycle_time)
