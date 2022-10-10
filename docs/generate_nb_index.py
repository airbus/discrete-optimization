#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import glob
import json
import logging
import os
import urllib.parse
from typing import List, Tuple

NOTEBOOKS_LIST_PLACEHOLDER = "[[notebooks-list]]"
NOTEBOOKS_PAGE_TEMPLATE_RELATIVE_PATH = "notebooks.template.md"
NOTEBOOKS_PAGE_RELATIVE_PATH = "notebooks.md"

doc_dir = os.path.dirname(os.path.abspath(__file__))
doc_source_dir = os.path.abspath(f"{doc_dir}/source")
rootdir = os.path.abspath(f"{doc_dir}/..")

logger = logging.getLogger(__name__)


def extract_notebook_title_n_description(
    notebook_filepath: str,
) -> Tuple[str, List[str]]:
    # load notebook
    with open(notebook_filepath, "rt") as f:
        notebook = json.load(f)

    # find title + description: from first cell,  h1 title + remaining text.
    # or title from filename else
    title = ""
    description_lines: List[str] = []
    cell = notebook["cells"][0]
    if cell["cell_type"] == "markdown":
        if cell["source"][0].startswith("# "):
            title = cell["source"][0][2:].strip()
            description_lines = cell["source"][1:]
        else:
            description_lines = cell["source"]
    if not title:
        title = os.path.splitext(os.path.basename(notebook_filepath))[0]

    return title, description_lines


def get_github_link(
    notebooks_repo_url: str,
    notebooks_branch: str,
    notebook_relative_path: str,
) -> str:
    return f"{notebooks_repo_url}/blob/{notebooks_branch}/{notebook_relative_path}"


def get_binder_link(
    binder_env_repo_name: str,
    binder_env_branch: str,
    notebooks_repo_url: str,
    notebooks_branch: str,
    notebook_relative_path: str,
) -> str:
    # binder hub url
    jupyterhub = urllib.parse.urlsplit("https://mybinder.org")

    # path to the binder env
    binder_path = f"v2/gh/{binder_env_repo_name}/{binder_env_branch}"

    # nbgitpuller query
    notebooks_repo_basename = os.path.basename(notebooks_repo_url)
    urlpath = f"tree/{notebooks_repo_basename}/{notebook_relative_path}"
    next_url_params = urllib.parse.urlencode(
        {
            "repo": notebooks_repo_url,
            "urlpath": urlpath,
            "branch": notebooks_branch,
        }
    )
    next_url = f"git-pull?{next_url_params}"
    query = urllib.parse.urlencode({"urlpath": next_url})

    # full link
    link = urllib.parse.urlunsplit(
        urllib.parse.SplitResult(
            scheme=jupyterhub.scheme,
            netloc=jupyterhub.netloc,
            path=binder_path,
            query=query,
            fragment="",
        )
    )

    return link


def get_repo_n_branches_for_binder_n_github_links() -> Tuple[bool, str, str, str, str]:
    # repos + branches to use for binder environment and notebooks content.
    creating_links = True
    try:
        binder_env_repo_name = os.environ["AUTODOC_BINDER_ENV_GH_REPO_NAME"]
    except KeyError:
        binder_env_repo_name = "airbus/discrete-optimization"
    try:
        binder_env_branch = os.environ["AUTODOC_BINDER_ENV_GH_BRANCH"]
    except KeyError:
        binder_env_branch = "binder"
    try:
        notebooks_repo_url = os.environ["AUTODOC_NOTEBOOKS_REPO_URL"]
        notebooks_branch = os.environ["AUTODOC_NOTEBOOKS_BRANCH"]
    except KeyError:
        # missing environment variables => no github and binder links creation
        notebooks_repo_url = ""
        notebooks_branch = ""
        creating_links = False
        logger.warning(
            "Missing environment variables AUTODOC_NOTEBOOKS_REPO_URL "
            "or AUTODOC_NOTEBOOKS_BRANCH to create github and binder links for notebooks."
        )
    return (
        creating_links,
        notebooks_repo_url,
        notebooks_branch,
        binder_env_repo_name,
        binder_env_branch,
    )


if __name__ == "__main__":

    # List existing notebooks and and write Notebooks page
    notebook_filepaths = sorted(glob.glob(f"{rootdir}/notebooks/*.ipynb"))
    notebooks_list_text = ""
    (
        creating_links,
        notebooks_repo_url,
        notebooks_branch,
        binder_env_repo_name,
        binder_env_branch,
    ) = get_repo_n_branches_for_binder_n_github_links()
    # loop on notebooks sorted alphabetically by filenames
    for notebook_filepath in notebook_filepaths:
        title, description_lines = extract_notebook_title_n_description(
            notebook_filepath
        )
        # subsection title
        notebooks_list_text += f"## {title}\n\n"
        # links
        if creating_links:
            notebook_path_prefix_len = len(f"{rootdir}/")
            notebook_relative_path = notebook_filepath[notebook_path_prefix_len:]
            binder_link = get_binder_link(
                binder_env_repo_name=binder_env_repo_name,
                binder_env_branch=binder_env_branch,
                notebooks_repo_url=notebooks_repo_url,
                notebooks_branch=notebooks_branch,
                notebook_relative_path=notebook_relative_path,
            )
            binder_badge = (
                f"[![Binder](https://mybinder.org/badge_logo.svg)]({binder_link})"
            )
            github_link = get_github_link(
                notebooks_repo_url=notebooks_repo_url,
                notebooks_branch=notebooks_branch,
                notebook_relative_path=notebook_relative_path,
            )
            github_badge = f"[![Github](https://img.shields.io/badge/see-Github-579aca?logo=github)]({github_link})"

            # markdown item
            # notebooks_list_text += f"{github_badge}\n{binder_badge}\n\n"
            notebooks_list_text += f"{github_badge}\n{binder_badge}\n\n"
        # description
        notebooks_list_text += "".join(description_lines)
        notebooks_list_text += "\n\n"

    with open(f"{doc_source_dir}/{NOTEBOOKS_PAGE_TEMPLATE_RELATIVE_PATH}", "rt") as f:
        readme_template_text = f.read()

    readme_text = readme_template_text.replace(
        NOTEBOOKS_LIST_PLACEHOLDER, notebooks_list_text
    )

    with open(f"{doc_source_dir}/{NOTEBOOKS_PAGE_RELATIVE_PATH}", "wt") as f:
        f.write(readme_text)
