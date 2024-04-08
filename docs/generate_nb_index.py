#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import glob
import json
import logging
import os
import re
import urllib.parse
from typing import List, Tuple

NOTEBOOKS_LIST_PLACEHOLDER = "[[notebooks-list]]"
NOTEBOOKS_PAGE_TEMPLATE_RELATIVE_PATH = "notebooks.template.md"
NOTEBOOKS_PAGE_RELATIVE_PATH = "notebooks.md"
NOTEBOOKS_SECTION_KEY_VAR_SEP = "_"
NOTEBOOKS_DIRECTORY_NAME = "notebooks"

DEFAULT_REPO_NAME = "airbus/discrete-optimization"

doc_dir = os.path.dirname(os.path.abspath(__file__))
doc_source_dir = os.path.abspath(f"{doc_dir}/source")
rootdir = os.path.abspath(f"{doc_dir}/..")
notebooksdir = f"{rootdir}/{NOTEBOOKS_DIRECTORY_NAME}"

logger = logging.getLogger(__name__)


def extract_notebook_title_n_description(
    notebook_filepath: str,
) -> Tuple[str, List[str]]:
    # load notebook
    with open(notebook_filepath, "rt", encoding="utf-8") as f:
        notebook = json.load(f)

    # find title + description: from first cell,  h1 title + remaining text.
    # or title from filename else
    title = ""
    description_lines: List[str] = []
    cell = notebook["cells"][0]
    if cell["cell_type"] == "markdown":
        firstline = cell["source"][0].strip()
        if firstline.startswith("# "):
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
    return f"{notebooks_repo_url}/blob/{notebooks_branch}/{urllib.parse.quote(notebook_relative_path)}"


def get_colab_link(
    notebooks_repo_name: str,
    notebooks_branch: str,
    notebook_relative_path: str,
) -> str:
    if notebooks_repo_name:
        return f"https://colab.research.google.com/github/{notebooks_repo_name}/blob/{notebooks_branch}/{urllib.parse.quote(notebook_relative_path)}"
    else:
        return ""


def get_binder_link(
    binder_env_repo_name: str,
    binder_env_branch: str,
    notebooks_repo_url: str,
    notebooks_branch: str,
    notebook_relative_path: str,
    notebooks_repo_name: str,
    use_nbgitpuller: bool = False,
) -> str:
    # binder hub url
    jupyterhub = urllib.parse.urlsplit("https://mybinder.org")

    if use_nbgitpuller:
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
    else:
        if notebooks_repo_name:
            # path to the binder env
            binder_path = f"v2/gh/{notebooks_repo_name}/{notebooks_branch}"

            # query to open proper notebook
            query = urllib.parse.urlencode({"labpath": notebook_relative_path})

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
        else:
            link = ""

    return link


def get_repo_n_branches_for_binder_n_github_links() -> Tuple[
    bool, str, str, str, str, str, bool
]:
    # repos + branches to use for binder environment and notebooks content.
    creating_links = True
    use_nbgitpuller = False
    try:
        use_nbgitpuller_str = os.environ["AUTODOC_BINDER_NBGITPULLER"]
        try:
            use_nbgitpuller_int = int(use_nbgitpuller_str)
        except ValueError:
            use_nbgitpuller_int = 1
        if (use_nbgitpuller_str.lower() != "false") and (use_nbgitpuller_int != 0):
            use_nbgitpuller = True
    except KeyError:
        pass
    try:
        binder_env_repo_name = os.environ["AUTODOC_BINDER_ENV_GH_REPO_NAME"]
    except KeyError:
        binder_env_repo_name = DEFAULT_REPO_NAME
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
    try:
        notebooks_repo_name = os.environ["AUTODOC_NOTEBOOKS_REPO_NAME"]
    except KeyError:
        notebooks_repo_name = ""
        match = re.match(".*/(.*/.*)/?", notebooks_repo_url)
        if match:
            notebooks_repo_name = match.group(1)
    return (
        creating_links,
        notebooks_repo_url,
        notebooks_repo_name,
        notebooks_branch,
        binder_env_repo_name,
        binder_env_branch,
        use_nbgitpuller,
    )


if __name__ == "__main__":

    # List existing notebooks and write Notebooks page
    notebook_filepaths = sorted(glob.glob(f"{notebooksdir}/**/*.ipynb", recursive=True))
    notebooks_list_text = ""
    notebooksdir_prefixlen = len(notebooksdir) + 1
    sections_baselevel = 2
    current_sections = []
    (
        creating_links,
        notebooks_repo_url,
        notebooks_repo_name,
        notebooks_branch,
        binder_env_repo_name,
        binder_env_branch,
        use_nbgitpuller,
    ) = get_repo_n_branches_for_binder_n_github_links()
    # loop on notebooks sorted alphabetically by filenames
    for notebook_filepath in notebook_filepaths:
        # get subsections arborescence
        notebook_relpath = notebook_filepath[notebooksdir_prefixlen:]
        notebook_arbo = notebook_relpath.split(os.path.sep)
        notebook_sections = notebook_arbo[:-1]
        #  write missing sections
        for i_section, section in enumerate(notebook_sections):
            if (
                i_section >= len(current_sections)
                or section != current_sections[i_section]
            ):
                section_prefix = (sections_baselevel + i_section) * "#"
                section_name = section.split(NOTEBOOKS_SECTION_KEY_VAR_SEP)[-1]
                notebooks_list_text += f"{section_prefix} {section_name}\n\n"
        current_sections = notebook_sections
        # extract title and description
        title, description_lines = extract_notebook_title_n_description(
            notebook_filepath
        )
        # write title
        title_prefix = (sections_baselevel + len(notebook_sections)) * "#"
        notebooks_list_text += f"{title_prefix} {title}\n\n"
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
                notebooks_repo_name=notebooks_repo_name,
                use_nbgitpuller=use_nbgitpuller,
            )
            if binder_link:
                binder_badge = (
                    f"[![Binder](https://mybinder.org/badge_logo.svg)]({binder_link})"
                )
            else:
                binder_badge = ""
            github_link = get_github_link(
                notebooks_repo_url=notebooks_repo_url,
                notebooks_branch=notebooks_branch,
                notebook_relative_path=notebook_relative_path,
            )
            github_badge = f"[![Github](https://img.shields.io/badge/see-Github-579aca?logo=github)]({github_link})"
            colab_link = get_colab_link(
                notebooks_repo_name=notebooks_repo_name,
                notebooks_branch=notebooks_branch,
                notebook_relative_path=notebook_relative_path,
            )
            if colab_link:
                colab_badge = f"[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link})"
            else:
                colab_badge = ""
            # markdown item
            # notebooks_list_text += f"{github_badge}\n{binder_badge}\n\n"
            notebooks_list_text += f"{github_badge}\n"
            if colab_badge:
                notebooks_list_text += f"{colab_badge}\n"
            if binder_badge:
                notebooks_list_text += f"{binder_badge}\n"
            notebooks_list_text += "\n"

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
