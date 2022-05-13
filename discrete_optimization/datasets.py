"""Fetch datasets for examples and tests."""

import glob
import os
import tempfile
import zipfile
from typing import Optional
from urllib.request import urlretrieve, urlcleanup

DO_DEFAULT_DATAHOME = "~/discrete_optimization_data"
DO_DEFAULT_DATAHOME_ENVVARNAME = "DISCRETE_OPTIMIZATION_DATA"

COURSERA_REPO_URL = "https://github.com/discreteoptimization/assignment"
COURSERA_REPO_URL_SHA1 = "f69378420ce2bb845abaef0f448eab303aa7a7e7"
COURSERA_DATASETS = ["coloring", "facility", "knapsack", "tsp", "vrp"]
COURSERA_DATADIRNAME = "data"

SKDECIDE_REPO_URL = "https://github.com/airbus/scikit-decide"
SKDECIDE_TAG = "v0.9.4"
SKDECIDE_RCPSP_DATADIR = "examples/discrete_optimization/data"


def get_data_home(data_home: Optional[str] = None) -> str:
    """Return the path of the discrete-optimization data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.
    By default the data dir is set to a folder named 'discrete_optimization_data' in the
    user home folder.
    Alternatively, it can be set by the 'DISCRETE_OPTIMIZATION_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.

    Params:
        data_home : The path to discrete-optimization data directory. If `None`, the default path
        is `~/discrete_optimization_data`.

    """
    if data_home is None:
        data_home = os.environ.get(DO_DEFAULT_DATAHOME_ENVVARNAME, DO_DEFAULT_DATAHOME)
    data_home = os.path.expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


def fetch_data_from_coursera(data_home: Optional[str] = None):
    """Fetch data from coursera repo.

    https://github.com/discreteoptimization/assignment

    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.

    """
    #  get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # download in a temporary file the repo data
    url = f"{COURSERA_REPO_URL}/archive/{COURSERA_REPO_URL_SHA1}.zip"
    try:
        local_file_path, headers = urlretrieve(url)
        # extract only data
        with zipfile.ZipFile(local_file_path) as zipf:
            namelist = zipf.namelist()
            rootdir = namelist[0].split("/")[0]
            for dataset in COURSERA_DATASETS:
                dataset_dir = f"{data_home}/{dataset}"
                os.makedirs(dataset_dir, exist_ok=True)
                dataset_prefix_in_zip = f"{rootdir}/{dataset}/{COURSERA_DATADIRNAME}/"
                for name in namelist:
                    if name.startswith(dataset_prefix_in_zip):
                        zipf.extract(name, path=dataset_dir)
                for datafile in glob.glob(f"{dataset_dir}/{dataset_prefix_in_zip}/*"):
                    os.replace(
                        src=datafile, dst=f"{dataset_dir}/{os.path.basename(datafile)}"
                    )
                os.removedirs(f"{dataset_dir}/{dataset_prefix_in_zip}")
    finally:
        urlcleanup()


def fetch_data_for_rcpsp(data_home: Optional[str] = None):
    """Fetch data for rcpsp and rcpsp_multiskill examples.

    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.

    """
    #  get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # download in a temporary file the repo data
    url = f"{SKDECIDE_REPO_URL}/archive/refs/tags/{SKDECIDE_TAG}.zip"
    try:
        local_file_path, headers = urlretrieve(url)
        # extract only data
        with zipfile.ZipFile(local_file_path) as zipf:
            namelist = zipf.namelist()
            rootdir = namelist[0].split("/")[0]
            prefix_in_zip = f"{rootdir}/{SKDECIDE_RCPSP_DATADIR}/"
            with tempfile.TemporaryDirectory() as tmpdir:
                for name in namelist:
                    if name.startswith(prefix_in_zip):
                        zipf.extract(name, path=tmpdir)
                # move to appropriate place
                for datafile in glob.glob(f"{tmpdir}/{prefix_in_zip}/*"):
                    os.replace(
                        src=datafile, dst=f"{data_home}/{os.path.basename(datafile)}"
                    )
    finally:
        urlcleanup()
