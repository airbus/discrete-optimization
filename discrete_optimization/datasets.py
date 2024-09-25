"""Fetch datasets for examples and tests."""


#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import glob
import gzip
import os
import shutil
import tempfile
import zipfile
from typing import Optional
from urllib.request import urlcleanup, urlretrieve

DO_DEFAULT_DATAHOME = "~/discrete_optimization_data"
DO_DEFAULT_DATAHOME_ENVVARNAME = "DISCRETE_OPTIMIZATION_DATA"

COURSERA_REPO_URL = "https://github.com/discreteoptimization/assignment"
COURSERA_REPO_URL_SHA1 = "f69378420ce2bb845abaef0f448eab303aa7a7e7"
COURSERA_DATASETS = ["coloring", "facility", "knapsack", "tsp", "vrp"]
COURSERA_DATADIRNAME = "data"

SOLUTIONSUPDATE_BASE_URL = (
    "http://solutionsupdate.ugent.be/sites/default/files/datasets/instances"
)
SOLUTIONSUPDATE_DATASETS = ["RG30", "RG300"]

PSPLIB_FILES_BASE_URL = "https://www.om-db.wi.tum.de/psplib/files"
PSPLIB_DATASETS = {
    "j10.mm": "j1010_",
    "j120.sm": "j1201_",
    "j30.sm": "j301_",
    "j60.sm": "j601_",
}


IMOPSE_REPO_URL = "https://github.com/imopse/iMOPSE"
IMOPSE_REPO_URL_SHA1 = "e58ace53202ec29aa548dd0678ae3164d8349f4e"
IMOPSE_DATASET_RELATIVE_PATH = "configurations/problems/MSRCPSP/Regular"

MSPSPLIB_REPO_URL = "https://github.com/youngkd/MSPSP-InstLib"
MSPSPLIB_REPO_URL_SHA1 = "f77644175b84beed3bd365315412abee1a15eea1"

JSPLIB_REPO_URL = "https://github.com/tamy0612/JSPLIB"
JSPLIB_REPO_URL_SHA1 = "eea2b60dd7e2f5c907ff7302662c61812eb7efdf"

MSLIB_DATASET_URL = "http://www.projectmanagement.ugent.be/sites/default/files/datasets/MSRCPSP/MSLIB.zip"
MSLIB_DATASET_RELATIVE_PATH = "MSLIB.zip"

MIS_FILES = [
    "https://oeis.org/A265032/a265032_1dc.64.txt.gz",
    "https://oeis.org/A265032/a265032_1dc.128.txt.gz",
    "https://oeis.org/A265032/a265032_1dc.256.txt.gz",
    "https://oeis.org/A265032/a265032_1dc.512.txt.gz",
    "https://oeis.org/A265032/a265032_1dc.1024.txt.gz",
    "https://oeis.org/A265032/a265032_1dc.2048.txt.gz",
    "https://oeis.org/A265032/a265032_2dc.128.txt.gz",
    "https://oeis.org/A265032/a265032_2dc.256.txt.gz",
    "https://oeis.org/A265032/a265032_2dc.512.txt.gz",
    "https://oeis.org/A265032/a265032_2dc.1024.txt.gz",
    "https://oeis.org/A265032/a265032_2dc.2048.txt.gz",
    "https://oeis.org/A265032/a265032_1tc.8.txt.gz",
    "https://oeis.org/A265032/a265032_1tc.16.txt.gz",
    "https://oeis.org/A265032/a265032_1tc.32.txt.gz",
    "https://oeis.org/A265032/a265032_1tc.64.txt.gz",
    "https://oeis.org/A265032/a265032_1tc.128.txt.gz",
    "https://oeis.org/A265032/a265032_1tc.256.txt.gz",
    "https://oeis.org/A265032/a265032_1tc.512.txt.gz",
    "https://oeis.org/A265032/a265032_1tc.1024.txt.gz",
    "https://oeis.org/A265032/a265032_1tc.2048.txt.gz",
    "https://oeis.org/A265032/a265032_1et.64.txt.gz",
    "https://oeis.org/A265032/a265032_1et.128.txt.gz",
    "https://oeis.org/A265032/a265032_1et.256.txt.gz",
    "https://oeis.org/A265032/a265032_1et.512.txt.gz",
    "https://oeis.org/A265032/a265032_1et.1024.txt.gz",
    "https://oeis.org/A265032/a265032_1et.2048.txt.gz",
    "https://oeis.org/A265032/a265032_1zc.128.txt.gz",
    "https://oeis.org/A265032/a265032_1zc.256.txt.gz",
    "https://oeis.org/A265032/a265032_1zc.512.txt.gz",
    "https://oeis.org/A265032/a265032_1zc.1024.txt.gz",
    "https://oeis.org/A265032/a265032_1zc.2048.txt.gz",
    "https://oeis.org/A265032/a265032_1zc.4096.txt.gz",
]


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


def fetch_data_from_solutionsupdate(data_home: Optional[str] = None):
    """Fetch data for rcpsp examples from solutionsupdate.

    cf http://solutionsupdate.ugent.be/index.php/solutions-update

    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.

    """
    # get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # get rcpsp data directory
    rcpsp_dir = f"{data_home}/rcpsp"
    os.makedirs(rcpsp_dir, exist_ok=True)

    try:
        # download each datasets
        for dataset in SOLUTIONSUPDATE_DATASETS:
            url = f"{SOLUTIONSUPDATE_BASE_URL}/{dataset}.zip"
            local_file_path, _ = urlretrieve(url)
            with zipfile.ZipFile(local_file_path) as zipf:
                namelist = zipf.namelist()
                for name in namelist:
                    zipf.extract(name, path=rcpsp_dir)
    finally:
        # remove temporary files
        urlcleanup()


def fetch_data_from_psplib(data_home: Optional[str] = None):
    """Fetch data for rcpsp examples from psplib.

    cf https://www.om-db.wi.tum.de/psplib/data.html

    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.

    """
    #  get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # get rcpsp data directory
    rcpsp_dir = f"{data_home}/rcpsp"
    os.makedirs(rcpsp_dir, exist_ok=True)

    try:
        # download each datasets
        for dataset, prefix in PSPLIB_DATASETS.items():
            url = f"{PSPLIB_FILES_BASE_URL}/{dataset}.zip"
            local_file_path, _ = urlretrieve(url)
            with zipfile.ZipFile(local_file_path) as zipf:
                namelist = zipf.namelist()
                for name in namelist:
                    if name.startswith(prefix):
                        zipf.extract(name, path=rcpsp_dir)
    finally:
        # remove temporary files
        urlcleanup()


def fetch_data_from_imopse(data_home: Optional[str] = None):
    """Fetch data from iMOPSE repo for rcpsp_multiskill examples.

    https://github.com/imopse/iMOPSE

    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.

    """
    # get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # get rcpsp_multiskill data directory
    rcpsp_multiskill_dir = f"{data_home}/rcpsp_multiskill"
    os.makedirs(rcpsp_multiskill_dir, exist_ok=True)
    dataset_dir = rcpsp_multiskill_dir

    # download in a temporary file the repo data
    url = f"{IMOPSE_REPO_URL}/archive/{IMOPSE_REPO_URL_SHA1}.zip"

    try:
        local_file_path, headers = urlretrieve(url)
        # extract only data
        with zipfile.ZipFile(local_file_path) as zipf:
            namelist = zipf.namelist()
            rootdir = namelist[0].split("/")[0]
            dataset_prefix_in_zip = f"{rootdir}/{IMOPSE_DATASET_RELATIVE_PATH}/"
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


def fetch_data_from_mspsplib_repo(data_home: Optional[str] = None):
    """Fetch data from youngkd repo. (for multiskill rcpsp)

    https://github.com/youngkd/MSPSP-InstLib

    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.

    """
    #  get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # download in a temporary file the repo data
    url = f"{MSPSPLIB_REPO_URL}/archive/{MSPSPLIB_REPO_URL_SHA1}.zip"
    try:
        local_file_path, headers = urlretrieve(url)
        # extract only data
        with zipfile.ZipFile(local_file_path) as zipf:
            namelist = zipf.namelist()
            rootdir = namelist[0].split("/")[0]
            dataset_dir = f"{data_home}/MSPSP_Instances"
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_prefix_in_zip = f"{rootdir}/instances/"
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


def fetch_data_from_mslib(data_home: Optional[str] = None):
    """Fetch data from MSLIB for rcpsp_multiskill examples.
    cf https://www.projectmanagement.ugent.be/research/project_scheduling/MSRCPSP
    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.
    """
    #  get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # get rcpsp_multiskill data directory
    rcpsp_multiskill_dir = f"{data_home}/rcpsp_multiskill_mslib"
    os.makedirs(rcpsp_multiskill_dir, exist_ok=True)

    try:
        # download dataset
        local_file_path, headers = urlretrieve(MSLIB_DATASET_URL)
        with tempfile.TemporaryDirectory() as tmpdir:
            # extract only data
            with zipfile.ZipFile(local_file_path) as zipf:
                zipf.extractall(path=rcpsp_multiskill_dir)
    finally:
        # remove temporary files
        urlcleanup()


def decompress_gz_to_folder(input_file, output_folder, url):
    with gzip.open(input_file, "rb") as f_in:
        # Get the base name of the gzipped file without the .gz extension
        base_name = url[33:]
        file_name = os.path.splitext(base_name)[0]
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        # Construct the output file path for each extracted file
        output_file = os.path.join(output_folder, f"{file_name}")
        # Open the output file in write-binary mode ('wb')
        with open(output_file, "wb") as f_out:
            # Write the extracted file data to the output file
            shutil.copyfileobj(f_in, f_out)


def fetch_data_for_mis(data_home: Optional[str] = None):
    #  get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # get mis data directory
    mis_dir = f"{data_home}/mis"
    os.makedirs(mis_dir, exist_ok=True)

    try:
        # download each datasets
        for url in MIS_FILES:
            filename, _ = urlretrieve(url)
            decompress_gz_to_folder(filename, mis_dir, url)
    finally:
        # remove temporary files
        urlcleanup()


def fetch_data_from_jsplib_repo(data_home: Optional[str] = None):
    """Fetch data from jsplib repo. (for jobshop problems)

    https://github.com/tamy0612/JSPLIB

    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.

    """
    #  get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # download in a temporary file the repo data
    url = f"{JSPLIB_REPO_URL}/archive/{JSPLIB_REPO_URL_SHA1}.zip"
    try:
        local_file_path, headers = urlretrieve(url)
        # extract only data
        with zipfile.ZipFile(local_file_path) as zipf:
            namelist = zipf.namelist()
            rootdir = namelist[0].split("/")[0]
            dataset_dir = f"{data_home}/jobshop"
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_prefix_in_zip = f"{rootdir}/instances/"
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


def fetch_all_datasets(data_home: Optional[str] = None):
    """Fetch data used by examples for all packages.

    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.

    """
    fetch_data_from_coursera(data_home=data_home)
    fetch_data_from_psplib(data_home=data_home)
    fetch_data_from_imopse(data_home=data_home)
    fetch_data_from_solutionsupdate(data_home=data_home)
    fetch_data_for_mis(data_home=data_home)
    fetch_data_from_jsplib_repo(data_home=data_home)


if __name__ == "__main__":
    fetch_all_datasets()
