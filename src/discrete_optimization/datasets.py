"""Fetch datasets for examples and tests."""


#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import glob
import gzip
import os
import shutil
import tarfile
import tempfile
import zipfile
from typing import Optional
from urllib.request import urlcleanup, urlretrieve

DO_DEFAULT_DATAHOME = "~/discrete_optimization_data"
DO_DEFAULT_DATAHOME_ENVVARNAME = "DISCRETE_OPTIMIZATION_DATA"

OR_LIBRARY_ROOT_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"

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

SALBP_OTTO_DATASET_URL = (
    "https://assembly-line-balancing.de/wp-content/uploads/2017/01/SALBP_benchmark.zip"
)
SALBP_OTTO_DATASET_RELATIVE_PATH = "SALBP_benchmark.zip"


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

BPPC_ZIP = "https://site.unibo.it/operations-research/en/research/library-of-codes-and-instances-1/bppc.zip/@@download/file/BPPC.zip"


FJSP_DATASET_PREFIX = "jfsp_openhsu"
MIS_DATASET_PREFIX = "mis"
VRPTW_DATASET_PREFIX = "vrptw/homberger_200_customer_instances"

ERROR_MSG_MISSING_DATASETS = (
    "\nYou probably have not downloaded the needed dataset.\n"
    "You can do it by using the proper function in discrete_optimization.datasets.\n"
    'Fetch all datasets used by discrete-optimization with the command "python -m discrete_optimization.datasets".'
)


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
    # get the proper data directory
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


def fetch_data_from_alb(data_home: Optional[str] = None):
    """Fetch data from SALPB for simple assembly line problem examples.
    cf https://assembly-line-balancing.de/salbp/benchmark-data-sets-2013/
    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.
    """
    #  get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # get rcpsp_multiskill data directory
    rcpsp_multiskill_dir = f"{data_home}/salpb"
    os.makedirs(rcpsp_multiskill_dir, exist_ok=True)

    try:
        # download dataset
        local_file_path, headers = urlretrieve(SALBP_OTTO_DATASET_URL)
        with tempfile.TemporaryDirectory() as tmpdir:
            # extract only data
            with zipfile.ZipFile(local_file_path) as zipf:
                zipf.extractall(path=rcpsp_multiskill_dir)
            for file in glob.glob(f"{rcpsp_multiskill_dir}/*"):
                if "zip" in file:
                    with zipfile.ZipFile(file) as zipf:
                        zipf.extractall(path=rcpsp_multiskill_dir)
                    os.remove(file)
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
    mis_dir = f"{data_home}/{MIS_DATASET_PREFIX}"
    os.makedirs(mis_dir, exist_ok=True)

    try:
        # download each datasets
        for url in MIS_FILES:
            filename, _ = urlretrieve(url)
            decompress_gz_to_folder(filename, mis_dir, url)
    finally:
        # remove temporary files
        urlcleanup()


def fetch_mis_from_repo(data_home: Optional[str] = None) -> None:
    """Fetch mis dataset stored in g-poveda repo."""
    fetch_datasets_from_repo(data_home=data_home, dataset_prefixes=[MIS_DATASET_PREFIX])


def fetch_fjsp_from_repo(data_home: Optional[str] = None) -> None:
    """Fetch fjsp dataset stored in g-poveda repo."""
    fetch_datasets_from_repo(
        data_home=data_home, dataset_prefixes=[FJSP_DATASET_PREFIX]
    )


def fetch_datasets_from_repo(
    data_home: Optional[str] = None, dataset_prefixes: Optional[list[str]] = None
) -> None:
    """Fetch all datasets stored in g-poveda repo."""
    url_repo = "https://github.com/g-poveda/do-data"
    sha_url_repo = "f078cf0ee5440aeae72af9b6c5c83c14acbb2888"
    url = f"{url_repo}/archive/{sha_url_repo}.zip"
    if dataset_prefixes is None:
        dataset_prefixes = [
            FJSP_DATASET_PREFIX,
            MIS_DATASET_PREFIX,
            VRPTW_DATASET_PREFIX,
        ]
    try:
        local_file_path, headers = urlretrieve(url)
        for dataset_prefix in dataset_prefixes:
            _extract_dataset_from_zipped_repo(
                zipped_repo_path=local_file_path,
                dataset_prefix=dataset_prefix,
                data_home=data_home,
            )
    finally:
        urlcleanup()


def _extract_dataset_from_zipped_repo(
    zipped_repo_path: str, dataset_prefix: str, data_home: Optional[str] = None
):
    data_home = get_data_home(data_home=data_home)
    # extract only dataset with given prefix
    with zipfile.ZipFile(zipped_repo_path) as zipf:
        namelist = zipf.namelist()
        rootdir = namelist[0].split("/")[0]
        dataset_dir = f"{data_home}/{dataset_prefix}"
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_prefix_in_zip = f"{rootdir}/{dataset_prefix}/"
        filename_to_move = []
        for name in namelist:
            if name.startswith(dataset_prefix_in_zip):
                zipf.extract(name, path=dataset_dir)
                filename_to_move.append(name)
        for datafile in filename_to_move:
            if os.path.isdir(datafile):
                continue
            if len(os.path.basename(datafile)) == 0:
                continue
            destination = os.path.join(
                dataset_dir, str(datafile).replace(dataset_prefix_in_zip, "")
            )
            if not os.path.exists(os.path.dirname(destination)):
                os.makedirs(os.path.dirname(destination))
            os.replace(src=os.path.join(dataset_dir, datafile), dst=destination)


def fetch_data_from_jsplib_repo(data_home: Optional[str] = None):
    """Fetch data from jsplib repo. (for jobshop problems)

    https://github.com/tamy0612/JSPLIB

    Params:
        data_home: Specify the cache folder for the datasets. By default
            all discrete-optimization data is stored in '~/discrete_optimization_data' subfolders.

    """
    # get the proper data directory
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


def fetch_data_fjsp(data_home: Optional[str] = None):
    data_home = get_data_home(data_home=data_home)
    url = "https://openhsu.ub.hsu-hh.de/bitstreams/4ed8d5b1-2546-4a30-8f3a-a8f3732ffbbd/download"
    try:
        local_file_path, headers = urlretrieve(url)
        # extract only data
        dataset_dir = f"{data_home}/{FJSP_DATASET_PREFIX}"
        os.makedirs(dataset_dir, exist_ok=True)
        with zipfile.ZipFile(local_file_path) as zipf:
            zipf.extractall(path=dataset_dir)
    finally:
        urlcleanup()


def fetch_data_from_bppc(data_home: Optional[str] = None):
    """Fetch data from bin packing problem with conflicts benchmark"""
    # get the proper data directory
    data_home = get_data_home(data_home=data_home)

    # download in a temporary file the repo data
    url = BPPC_ZIP
    try:
        local_file_path, headers = urlretrieve(url)
        # extract only data
        with zipfile.ZipFile(local_file_path) as zipf:
            namelist = zipf.namelist()
            rootdir = namelist[0].split("/")[0]
            dataset_dir = f"{data_home}/bppc"
            os.makedirs(dataset_dir, exist_ok=True)
            # dataset_prefix_in_zip = f"{rootdir}/Istanze/"
            for name in namelist:
                if "Istanze" in name:
                    zipf.extract(name, path=dataset_dir)
            for datafile in glob.glob(f"{dataset_dir}/Istanze/*"):
                os.replace(
                    src=datafile, dst=f"{dataset_dir}/{os.path.basename(datafile)}"
                )
            os.removedirs(f"{dataset_dir}/Istanze")
    except Exception as e:
        print(e)
    finally:
        urlcleanup()


def fetch_data_from_cp25(data_home: Optional[str] = None):
    data_home = get_data_home(data_home=data_home)
    CP25_REPO_URL = "https://github.com/ML-KULeuven/Explainable-Workforce-Scheduling/"
    CP25_REPO_URL_SHA1 = "9b1fc7e38fa7e80f75de88ee206f7cd8a9cf51a9"
    url = f"{CP25_REPO_URL}/archive/{CP25_REPO_URL_SHA1}.zip"
    try:
        local_file_path, headers = urlretrieve(url)
        # extract only data
        with zipfile.ZipFile(local_file_path) as zipf:
            namelist = zipf.namelist()
            rootdir = namelist[0].split("/")[0]
            dataset_dir = f"{data_home}/workforce"
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_prefix_in_zip = f"{rootdir}/data/anon_jsons"
            for name in namelist:
                if name.startswith(dataset_prefix_in_zip):
                    zipf.extract(name, path=dataset_dir)
            for datafile in glob.glob(f"{dataset_dir}/{dataset_prefix_in_zip}/*"):
                os.replace(
                    src=datafile, dst=f"{dataset_dir}/{os.path.basename(datafile)}"
                )
            os.removedirs(f"{dataset_dir}/{dataset_prefix_in_zip}")
    except Exception as e:
        print(e)
    finally:
        urlcleanup()


def fetch_data_weighted_tardiness_single_machine(data_home: Optional[str] = None):
    data_home = get_data_home(data_home=data_home)
    folder = os.path.join(data_home, "wt/")
    if not os.path.exists(folder):
        os.makedirs(folder)
    for file in ["wt40", "wt50", "wt100"]:
        try:
            urlretrieve(
                url=OR_LIBRARY_ROOT_URL + file + ".txt",
                filename=os.path.join(folder, file + ".txt"),
            )
        except Exception as e:
            print(e)
        finally:
            urlcleanup()


def fetch_data_tsptw(data_home: Optional[str] = None):
    data_home = get_data_home(data_home=data_home)
    tsp_tw_folder = os.path.join(data_home, "tsptw")
    if not os.path.exists(tsp_tw_folder):
        os.makedirs(tsp_tw_folder)
    urls = [
        "SolomonPotvinBengio.tar.gz",
        "Langevin.tar.gz",
        "Dumas.tar.gz",
        "GendreauDumasExtended.tar.gz",
        "OhlmannThomas.tar.gz",
        "AFG.tar.gz",
        "SolomonPesant.tar.gz",
    ]
    base_url = "https://lopez-ibanez.eu/files/TSPTW/"
    try:
        for url in urls:
            # base_name = url.removesuffix(".tar.gz")
            file_name = os.path.join(tsp_tw_folder, url)
            urlretrieve(base_url + url, filename=file_name)
            with tarfile.open(file_name, "r:gz") as tar:
                tar.extractall(path=tsp_tw_folder)
            os.remove(file_name)
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
    fetch_datasets_from_repo(data_home=data_home)
    fetch_data_from_jsplib_repo(data_home=data_home)
    fetch_data_from_bppc(data_home=data_home)
    fetch_data_from_cp25(data_home=data_home)
    fetch_data_weighted_tardiness_single_machine(data_home=data_home)
    fetch_data_tsptw(data_home=data_home)
    fetch_data_from_mslib(data_home=data_home)
    fetch_data_from_mspsplib_repo(data_home=data_home)
    fetch_data_from_alb(data_home=data_home)


if __name__ == "__main__":
    fetch_all_datasets()
