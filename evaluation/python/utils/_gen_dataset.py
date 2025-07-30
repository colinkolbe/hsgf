"""Util for HSGF Eval

- Only once used functions to create a 3M subset from the LAION-10M subset and calculate its
  groundtruth (via hsgf_py/graphidx)

- Dev note: See git history for code to double check the calculated groundtruth with other
  already existing groundtruth data from other subsets
"""

import time

import h5py
import numpy as np
from utils.eval_configs import BASE_PATH_DATASETS, SISAP_DATASET_PATH
from utils.util import (
    LAION_SISAP_Subset,
    _load_hdf5_sisap_data_or_queries,
    _load_hdf5_sisap_groundtruth,
)

import hsgf


def create_subset():
    subset_size = 3_000_000
    data = _load_hdf5_sisap_data_or_queries(
        f"../{SISAP_DATASET_PATH}/data/clip768/{LAION_SISAP_Subset.ten_m}/dataset.h5"
    )
    with h5py.File(f"../{SISAP_DATASET_PATH}/3M_dataset.h5", "w") as data_file:
        data_file.create_dataset("laion_3M_subset", data=data[:subset_size, :])


def calc_ground_truth():
    data = _load_hdf5_sisap_data_or_queries(
        f"../{SISAP_DATASET_PATH}/data/clip768/10M/dataset.h5"
    )
    queries = _load_hdf5_sisap_data_or_queries(
        f"../{SISAP_DATASET_PATH}/data/clip768/query.h5"
    )

    (ground_truth, dists) = hsgf.bruteforce_groundtruth(data, queries[0:2], k=1000)

    # so that it is also index 1 based
    ground_truth = (ground_truth + np.ones(shape=ground_truth.shape)).astype(
        np.uint64, copy=False
    )
    print("Groundtruth Shape: ", ground_truth.shape)
    with h5py.File(f"../{SISAP_DATASET_PATH}/ground_truth_10M_2.h5", "w") as data_file:
        data_file.create_dataset("dists", data=dists)
        data_file.create_dataset("knns", data=ground_truth)


def sample_subset_from_hdf5():
    create_subset()
    calc_ground_truth()


# -----


def create_normal_data_dataset():
    """Data, Queries, Groundtruth to H5 via hsgf"""
    nd, nq, d, k = 1_000_000, 10_000, 30, 1000
    indexed_queries = False
    (data, queries, ground_truth, dists) = hsgf.load_normal_data(
        nd, nq, d, k, indexed_queries
    )

    with h5py.File(
        f"../{BASE_PATH_DATASETS}/normaldata/normal_1M_base.h5", "w"
    ) as data_file:
        data_file.create_dataset("normal_1M", data=data)

    with h5py.File(
        f"../{BASE_PATH_DATASETS}/normaldata/normal_1M_query.h5", "w"
    ) as data_file:
        data_file.create_dataset("query", data=queries)

    # adding ones because of LAION dataset and using the same ground_truth loader in utils/util
    ground_truth = (ground_truth + np.ones(shape=ground_truth.shape)).astype(
        np.uint64, copy=False
    )

    with h5py.File(
        f"../{BASE_PATH_DATASETS}/normaldata/normal_1M_groundtruth.h5", "w"
    ) as data_file:
        data_file.create_dataset("dists", data=dists)
        data_file.create_dataset("knns", data=ground_truth)


def main():
    overall_time = time.time()
    # sample_subset_from_hdf5()
    create_normal_data_dataset()
    print(f"Gen_dataset Execution time: {time.time() - overall_time}")


if __name__ == "__main__":
    main()
