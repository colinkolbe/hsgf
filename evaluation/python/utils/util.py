"""Util for HSGF Eval

- Loaders for fvecs and h5 (specifically here SISAP LAION) datasets
- Search loops for evaluating/calculating QPS and Recall
    - Plus helpers

"""

import logging
import os

import h5py
import hsgf
import numpy as np
import pandas as pd

# ---------------------------- Init ----------------------------
RESULT_PATH = "../results/python"

# Make sure that directories to save the experiment results .csv files exist.
os.makedirs(f"{RESULT_PATH}/base_graphs_eval", exist_ok=True)
os.makedirs(f"{RESULT_PATH}/hnsw_eval", exist_ok=True)
os.makedirs(f"{RESULT_PATH}/hsgf_eval", exist_ok=True)
os.makedirs(f"{RESULT_PATH}/log", exist_ok=True)
logging.basicConfig(
    filename=f"{RESULT_PATH}/log/log.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,  # do not let other (imported) modules log their DEBUG
)

# ---------------------------- Dataset Util ----------------------------

# Dataset paths
BASE_PATH_DATASETS = "../datasets/"
SISAP_DATASET_PATH = "../datasets/"


class Fvecs_Dataset:
    AUDIO = "audio"
    ENRON = "enron"
    DEEP = "deep1m"
    # GIST = "gist" # currently not available
    GLOVE = "glove-100"
    SIFT = "sift"
    SIFTSMALL = "siftsmall"


class LAION_SISAP_Subset:
    three_hun_k = "300K"
    three_m = "3M"
    ten_m = "10M"
    hundred_m = "100M"


def is_laion(dataset):
    if (
        dataset == LAION_SISAP_Subset.three_hun_k
        or dataset == LAION_SISAP_Subset.three_m
        or dataset == LAION_SISAP_Subset.ten_m
        or dataset == LAION_SISAP_Subset.hundred_m
    ):
        return True
    else:
        return False


def get_data(
    dataset,
    reduce_k_audio=20,
):
    """The audio dataset is always reduced from k=100 to k=20 to be line with the literature"""
    print("Dataset: ", dataset)  # just logging to the console
    match dataset:
        case (
            Fvecs_Dataset.AUDIO
            | Fvecs_Dataset.ENRON
            | Fvecs_Dataset.DEEP
            | Fvecs_Dataset.GLOVE
            | Fvecs_Dataset.SIFT
            | Fvecs_Dataset.SIFTSMALL
        ):
            reduce_k = reduce_k_audio if dataset == Fvecs_Dataset.AUDIO else None
            return load_fvecs_dataset(dataset, reduce_k)
        case (
            LAION_SISAP_Subset.three_hun_k
            | LAION_SISAP_Subset.three_m
            | LAION_SISAP_Subset.ten_m
            | LAION_SISAP_Subset.hundred_m
        ):
            return load_laion_sisap_h5(dataset)
        case "NormalData":
            # (nd, nq, d, k) = (100_000, 10_000, 30, 1000)
            # indexed_queries = False
            # Note this computes the ground truth! (in Rust at least though)
            # (data, queries, ground_truth, _dists) = hsgf.load_normal_data(
            #     nd, nq, d, k, indexed_queries=indexed_queries
            # )
            # return (data, queries, ground_truth, nd, nq, d, k)
            """Uses fixed, precomputed normal data, so it can be used for fair comparisons
            with different ANNS graph builders and additionally avoids re-computing.
            However, adds the risk of over-fitting (as with the other datasets).

            Currently saved dataset:
            nd, nq, d, k = 1_000_000, 10_000, 30, 1000
            indexed_queries = False
            """
            data = _load_hdf5_sisap_data_or_queries(
                f"{BASE_PATH_DATASETS}/normaldata/normal_1M_base.h5"
            )
            queries = _load_hdf5_sisap_data_or_queries(
                f"{BASE_PATH_DATASETS}/normaldata/normal_1M_query.h5"
            )
            ground_truth = _load_hdf5_sisap_groundtruth(
                f"{BASE_PATH_DATASETS}/normaldata/normal_1M_groundtruth.h5"
            )
            return (
                data,
                queries,
                ground_truth,
                data.shape[0],
                queries.shape[0],
                data.shape[1],
                ground_truth.shape[1],
            )

        case _:
            raise Exception("Specify available dataset.")


def load_fvecs_dataset(dataset: Fvecs_Dataset, reduce_k=None):
    (data, queries, ground_truth) = (
        _fvecs_read(f"{BASE_PATH_DATASETS}/{dataset}/{dataset}_base.fvecs"),
        _fvecs_read(f"{BASE_PATH_DATASETS}/{dataset}/{dataset}_query.fvecs"),
        _ivecs_read(dataset, reduce_k=reduce_k),
    )
    return (
        data,
        queries,
        ground_truth,
        data.shape[0],
        queries.shape[0],
        data.shape[1],
        ground_truth.shape[1],
    )


def load_laion_sisap_h5(dataset: LAION_SISAP_Subset):
    if dataset == LAION_SISAP_Subset.three_m:
        data = _load_hdf5_sisap_data_or_queries(f"{SISAP_DATASET_PATH}/3M_dataset.h5")
    else:
        data = _load_hdf5_sisap_data_or_queries(
            f"{SISAP_DATASET_PATH}/data/clip768/{dataset}/dataset.h5"
        )
    queries = _load_hdf5_sisap_data_or_queries(
        f"{SISAP_DATASET_PATH}/data/clip768/query.h5"
    )
    ground_truth = _load_hdf5_sisap_groundtruth(
        f"{SISAP_DATASET_PATH}/ground_truth_{dataset}.h5"
    )

    return (
        data,
        queries,
        ground_truth,
        data.shape[0],
        queries.shape[0],
        data.shape[1],
        ground_truth.shape[1],
    )


def get_reduced_groundtruth(ground_truth, reduce_k):
    return np.array(ground_truth[:, 0:reduce_k])


# ----- Helpers -----


# Source: https://gist.github.com/danoneata/49a807f47656fedbb389
def _fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


# See _fvecs_read()
def _ivecs_read(dataset, c_contiguous=True, reduce_k=None):
    filename = f"{BASE_PATH_DATASETS}/{dataset}/{dataset}_groundtruth.ivecs"
    ground_truth = np.fromfile(filename, dtype="int32")
    if ground_truth.size == 0:
        return np.zeros((0, 0))
    dim = ground_truth.view(np.int32)[0]
    assert dim > 0
    ground_truth = ground_truth.reshape(-1, 1 + dim)
    if not all(ground_truth.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    ground_truth = ground_truth[:, 1:]
    if c_contiguous:
        ground_truth = ground_truth.copy()
    if reduce_k is not None:
        assert reduce_k > 0 and reduce_k <= ground_truth.shape[1]
        ground_truth = ground_truth[:, 0:reduce_k]
    ground_truth = ground_truth.astype("uint64", copy=False)

    return ground_truth


# See https://sisap-challenges.github.io/2024/datasets/ for more information
def _load_hdf5_sisap_data_or_queries(filename):
    with h5py.File(filename, "r") as f:
        return (f[list(f.keys())[0]][()]).astype(np.float32, copy=False)


def _load_hdf5_sisap_groundtruth(filename):
    """
    SISAP 2024 LAION2B Dataset specifics:
    - "The knns identifiers start indexing on 1." so subtracts 1 from each element
    - The groundtruth data (aka "gold standard") includes two matrices 'dists' and
    'knns'

    reduce_k - select a smaller k from the available ground_truth to evaluate on
    the correct amount of knn
    """
    with h5py.File(filename, "r") as f:
        ground_truth = f["knns"][()]
        return (ground_truth - np.ones(shape=ground_truth.shape)).astype(
            np.uint64, copy=False
        )


# ---------------------------- Running Eval ----------------------------


def iterate_search(
    dataset,
    data,
    queries,
    ground_truth,
    index,
    config,
    nq,
    nd,
    d,
    graph_builder_type,
    df_local,
    df_overall,
    filename_res,
    elapsed_time,
    params_str,
    print_output=False,
):
    """
    Runs the search (QPS and Recall) over the configured:
    k, higher_level_max_heap_size and max_heap_size values

    and intermediate saves the results while finally returning the appended overall dataframe
    """
    (
        n_vertices,
        n_edges,
        total_avg_out_degree,
        avg_out_degrees,
        min_max_out_degrees,
    ) = extract_index_stats(index.get_index_stats(reduced_stats=True))

    for k in config["k_values"]:
        # taking the top-k elements, thus reducing the ground_truth in each step
        # therefore, the values in config["k_values"] need to desc sorted
        ground_truth = get_reduced_groundtruth(ground_truth, reduce_k=k)
        for higher_level_max_heap_size in config["higher_level_max_heap_sizes"]:
            if "HSGF" in graph_builder_type or "HNSW" in graph_builder_type:
                index.set_higher_level_max_heap_size(higher_level_max_heap_size)
            for max_heap_size in get_max_heap_sizes(
                k, config["override_max_heap_size_values"]
            ):
                if "HSGF" in graph_builder_type or "HNSW" in graph_builder_type:
                    qps_recall_stats = None
                    if config["eval_powerset"]:
                        qps_recall_stats = (
                            index.evaluate_level_graphs_performance_powerset(
                                data,
                                queries,
                                ground_truth,
                                k,
                                max_heap_size,
                                nq,
                                graph_builder_type,
                                print_output,
                                config["point_zero_entry"],
                            )
                        )
                    else:
                        qps_recall_stats = index.evaluate_level_graphs_performance(
                            data,
                            queries,
                            ground_truth,
                            k,
                            max_heap_size,
                            nq,
                            graph_builder_type,
                            print_output,
                            False,  # only_top_level: change to true to evaluate only the complete graph
                            config["point_zero_entry"],
                        )
                else:
                    qps_recall_stats = [
                        index.evaluate_performance(
                            queries,
                            ground_truth,
                            k,
                            max_heap_size,
                            nq,
                            graph_builder_type,
                            print_output,
                        )
                    ]

                for stat in qps_recall_stats:
                    df_local.loc[len(df_local)] = [
                        stat.recall,
                        stat.qps,
                        graph_builder_type,
                        dataset,
                        elapsed_time,
                        nd,
                        nq,
                        d,
                        k,
                        max_heap_size,
                        stat.layer_count,
                        stat.query_time,
                        n_vertices,
                        n_edges,
                        total_avg_out_degree,
                        avg_out_degrees,
                        min_max_out_degrees,
                        params_str,
                        higher_level_max_heap_size,
                    ]
    df_local.to_csv(
        filename_res,
        sep=";",
        index=False,
        header=False,
        mode="a",
    )
    df_overall = pd.concat([df_overall, df_local], ignore_index=True)
    return df_overall


# --- Helpers


def init_recall_qps_dataframe():
    df_fields = {
        "recall": [],
        "qps": [],
        "builder_type": [],
        "dataset": [],
        "elapsed_time": [],
        "n_data": [],
        "n_queries": [],
        "dim": [],
        "k": [],
        "max_heap_size": [],
        "layer_count": [],
        "query_time": [],
        "n_vertices": [],
        "n_edges": [],
        "total_avg_out_degree": [],
        "avg_out_degrees": [],
        "min_max_out_degrees": [],
        "params": [],
        "higher_level_max_heap_size": [],
    }
    return pd.DataFrame(df_fields)


def extract_index_stats(index_stats):
    return (
        str(index_stats.n_vertices),
        str(index_stats.n_edges),
        str(index_stats.total_avg_out_degree),
        str(index_stats.avg_out_degrees),
        str(index_stats.min_max_out_degrees),
    )


# Quick access for defaults
def get_max_heap_sizes(k, override_max_heap_size_values=None):
    """For a given k returns a list of 'useful' max_heap_sizes to evaluate on"""
    if override_max_heap_size_values is not None:
        return override_max_heap_size_values
    else:
        match k:
            case 1:
                return [10, 30, 50, 100, 200]
            case 5:
                return [10, 30, 50, 100, 200]
            case 10:
                return [10, 30, 50, 100, 150, 200]
            case 20:
                return [20, 30, 40, 60, 100, 150, 200]
            case 30:
                return [30, 40, 60, 100, 150, 200]
            case 50:
                return [50, 75, 100, 150, 250]
            case 100:
                return [100, 150, 200, 300, 500, 700, 1000]
            case 300:
                return [300, 500, 750]
            case 500:
                return [500, 750, 1000]
            case 1000:
                return [1000, 1200]
            case _:
                return [k]


def write_df_overall(df_overall, filename_res, write_mode, config_str):
    if len(df_overall) > 0:
        if df_overall.isna().any().any():
            print("Warning dataframe contains NaN values.")
        df_overall = df_overall.convert_dtypes()
        df_overall.to_csv(
            filename_res,
            sep=";",
            index=False,
            header=True,
            mode=write_mode,
        )
        # append some config info as comment via '#'
        # for pd.read_csv() the '#' needs to be manually set
        with open(filename_res, "a") as f:
            f.write(f"# {config_str}")


# ------------ Misc ------------


def extract_params_from_result_df(df):
    params_objs = {}
    for bt in df["builder_type"].unique():
        df_bt = df.loc[df["builder_type"] == bt]
        params_objs[bt] = df_bt["params"].iloc[0]
    return params_objs
