"""Combine a bottom graph with the higher layers of a HNSW graph

--- Currently unused ---
Beyond few tests.

Does not use CONFIGs via eval_configs
"""

import logging
import time

from utils.graph_params import into_hsgf_level_params
from utils.hsgf_util import hsgf_selector
from utils.misc import post_to_email
from utils.util import (
    RESULT_PATH,
    Fvecs_Dataset,
    LAION_SISAP_Subset,
    extract_index_stats,
    get_data,
    init_recall_qps_dataframe,
    iterate_search,
    write_df_overall,
)

import hsgf

CONFIG = {
    "id": "HNSW_Merged",
    "iters": 1,
    "different_k_values": True,
    "k_values": [100, 10, 1],  # needs to be desc sorted
    "override_max_heap_size_values": None,
    "higher_level_max_heap_sizes": [1],
    "point_zero_entry": True,
    "datasets": [
        # Fvecs_Dataset.AUDIO,
        # Fvecs_Dataset.ENRON,
        # Fvecs_Dataset.DEEP,
        # Fvecs_Dataset.GLOVE,
        # Fvecs_Dataset.SIFT,
        # Fvecs_Dataset.SIFTSMALL,
        # LAION_SISAP_Subset.three_hun_k,
        LAION_SISAP_Subset.three_m,
        # LAION_SISAP_Subset.ten_m
    ],
}

PARAMS_OBJECTS_TO_ITERATE = [
    (
        into_hsgf_level_params(
            hsgf.PyDEGParams(
                edges_per_vertex=60,
                max_build_heap_size=100,
            ),
            "DEG",
        ),
        hsgf.PyHNSWParams(
            higher_max_degree=55, lowest_max_degree=55, max_build_heap_size=100
        ),
        hsgf_selector(random=True),
        100_000,
        "DEG_Rand_HNSW",
    )
]


def eval_merged_hnsw():
    overall_time = time.time()
    for (
        params_bottom,
        params_hnsw,
        selector,
        level_subset_size,
        graph_id,
    ) in PARAMS_OBJECTS_TO_ITERATE:
        try:
            # Note: this can result in the same filename for run-times per graph of < 1s
            filename_res = f"{RESULT_PATH}/hsgf_eval/_special/hnsw_merged_{str(int(time.time()))}.csv"
            write_mode, iter_time = "w", time.time()
            df_overall = init_recall_qps_dataframe()

            dataset = CONFIG["datasets"][0]
            (data, queries, ground_truth, nd, nq, d, k) = get_data(dataset)
            if not CONFIG["different_k_values"]:
                CONFIG["k_values"] = [k]
            assert k >= CONFIG["k_values"][0]

            for _iter in range(0, CONFIG["iters"]):
                df_local = init_recall_qps_dataframe()
                # Construct the merged index
                start_time = time.time()
                index = hsgf.merge_hnsw_with_new_bottom_level(
                    data, params_bottom, params_hnsw, selector, 1, level_subset_size
                )
                elapsed_time = time.time() - start_time

                CONFIG["graph_id"] = graph_id

                params_str = index.get_level_builder_as_str()
                # Search the graph
                df_overall = iterate_search(
                    dataset,
                    data,
                    queries,
                    ground_truth,
                    index,
                    CONFIG,
                    nq,
                    nd,
                    d,
                    graph_id,
                    df_local,
                    df_overall,
                    filename_res,
                    elapsed_time,
                    params_str,
                )
        except Exception as e:
            logging.exception(e)
            print(e)
            write_mode = "a"
        finally:
            iter_time = time.time() - iter_time
            CONFIG["iter_time"] = iter_time
            config_str = f"Config: {str(CONFIG)}"

            write_df_overall(df_overall, filename_res, write_mode, config_str)

            print(f"Index time: {(iter_time):.3f}")
            msg = f"hnsw_merged_eval Execution time: {iter_time}; Filename: {filename_res}; \
                {config_str}"
            logging.info(msg)

    overall_time = time.time() - overall_time
    logging.info(f"hnsw_merged_eval Overall Execution time: {overall_time}")
    post_to_email(msg)


def main():
    logging.info(f"Execution of: eval_merged_graph()")
    eval_merged_hnsw()


if __name__ == "__main__":
    main()
