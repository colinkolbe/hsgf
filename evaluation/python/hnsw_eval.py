"""Eval

Eval HNSW over all its levels

Technically HNSW could as well be evaluated via base_graphs_eval but this seemed cleaner
"""

import logging
import time

from utils.eval_configs import HNSW_CONFIG, HNSW_PARAMS_OBJECTS_TO_ITERATE
from utils.misc import post_to_email
from utils.util import (
    RESULT_PATH,
    get_data,
    init_recall_qps_dataframe,
    iterate_search,
    write_df_overall,
)

import hsgf

CONFIG = HNSW_CONFIG
PARAMS_OBJECTS_TO_ITERATE = lambda _curr_config: HNSW_PARAMS_OBJECTS_TO_ITERATE(
    _curr_config
)


def eval_all_level_hnsw():
    # Note: this can result in the same filename for run-times per graph of < 1s
    filename_res = f"{RESULT_PATH}/hnsw_eval/hnsw_{str(int(time.time()))}.csv"
    write_mode, overall_time = "w", time.time()
    df_overall = init_recall_qps_dataframe()

    try:
        for dataset in CONFIG["datasets"]:
            (data, queries, ground_truth, nd, nq, d, k) = get_data(dataset)
            if not CONFIG["different_k_values"]:
                CONFIG["k_values"] = [k]
            assert k >= CONFIG["k_values"][0]

            # add additional information to CONFIG if needed
            # just be careful not to override other fields
            # this line here is mostly for logging the config in the results file
            CONFIG["curr_dataset"] = dataset

            for params in PARAMS_OBJECTS_TO_ITERATE(CONFIG):
                for _iter in range(0, CONFIG["iters"]):
                    df_local = init_recall_qps_dataframe()
                    # Construct the HNSW index
                    start_time = time.time()
                    index = hsgf.PyHNSW(
                        data,
                        params,
                        max_frontier_size=None,
                        higher_level_max_heap_size=CONFIG[
                            "higher_level_max_heap_sizes"
                        ][0],
                    )
                    elapsed_time = time.time() - start_time
                    params_str = params.params_as_str()

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
                        CONFIG["id"],
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
        overall_time = time.time() - overall_time
        CONFIG["overall_time"] = overall_time
        config_str = f"Config: {str(CONFIG)}"

        write_df_overall(df_overall, filename_res, write_mode, config_str)

        print(f"Overall time: {(overall_time):.3f}")
        msg = f"hnsw_eval Execution time: {overall_time}; Filename: {filename_res}; {config_str}"
        logging.info(msg)
        post_to_email(msg)


def main():
    logging.info(f"Execution of: hnsw_eval()")
    eval_all_level_hnsw()


if __name__ == "__main__":
    main()
