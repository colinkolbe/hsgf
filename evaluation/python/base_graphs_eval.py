"""Eval

Evaluating DEG, EFANNA and NSSG as 'new' implementations while also comparing to HNSW and
RNN (the latter two are 'ported' from graphindexbaselines see hsgf::py::py_hsgf)
on different datasets

Note
For the base graphs evaluation we always query with k=100 for the LAION subsets.
On that note, the audio dataset has a full groundtruth with k=100 but only k=20 seems to be used
in the literature - maybe because the dataset is so small and QPS is already quite high regardless.
"""

import logging
import time

from utils.eval_configs import BGE_CONFIG
from utils.misc import post_to_email
from utils.util import (
    RESULT_PATH,
    get_data,
    init_recall_qps_dataframe,
    is_laion,
    iterate_search,
    write_df_overall,
)

CONFIG = BGE_CONFIG


def eval_graphs_on_standard_datasets():
    # Note: this can result in the same filename for run-times per graph of < 1s
    filename_res = f"{RESULT_PATH}/base_graphs_eval/bge_{str(int(time.time()))}.csv"
    write_mode, overall_time = "w", time.time()
    df_overall = init_recall_qps_dataframe()

    try:
        for dataset in CONFIG["datasets"]:
            (data, queries, ground_truth, nd, nq, d, k) = get_data(dataset)
            if not CONFIG["different_k_values"]:
                if is_laion(dataset):
                    k = 100
                CONFIG["k_values"] = [k]
            assert k >= CONFIG["k_values"][0]

            for index_fn in CONFIG["graphs_to_eval"]:
                print("Graph: ", str(index_fn).split(" ")[1])
                for _iter in range(0, CONFIG["iters"]):
                    df_local = init_recall_qps_dataframe()
                    # Construct the current index
                    start_time = time.time()
                    (index, graph_builder_type, params_str) = index_fn(data, dataset)
                    elapsed_time = time.time() - start_time

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
                        graph_builder_type,
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
        msg = f"""base_graphs_eval Execution time: {overall_time}; Filename: {filename_res};\
            {config_str}"""
        logging.info(msg)
        post_to_email(msg)


def main():
    logging.info(f"Execution of: base_graphs_eval()")
    eval_graphs_on_standard_datasets()


if __name__ == "__main__":
    main()
