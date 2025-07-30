"""HSGF Eval

Evaluate most graph and parameter configurations

Open Issues
    # Currently not using command line arguments, as there is not need for it
    # still storing this here:
    # import argparse
    # parser = argparse.ArgumentParser(prog="HSGF Eval")
    # parser.add_argument("-ta", "--test_arg", action="store_true", help="Description")
    # test_arg = pargs.test_arg
"""

import logging
import time

from utils.eval_configs import HSGF_CONFIG, HSGF_PARAMS_CONFIG
from utils.misc import post_to_email
from utils.util import (
    RESULT_PATH,
    get_data,
    init_recall_qps_dataframe,
    iterate_search,
    write_df_overall,
)

import hsgf

CONFIG = HSGF_CONFIG
PARAMS_OBJECTS_TO_ITERATE = HSGF_PARAMS_CONFIG


def eval_hsgf_all_levels():
    overall_time = time.time()
    prev_dataset = CONFIG["datasets"][0]
    (data, queries, ground_truth, nd, nq, d, k) = get_data(prev_dataset)
    for local_config, params_lambda_obj in PARAMS_OBJECTS_TO_ITERATE:
        try:
            # Note: this can result in the same filename for run-times per graph of < 1s
            filename_res = f"{RESULT_PATH}/hsgf_eval/hsgf_{str(int(time.time()))}.csv"
            write_mode, iter_time = "w", time.time()
            df_overall = init_recall_qps_dataframe()

            # Copy and override default config
            curr_config = CONFIG.copy()
            for key in local_config.keys():
                curr_config[key] = local_config[key]

            dataset = curr_config["datasets"][0]
            if dataset != prev_dataset:
                # Load dataset only if different (aka avoid reloading)
                (data, queries, ground_truth, nd, nq, d, k) = get_data(dataset)
                prev_dataset = dataset

            if not curr_config["different_k_values"]:
                curr_config["k_values"] = [k]
            assert k >= curr_config["k_values"][0]

            # Add additional information to CONFIG if needed, e.g.:
            # curr_config["k"] = k
            (
                hsgf_params,
                level_builder,
                level_subset_sizes,
                curr_graph_name,
            ) = params_lambda_obj(curr_config)

            # _knn_prep()

            print("Graph-ID: ", curr_graph_name)
            curr_config["graph_id"] = curr_graph_name
            for _iter in range(0, curr_config["iters"]):
                df_local = init_recall_qps_dataframe()
                # Construct the HSGF index
                start_time = time.time()
                index = hsgf.PyHSGF(
                    data,
                    hsgf_params,
                    level_builder,
                    level_subset_sizes,
                    max_frontier_size=None,
                    higher_level_max_heap_size=curr_config[
                        "higher_level_max_heap_sizes"
                    ][0],
                )
                elapsed_time = time.time() - start_time
                params_str = (
                    hsgf_params.params_as_str() + "|" + index.get_level_builder_as_str()
                )

                CONFIG["graph_id"] = curr_graph_name

                # Search the graph
                df_overall = iterate_search(
                    dataset,
                    data,
                    queries,
                    ground_truth,
                    index,
                    curr_config,
                    nq,
                    nd,
                    d,
                    f"{curr_config['id']}_{curr_graph_name}",
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
            curr_config["iter_time"] = iter_time
            config_str = f"Config: {str(curr_config)}"

            write_df_overall(df_overall, filename_res, write_mode, config_str)

            print(f"Index time: {(iter_time):.3f}")
            msg = f"hsgf_eval Execution time: {iter_time}; Filename: {filename_res}; {config_str}"
            logging.info(msg)

    overall_time = time.time() - overall_time
    logging.info(f"hsgf_eval Overall Execution time: {overall_time}")
    # post_to_email(msg)


def main():
    logging.info(f"Execution of: hsgf_eval")
    eval_hsgf_all_levels()


if __name__ == "__main__":
    main()
