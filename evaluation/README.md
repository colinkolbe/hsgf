# HSGF - Evaluation

All experiments for this master's thesis were done via the Python binding of hsgf.

## TL;DR 
- Rerunning experiments
    - (Install the hsgf module by compiling via maturin develop)
    - Have the datasets at the right location available
        - Under /datasets/
        - Otherwise change BASE_PATH_DATASETS and SISAP_DATASET_PATH in /eval/python/utils/eval_configs
    - Change the respective configs in `python/utils/eval_configs.py`
    - Start an experiment by running `python3 hsgf_eval.py` or hnsw_eval.py or base_graphs_eval.py
    - Once finished, copy filepath to result CSV from log into /utils/results_dict.py 
    - Add identifier in view_results.ipynb and view new results

## Quick Overview

`Evaluation is done by:`
- Main metrics: Query-per-Second and Recall
    - Further, information (the used configurations and parameter settings) for each search pass is reported in result files (.csv)
- By evaluating the following available graphs on standard datasets comparing the recall, QPS, and build time to already established results from the literature, and, therefore, as baselines (DEG, EFANNA, HNSW, NSSG, RNN)
    - All graphs are built in parallel mode (in contrast to the "usual" single thread setting in the literature)
- HNSW is evaluated and analyzed by starting a search from each layer of the hierarchy graph separately, which allows us to observe the hierarchy effect on the graph's search performance 
- HSGF is evaluated the same way as HNSW
    - Furthermore, we evaluate the HSGF's different parameter configurations, primarily the different level builder and the different level subset selection heuristics 
        - Consequently, the parameter configuration of the individual levels is another parameter 
        
- All experiments were run on the provided hardware from the university/Data Mining Group 
    - Intel(R) Xeon(R) W-2155 CPU @ 3.30GHz and 64GB of RAM
        - Semi-Exclusive usage, but the final experiments were (basically) run with exclusive usage of the machine
    - VS Code Remote Explorer was used to work on the machine
    - The experiments were started inside a `tmux` session to keep them alive after logging out from the machine

## Contents

- `/python`
    - `hsgf_eval`, `hnsw_eval`, and `base_graphs_eval` are the main evaluation scripts for running this work's experiments 
        - For each scripts exists a *CONFIG* and *PARAMS_OBJECTS_TO_ITERATE* object (although the latter does not exist for base_graphs_eval) in `/utils/eval_configs`
        - An experiment's run is working the following way:
            - For each configured dataset, it loads the data, queries, and groundtruth
            - It then iterates over the specified PARAMS_OBJECTS_TO_ITERATE, each time building an index, which is then used for running the queries via iterate_search()
            - In iterate_search(), the index is queried for all combinations of k, max_heap_size (mhs) and higher_level_max_heap_size (hlmhs) of which the result is appended to the current result CSV as an intermediate save, lastly appending the CONFIG object to the dataframe as a comment
            - The parameters, configurations for each graph builder are stored in a column of the result CSV
    - `/utils/results.dict.py` contains a mapping of result CSV filepaths to more helpful identifiers
    - `/view_results.ipynb` - A Jupyter notebook used for viewing the results and saving them to .pdf and .html (via pandas and plotly.express)
    - `/utils/graph_params` contains the parameter configuration as a quick access matched on a dataset (used almost exclusively for base_graphs_eval)
        - In the file description, you can find the default parameters for each graph, and you can see which are the "standard" parameters to change
        - Different parameter configurations can, of course, be passed to the HSGF builders manually as well (besides in base_graphs_eval, which is primarily meant to reproduce existing results)
    - `/utils/misc` - we send ourselves an email when an experiment has finished by making a simple post request to a private web server
    - For more documentation, see the individual files
    - Note: We currently do not use the Python module argparse for specifying commandline arguments but instead change the configuration objects via /utils/eval_configs
- `/results`
    - `/results/python/*` Contains the results of all experiments
    - A log file is written to for each experiment (not committed in git)
        - Mostly time-stamps, the script, and the CONFIG
        - Filepath of the result file can be easily copied from there to a new entry in RESULTS_TO_FILENAME in /utils/results_dict
    - A result .csv file uses ";" as separator
        - Has always the same header (see file or below for details)
        - The parameters for each graph builder is part of each row (also true for the complete parameter list of HSGF)
        - The CONFIG is attached at the end of each .csv file as a comment 
            - Via '#'; needs to be taken care of when reading in the file: `df = pd.read_csv(filename, sep=";", comment="#")`

## About the configuration objects
- `CONFIG`
    - id: str - identifier
    - iters: uint - how many times an index should be build and evaluated
    - different_k_values: bool - if just the default/max available k of the ground_truth should be used, or k_values
    - k_values: None or [uint],  # needs to be desc sorted
    - override_max_heap_size_values: None or [uint] - the max_heap_sizes to be used during the search, if None, a default list based on the current _k_ value will be used 
    - higher_level_max_heap_sizes: [uint] - a search parameter for hierarchy graphs (HNSW, HSGF), for each value, a separate search pass will be run
    - point_zero_entry: bool - If the first point of the dataset should be used as the entry point on all level-stacked subgraphs
    - datasets: [LAION_SISAP_Subset or Fvecs_Dataset] - datasets to be evaluated
    - indices_to_eval: [function to build the different graphs] - the functions are defined in `utils/util` (only used in base_graphs_eval)

- `PARAMS_OBJECTS_TO_ITERATE (HNSW)`
    - Uses lambdas to pass in the dataset identifier
    - A list of one (or more) hsgf.PyHNSWParams() objects, which are to be used on this dataset
        - hnsw_params(CONFIG["dataset"])[0] can be used as a shortcut to the "defaults" for the current dataset

- `PARAMS_OBJECTS_TO_ITERATE (HSGF)`
    - Uses lambdas to pass in the dataset identifier
    - Specifies the level_builder and level_selector objects for the HSGF graph
    - A list of one (or more) tuples of the (optional_config_override_dict, hsgf.PyHSGFParams(), [(hsgf.PyHSGFLevelGraphParams, hsgf.PySubsetSelector)], level_subset_size=[uint], hsgf_graph_id=str)
        - Note (again) that the list for the level_builder and level_selector can contain one or more tuples, but is independent of the final level-count of the hierarchy graphs, as the last element in the list will be used if the last index is exhausted
    - An example can be found in `utils/eval_configs`

## Header of the Result CSV
- recall, qps, builder_type, dataset, elapsed_time, n_data, n_queries, dim, k, max_heap_size, layer_count, query_time, n_vertices, n_edges, total_avg_out_degree, avg_out_degrees, min_max_out_degrees, params, higher_level_max_heap_size
    - Note the actual header has no spaces and is separated with ";" and has comments via "#" 
    - elapsed_time is (only!) the construction time for that row's index/graph
    - A lot of entries per row are duplicates of previous rows due to the different search configurations (of k, mhs, and hlmhs), plus the layer_count (= the layer used for starting the search on a hierarchy graph, which defaults to 1)
    - Older results might use "data_set" instead of "dataset" as column-id