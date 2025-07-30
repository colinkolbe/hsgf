"""Storing the configs for the different evaluation scripts in one place
See eval/README for more information
"""

from utils.graph_params import (
    deg_index,
    deg_params,
    efanna_index,
    efanna_params,
    hnsw_index,
    hnsw_params,
    into_hsgf_level_params,
    nssg_index,
    nssg_params,
    rngg_index,
    rnn_index,
    rnn_params,
)
from utils.hsgf_util import get_level_subset_sizes, hsgf_selector
from utils.util import Fvecs_Dataset, LAION_SISAP_Subset

import hsgf

# -------------------- HSGF --------------------
HSGF_CONFIG = {
    "id": "HSGF",
    "iters": 1,
    "different_k_values": True,
    "k_values": [10],  # needs to be desc sorted
    "override_max_heap_size_values": None,
    "higher_level_max_heap_sizes": [1],  # [1, 3, 10, 30],
    # experimental; currently not yet working; also expect significantly
    # longer runtime because of number of combinations
    "point_zero_entry": False,
    "eval_powerset": False,  # not implemented yet
    "datasets": [  # Note: for hsgf_eval only one dataset must be specified
        #            otherwise use config_override
        # Fvecs_Dataset.AUDIO,
        # Fvecs_Dataset.ENRON,
        # Fvecs_Dataset.DEEP,
        # Fvecs_Dataset.GLOVE,
        # Fvecs_Dataset.SIFT,
        Fvecs_Dataset.SIFTSMALL,
        # LAION_SISAP_Subset.three_hun_k,
        # LAION_SISAP_Subset.three_m,
        # LAION_SISAP_Subset.ten_m
        # "NormalData",
    ],
}

HSGF_PARAMS_CONFIG = [
    (
        {},
        lambda _config: (
            hsgf.PyHSGFParams(
                min_layers=5,
                max_layers=5,
                min_n_vertices_layer=30,
            ),
            [
                (
                    into_hsgf_level_params(
                        hsgf.PyDEGParams(
                            edges_per_vertex=30,
                            max_build_heap_size=60,
                        ),
                        "DEG",
                    ),
                    hsgf_selector(flood=1),
                ),
            ],
            [None],
            "DEG_TBD",
        ),
    ),
]

# Example/Placeholder to quickly copy the respective level graph builder config
HSGF_EXAMPLE = [
    (
        # Override keys in default HSGF_CONFIG
        {},
        #  Example: {"datasets": [Fvecs_Dataset.AUDIO], "different_k_values": True, "k_values": [20, 10, 1]},
        lambda _config: (
            # HSGF params
            hsgf.PyHSGFParams(
                min_layers=5,
                max_layers=5,
                # higher_max_degree=40, # (only) used for the level subset size calculation
                # specifies the threshold of points for which a level is still build for
                # might result in less level that specified by min_- max_layers
                min_n_vertices_layer=30,
            ),
            # Levels: List of tuples of (<Level-Builder>, <Level-Selector>)
            [
                (
                    into_hsgf_level_params(
                        hsgf.PyDEGParams(
                            edges_per_vertex=50,
                            max_build_heap_size=80,
                        ),
                        "DEG",
                    ),
                    hsgf_selector(random=True),
                ),
                (
                    into_hsgf_level_params(
                        hsgf.PyRNNParams(
                            initial_degree=120,
                            reduce_degree=70,
                            n_outer_loops=4,
                            n_inner_loops=15,
                        ),
                        "RNN",
                    ),
                    hsgf_selector(hubs=25),
                ),
                (
                    into_hsgf_level_params(
                        hsgf.PyNSSGParams(
                            range=50,
                            l=200,
                            angle=60.0,
                            input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                                hsgf.PyEFANNAParams(k=150, l=150, iter=3, s=10, r=100)
                            ),
                        ),
                        "NSSG",
                    ),
                    hsgf_selector(flood=1),
                ),
                (
                    into_hsgf_level_params(
                        hsgf.PyEFANNAParams(k=150, l=150, iter=3, s=10, r=100),
                        "EFANNA",
                    ),
                    hsgf_selector(floodRepeat=1),
                ),
                (
                    into_hsgf_level_params(
                        hsgf.PyBruteforceKNNParams(
                            degree=int(_config["k_values"][0] * 0.5)
                        ),
                        "KNN",
                    ),
                    hsgf_selector(flood=1),
                ),
            ],
            # Level-Subset-Sizes argument
            [None],
            # [None] : let the subset selection determine the subset sizes (use this for flooding)
            # [uint/None, unit/None,..] : specify level sizes; Note the sizes are only for the higher levels, so that the bottom level does not be specified
            # A combination, e.g. [50_000, None, 40, None], is also possible
            # get_level_subset_sizes(_config["datasets"][0]) : use HNSW derived subset sizes
            # None : let HSGF itself pre-calculate subset sizes
            "HSGF_EXAMPLE_GRAPH",  # graph_id, e.g. HSGF_DEG_Flood_2_Rand_high_params
        ),
    )
]

# -------------------- BASE GRAPHS --------------------
BGE_CONFIG = {
    "id": "base_graphs",
    "iters": 1,
    "different_k_values": False,
    "k_values": [10],  # needs to be desc sorted
    "override_max_heap_size_values": None,
    "higher_level_max_heap_sizes": [1],
    "point_zero_entry": True,
    "eval_powerset": False,  # not implemented yet
    "datasets": [
        # Fvecs_Dataset.AUDIO,
        # Fvecs_Dataset.ENRON,
        # Fvecs_Dataset.DEEP,
        # Fvecs_Dataset.GLOVE,
        # Fvecs_Dataset.SIFT,
        Fvecs_Dataset.SIFTSMALL,
        # LAION_SISAP_Subset.three_hun_k,
        # LAION_SISAP_Subset.three_m,
        # LAION_SISAP_Subset.ten_m
        # "NormalData",
    ],
    # takes the parameters specified for each dataset and graph in utils/graph_params
    "graphs_to_eval": [
        deg_index,
        # rnn_index,
        # hnsw_index,
        # efanna_index,
        # nssg_index,
    ],
}

# -------------------- HNSW --------------------
HNSW_CONFIG = {
    "id": "HNSW",
    "iters": 1,
    "different_k_values": True,
    "k_values": [10],  # needs to be desc sorted
    "override_max_heap_size_values": None,
    "higher_level_max_heap_sizes": [1],
    "point_zero_entry": True,
    "eval_powerset": False,  # not implemented yet
    "datasets": [
        # Fvecs_Dataset.AUDIO,
        # Fvecs_Dataset.ENRON,
        # Fvecs_Dataset.DEEP,
        # Fvecs_Dataset.GLOVE,
        # Fvecs_Dataset.SIFT,
        Fvecs_Dataset.SIFTSMALL,
        # LAION_SISAP_Subset.three_hun_k,
        # LAION_SISAP_Subset.three_m,
        # LAION_SISAP_Subset.ten_m
        # "NormalData",
    ],
}

# Note: this is not the same kind of params_config object as for hsgf_eval
HNSW_PARAMS_OBJECTS_TO_ITERATE = lambda _config: [
    hnsw_params(_config["curr_dataset"])[0],
    # hsgf.PyHNSWParams(
    #     higher_max_degree=40, lowest_max_degree=55, max_build_heap_size=200
    # ),
]
