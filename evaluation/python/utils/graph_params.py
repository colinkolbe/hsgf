"""HSGF Eval

- Roughly using the parameter as specified in https://arxiv.org/abs/2307.10479 for DEG, EFANNA, NSSG
and HNSW, and https://arxiv.org/abs/2310.20419 for RNN (the same parameters for all datasets given
the lack for more specific ones)
    - However, tuning many parameters over time to better fit recall, qps and construction time
    performance/constraints
- For the DEEP1M dataset the same parameters as for GLOVE-100 are used as it has
roughly the same LID
- For LAION parameters have been roughly tuned on the 300K but derived from DEEP1M

Contains a params getter which returns graph parameters for the named graph builders
specific to a dataset
    - Datasets
            - AUDIO, ENRON, DEEP1M, (GIST), GLOVE-100, SIFT,
                and SIFTSMALL (which is covered by the base case in the match block)
            - LAION2B in the 100M, 10M, 3M and 300K subsets

Contains a function for each graph to build an index with dataset specific graph-parameters

Besides into_hsgf_level_params()
the other functions are used almost exclusively for the base_graphs_eval to have a quick and
organized access and specification of the parameters depending on the dataset

---------------- Default Parameters for each graph builder ----------------
- See further down in this file which parameters for each graph are the "standard ones" to tune)
- Use, e.g., hsgf.PyDEGParams().params_as_str() to get a string for the default parameters


- DEG: edges_per_vertex=40|max_build_heap_size=60|max_build_frontier_size=None|extend_k=0|
extend_eps=0|improve_k=0|improve_eps=0|lid=High|additional_swap_tries=0|swap_tries=0|
use_range_search=false
- EFANNA: k=50|l=50|iter=10|s=10|r=100
- HNSW: higher_max_degree=50|lowest_max_degree=100|max_layers=10|n_parallel_burnin=0|
max_build_heap_size=50|max_build_frontier_size=None|level_norm_param_override=None|
insert_heuristic=true|insert_heuristic_extend=true|post_prune_heuristic=false|
insert_minibatch_size=100|n_rounds=1|finetune_rnn=false|finetune_sen=false
- NSSG: range=70|l=250|angle=60|n_try=10|derive_angle_from_dim:false|
Input-graph params:EFANNA:k=50|l=50|iter=10|s=10|r=100
- RNN: initial_degree=100|reduce_degree=50|n_outer_loops=4|n_inner_loops=15|
concurrent_batch_size=100
- HSGF: min_layers=1|max_layers=6|higher_max_degree=40|min_n_vertices_layer=50|
level_norm_param_override=None|create_artificial_hub_nodes=None
"""

from utils.util import Fvecs_Dataset, LAION_SISAP_Subset

import hsgf

# ----------- Graph as Index per Dataset -----------


def deg_index(data, dataset):
    (params, graph_type) = deg_params(dataset)
    index = hsgf.PyDEG(data, params)
    return (index, graph_type, params.params_as_str())


def efanna_index(data, dataset):
    (params, graph_type) = efanna_params(dataset)
    index = hsgf.PyEFANNA(data, params)
    return (index, graph_type, params.params_as_str())


def hnsw_index(data, dataset):
    (params, graph_type) = hnsw_params(dataset)
    index = hsgf.PyHNSW(data, params)
    return (index, graph_type, params.params_as_str())


def nssg_index(data, dataset):
    (params, graph_type) = nssg_params(dataset)
    index = hsgf.PyNSSG(data, params)
    return (index, graph_type, params.params_as_str())


def rnn_index(data, dataset):
    (params, graph_type) = rnn_params(dataset)
    index = hsgf.PyRNN(data, params)
    return (index, graph_type, params.params_as_str())


def rngg_index(data, dataset):
    params = hsgf.PyRNGGParams()
    index = hsgf.PyRNGG(data, params)
    return (index, "RNGG", params.params_as_str())


# ----------- Graph Params per Dataset -----------


def deg_params(dataset):
    """
    Note that `improve_k` is set to zero (by default) for all datasets
    The optional optimization phase of the DEG is therefore skipped
    for the sake of significant shorter construction times and limited improvements

    Unless use_range_search is set to true, the current DEG Builder does not use
    the range_search so the _eps parameter has no effect
    """
    match dataset:
        case Fvecs_Dataset.AUDIO:
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.Low,
                edges_per_vertex=30,
                max_build_heap_size=40,
            )
        case Fvecs_Dataset.ENRON:
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.High,
                edges_per_vertex=30,
                max_build_heap_size=60,
            )
        case Fvecs_Dataset.DEEP:
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.High,
                edges_per_vertex=50,
                max_build_heap_size=70,
            )
        case Fvecs_Dataset.GLOVE:
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.High,
                edges_per_vertex=30,
                max_build_heap_size=30,
            )
        # case Fvecs_Dataset.GIST:
        #     params = hsgf.PyDEGParams(
        #         lid=hsgf.PyLID.High,
        #         edges_per_vertex=50,
        #         max_build_heap_size=75,
        #     )
        case Fvecs_Dataset.SIFT:
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.Low,
                edges_per_vertex=30,
                max_build_heap_size=60,
            )
        case "NormalData":
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.High,
                edges_per_vertex=40,
                max_build_heap_size=60,
            )
        case LAION_SISAP_Subset.three_hun_k:
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.High,
                edges_per_vertex=35,
                max_build_heap_size=60,
            )
        case LAION_SISAP_Subset.three_m:
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.High,
                edges_per_vertex=35,
                max_build_heap_size=60,
            )
        case LAION_SISAP_Subset.ten_m:
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.High,
                edges_per_vertex=40,
                max_build_heap_size=70,
            )
        case LAION_SISAP_Subset.hundred_m:
            params = hsgf.PyDEGParams(
                lid=hsgf.PyLID.High,
                edges_per_vertex=40,
                max_build_heap_size=70,
            )
        case _:
            params = hsgf.PyDEGParams()
    return (params, "DEG")


def efanna_params(dataset):
    match dataset:
        case Fvecs_Dataset.AUDIO:
            params = hsgf.PyEFANNAParams(k=80, l=120, iter=8, s=30, r=100)
        case Fvecs_Dataset.ENRON:
            hsgf.PyEFANNAParams(k=130, l=150, iter=6, s=20, r=300)
        case Fvecs_Dataset.DEEP:
            params = hsgf.PyEFANNAParams(k=140, l=200, iter=4, s=15, r=100)
        case Fvecs_Dataset.GLOVE:
            params = hsgf.PyEFANNAParams(k=400, l=420, iter=12, s=15, r=200)
        # case Fvecs_Dataset.GIST:
        #     params = hsgf.PyEFANNAParams(k=100, l=150, iter=7, s=12, r=100)
        case Fvecs_Dataset.SIFT:
            params = hsgf.PyEFANNAParams(k=80, l=140, iter=12, s=10, r=100)
        case "NormalData":
            params = hsgf.PyEFANNAParams(k=150, l=150, iter=3, s=10, r=100)
        case LAION_SISAP_Subset.three_hun_k:
            params = hsgf.PyEFANNAParams(k=100, l=150, iter=3, s=10, r=100)
        case LAION_SISAP_Subset.three_m:
            params = hsgf.PyEFANNAParams(k=150, l=150, iter=3, s=10, r=100)
        case LAION_SISAP_Subset.ten_m:
            params = hsgf.PyEFANNAParams(k=100, l=150, iter=3, s=10, r=100)
        case LAION_SISAP_Subset.hundred_m:
            params = hsgf.PyEFANNAParams(k=100, l=180, iter=4, s=12, r=100)
        case _:
            params = hsgf.PyEFANNAParams()
    return (params, "EFANNA")


def hnsw_params(dataset):
    match dataset:
        case Fvecs_Dataset.AUDIO:
            params = hsgf.PyHNSWParams(
                higher_max_degree=20, lowest_max_degree=40, max_build_heap_size=60
            )
        case Fvecs_Dataset.ENRON:
            params = hsgf.PyHNSWParams(
                higher_max_degree=50, lowest_max_degree=80, max_build_heap_size=200
            )
        case Fvecs_Dataset.DEEP:
            params = hsgf.PyHNSWParams(
                higher_max_degree=40, lowest_max_degree=55, max_build_heap_size=200
            )
        case Fvecs_Dataset.GLOVE:
            params = hsgf.PyHNSWParams(
                higher_max_degree=60, lowest_max_degree=80, max_build_heap_size=300
            )
        # case Fvecs_Dataset.GIST:
        #     params = hsgf.PyHNSWParams(
        #         higher_max_degree=40, lowest_max_degree=75, max_build_heap_size=300
        #     )
        case Fvecs_Dataset.SIFT:
            params = hsgf.PyHNSWParams(
                higher_max_degree=35, lowest_max_degree=45, max_build_heap_size=200
            )
        case "NormalData":
            params = hsgf.PyHNSWParams(
                higher_max_degree=30, lowest_max_degree=50, max_build_heap_size=100
            )
        case LAION_SISAP_Subset.three_hun_k:
            params = hsgf.PyHNSWParams(
                higher_max_degree=30, lowest_max_degree=45, max_build_heap_size=100
            )
        case LAION_SISAP_Subset.three_m:
            params = hsgf.PyHNSWParams(
                higher_max_degree=30, lowest_max_degree=55, max_build_heap_size=100
            )
        case LAION_SISAP_Subset.ten_m:
            params = hsgf.PyHNSWParams(
                higher_max_degree=30, lowest_max_degree=60, max_build_heap_size=200
            )
        case LAION_SISAP_Subset.hundred_m:
            params = hsgf.PyHNSWParams(
                higher_max_degree=50, lowest_max_degree=80, max_build_heap_size=400
            )
        case _:
            params = hsgf.PyHNSWParams()
    return (params, "HNSW")


def nssg_params(dataset):
    match dataset:
        case Fvecs_Dataset.AUDIO:
            params = hsgf.PyNSSGParams(
                range=30,
                l=150,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    # hsgf.PyEFANNAParams(k=150, l=200, iter=5, s=25, r=200)
                    hsgf.PyEFANNAParams(k=80, l=120, iter=8, s=30, r=100)
                ),
            )
        case Fvecs_Dataset.ENRON:
            params = hsgf.PyNSSGParams(
                range=50,
                l=300,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    hsgf.PyEFANNAParams(k=130, l=150, iter=6, s=20, r=300)
                    # hsgf.PyEFANNAParams(k=140, l=180, iter=7, s=20, r=200)
                ),
            )
        case Fvecs_Dataset.DEEP:
            params = hsgf.PyNSSGParams(
                range=60,
                l=350,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    # hsgf.PyEFANNAParams(k=240, l=260, iter=7, s=15, r=100)
                    hsgf.PyEFANNAParams(k=140, l=200, iter=4, s=15, r=100)
                ),
            )
        # case Fvecs_Dataset.GIST:
        #     params = hsgf.PyNSSGParams(
        #         range=60,
        #         l=500,
        #         angle=60.0,
        #         input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
        #             hsgf.PyEFANNAParams(k=100, l=150, iter=3, s=10, r=100)
        #         ),
        #     )
        case Fvecs_Dataset.GLOVE:
            params = hsgf.PyNSSGParams(
                range=50,
                l=500,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    hsgf.PyEFANNAParams(k=400, l=420, iter=12, s=15, r=200)
                ),
            )
        case Fvecs_Dataset.SIFT:
            params = hsgf.PyNSSGParams(
                range=50,
                l=100,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    # hsgf.PyEFANNAParams(k=200, l=200, iter=12, s=10, r=100)
                    hsgf.PyEFANNAParams(k=80, l=140, iter=12, s=10, r=100)
                ),
            )
        case "NormalData":
            params = hsgf.PyNSSGParams(
                range=50,
                l=250,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    hsgf.PyEFANNAParams(k=150, l=150, iter=3, s=10, r=100)
                ),
            )
        case LAION_SISAP_Subset.three_hun_k:
            params = hsgf.PyNSSGParams(
                range=50,
                l=250,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    hsgf.PyEFANNAParams(k=150, l=150, iter=3, s=10, r=100)
                ),
            )
        case LAION_SISAP_Subset.three_m:
            params = hsgf.PyNSSGParams(
                range=60,
                l=400,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    hsgf.PyEFANNAParams(k=150, l=150, iter=3, s=10, r=100)
                ),
            )
        case LAION_SISAP_Subset.ten_m:
            params = hsgf.PyNSSGParams(
                range=60,
                l=400,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    hsgf.PyEFANNAParams(k=150, l=150, iter=3, s=10, r=100)
                ),
            )
        case LAION_SISAP_Subset.hundred_m:
            params = hsgf.PyNSSGParams(
                range=50,
                l=400,
                angle=60.0,
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                    hsgf.PyEFANNAParams(k=180, l=180, iter=4, s=12, r=100)
                ),
            )
        case _:
            params = hsgf.PyNSSGParams(
                input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(None)
            )
    return (params, "NSSG")


def rnn_params(dataset):
    match dataset:
        case Fvecs_Dataset.AUDIO:
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=30, n_outer_loops=4, n_inner_loops=15
            )
        case Fvecs_Dataset.ENRON:
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=30, n_outer_loops=4, n_inner_loops=15
            )
        case Fvecs_Dataset.DEEP:
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=40, n_outer_loops=4, n_inner_loops=15
            )
        # case Fvecs_Dataset.GIST:
        #     params = hsgf.PyRNNParams(
        #         initial_degree=96, reduce_degree=45, n_outer_loops=4, n_inner_loops=15
        #     )
        case Fvecs_Dataset.GLOVE:
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=50, n_outer_loops=4, n_inner_loops=15
            )
        case Fvecs_Dataset.SIFT:
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=40, n_outer_loops=4, n_inner_loops=15
            )
        case "NormalData":
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=30, n_outer_loops=4, n_inner_loops=15
            )
        case LAION_SISAP_Subset.three_hun_k:
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=30, n_outer_loops=4, n_inner_loops=15
            )
        case LAION_SISAP_Subset.three_m:
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=40, n_outer_loops=4, n_inner_loops=15
            )
        case LAION_SISAP_Subset.ten_m:
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=40, n_outer_loops=4, n_inner_loops=15
            )
        case LAION_SISAP_Subset.hundred_m:
            params = hsgf.PyRNNParams(
                initial_degree=96, reduce_degree=40, n_outer_loops=4, n_inner_loops=15
            )
        case _:
            params = hsgf.PyRNNParams()
    return (params, "RNN")


# ------ Params Util ------


# wrapper
def into_hsgf_level_params(params, graph_type):
    match graph_type:
        case "DEG":
            params = hsgf.PyHSGFLevelGraphParams.DEG(params)
        case "EFANNA":
            params = hsgf.PyHSGFLevelGraphParams.EFANNA(params)
        case "NSSG":
            params = hsgf.PyHSGFLevelGraphParams.NSSG(params)
        case "KNN":
            params = hsgf.PyHSGFLevelGraphParams.KNN(params)
        case "RNN":
            params = hsgf.PyHSGFLevelGraphParams.RNN(params)
    return params
