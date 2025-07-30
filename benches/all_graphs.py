"""
Running basic integration tests via the Python module of hsgf

This is therefore more of a Python bindings integration test than a benchmark.

Script is adapted from GraphIndexBaselines.
"""

import time

import hsgf
import numpy as np

# BASE_PATH_DATA_SETS = "../evaluation/datasets"
PRINT_PARAMS = False


def easy_bench(f):
    start_time = time.time()
    f()
    elapsed_time = time.time() - start_time
    print(f"{f.__name__}: {elapsed_time:.3f}s")


def init_data(nd=100_000):
    """
    if nd=0, knn_query_batch() will fail but not the graph construction itself
    """
    nq, d, k = 10_000, 20, 100
    data = np.random.normal(0, 1, (nd, d)).astype(np.float32)
    queries = np.random.normal(0, 1, (nq, d)).astype(np.float32)
    # hsgf also includes a function to supply to normal data:
    # indexed_queries = True
    # (data, queries, ground_truth, dists) = hsgf.load_normal_data(nd, nq, d, k, indexed_queries)
    return data, queries, k


def deg_construction():
    data, queries, k = init_data()
    params = hsgf.PyDEGParams()
    index = hsgf.PyDEG(data, params)
    if PRINT_PARAMS:
        print(params.params_as_str())
    _ = index.knn_query_batch(queries, k)


def nssg_construction():
    data, queries, k = init_data()
    params = hsgf.PyNSSGParams(
        input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(hsgf.PyEFANNAParams())
    )
    if PRINT_PARAMS:
        print(params.params_as_str())
    index = hsgf.PyNSSG(data, params)
    _ = index.knn_query_batch(queries, k)


def efanna_construction():
    data, queries, k = init_data()
    params = hsgf.PyEFANNAParams(k=50, l=50)
    if PRINT_PARAMS:
        print(params.params_as_str())
    index = hsgf.PyEFANNA(data, params)
    _ = index.knn_query_batch(queries, k)


def rngg_construction():
    data, queries, k = init_data()
    params = hsgf.PyRNGGParams()
    if PRINT_PARAMS:
        print(params.params_as_str())
    index = hsgf.PyRNGG(data, params)
    _ = index.knn_query_batch(queries, k)


# constructing a arbitrary hsgf_graph with all available level_builders (see hsgf.PyHSGFLevelGraphParams)
def hsgf_construction():
    data, queries, k = init_data()

    # using RNN on two lowest levels for faster construction time
    level_input_graph_params = hsgf.PyDEGParams()
    level_input_graph = hsgf.PyDEG(data, level_input_graph_params).get_graph()

    flooding_range = 1

    level_builder = [
        (
            hsgf.PyHSGFLevelGraphParams.ExistingGraph(
                (level_input_graph, level_input_graph_params.params_as_str())
            ),
            hsgf.PySubsetSelector.Flooding(flooding_range),
        ),
        (
            hsgf.PyHSGFLevelGraphParams.RNN(hsgf.PyRNNParams()),
            hsgf.PySubsetSelector.Random(),
        ),
        (
            hsgf.PyHSGFLevelGraphParams.NSSG(
                hsgf.PyNSSGParams(
                    input_graph=hsgf.PyNSSGInputGraphDataForHSGF.EFANNA(
                        hsgf.PyEFANNAParams()
                    )
                )
            ),
            hsgf.PySubsetSelector.Random(),
        ),
        (
            hsgf.PyHSGFLevelGraphParams.DEG(hsgf.PyDEGParams()),
            hsgf.PySubsetSelector.Random(),
        ),
        (
            hsgf.PyHSGFLevelGraphParams.EFANNA(hsgf.PyEFANNAParams(k=50, l=50)),
            hsgf.PySubsetSelector.Random(),
        ),
        (
            hsgf.PyHSGFLevelGraphParams.RNGG(hsgf.PyRNGGParams()),
            hsgf.PySubsetSelector.Random(),
        ),
        (
            hsgf.PyHSGFLevelGraphParams.KNN(hsgf.PyBruteforceKNNParams(degree=5)),
            hsgf.PySubsetSelector.Random(),
        ),
    ]
    level_subset_sizes = [20_000, 5000, 1000, 300, 50, 25]
    params = hsgf.PyHSGFParams(min_layers=7)

    index = hsgf.PyHSGF(
        data,
        params,
        level_builder,
        level_subset_sizes=[None],  # level_subset_sizes
        max_frontier_size=None,
        higher_level_max_heap_size=1,
    )
    if PRINT_PARAMS:
        print(params.params_as_str())
        print(index.get_level_builder_as_str())
    _ = index.knn_query_batch(queries, k)

    graph_stats = index.get_graph_stats()
    for i, level_stats in enumerate(graph_stats):
        print(f"Level: {i}")
        print(
            f"n_nodes: {level_stats.n_nodes}, n_edges: {level_stats.n_edges}, max_degree: {level_stats.max_degree}, min_degree: {level_stats.min_degree}, avg_degree: {level_stats.avg_degree:.2f}, std_degree: {level_stats.std_degree:.2f}"
        )


def rnn_construction():
    data, queries, k = init_data()
    params = hsgf.PyRNNParams()
    index = hsgf.PyRNN(data, params)
    if PRINT_PARAMS:
        print(params.params_as_str())
    _ = index.knn_query_batch(queries, k)


def hnsw_construction():
    data, queries, k = init_data()
    params = hsgf.PyHNSWParams()
    index = hsgf.PyHNSW(data, params)
    if PRINT_PARAMS:
        print(params.params_as_str())
    _ = index.knn_query_batch(queries, k)


def knn_construction():
    data, queries, k = init_data(nd=10_000)
    params = hsgf.PyBruteforceKNNParams()
    index = hsgf.PyBruteforceKNN(data, params, 1)
    if PRINT_PARAMS:
        print(params.params_as_str())
    _ = index.knn_query_batch(queries, k)


def merge_hnsw_with_new_bottom_level_construction():
    data, queries, k = init_data()
    params_bottom = hsgf.PyHSGFLevelGraphParams.DEG(
        hsgf.PyDEGParams(lid=hsgf.PyLID.High, improve_k=0)
    )
    selector = hsgf.PySubsetSelector.Flooding(1)
    params_hnsw = hsgf.PyHNSWParams()
    index = hsgf.merge_hnsw_with_new_bottom_level(
        data, params_bottom, params_hnsw, selector, 1, None
    )
    if PRINT_PARAMS:
        print(params_bottom.params_as_str())
        print(params_hnsw.params_as_str())
    _ = index.knn_query_batch(queries, k)


def main():
    easy_bench(deg_construction)
    print("---")
    easy_bench(efanna_construction)
    print("---")
    easy_bench(hnsw_construction)
    print("---")
    easy_bench(knn_construction)
    print("---")
    easy_bench(nssg_construction)
    print("---")
    easy_bench(rngg_construction)
    print("---")
    easy_bench(rnn_construction)
    print("---")
    easy_bench(hsgf_construction)
    print("---")
    easy_bench(merge_hnsw_with_new_bottom_level_construction)
    print("---")


if __name__ == "__main__":
    main()
