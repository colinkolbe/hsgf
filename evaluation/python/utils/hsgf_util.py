"""HSGF related util"""

from utils.util import Fvecs_Dataset, LAION_SISAP_Subset

import hsgf


# wrapper
def hsgf_selector(flood=None, random=False, hubs=None, floodRepeat=None):
    if flood is not None:
        return hsgf.PySubsetSelector.Flooding(flood)
    if floodRepeat is not None:
        return hsgf.PySubsetSelector.FloodingRepeat(floodRepeat)
    if random:
        return hsgf.PySubsetSelector.Random()
    if hubs is not None:
        return hsgf.PySubsetSelector.HubNodes(hubs)
    print("Specify selector.")
    return None


# Using already calculated HNSW level sizes as "good" level sizes
# which are taken from their corresponding runs of HNSW
# Always (of course) omitting the bottom-level size
# HSGF offers feature similar to the automatic calculations
# of the level sizes in HNSW as well though
def get_level_subset_sizes(dataset):
    match dataset:
        case Fvecs_Dataset.AUDIO:
            level_subset_sizes = [5300, 550, 40]
        case Fvecs_Dataset.ENRON:
            level_subset_sizes = [1800, 40]
        case Fvecs_Dataset.DEEP:
            level_subset_sizes = [20_000, 400, 40]
        case Fvecs_Dataset.GLOVE:
            level_subset_sizes = [24_000, 500, 40]
        # case Fvecs_Dataset.GIST:
        #     level_subset_sizes = [16763, 283, 40]
        case Fvecs_Dataset.SIFT:
            level_subset_sizes = [25_000, 600, 40]
        case Fvecs_Dataset.SIFTSMALL:
            level_subset_sizes = [200, 40]
        case "NormalData":
            # dependent on chosen n_data
            level_subset_sizes = None
        case LAION_SISAP_Subset.three_hun_k:
            level_subset_sizes = [10_000, 400, 40]
        case LAION_SISAP_Subset.three_m:
            level_subset_sizes = [
                100_000,
                3_300,
                150,
                30,
            ]
        case LAION_SISAP_Subset.ten_m:
            level_subset_sizes = [
                350_000,
                12_000,
                500,
                30,
            ]
        case LAION_SISAP_Subset.hundred_m:
            level_subset_sizes = None
        case _:
            level_subset_sizes = None
    return level_subset_sizes


def _knn_prep():
    """TODO-Feature
    - For KNN-Bruteforce calculate ground_truth only once, keep it in memory and pass it at least for the bottom level as Existing graph back into the HSGF builder
    """
    pass
    # if PARAMS_OBJECTS_TO_ITERATE == KNN_CONFIG:
    #     pass
    #     # only calculates the bottom level knn graph to save at least some computations
    #     # todo tests on siftsmall with this and without this
    #     # Assumes maximum degree <= maximum query k
    #     params = hsgf.PyBruteforceKNNParams()
    #     knn_bottom_level_input_graph = (
    #         hsgf.PyBruteforceKNN(data, params, 1).get_graph().adjacency
    #     )
    #     # not sure if we can access the adjacency directly without implementing a getter in py_hsgf.rs

    #     exact_neighbors = []  # todo
    #     updated_config = []
    #     for (
    #         hsgf_params,
    #         level_builder,
    #         level_subset_sizes,
    #         level_builder_name,
    #     ) in KNN_CONFIG:
    #         (bottom_builder_params, bottom_selector) = level_builder[0]
    #         degree = bottom_builder_params.params.degree
    #         new_bottom_builder = hsgf.PyHSGFLevelGraphParams.ExistingGraph(
    #             hsgf.PyDirLoLGraph(
    #                 adjacency=exact_neighbors[:, degree], n_edges=nd * degree
    #             ),
    #             f"ExistingGraph:{bottom_builder_params.params.params_as_str()}",
    #         )
    #         level_builder[0] = (new_bottom_builder, bottom_selector)
    #         updated_config.append(
    #             (
    #                 hsgf_params,
    #                 level_builder,
    #                 level_subset_sizes,
    #                 level_builder_name,
    #             )
    #         )
    #     KNN_CONFIG = updated_config
