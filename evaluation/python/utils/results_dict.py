"""Results dictionary

Mapping result files to more understandable identifiers
which makes importing them in view_results.ipynb easier
"""

from utils.util import RESULT_PATH

OLD_RESULT_PATH = RESULT_PATH + "/old"

# Better selection of results
RESULTS_TO_FILENAME = {
    # -------------------------
    "NSSG_Test": f"{RESULT_PATH}/base_graphs_eval/bge_1753546348.csv",
    "HSGF_DEG_Flood_2_Flood_1_higher_params_10M_2": f"{RESULT_PATH}/hsgf_eval/hsgf_1753820971.csv",
    # ----- HSGF -----
    # -- 10M
    "HSGF_DEG_Rand_higher_params_10M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752918273.csv",
    "HSGF_DEG_Flood_2_Flood_1_higher_params_10M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752880410.csv",
    "HSGF_DEG_FloodRepeat_2_FloodRepeat_1_higher_params_10M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752909084.csv",
    "HSGF_DEG_Rand_10M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752786777.csv",
    "HSGF_DEG_Flood_2_Flood_1_10M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752852457.csv",
    "HSGF_DEG_Flood_2_Flood_1_10M_2": f"{RESULT_PATH}/hsgf_eval/hsgf_1752784217.csv",
    "HSGF_DEG_Rand_larger_subsets_10M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752789514.csv",
    "HSGF_DEG_Flood_2_Flood_1_multi_hlmhs_10M": f"{RESULT_PATH}/hsgf_eval/hsgf_1753531917.csv",
    "HSGF_DEG_FloodRepeat_2_FloodRepeat_1_10M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752554726.csv",
    "HSGF_DEG_FloodRepeat_2_FloodRepeat_2_larger_level_sizes_10M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752867950.csv",
    # --  DEG - 3M
    "HSGF_DEG_Rand_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752744359.csv",
    "HSGF_DEG_Flood_1_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752750631.csv",
    "HSGF_DEG_Flood_2_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752757172.csv",
    "HSGF_DEG_Flood_2_Flood_1_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752678935.csv",
    "HSGF_DEG_Flood_2_NSSG_Flood_1_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752665200.csv",
    "HSGF_DEG_Rand_Compare_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752778379.csv",
    "HSGF_DEG_Rand_Compare_2_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752780263.csv",
    "HSGF_DEG_Flood_2_Flood_1_same_degree_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752774128.csv",
    "HSGF_DEG_Flood_2_Flood_1_higher_high_degree_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752775104.csv",
    "HSGF_DEG_Rand_many_levels_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752821606.csv",
    "HSGF_DEG_Flood_2_Flood_1_low_degree_higher_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752823018.csv",
    "HSGF_DEG_Flood_2_Flood_1_low_degree_higher_lower_mhs_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752830776.csv",
    "HSGF_DEG_Flood_2_Flood_1_very_low_degree_higher_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752827704.csv",
    "HSGF_DEG_Rand_small_level_sizes_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752833206.csv",
    # -- RNN
    "HSGF_RNN_Rand_low_params_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752793083.csv",
    "HSGF_RNN_Flood_2_Flood_1_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752681940.csv",
    "HSGF_RNN_Flood_2_NSSG_Flood_1_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1752684928.csv",
    # -- NSSG/EFANNA
    "Multi_Search_Variants_3M_NSSG_Rand": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751084245_NSSG.csv",
    "HSGF_NSSG_Rand_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752095038_nssg_rand_3m.csv",
    "HSGF_NSSG_Flood_1_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752095038_nssg_flood_3m.csv",
    "HSGF_EFANNA_Rand_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752095038_efanna_rand_3m.csv",
    "HSGF_EFANNA_Rand_3M_multi": f"{RESULT_PATH}/hsgf_eval/hsgf_1752761107.csv",
    "HSGF_EFANNA_Flood_1_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752095038_efanna_flood_3m.csv",
    # -- KNN
    "HSGF_KNN_Flood_degree_100_300K": f"{RESULT_PATH}/hsgf_eval/hsgf_1752709889.csv",
    "HSGF_KNN_Rand_degree_100_300K": f"{RESULT_PATH}/hsgf_eval/hsgf_1752714156.csv",
    "HSGF_KNN_Flood_degree_50_300K": f"{RESULT_PATH}/hsgf_eval/hsgf_1752718299.csv",
    "HSGF_KNN_Rand_degree_50_300K": f"{RESULT_PATH}/hsgf_eval/hsgf_1752721917.csv",
    # HSGF-DEG -- different datasets
    "HSGF_DEG_Flood_1_AUDIO": f"{RESULT_PATH}/hsgf_eval/hsgf_1752698887.csv",
    "HSGF_DEG_Flood_1_ENRON": f"{RESULT_PATH}/hsgf_eval/hsgf_1752698894.csv",
    "HSGF_DEG_Flood_1_DEEP": f"{RESULT_PATH}/hsgf_eval/hsgf_1752698963.csv",
    "HSGF_DEG_Flood_1_GLOVE": f"{RESULT_PATH}/hsgf_eval/hsgf_1752699754.csv",
    "HSGF_DEG_Flood_1_SIFT": f"{RESULT_PATH}/hsgf_eval/hsgf_1752700885.csv",
    # -
    "HSGF_DEG_Flood_2_Flood_1_AUDIO": f"{RESULT_PATH}/hsgf_eval/hsgf_1752847797.csv",
    "HSGF_DEG_Flood_2_Flood_1_ENRON": f"{RESULT_PATH}/hsgf_eval/hsgf_1752847800.csv",
    "HSGF_DEG_Flood_2_Flood_1_DEEP": f"{RESULT_PATH}/hsgf_eval/hsgf_1752847824.csv",
    "HSGF_DEG_Flood_2_Flood_1_GLOVE": f"{RESULT_PATH}/hsgf_eval/hsgf_1752848053.csv",
    "HSGF_DEG_Flood_2_Flood_1_SIFT": f"{RESULT_PATH}/hsgf_eval/hsgf_1752848407.csv",
    # -
    "HSGF_DEG_Rand_AUDIO": f"{RESULT_PATH}/hsgf_eval/hsgf_1752850468.csv",
    "HSGF_DEG_Rand_ENRON": f"{RESULT_PATH}/hsgf_eval/hsgf_1752850472.csv",
    "HSGF_DEG_Rand_DEEP": f"{RESULT_PATH}/hsgf_eval/hsgf_1752850501.csv",
    "HSGF_DEG_Rand_GLOVE": f"{RESULT_PATH}/hsgf_eval/hsgf_1752850784.csv",
    "HSGF_DEG_Rand_SIFT": f"{RESULT_PATH}/hsgf_eval/hsgf_1752851214.csv",
    # ----- HNSW -----
    # A filter version for HNSW of BGE_ALL_GRAPHS_ALL_DATASETS
    "HNSW_ALL_DATASETS": f"{RESULT_PATH}/hnsw_eval/hnsw_1753537740_all_datasets.csv",
    "HNSW_multi_merged_3M": f"{RESULT_PATH}/hnsw_eval/hnsw_3m_multi_merged.csv",
    "HNSW_Low_params_3M": f"{OLD_RESULT_PATH}/hnsw_eval/hnsw_qpsr_1751202461_low_params.csv",
    "HNSW_Medium_params_3M": f"{OLD_RESULT_PATH}/hnsw_eval/hnsw_qpsr_1751068460_medium_params.csv",
    "HNSW_High_params_3M": f"{OLD_RESULT_PATH}/hnsw_eval/hnsw_qpsr_1751202461_high_params.csv",
    "HNSW_Multi_medium_10M": f"{OLD_RESULT_PATH}/hnsw_eval/hnsw_qpsr_1752326120_10m.csv",
    # -
    "HNSW_DEEP_k_10_k1": f"{RESULT_PATH}/hnsw_eval/hnsw_1753525971.csv",
    # ----- Base Graphs -----
    "BGE_ALL_GRAPHS_ALL_DATASETS": f"{RESULT_PATH}/base_graphs_eval/bge_all_graphs_all_datasets_merged_new.csv",
    # The older version has different parameters configurations for some of the graphs
    "BGE_ALL_GRAPHS_ALL_DATASETS_Old": f"{RESULT_PATH}/base_graphs_eval/bge_all_graphs_all_datasets_merged.csv",
    # ----- HSGF Special (used for tables) -----
    "HSGF_HNSW_merged_10M": f"{RESULT_PATH}/hsgf_eval/_special/hsgf_hnsw_10m_merged.csv",
    "HSGF_HNSW_compare_gains_10m": f"{RESULT_PATH}/hsgf_eval/_special/hsgf_hnsw_compare_gains_10m.csv",
    # ----- Additional Testing -----
    "HSGF_RNN_Flood_2_DEG_Flood_1_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1753013351.csv",
    "HSGF_RNN_Flood_3_DEG_Flood_2_Flood_1_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1753021238.csv",
    "HSGF_RNN_Rand_DEG_Rand_3M": f"{RESULT_PATH}/hsgf_eval/hsgf_1753022407.csv",
    "Merged_DEG_Flood_1_HNSW_3M": f"{RESULT_PATH}/additional/hnsw_merged_1753112815.csv",
    "Merged_DEG_Rand_HNSW_3M": f"{RESULT_PATH}/additional/hnsw_merged_1753132274.csv",
    "HSGF_DEG_Flood_2_Flood_1_point_zero_false_3M": f"{RESULT_PATH}/additional/hsgf_1753137614.csv",
    "HNSW_point_zero_false_3M": f"{RESULT_PATH}/additional/hnsw_1753169740.csv",
    # HSGF-DEG very high params
    "HSGF_DEG_Flood_1_very_high_params_10M": f"{RESULT_PATH}/additional/hsgf_1753304610.csv",
    "HSGF_DEG_Rand_very_high_params_10M": f"{RESULT_PATH}/additional/hsgf_1753327910.csv",
    "HSGF_DEG_Rand_big_levels_10M": f"{RESULT_PATH}/additional/hsgf_1753353740.csv",
    # ----
    "HSGF_DEG_Flood_2_Flood_1_large_hlmhs_3M": f"{RESULT_PATH}/additional/hsgf_1753297349.csv",
    # -- HSGF-DEG-KNN
    "HSGF_DEG_Flood_2_KNN_Rand_3M": f"{RESULT_PATH}/additional/hsgf_1753193544.csv",
    # -- Normal data with un-indexed queries
    "HSGF_DEG_Flood_2_Flood_1_NormalData": f"{RESULT_PATH}/additional/hsgf_1753176239.csv",
    "HNSW_NormalData": f"{RESULT_PATH}/additional/hnsw_1753176897.csv",
    "HSGF_DEG_Flood_2_Flood_1_higher_params_NormalData": f"{RESULT_PATH}/additional/hsgf_1753178816.csv",
    "HNSW_higher_params_NormalData": f"{RESULT_PATH}/additional/hnsw_1753179626.csv",
}

# ------------------------------------------------

# Messy collection with no guarantee that the file path is correct or even still exists
RESULTS_TO_FILENAME_OLD = {
    "HNSW": {
        "300K_1": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1750435555.csv",
        "10M_1": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1750498799.csv",
        "3M_1": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1750839656.csv",
        "3M_Multiple_k": f"{OLD_RESULT_PATH}/hnsw_eval/hnsw_qpsr_1751019345.csv",
        "GIST_1": f"{OLD_RESULT_PATH}/hnsw_eval/hnsw_qpsr_1751039212.csv",
    },
    "HSGF": {
        # Curr testing
        "DEG_Flood_2_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752501208.csv",
        "DEG_Flood_2_Flood_1_3M_new": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752502732.csv",
        "DEG_Flood_2_Flood_1_3M_new_2": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752505592.csv",
        "DEG_FloodRepeat_2_FloodRepeat_1_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752508521.csv",
        "RNN_Stuff_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752512222.csv",
        # HSGF-KNN
        "KNN_Siftsmall_Flood_and_Rand": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751459603.csv",
        "KNN_Siftsmall_Flood_and_Rand_Mixed_Degrees": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751570953.csv",
        "KNN_Audio_Flood_and_Rand_": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751571604.csv",
        # HSGF-KNN-ELSE
        "KNN_Flood_multi_k_KNN_Rand": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751913635.csv",
        "RNN_KNN_Flood_DEG_KNN_Flood_300K": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752169851.csv",
        # Else
        "DEG_300K_Flood": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1750451970.csv",
        "DEG_Hubs_10M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1750503990.csv",
        "DEG_Rand_10M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1750704819.csv",
        "DEG_Flood_2_Flood_1_10M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1750754422.csv",
        "DEG_Flood_2_Flood_2_Flood_1_10M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752433265.csv",
        "ALL_GRAPHS_300K": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1750448436.csv",
        "ALL_GRAPHS_3M_Flood": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1750842419.csv",
        "ALL_GRAPHS_3M_Hubs": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1750921188.csv",
        "DEG_3M_Flood": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1750950347.csv",
        "RNN_Hubs_DEG_Hubs_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1750960813.csv",
        "Multi_Search_Variants_300K_RNN_Rand_low_params": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751285204.csv",
        "Multi_Search_Variants_300K_RNN_FloodRepeatCompare": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751296995.csv",
        "300K_DEG_FloodRepeatCompare": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751298452.csv",
        "300K_DEG_Rand_Iters": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751304496.csv",
        # ---
        "DEG_RNN_FloodRepeat_high_low_Rand_Hubs_0_50_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752010490_merged.csv",
        "NSSG_EFANNA_Flood_Rand_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752095038.csv",
        "DEG_RNN_Rand_Multi_Datasets": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752092341.csv",
        "DEG_RNN_Flood2_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752183132.csv",
        "MIXED_STUFF_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752357254.csv",
        # THIS HAS INTERESTING RESULTS:
        "DEG_Flood_1_Rand_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752217962.csv",
        "DEG_Flood_2_Flood_1_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752223445.csv",
        "DEG_Flood_2_Flood_1_3M_2": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752304001.csv",
        "DEG_Rand_higher_degree_higher_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752318711.csv",
        "RNN_Flood_2_Flood_1_3M": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1752232342.csv",
        # --
        # "DEG_NSSG_RNN_3M_Multi_Search_Variants": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751084245.csv",
        "Multi_Search_Variants_3M_DEG_Rand": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751084245_DEG.csv",
        "Multi_Search_Variants_3M_NSSG_Rand": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751084245_NSSG.csv",
        "Multi_Search_Variants_3M_RNN_Rand": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751146802.csv",
        "Multi_Search_Variants_3M_RNN_Flood_1": f"{OLD_RESULT_PATH}/hsgf_eval/hsgf_1751192947.csv",
    },
    # Mostly includes data for HNSW as well
    "BASE_GRAPHS": {
        "ALL_GRAPHS_ALL_DATASETS_1": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1750537100.csv",
        "ALL_GRAPHS_ALL_DATASETS_2": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1750624588.csv",
        "ALL_GRAPHS_GIST": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1751047525.csv",
        "ALL_GRAPHS_ALL_DATASETS_3": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1751457654_merged.csv",
        "SOMETHING_REPEAT_1": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1751457654.csv",
        "SOMETHING_REPEAT_2": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1752071715.csv",
        "TEST_NSSG": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1752491399.csv",
        # "CHECK_ALL_ALL": f"{OLD_RESULT_PATH}/base_graphs_eval/bge_1752516951.csv",
    },
}
