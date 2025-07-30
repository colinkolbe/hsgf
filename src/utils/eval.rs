//! Util
//!
//! Notes
//!     - get_test_data() - helper for running tests/benches to test on similar test data
//!         - This would be interesting but does not seem too important:
//!             - https://stackoverflow.com/questions/59020767/
//!     - print_recall() - helper when running tests/benches
//!     - init_with_dataset() - helper to load .fvecs and .ivecs specified by `EvalDatasets`
//!         - No HDF5 data loader (this is only done in in /src/eval via the Python module h5py)
//!     - hierarchy_graph_level_wise_performance()
//!         - Queries each level-stacked subgraph of a hierarchy graph like HSGF or HNSW
//!
//! Open Issues
//!         -
//!
//! TODOs
//!     - Merge evaluate_qps_recall_*() into one function (-> macro?)
//!     - Feature: hierarchy_graph_level_wise_performance_powerset()
//!         - Still needs layer_ids mapping for non-sequential
//!           level-stacked subgraph
//!
use graphidx::indices::{
    GreedyCappedLayeredGraphIndex, GreedyCappedSingleGraphIndex, GreedySingleGraphIndex,
};
use graphidx::measures::Distance;
use graphidx::types::{SyncFloat, SyncUnsignedInteger, UnsignedInteger};
use graphidx::{
    data::MatrixDataSource,
    graphs::Graph,
    indices::{GraphIndex, GreedyLayeredGraphIndex},
    measures::SquaredEuclideanDistance,
};
#[allow(unused)]
use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_rand::rand::prelude::Distribution;
use ndarray_rand::rand_distr::{Normal, Uniform};
use num::ToPrimitive;
use std::time::Instant;
use vecs_file::read_vecs_file;

pub const BASE_PATH_DATA_SETS: &str = "../evaluation/datasets/";

/// To use standard test values across crates' test modules
/// Currently returns: (nd=50_000, nq=1000, d=30, k=10, data)  
#[inline(always)]
pub fn get_test_data(normal_distribution: bool) -> (usize, usize, usize, usize, Array2<f32>) {
    let init_time = Instant::now();
    let (nd, nq, d, k) = (50_000, 1000, 30, 10);
    let data = if normal_distribution {
        let rng = Normal::new(0.0, 1.0).unwrap();
        Array2::from_shape_fn((nd, d), |_| rng.sample(&mut rand::thread_rng()))
    } else {
        let rng = Uniform::new(0.0, 1.0);
        Array2::from_shape_fn((nd, d), |_| rng.sample(&mut rand::thread_rng()))
    };
    println!("Data initialization: {:.2?}", init_time.elapsed());
    (nd, nq, d, k, data)
}

/// Helper to calculate and print the recall
pub fn calc_recall<R: UnsignedInteger>(
    bruteforce_ids: ArrayView2<R>,
    found_ids: &Array2<R>,
    nq: usize,
    k: usize,
    text: &str,
    print_output: bool,
) -> f32 {
    let mut same = 0;
    bruteforce_ids
        .axis_iter(Axis(0))
        .zip(found_ids.axis_iter(Axis(0)))
        .for_each(|(bf, res)| {
            let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
            let res_set = res.iter().collect::<std::collections::HashSet<_>>();
            same += bf_set.intersection(&res_set).count();
        });
    let recall = (same as f32 / (nq * k) as f32) * 100f32;
    if print_output {
        println!("Recall {}: {:.2}%", text, recall);
    }
    recall
}

/// print graph/index statistics
#[macro_export]
macro_rules! print_index_build_info {
    ($index:ident) => {{
        let (n_edges, counted_edges_in_adj, hashset_counted_edges) = $index.get_n_edges();
        println!(
            "Graph sizes: {:.2?}, Edges: {}, {}, {}, \
            Total avg. degree: {:.1?}, Avg. degrees: {:.1?}",
            $index.get_graph_n_vertices(),
            n_edges,
            counted_edges_in_adj,
            hashset_counted_edges,
            $index.get_total_avg_out_degree(),
            $index.get_avg_out_degrees(),
        );
        println!(
            "Duplicate neighbors, Loops, Escaping Edges: {}, {}, {}",
            $index.get_duplicate_neighbor_total_count(),
            $index.has_loops(),
            $index.has_escaping_edges(),
        );
    }};
}

#[allow(unused)]
/// Return data and (un-)indexed queries
pub fn init_with_normal_data(
    nd: usize,
    nq: usize,
    d: usize,
    indexed_queries: bool,
) -> (Array2<f32>, Array2<f32>) {
    let rng = Normal::new(0.0, 1.0).unwrap();
    let data: Array2<f32> = Array2::from_shape_fn((nd, d), |_| rng.sample(&mut rand::thread_rng()));
    let queries: Array2<f32> = if indexed_queries {
        data.slice_axis(Axis(0), ndarray::Slice::from(0..nq))
            .to_owned()
    } else {
        Array2::from_shape_fn((nq, d), |_| rng.sample(&mut rand::thread_rng()))
    };
    (data, queries)
}

/// Panics if filepaths are incorrect
pub fn load_data_set(
    d: usize,
    k: usize,
    filepaths: Vec<String>,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<usize>,
    usize,
    usize,
    usize,
    usize,
) {
    // load data vectors
    assert!(filepaths.len() == 3);
    let data_base: vecs_file::Vectors<f32> = read_vecs_file(&filepaths[0]).unwrap();
    let data_queries: vecs_file::Vectors<f32> = read_vecs_file(&filepaths[1]).unwrap();
    let data_ground_truth: vecs_file::Vectors<i32> = read_vecs_file(&filepaths[2]).unwrap();

    let (n_data, n_queries) = (data_base.len(), data_queries.len());
    assert!(n_queries == data_ground_truth.len());

    let data: Array2<f32> =
        Array2::from_shape_fn((n_data, d), |(i, j)| data_base.get_vector(i).unwrap()[j]);
    let queries: Array2<f32> = Array2::from_shape_fn((n_queries, d), |(i, j)| {
        data_queries.get_vector(i).unwrap()[j]
    });
    let ground_truth: Array2<usize> = Array2::from_shape_fn((n_queries, k), |(i, j)| {
        data_ground_truth.get_vector(i).unwrap()[j]
            .to_usize()
            .unwrap()
    });
    println!("Successfully loaded data.");
    (data, queries, ground_truth, n_data, n_queries, d, k)
}

macro_rules! data_set_loader {
    ($base_path_data_sets:ident, $data_set:ident, $d:ident, $k:ident) => {{
        // unpretty string building..
        let data_path_base =
            $base_path_data_sets.to_string() + $data_set + "/" + $data_set + "_base.fvecs";
        let data_path_queries =
            $base_path_data_sets.to_string() + $data_set + "/" + $data_set + "_query.fvecs";
        let data_path_groundtruth =
            $base_path_data_sets.to_string() + $data_set + "/" + $data_set + "_groundtruth.ivecs";
        load_data_set(
            $d,
            $k,
            vec![data_path_base, data_path_queries, data_path_groundtruth],
        )
    }};
}

#[allow(unused)]
#[derive(Copy, Clone)]
pub enum EvalDataSet {
    AUDIO,
    ENRON,
    DEEP1M,
    GLOVE100,
    SIFT,
    SIFTSmall,
    Normal100kIndexedQueries,
    Normal100kUnIndexedQueries,
    Normal1mioIndexedQueries,
    Normal1mioUnIndexedQueries,
    Normal100kHighDim,
}

pub fn init_with_dataset(
    data_set: &EvalDataSet,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<usize>,
    usize,
    usize,
    usize,
    usize,
) {
    match data_set {
        EvalDataSet::AUDIO => {
            // nb base vector: 53,387, nb query vectors: 200
            let (data_set, d, k) = ("audio", 192, 20);
            data_set_loader!(BASE_PATH_DATA_SETS, data_set, d, k)
        }
        EvalDataSet::ENRON => {
            // nb base vector: 94,987, nb query vectors: 200
            let (data_set, d, k) = ("enron", 1369, 20);
            data_set_loader!(BASE_PATH_DATA_SETS, data_set, d, k)
        }
        EvalDataSet::DEEP1M => {
            // nb base vector: 1,000,000, nb query vectors: 10,000
            let (data_set, d, k) = ("deep1m", 96, 100);
            data_set_loader!(BASE_PATH_DATA_SETS, data_set, d, k)
        }
        EvalDataSet::GLOVE100 => {
            // nb base vector: 1,183,514, nb query vectors: 10,000
            let (data_set, d, k) = ("glove-100", 100, 100);
            data_set_loader!(BASE_PATH_DATA_SETS, data_set, d, k)
        }
        EvalDataSet::SIFT => {
            // nb base vector: 1,000,000, nb query vectors: 10,000
            let (data_set, d, k) = ("sift", 128, 100);
            data_set_loader!(BASE_PATH_DATA_SETS, data_set, d, k)
        }
        EvalDataSet::SIFTSmall => {
            // nb base vector: 10,000, nb query vectors: 100
            let (data_set, d, k) = ("siftsmall", 128, 100);
            data_set_loader!(BASE_PATH_DATA_SETS, data_set, d, k)
        }
        EvalDataSet::Normal100kIndexedQueries => {
            let (nd, nq, d, k) = (100_000, 10_000, 30, 100);
            let (data, queries) = init_with_normal_data(nd, nq, d, true);
            let (ground_truth, _) = graphidx::indices::bruteforce_neighbors(
                &data,
                &queries,
                &graphidx::measures::SquaredEuclideanDistance::new(),
                k,
            );
            (data, queries, ground_truth, nd, nq, d, k)
        }
        EvalDataSet::Normal100kUnIndexedQueries => {
            let (nd, nq, d, k) = (100_000, 10_000, 30, 100);
            let (data, queries) = init_with_normal_data(nd, nq, d, false);
            let (ground_truth, _) = graphidx::indices::bruteforce_neighbors(
                &data,
                &queries,
                &graphidx::measures::SquaredEuclideanDistance::new(),
                k,
            );
            (data, queries, ground_truth, nd, nq, d, k)
        }
        EvalDataSet::Normal1mioIndexedQueries => {
            let (nd, nq, d, k) = (1_000_000, 10_000, 30, 100);
            let (data, queries) = init_with_normal_data(nd, nq, d, true);
            let (ground_truth, _) = graphidx::indices::bruteforce_neighbors(
                &data,
                &queries,
                &graphidx::measures::SquaredEuclideanDistance::new(),
                k,
            );
            (data, queries, ground_truth, nd, nq, d, k)
        }
        EvalDataSet::Normal1mioUnIndexedQueries => {
            let (nd, nq, d, k) = (1_000_000, 10_000, 30, 100);
            let (data, queries) = init_with_normal_data(nd, nq, d, false);
            let (ground_truth, _) = graphidx::indices::bruteforce_neighbors(
                &data,
                &queries,
                &graphidx::measures::SquaredEuclideanDistance::new(),
                k,
            );
            (data, queries, ground_truth, nd, nq, d, k)
        }
        EvalDataSet::Normal100kHighDim => {
            let (nd, nq, d, k) = (10_000, 1_000, 500, 10);
            let (data, queries) = init_with_normal_data(nd, nq, d, true);
            let (ground_truth, _) = graphidx::indices::bruteforce_neighbors(
                &data,
                &queries,
                &graphidx::measures::SquaredEuclideanDistance::new(),
                k,
            );
            (data, queries, ground_truth, nd, nq, d, k)
        }
    }
}

// ------------------------------ Eval functions ------------------------------
// which are used via the Python module hsgf

// Searches an index in non-batch mode and therefore allows measuring the QPS
#[macro_export]
macro_rules! search_as_qps {
    ($index:ident, $queries:ident, $ground_truth:ident, $k:ident, $max_heap_size:ident, $nq:ident, $graph_name:ident, $print_output:ident) => {{
        let mut cache = $index._new_search_cache($k);
        // measure QPS
        let query_time = Instant::now();
        let _ids = (0..$nq)
            .map(|i| {
                $index
                    .greedy_search(&$queries.row(i), $k, $max_heap_size, &mut cache)
                    .0
            })
            .collect::<Vec<Array1<R>>>();
        let elapsed_time = query_time.elapsed();

        let ids: Array2<R> = Array2::from_shape_fn(($nq, $k), |(i, j)| _ids[i][j]);
        let qps = $nq as f32 / elapsed_time.as_secs_f32();

        if $print_output {
            println!(
                "{} queries: {:.2?} | QPS: ~{:.0}",
                $graph_name, elapsed_time, qps,
            );
        }

        let recall = calc_recall($ground_truth, &ids, $nq, $k, $graph_name, $print_output);

        (elapsed_time.as_secs_f32(), qps, recall)
    }};
}

#[allow(unused)]
pub struct QPSRecallStats {
    pub layer_count: usize,
    pub k: usize,
    pub max_heap_size: usize,
    pub nq: usize,
    pub query_time: f32,
    pub qps: f32,
    pub recall: f32,
}

/// Calculate recall and QPS starting from each level of a hierarchy graph
pub fn hierarchy_graph_level_wise_performance<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
    M: MatrixDataSource<F> + Sync,
    G: Graph<R> + Sync,
>(
    input_index: &GreedyLayeredGraphIndex<R, F, Dist, M, G>,
    data: ArrayView2<F>,
    queries: ArrayView2<F>,
    ground_truth: ArrayView2<R>,
    k: usize,
    max_heap_size: usize,
    nq: usize,
    graph_name: &str,
    print_output: bool,
    only_top_level: bool,
    point_zero_entry: bool,
) -> Vec<QPSRecallStats> {
    let mut res: Vec<QPSRecallStats> = Vec::new();
    let n_layers = input_index.layer_count();
    // note layers are index-one based
    let lowest_search_level = if only_top_level { n_layers } else { 1 };
    let top_entry_points: Option<Vec<R>> = if point_zero_entry {
        Some(vec![R::zero()])
    } else {
        None
    };
    for layer in (lowest_search_level..input_index.layer_count() + 1).rev() {
        let index = GreedyLayeredGraphIndex::new(
            data.view(),
            input_index
                .graphs()
                .iter()
                .take(layer)
                .map(|g| g.as_dir_lol_graph())
                .collect(),
            (1..layer)
                .map(|i| input_index.get_local_layer_ids(i).unwrap().clone())
                .collect(),
            (1..layer)
                .map(|i| input_index.get_global_layer_ids(i).unwrap().clone())
                .collect(),
            SquaredEuclideanDistance::new(),
            input_index.higher_level_max_heap_size(),
            top_entry_points.clone(),
        );
        if print_output {
            println!("### Layer {}={}", layer, index.layer_count());
        }
        // hsgf_lib::print_index_build_info!(index);
        let (query_time, qps, recall) = search_as_qps!(
            index,
            queries,
            ground_truth,
            k,
            max_heap_size,
            nq,
            graph_name,
            print_output
        );
        res.push(QPSRecallStats {
            layer_count: index.layer_count(),
            k,
            max_heap_size,
            nq,
            query_time,
            qps,
            recall,
        })
    }
    res
}

pub fn hierarchy_graph_level_wise_performance_capped<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
    M: MatrixDataSource<F> + Sync,
    G: Graph<R> + Sync,
>(
    input_index: &GreedyCappedLayeredGraphIndex<R, F, Dist, M, G>,
    data: ArrayView2<F>,
    queries: ArrayView2<F>,
    ground_truth: ArrayView2<R>,
    k: usize,
    max_heap_size: usize,
    nq: usize,
    graph_name: &str,
    print_output: bool,
    only_top_level: bool,
    point_zero_entry: bool,
) -> Vec<QPSRecallStats> {
    let mut res: Vec<QPSRecallStats> = Vec::new();
    let n_layers = input_index.layer_count();
    let lowest_search_level = if only_top_level { n_layers } else { 1 };
    let top_entry_points: Option<Vec<R>> = if point_zero_entry {
        Some(vec![R::zero()])
    } else {
        None
    };
    for layer in (lowest_search_level..input_index.layer_count() + 1).rev() {
        let index = GreedyLayeredGraphIndex::new(
            data.view(),
            input_index
                .graphs()
                .iter()
                .take(layer)
                .map(|g| g.as_dir_lol_graph())
                .collect(),
            (1..layer)
                .map(|i| input_index.get_local_layer_ids(i).unwrap().clone())
                .collect(),
            (1..layer)
                .map(|i| input_index.get_global_layer_ids(i).unwrap().clone())
                .collect(),
            SquaredEuclideanDistance::new(),
            input_index.higher_level_max_heap_size(),
            top_entry_points.clone(),
        );
        if print_output {
            println!("### Layer {}={}", layer, index.layer_count());
        }
        // hsgf_lib::print_index_build_info!(index);
        let (query_time, qps, recall) = search_as_qps!(
            index,
            queries,
            ground_truth,
            k,
            max_heap_size,
            nq,
            graph_name,
            print_output
        );
        res.push(QPSRecallStats {
            layer_count: index.layer_count(),
            k,
            max_heap_size,
            nq,
            query_time,
            qps,
            recall,
        })
    }
    res
}

pub fn evaluate_qps_recall<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
    M: MatrixDataSource<F> + Sync,
    G: Graph<R> + Sync,
>(
    input_index: &GreedySingleGraphIndex<R, F, Dist, M, G>,
    queries: ArrayView2<F>,
    ground_truth: ArrayView2<R>,
    k: usize,
    max_heap_size: usize,
    nq: usize,
    graph_name: &str,
    print_output: bool,
) -> QPSRecallStats {
    let (query_time, qps, recall) = search_as_qps!(
        input_index,
        queries,
        ground_truth,
        k,
        max_heap_size,
        nq,
        graph_name,
        print_output
    );
    QPSRecallStats {
        layer_count: input_index.layer_count(),
        k,
        max_heap_size,
        nq,
        query_time,
        qps,
        recall,
    }
}

pub fn evaluate_qps_recall_capped<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
    M: MatrixDataSource<F> + Sync,
    G: Graph<R> + Sync,
>(
    input_index: &GreedyCappedSingleGraphIndex<R, F, Dist, M, G>,
    queries: ArrayView2<F>,
    ground_truth: ArrayView2<R>,
    k: usize,
    max_heap_size: usize,
    nq: usize,
    graph_name: &str,
    print_output: bool,
) -> QPSRecallStats {
    let (query_time, qps, recall) = search_as_qps!(
        input_index,
        queries,
        ground_truth,
        k,
        max_heap_size,
        nq,
        graph_name,
        print_output
    );
    QPSRecallStats {
        layer_count: input_index.layer_count(),
        k,
        max_heap_size,
        nq,
        query_time,
        qps,
        recall,
    }
}

// currently unused because of hierarchy_graph_level_wise_performance
pub fn evaluate_qps_recall_layered<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
    M: MatrixDataSource<F> + Sync,
    G: Graph<R> + Sync,
>(
    input_index: &GreedyLayeredGraphIndex<R, F, Dist, M, G>,
    queries: ArrayView2<F>,
    ground_truth: ArrayView2<R>,
    k: usize,
    max_heap_size: usize,
    nq: usize,
    graph_name: &str,
    print_output: bool,
) -> QPSRecallStats {
    let (query_time, qps, recall) = search_as_qps!(
        input_index,
        queries,
        ground_truth,
        k,
        max_heap_size,
        nq,
        graph_name,
        print_output
    );
    QPSRecallStats {
        layer_count: input_index.layer_count(),
        k,
        max_heap_size,
        nq,
        query_time,
        qps,
        recall,
    }
}

// currently unused because of hierarchy_graph_level_wise_performance
pub fn evaluate_qps_recall_layered_capped<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
    M: MatrixDataSource<F> + Sync,
    G: Graph<R> + Sync,
>(
    input_index: &GreedyCappedLayeredGraphIndex<R, F, Dist, M, G>,
    queries: ArrayView2<F>,
    ground_truth: ArrayView2<R>,
    k: usize,
    max_heap_size: usize,
    nq: usize,
    graph_name: &str,
    print_output: bool,
) -> QPSRecallStats {
    let (query_time, qps, recall) = search_as_qps!(
        input_index,
        queries,
        ground_truth,
        k,
        max_heap_size,
        nq,
        graph_name,
        print_output
    );
    QPSRecallStats {
        layer_count: input_index.layer_count(),
        k,
        max_heap_size,
        nq,
        query_time,
        qps,
        recall,
    }
}

/// Calculate recall and QPS starting from each bottom-up stacked level graph combination
/// of a hierarchy graph, like levels: [1,3,5], or [1,2,5] (always needs the bottom level)
pub fn hierarchy_graph_level_wise_performance_powerset<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
    M: MatrixDataSource<F> + Sync,
    G: Graph<R> + Sync,
>(
    _input_index: &GreedyLayeredGraphIndex<R, F, Dist, M, G>,
    _data: ArrayView2<F>,
    _queries: ArrayView2<F>,
    _ground_truth: ArrayView2<R>,
    _k: usize,
    _max_heap_size: usize,
    _nq: usize,
    _graph_name: &str,
    _print_output: bool,
    _point_zero_entry: bool,
) -> Vec<QPSRecallStats> {
    unimplemented!("TODO. Finish mapping over layer ids.");
    // let mut res: Vec<QPSRecallStats> = Vec::new();
    // let n_layers = input_index.layer_count();
    // let top_entry_points: Option<Vec<R>> = if point_zero_entry {
    //     Some(vec![R::zero()])
    // } else {
    //     None
    // };

    // // unimplemented!("Not fully implemented yet. Fix layer id mapping.");

    // /* Note layers are index-one based.
    // Get power-set for (1..n_layers+1) but only those which also
    // contain the bottom level; For (1..5+1) this results in 16 (!) combinations!
    // E.g., [[1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4],
    // [1, 3, 5], [1, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [1, 2, 3, 4, 5]]
    // */
    // let all_indexes = (1..n_layers)
    //     .powerset()
    //     .filter(|e| e.contains(&1))
    //     .collect::<Vec<Vec<usize>>>();
    // println!("H1 {:?}", all_indexes);
    // for indexes in all_indexes {
    //     println!("H2: {:?}", indexes);
    //     let graphs: Vec<DirLoLGraph<R>> = indexes
    //         .iter()
    //         .map(|i| input_index.graphs()[*i].as_dir_lol_graph())
    //         .collect();
    //     let mut local_layer_ids: Vec<Vec<R>> = Vec::new();
    //     let mut global_layer_ids: Vec<Vec<R>> = Vec::new();

    //     // TODO still needs to remap the layer ids for non sequential layer-stacked subgraphs (!!)

    //     let index = GreedyLayeredGraphIndex::new(
    //         data.view(),
    //         graphs,
    //         local_layer_ids,
    //         global_layer_ids,
    //         SquaredEuclideanDistance::new(),
    //         input_index.higher_level_max_heap_size(),
    //         top_entry_points.clone(),
    //     );
    //     // current indices as one number like [1,3,5] => 135
    //     let layer_cnt_number: usize = indexes
    //         .iter()
    //         .enumerate()
    //         .map(|(i, j)| (10 as i32).pow((indexes.len() - 1 - i) as u32) as usize * j)
    //         .sum();
    //     if print_output {
    //         println!(
    //             "### Layers: {}, {}={}",
    //             layer_cnt_number,
    //             indexes.len(),
    //             index.layer_count()
    //         );
    //     }
    //     // hsgf_lib::print_index_build_info!(index);
    //     let (query_time, qps, recall) = search_as_qps!(
    //         index,
    //         queries,
    //         ground_truth,
    //         k,
    //         max_heap_size,
    //         nq,
    //         graph_name,
    //         print_output
    //     );
    //     res.push(QPSRecallStats {
    //         layer_count: layer_cnt_number, // note, it is not index.layer_count() here!
    //         k,
    //         max_heap_size,
    //         nq,
    //         query_time,
    //         qps,
    //         recall,
    //     })
    // }
    // res
}
