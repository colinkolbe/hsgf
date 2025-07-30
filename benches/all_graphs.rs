//! Bench for hsgf
//!
//! Benchmark graph-indices on common datasets
//! (although more a integration test of the library; see also all_graphs.py)
//!     - Construction and query for
//!         - DEG
//!         - NSSG
//!         - Efanna
//!         - RNN (copied from GraphIndexBaselines)
//!         - HNSW (copied from GraphIndexBaselines)
//!         - HSGF
//!             - Currently uses an 'arbitrary' combination of graph-builders and subset-selectors
//!
//! Notes
//!     - Measuring QPS and therefore querying in single-threaded mode
//!
//! Open Issues
//!
//! TODOs
//!
use graphidx::{indices::GraphIndex, measures::SquaredEuclideanDistance};
use hsgf::{
    print_index_build_info, search_as_qps,
    utils::{
        eval::{calc_recall, init_with_dataset, EvalDataSet},
        index_stats::IndexGraphStats,
    },
};
use ndarray::{Array1, Array2};
use std::time::Instant;

type R = usize;
type F = f32;
type Dist = SquaredEuclideanDistance<F>;

const MAX_HEAP_SIZE_FACTOR: usize = 1;

#[allow(unused)]
fn deg_query(data_set: EvalDataSet) {
    use hsgf::deg::*;

    let (data, queries, ground_truth, nd, nq, _d, k) = init_with_dataset(&data_set);

    let degree = 30;

    let build_time = Instant::now();

    let params = DEGParams::new()
        .with_lid(LID::High) // LID::Unknown
        .with_edges_per_vertex(degree)
        .with_extend_k(degree * 2)
        .with_extend_eps(0.3)
        .with_max_build_heap_size(degree);
    // .with_improve_k(0)
    // .with_improve_eps(0.001)
    // .with_swap_tries(5)
    // .with_additional_swap_tries(5)
    // .with_max_build_frontier_size(Some(200));

    type BuilderType = DEGParallelBuilder<R, F, Dist>;
    let index = BuilderType::build(data.view(), Dist::new(), params);

    println!(
        "Graph construction ({:?}): {:.2?}",
        std::any::type_name::<BuilderType>(),
        build_time.elapsed()
    );
    print_index_build_info!(index);

    let graph_name = "DEG";
    let ground_truth_view = ground_truth.view();
    let max_heap_size = k * MAX_HEAP_SIZE_FACTOR;
    let (_query_time, _qps, _recall) = search_as_qps!(
        index,
        queries,
        ground_truth_view,
        k,
        max_heap_size,
        nq,
        graph_name,
        true
    );
}

#[allow(unused)]
fn nssg_query(data_set: EvalDataSet) {
    use hsgf::efanna::*;
    use hsgf::nssg::*;
    use hsgf::rngg::*;

    let use_efanna = true;

    let (data, queries, ground_truth, _nd, nq, _d, k) = init_with_dataset(&data_set);

    let build_time = Instant::now();

    let params_efanna = EfannaParams::new()
        .with_k(200)
        .with_l(200)
        .with_iter(12)
        .with_s(10)
        .with_r(100);

    let params = NSSGParams::new()
        .with_l(100)
        .with_range(50)
        .with_angle(60.0);
    type BuilderType = NSSGParallelBuilder<R, F, Dist>;
    let index = BuilderType::build(
        data.view(),
        Dist::new(),
        params,
        NSSGInputGraphData::EFANNA(Some(params_efanna)),
    );

    println!(
        "Graph construction ({:?}): {:.2?}",
        std::any::type_name::<BuilderType>(),
        build_time.elapsed(),
    );
    print_index_build_info!(index);
    let graph_name = "NSSG";
    let ground_truth_view = ground_truth.view();
    let max_heap_size = k * MAX_HEAP_SIZE_FACTOR;
    let (_query_time, _qps, _recall) = search_as_qps!(
        index,
        queries,
        ground_truth_view,
        k,
        max_heap_size,
        nq,
        graph_name,
        true
    );
}

#[allow(unused)]
fn efanna_query(data_set: EvalDataSet) {
    use hsgf::efanna::*;

    let (data, queries, ground_truth, _nd, nq, _d, k) = init_with_dataset(&data_set);

    let build_time = Instant::now();
    let params_efanna = EfannaParams::new()
        .with_k(50)
        .with_l(70)
        .with_iter(10)
        .with_s(10)
        .with_r(50);
    type BuilderType = EfannaParallelMaxHeapBuilder<R, F, Dist>;
    let index = BuilderType::build(data.view(), Dist::new(), params_efanna);

    println!(
        "Graph construction ({:?}): {:.2?}",
        std::any::type_name::<BuilderType>(),
        build_time.elapsed(),
    );
    print_index_build_info!(index);
    let graph_name = "Efanna";
    let ground_truth_view = ground_truth.view();
    let max_heap_size = k * MAX_HEAP_SIZE_FACTOR;
    let (_query_time, _qps, _recall) = search_as_qps!(
        index,
        queries,
        ground_truth_view,
        k,
        max_heap_size,
        nq,
        graph_name,
        true
    );
}

// Copied from graphIndexBaselines to test and compare on EvalDataSets
#[allow(unused)]
fn rnn_query(data_set: EvalDataSet) {
    use graphidxbaselines::rnn::*;

    let (data, queries, ground_truth, _nd, nq, _d, k) = init_with_dataset(&data_set);

    let build_time = Instant::now();
    let params = RNNParams::new()
        .with_n_outer_loops(4)
        .with_n_inner_loops(15)
        .with_reduce_degree(20)
        .with_initial_degree(96);

    type BuilderType = RNNDescentBuilder<R, F, Dist>;
    let index = BuilderType::build(data.view(), Dist::new(), params);

    println!(
        "Graph construction ({:?}): {:.2?}",
        std::any::type_name::<BuilderType>(),
        build_time.elapsed()
    );
    print_index_build_info!(index);

    let graph_name = "RNN";
    let ground_truth_view = ground_truth.view();
    let max_heap_size = k * MAX_HEAP_SIZE_FACTOR;
    let (_query_time, _qps, _recall) = search_as_qps!(
        index,
        queries,
        ground_truth_view,
        k,
        max_heap_size,
        nq,
        graph_name,
        true
    );
}

// Copied from graphIndexBaselines to test and compare on EvalDataSets
#[allow(unused)]
fn hnsw_query(data_set: EvalDataSet) {
    use graphidxbaselines::hnsw::*;

    let (data, queries, ground_truth, _nd, nq, _d, k) = init_with_dataset(&data_set);

    let build_time = Instant::now();
    let params = HNSWParams::new()
        .with_higher_max_degree(40)
        .with_lowest_max_degree(50)
        .with_max_build_heap_size(800)
        .with_insert_heuristic(true)
        .with_insert_heuristic_extend(false);

    type BuilderType = HNSWParallelHeapBuilder<R, F, Dist>;
    let index = BuilderType::build(data.view(), Dist::new(), params, 1);

    println!(
        "Graph construction ({:?}): {:.2?}",
        std::any::type_name::<BuilderType>(),
        build_time.elapsed()
    );
    print_index_build_info!(index);

    let graph_name = "HNSW";
    let ground_truth_view = ground_truth.view();
    let max_heap_size = k * MAX_HEAP_SIZE_FACTOR;
    let (_query_time, _qps, _recall) = search_as_qps!(
        index,
        queries,
        ground_truth_view,
        k,
        max_heap_size,
        nq,
        graph_name,
        true
    );
}

#[allow(unused)]
fn hsgf_query(data_set: EvalDataSet) {
    use graphidxbaselines::rnn::RNNParams;
    use hsgf::deg::{DEGParams, LID};
    use hsgf::hsgf::*;
    use hsgf::rngg::RNGGParams;
    use hsgf::selectors::{FloodingSelectorParallelChunked, RandomSelector};

    let (data, queries, ground_truth, _nd, nq, _d, k) = init_with_dataset(&data_set);

    type FloodS = FloodingSelectorParallelChunked<R, F>;
    type RandS = RandomSelector<R, F>;

    let build_time = Instant::now();

    let params = HSGFParams::new().with_min_layers(4);
    let higher_level_max_heap_size = None;
    let level_subset_sizes = None;
    let flooding_range = 2;

    let level_builders: Vec<HSGFLevelGraphBuilder<R, F>> = vec![
        HSGFLevelGraphBuilder::new_deg(
            DEGParams::new().with_lid(LID::High).with_improve_k(0),
            Box::new(FloodS::new(flooding_range)),
        ),
        HSGFLevelGraphBuilder::new_rnn(RNNParams::new(), Box::new(RandS::new())),
        HSGFLevelGraphBuilder::new_rngg(RNGGParams::new(), Box::new(RandS::new())),
    ];

    type BuilderType = HSGFEnumBuilder<R, F, Dist>;
    let index = BuilderType::build(
        data.view(),
        Dist::new(),
        params,
        level_builders,
        level_subset_sizes,
        Some(vec![0]),
        higher_level_max_heap_size.unwrap_or(1),
    );

    println!(
        "Graph construction ({:?}): {:.2?}",
        std::any::type_name::<BuilderType>(),
        build_time.elapsed()
    );
    print_index_build_info!(index);

    let graph_name = "HSGF";
    let ground_truth_view = ground_truth.view();
    let max_heap_size = k * MAX_HEAP_SIZE_FACTOR;
    let (_query_time, _qps, _recall) = search_as_qps!(
        index,
        queries,
        ground_truth_view,
        k,
        max_heap_size,
        nq,
        graph_name,
        true
    );
}

fn main() {
    let curr_dataset = EvalDataSet::SIFTSmall;
    deg_query(curr_dataset);
    efanna_query(curr_dataset);
    hnsw_query(curr_dataset);
    hsgf_query(curr_dataset);
    nssg_query(curr_dataset);
    rnn_query(curr_dataset);
}
