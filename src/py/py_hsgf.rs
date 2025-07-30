//! Python bindings of HSGF
//!
//! Contains Python Graph Builders and Parameters objects for
//!     - HSGF, DEG, NSSG, EFANNA, RNGG
//!     - RNN, HNSW and BruteforceKNN from GraphIndexBaselines as re-exports
//!     - Additionally via crate::py_util and crate::py_graph_params
//!         - A few helper enums to organize input data
//!         - A helper to load .fvecs/.ivecs datasets by specifying three file_paths (unused)
//!         - A helper to get normal data including queries and groundtruth (mostly unused)
//!         - merge_hnsw_with_new_bottom_level() as an eval/test function
//!
//! Notes
//!     - Currently used data types: R:=usize, F:f32, Dist:SquaredEuclideanDistance
//!         - Defined in crate::py::py_util
//!
//! Open Issues
//!     - The formatting via `cargo fmt` might sometimes need to be triggered manually because
//!       of the pyo3 annotations
//!
//! TODOs
//!
use crate::generic_graph_index_funs;
use crate::py::py_graph_params::*;
use crate::py::py_util::*;
use crate::selectors::{
    FloodingSelectorParallelChunked, FloodingSelectorParallelChunkedRepeat, HubNodesSelector,
    RandomSelector, SubsetSelector,
};
use crate::utils::{
    eval::{
        evaluate_qps_recall, evaluate_qps_recall_capped, hierarchy_graph_level_wise_performance,
        hierarchy_graph_level_wise_performance_capped,
        hierarchy_graph_level_wise_performance_powerset, init_with_normal_data, QPSRecallStats,
    },
    index_stats::IndexGraphStats,
};
use crate::{
    deg::{DEGStyleBuilder, LID},
    efanna::EfannaStyleBuilder,
    hsgf::{HSGFLevelGraphBuilder, HSGFStyleBuilder},
    nssg::{NSSGInputGraphData, NSSGInputGraphDataForHSGF, NSSGStyleBuilder},
    rngg::RNGGStyleBuilder,
};
use graphidx::{
    graphs::{DirLoLGraph, Graph},
    indices::GraphIndex,
    measures::SquaredEuclideanDistance,
};
use graphidxbaselines::{
    hnsw::{HNSWParallelHeapBuilder, HNSWStyleBuilder},
    python::GraphStats,
    rnn::RNNStyleBuilder,
};
use ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

#[pyclass]
pub struct PyHNSW {
    index: IndexOneOf<GLIndex<ArrayView2<'static, F>>, GCLIndex<ArrayView2<'static, F>>>,
    max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyHNSW {
    #[new]
    #[pyo3(signature = (data, params, max_frontier_size=None, higher_level_max_heap_size=None))]
    fn new<'py>(
        data: Bound<'py, PyArray2<F>>,
        params: PyHNSWParams,
        max_frontier_size: Option<usize>,
        higher_level_max_heap_size: Option<usize>,
    ) -> Self {
        let params = params.params;
        unsafe {
            let index = HNSWParallelHeapBuilder::build(
                arrview2_py_to_rust(data.as_array()),
                SquaredEuclideanDistance::new(),
                params,
                higher_level_max_heap_size.unwrap_or(1),
            );
            if max_frontier_size.is_none() {
                PyHNSW {
                    index: IndexOneOf::A(index),
                    max_frontier_size: None,
                }
            } else {
                let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
                PyHNSW {
                    index: IndexOneOf::B(capped_index),
                    max_frontier_size: max_frontier_size,
                }
            }
        }
    }
}
generic_graph_index_funs!(layered PyHNSW);

#[pyclass]
pub struct PyRNN {
    index: IndexOneOf<GSIndex<ArrayView2<'static, F>>, GCSIndex<ArrayView2<'static, F>>>,
    max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyRNN {
    #[new]
    #[pyo3(signature = (data, params, max_frontier_size=None))]
    fn new<'py>(
        data: Bound<'py, PyArray2<F>>,
        params: PyRNNParams,
        max_frontier_size: Option<usize>,
    ) -> Self {
        let params = params.params;
        unsafe {
            let index = graphidxbaselines::rnn::RNNDescentBuilder::build(
                arrview2_py_to_rust(data.as_array()),
                SquaredEuclideanDistance::new(),
                params,
            );
            if max_frontier_size.is_none() {
                PyRNN {
                    index: IndexOneOf::A(index),
                    max_frontier_size: None,
                }
            } else {
                let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
                PyRNN {
                    index: IndexOneOf::B(capped_index),
                    max_frontier_size: max_frontier_size,
                }
            }
        }
    }
}

generic_graph_index_funs!(single PyRNN);

/// For PyLID::Unknown it uses a single threaded version
#[pyclass]
pub struct PyDEG {
    index: IndexOneOf<GSIndex<ArrayView2<'static, F>>, GCSIndex<ArrayView2<'static, F>>>,
    max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyDEG {
    #[new]
    #[pyo3(signature = (data, params, max_frontier_size=None))]
    fn new<'py>(
        data: Bound<'py, PyArray2<F>>,
        params: PyDEGParams,
        max_frontier_size: Option<usize>,
    ) -> Self {
        let params = params.params;
        unsafe {
            let index = if params.lid == LID::Unknown {
                /* Use single threaded version because LID::Unknown is not (meant to be) implemented
                for DEGParallelBuilder */
                println!("INFO: Using single-threaded builder because LID::Unknown was specified.");
                crate::deg::single_threaded::DEGBuilder::build(
                    arrview2_py_to_rust(data.as_array()),
                    SquaredEuclideanDistance::new(),
                    params,
                )
            } else {
                crate::deg::DEGParallelBuilder::build(
                    arrview2_py_to_rust(data.as_array()),
                    SquaredEuclideanDistance::new(),
                    params,
                )
            };
            if max_frontier_size.is_none() {
                PyDEG {
                    index: IndexOneOf::A(index),
                    max_frontier_size: None,
                }
            } else {
                let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
                PyDEG {
                    index: IndexOneOf::B(capped_index),
                    max_frontier_size: max_frontier_size,
                }
            }
        }
    }
}
generic_graph_index_funs!(single PyDEG);

#[pyclass]
pub struct PyNSSG {
    index: IndexOneOf<GSIndex<ArrayView2<'static, F>>, GCSIndex<ArrayView2<'static, F>>>,
    max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyNSSG {
    #[new]
    #[pyo3(signature = (data, params, max_frontier_size=None))]
    fn new<'py>(
        data: Bound<'py, PyArray2<F>>,
        params: PyNSSGParams,
        max_frontier_size: Option<usize>,
    ) -> Self {
        unsafe {
            let index = crate::nssg::NSSGParallelBuilder::build(
                arrview2_py_to_rust(data.as_array()),
                SquaredEuclideanDistance::new(),
                params.params,
                match params.input_graph {
                    NSSGInputGraphDataForHSGF::DEG(params) => NSSGInputGraphData::DEG(params),
                    NSSGInputGraphDataForHSGF::EFANNA(params) => NSSGInputGraphData::EFANNA(params),
                    NSSGInputGraphDataForHSGF::GRAPH(graph) => {
                        let mut input_level_graph: DirLoLGraph<R> = DirLoLGraph::new();
                        let n_vertices = graph.adjacency.len();
                        for v in 0..n_vertices {
                            input_level_graph.add_node_with_capacity(graph.adjacency[v].len());
                        }
                        for v in 0..n_vertices {
                            for n in graph.adjacency[v].iter() {
                                input_level_graph.add_edge(v, *n);
                            }
                        }
                        NSSGInputGraphData::GRAPH(input_level_graph)
                    }
                    NSSGInputGraphDataForHSGF::RANDOM(degree) => NSSGInputGraphData::RANDOM(degree),
                    NSSGInputGraphDataForHSGF::RNN(params) => NSSGInputGraphData::RNN(params),
                },
            );

            if max_frontier_size.is_none() {
                PyNSSG {
                    index: IndexOneOf::A(index),
                    max_frontier_size: None,
                }
            } else {
                let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
                PyNSSG {
                    index: IndexOneOf::B(capped_index),
                    max_frontier_size: max_frontier_size,
                }
            }
        }
    }
}
generic_graph_index_funs!(single PyNSSG);

#[pyclass]
pub struct PyEFANNA {
    index: IndexOneOf<GSIndex<ArrayView2<'static, F>>, GCSIndex<ArrayView2<'static, F>>>,
    max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyEFANNA {
    #[new]
    #[pyo3(signature = (data, params, max_frontier_size=None))]
    fn new<'py>(
        data: Bound<'py, PyArray2<F>>,
        params: PyEFANNAParams,
        max_frontier_size: Option<usize>,
    ) -> Self {
        let params = params.params;

        unsafe {
            let index = crate::efanna::EfannaParallelMaxHeapBuilder::build(
                arrview2_py_to_rust(data.as_array()),
                SquaredEuclideanDistance::new(),
                params,
            );

            if max_frontier_size.is_none() {
                PyEFANNA {
                    index: IndexOneOf::A(index),
                    max_frontier_size: None,
                }
            } else {
                let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
                PyEFANNA {
                    index: IndexOneOf::B(capped_index),
                    max_frontier_size: max_frontier_size,
                }
            }
        }
    }
}
generic_graph_index_funs!(single PyEFANNA);

#[pyclass]
pub struct PyRNGG {
    index: IndexOneOf<GSIndex<ArrayView2<'static, F>>, GCSIndex<ArrayView2<'static, F>>>,
    max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyRNGG {
    #[new]
    #[pyo3(signature = (data, params, max_frontier_size=None))]
    fn new<'py>(
        data: Bound<'py, PyArray2<F>>,
        params: PyRNGGParams,
        max_frontier_size: Option<usize>,
    ) -> Self {
        let params = params.params;
        unsafe {
            let index = crate::rngg::RNGGBuilder::build(
                arrview2_py_to_rust(data.as_array()),
                SquaredEuclideanDistance::new(),
                params,
            );

            if max_frontier_size.is_none() {
                PyRNGG {
                    index: IndexOneOf::A(index),
                    max_frontier_size: None,
                }
            } else {
                let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
                PyRNGG {
                    index: IndexOneOf::B(capped_index),
                    max_frontier_size: max_frontier_size,
                }
            }
        }
    }
}
generic_graph_index_funs!(single PyRNGG);

/* level_builders and level_subset_sizes cannot (easily) be moved under (Py)HSGFParams because
the Copy trait cannot be implemented for Vec */
#[pyclass]
pub struct PyHSGF {
    index: IndexOneOf<GLIndex<ArrayView2<'static, F>>, GCLIndex<ArrayView2<'static, F>>>,
    max_frontier_size: Option<usize>,
    level_builder_as_str: String,
}
#[pymethods]
impl PyHSGF {
    #[new]
    #[pyo3(signature = (data, params, level_builders, level_subset_sizes=None,
        max_frontier_size=None, higher_level_max_heap_size=None))]
    fn new<'py>(
        data: Bound<'py, PyArray2<F>>,
        params: PyHSGFParams,
        mut level_builders: Vec<(PyHSGFLevelGraphParams, PySubsetSelector)>,
        level_subset_sizes: Option<Vec<Option<usize>>>,
        max_frontier_size: Option<usize>,
        higher_level_max_heap_size: Option<usize>,
    ) -> Self {
        assert!(
            level_builders.len() >= 1,
            "Supply at least one level_builder."
        );

        let hsgf_params = params.params;

        // local helper
        // TODO next build level_builder_str alongside here
        fn get_level_builder(
            a: &mut PyHSGFLevelGraphParams,
            b: &PySubsetSelector,
            level_builder_as_str: &mut String,
        ) -> HSGFLevelGraphBuilder<R, F> {
            level_builder_as_str.push_str("(");
            let selector: Box<dyn SubsetSelector<R, F>> = match b {
                PySubsetSelector::Flooding(flooding_range) => {
                    level_builder_as_str.push_str(&format!(
                        "FloodingSelectorParallelChunked=\
                    {flooding_range}"
                    ));
                    Box::new(FloodingSelectorParallelChunked::new(*flooding_range))
                }
                PySubsetSelector::FloodingRepeat(flooding_range) => {
                    level_builder_as_str.push_str(&format!(
                        "FloodingSelectorParallelChunkedRepeat=\
                    {flooding_range}"
                    ));
                    Box::new(FloodingSelectorParallelChunkedRepeat::new(*flooding_range))
                }
                PySubsetSelector::Random() => {
                    level_builder_as_str.push_str("Random");
                    Box::new(RandomSelector::new())
                }
                PySubsetSelector::HubNodes(percentage_low_degree_nodes) => {
                    level_builder_as_str.push_str(&format!(
                        "HubNodesSelector:\
                    percentage_low_degree_nodes:{percentage_low_degree_nodes}"
                    ));
                    Box::new(HubNodesSelector::new(*percentage_low_degree_nodes))
                }
            };
            level_builder_as_str.push_str("|");
            let hsgf_level_builder = match a {
                PyHSGFLevelGraphParams::DEG(params) => {
                    let s = params.params_as_str();
                    level_builder_as_str.push_str(&format!("DEG:{s}"));
                    HSGFLevelGraphBuilder::new_deg(params.params, selector)
                }
                PyHSGFLevelGraphParams::EFANNA(params) => {
                    let s = params.params_as_str();
                    level_builder_as_str.push_str(&format!("EFANNA: {s}"));
                    HSGFLevelGraphBuilder::new_efanna(params.params, selector)
                }
                PyHSGFLevelGraphParams::KNN(params) => {
                    let s = params.params_as_str();
                    level_builder_as_str.push_str(&format!("KNN: {s}"));
                    HSGFLevelGraphBuilder::new_knn(params.params, selector)
                }
                PyHSGFLevelGraphParams::NSSG(params) => {
                    let s = params.params_as_str();
                    level_builder_as_str.push_str(&format!("NSSG:{s}"));
                    HSGFLevelGraphBuilder::new_nssg(
                        // NOTE cloning a complete input graph here potentially
                        params.params,
                        selector,
                        params.input_graph.clone(),
                    )
                }
                PyHSGFLevelGraphParams::RNGG(params) => {
                    let s = params.params_as_str();
                    level_builder_as_str.push_str(&format!("RNGG:{s}"));
                    HSGFLevelGraphBuilder::new_rngg(params.params, selector)
                }
                PyHSGFLevelGraphParams::RNN(params) => {
                    let s = params.params_as_str();
                    level_builder_as_str.push_str(&format!("RNN:{s}"));
                    HSGFLevelGraphBuilder::new_rnn(params.params, selector)
                }
                PyHSGFLevelGraphParams::ExistingGraph((py_dir_lol_graph, params_str)) => {
                    level_builder_as_str.push_str(&format!("ExistingGraph:{params_str}"));
                    // This should be slow; Could maybe be parallelized as the
                    // data-write would always be to a unique index?
                    // Same issue as in crate::hsgf for HSGFLevelGraphBuilder::NSSG
                    let mut input_level_graph = DirLoLGraph::new();
                    let n_vertices = py_dir_lol_graph.adjacency.len();
                    for v in 0..n_vertices {
                        input_level_graph
                            .add_node_with_capacity(py_dir_lol_graph.adjacency[v].len());
                    }
                    for v in 0..n_vertices {
                        for n in py_dir_lol_graph.adjacency[v].iter() {
                            input_level_graph.add_edge(v, *n);
                        }
                    }
                    // try to help by freeing up memory after copying all the data
                    py_dir_lol_graph.adjacency = Vec::new();
                    py_dir_lol_graph.n_edges = 0;
                    HSGFLevelGraphBuilder::new_existing_input_graph(input_level_graph, selector)
                }
            };
            level_builder_as_str.push_str(")");
            hsgf_level_builder
        }

        let mut level_builder_as_str = format!(
            "Subset-Sizes:{:?}|\
            higher_level_max_heap_size:{:?}|Level-Builder-Params:[",
            level_subset_sizes, higher_level_max_heap_size
        )
        .to_owned();

        /* Info: If only one level_builder is supplied, it is used for the other
        (possibly remaining) levels as well */
        let level_builders_len = level_builders.len();
        let level_builders: Vec<HSGFLevelGraphBuilder<R, F>> = level_builders[0..]
            .iter_mut()
            .enumerate()
            .map(|(i, (a, b))| {
                let res = get_level_builder(a, b, &mut level_builder_as_str);
                if i + 1 < level_builders_len {
                    level_builder_as_str.push_str("|");
                }
                res
            })
            .collect();
        level_builder_as_str.push_str("]");
        unsafe {
            let data = arrview2_py_to_rust(data.as_array());

            let index = crate::hsgf::HSGFEnumBuilder::build(
                data,
                SquaredEuclideanDistance::new(),
                hsgf_params,
                level_builders,
                level_subset_sizes,
                Some(vec![0]), // always the same entry point
                higher_level_max_heap_size.unwrap_or(1),
            );

            if max_frontier_size.is_none() {
                PyHSGF {
                    index: IndexOneOf::A(index),
                    max_frontier_size: None,
                    level_builder_as_str,
                }
            } else {
                let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
                PyHSGF {
                    index: IndexOneOf::B(capped_index),
                    max_frontier_size: max_frontier_size,
                    level_builder_as_str,
                }
            }
        }
    }
    fn get_level_builder_as_str(&self) -> String {
        self.level_builder_as_str.clone()
    }
}
generic_graph_index_funs!(layered PyHSGF);

#[pyclass]
pub struct PyBruteforceKNN {
    index: IndexOneOf<GSIndex<ArrayView2<'static, F>>, GCSIndex<ArrayView2<'static, F>>>,
    max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyBruteforceKNN {
    #[new]
    #[pyo3(signature = (data, params, max_frontier_size=None))]
    fn new<'py>(
        data: Bound<'py, PyArray2<F>>,
        params: PyBruteforceKNNParams,
        max_frontier_size: Option<usize>,
    ) -> Self {
        let params = params.params;
        unsafe {
            let index = graphidxbaselines::knn::BruteforceKNNGraphBuilder::build(
                arrview2_py_to_rust(data.as_array()),
                SquaredEuclideanDistance::new(),
                params,
            );

            if max_frontier_size.is_none() {
                PyBruteforceKNN {
                    index: IndexOneOf::A(index),
                    max_frontier_size: None,
                }
            } else {
                let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
                PyBruteforceKNN {
                    index: IndexOneOf::B(capped_index),
                    max_frontier_size: max_frontier_size,
                }
            }
        }
    }
}
generic_graph_index_funs!(single PyBruteforceKNN);

// ---------- Hierarchy Index Merger ----------
#[pyfunction]
/// Returns a HSGF graph
/// NOTE currently is cloning the data
#[pyo3(signature = (data, index_bottom_params, index_hierarchy_params, bottom_subset_selector, higher_level_max_heap_size, level_subset_size=None))]
fn merge_hnsw_with_new_bottom_level<'py>(
    data: Bound<'py, PyArray2<F>>,
    index_bottom_params: PyHSGFLevelGraphParams,
    index_hierarchy_params: PyHNSWParams,
    bottom_subset_selector: PySubsetSelector,
    higher_level_max_heap_size: usize,
    level_subset_size: Option<usize>,
) -> PyHSGF {
    unsafe {
        let mut level_builder_as_str = "HNSW-Bottom-Level-Merged:".to_owned();
        let index_bottom = match index_bottom_params {
            PyHSGFLevelGraphParams::DEG(params) => {
                let index = if params.params.lid == LID::Unknown {
                    /* Use single threaded version because LID::Unknown is not implemented
                    for DEGParallelBuilder */
                    crate::deg::single_threaded::DEGBuilder::build(
                        arrview2_py_to_rust(data.clone().as_array()),
                        SquaredEuclideanDistance::new(),
                        params.params,
                    )
                } else {
                    crate::deg::DEGParallelBuilder::build(
                        arrview2_py_to_rust(data.clone().as_array()),
                        SquaredEuclideanDistance::new(),
                        params.params,
                    )
                };
                let s = params.params_as_str();
                level_builder_as_str.push_str(&format!("DEG:{s}"));
                index
            }
            PyHSGFLevelGraphParams::EFANNA(params) => {
                let index = crate::efanna::EfannaParallelMaxHeapBuilder::build(
                    arrview2_py_to_rust(data.clone().as_array()),
                    SquaredEuclideanDistance::new(),
                    params.params,
                );
                let s = params.params_as_str();
                level_builder_as_str.push_str(&format!("EFANNA:{s}"));
                index
            }
            PyHSGFLevelGraphParams::KNN(params) => {
                let index = graphidxbaselines::knn::BruteforceKNNGraphBuilder::build(
                    arrview2_py_to_rust(data.clone().as_array()),
                    SquaredEuclideanDistance::new(),
                    params.params,
                );
                let s = params.params_as_str();
                level_builder_as_str.push_str(&format!("EFANNA:{s}"));
                index
            }
            PyHSGFLevelGraphParams::NSSG(ref params) => {
                let index = crate::nssg::NSSGParallelBuilder::build(
                    arrview2_py_to_rust(data.clone().as_array()),
                    SquaredEuclideanDistance::new(),
                    params.params,
                    match &params.input_graph {
                        NSSGInputGraphDataForHSGF::DEG(params) => NSSGInputGraphData::DEG(*params),
                        NSSGInputGraphDataForHSGF::EFANNA(params) => {
                            NSSGInputGraphData::EFANNA(*params)
                        }
                        NSSGInputGraphDataForHSGF::GRAPH(graph) => {
                            let mut input_level_graph: DirLoLGraph<R> = DirLoLGraph::new();
                            let n_vertices = graph.adjacency.len();
                            for v in 0..n_vertices {
                                input_level_graph.add_node_with_capacity(graph.adjacency[v].len());
                            }
                            for v in 0..n_vertices {
                                for n in graph.adjacency[v].iter() {
                                    input_level_graph.add_edge(v, *n);
                                }
                            }
                            NSSGInputGraphData::GRAPH(input_level_graph)
                        }
                        NSSGInputGraphDataForHSGF::RANDOM(degree) => {
                            NSSGInputGraphData::RANDOM(*degree)
                        }
                        NSSGInputGraphDataForHSGF::RNN(params) => NSSGInputGraphData::RNN(*params),
                    },
                );
                let s = params.params_as_str();
                level_builder_as_str.push_str(&format!("NSSG:{s}"));
                index
            }
            PyHSGFLevelGraphParams::RNGG(params) => {
                let index = crate::rngg::RNGGBuilder::build(
                    arrview2_py_to_rust(data.clone().as_array()),
                    SquaredEuclideanDistance::new(),
                    params.params,
                );
                let s = params.params_as_str();
                level_builder_as_str.push_str(&format!("RNGG:{s}"));
                index
            }
            PyHSGFLevelGraphParams::RNN(params) => {
                let index = graphidxbaselines::rnn::RNNDescentBuilder::build(
                    arrview2_py_to_rust(data.clone().as_array()),
                    SquaredEuclideanDistance::new(),
                    params.params,
                );
                let s = params.params_as_str();
                level_builder_as_str.push_str(&format!("RNN:{s}"));
                index
            }
            PyHSGFLevelGraphParams::ExistingGraph((_, _)) => {
                unimplemented!()
            }
        };

        level_builder_as_str.push_str("|");
        let bottom_subset_selector: Box<dyn SubsetSelector<R, F>> = match bottom_subset_selector {
            PySubsetSelector::Flooding(flooding_range) => {
                level_builder_as_str.push_str(&format!(
                    "FloodingSelectorParallelChunked=\
                {flooding_range}"
                ));
                Box::new(FloodingSelectorParallelChunked::new(flooding_range))
            }
            PySubsetSelector::FloodingRepeat(flooding_range) => {
                level_builder_as_str.push_str(&format!(
                    "FloodingSelectorParallelChunkedRepeat=\
                {flooding_range}"
                ));
                Box::new(FloodingSelectorParallelChunkedRepeat::new(flooding_range))
            }
            PySubsetSelector::Random() => {
                level_builder_as_str.push_str("Random");
                Box::new(RandomSelector::new())
            }
            PySubsetSelector::HubNodes(percentage_low_degree_nodes) => {
                level_builder_as_str.push_str(&format!(
                    "HubNodesSelector:\
                percentage_low_degree_nodes:{percentage_low_degree_nodes}"
                ));
                Box::new(HubNodesSelector::new(percentage_low_degree_nodes))
            }
        };

        let s = index_hierarchy_params.params_as_str();
        level_builder_as_str.push_str(&format!("|HNSW:{s}"));

        let index = crate::hsgf::hierarchy_merger::merge_hnsw_with_new_bottom_level(
            arrview2_py_to_rust(data.as_array()),
            index_bottom,
            index_hierarchy_params.params,
            bottom_subset_selector,
            higher_level_max_heap_size,
            level_subset_size,
        );
        PyHSGF {
            index: IndexOneOf::A(index),
            max_frontier_size: None,
            level_builder_as_str: level_builder_as_str,
        }
    }
}

// ---------- UTIL ----------

#[pyfunction]
#[pyo3(signature = (nd, nq, d, k, indexed_queries))]
fn load_normal_data<'py>(
    py: Python<'py>,
    nd: usize,
    nq: usize,
    d: usize,
    k: usize,
    indexed_queries: bool,
) -> (
    Bound<'py, PyArray2<F>>,
    Bound<'py, PyArray2<F>>,
    Bound<'py, PyArray2<R>>,
    Bound<'py, PyArray2<F>>,
) {
    let (data, queries) = init_with_normal_data(nd, nq, d, indexed_queries);
    let (ground_truth, dists) = graphidx::indices::bruteforce_neighbors(
        &data,
        &queries,
        &SquaredEuclideanDistance::new(),
        k,
    );
    (
        PyArray2::from_owned_array(py, arr2_rust_to_py(data)),
        PyArray2::from_owned_array(py, arr2_rust_to_py(queries)),
        PyArray2::from_owned_array(py, arr2_rust_to_py(ground_truth)),
        PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
    )
}

#[pyfunction]
fn bruteforce_groundtruth<'py>(
    py: Python<'py>,
    data: Bound<'py, PyArray2<F>>,
    queries: Bound<'py, PyArray2<F>>,
    k: usize,
) -> (Bound<'py, PyArray2<R>>, Bound<'py, PyArray2<F>>) {
    unsafe {
        let (ids, dists) = graphidx::indices::bruteforce_neighbors(
            &arrview2_py_to_rust(data.as_array()),
            &arrview2_py_to_rust(queries.as_array()),
            &SquaredEuclideanDistance::new(),
            k,
        );
        (
            PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
            PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
        )
    }
}

// ---------- Exports ----------

#[pymodule(name = "hsgf")]
fn hsgf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Helper enums and structs
    m.add_class::<PyDirLoLGraph>()?;
    m.add_class::<PyLID>()?;
    m.add_class::<PyKeepEdges>()?;
    m.add_class::<PyNSSGInputGraphDataForHSGF>()?;
    m.add_class::<PySubsetSelector>()?;
    m.add_class::<PyHSGFLevelGraphParams>()?;
    m.add_class::<PyQPSRecallStats>()?;

    // Graph builder parameters
    m.add_class::<PyDEGParams>()?;
    m.add_class::<PyEFANNAParams>()?;
    m.add_class::<PyHSGFParams>()?;
    m.add_class::<PyNSSGParams>()?;
    m.add_class::<PyRNGGParams>()?;
    m.add_class::<PyRNNParams>()?; // reexport
    m.add_class::<PyHNSWParams>()?; // reexport
    m.add_class::<PyBruteforceKNNParams>()?; // "reexport"

    // Graph builders
    m.add_class::<PyDEG>()?;
    m.add_class::<PyEFANNA>()?;
    m.add_class::<PyHSGF>()?;
    m.add_class::<PyNSSG>()?;
    m.add_class::<PyRNGG>()?;
    m.add_class::<PyRNN>()?; // wrapped reexport
    m.add_class::<PyHNSW>()?; // wrapped reexport
    m.add_class::<PyBruteforceKNN>()?; // "wrapped reexport"

    // Hierarchy Merger
    m.add_function(wrap_pyfunction!(merge_hnsw_with_new_bottom_level, m)?)?;

    // Util functions
    m.add_function(wrap_pyfunction!(load_normal_data, m)?)?;
    m.add_function(wrap_pyfunction!(bruteforce_groundtruth, m)?)?;
    Ok(())
}
