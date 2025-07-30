//! Params and helper structs for py_hsgf
//!
//! Notes
//!     - A lot of fields on the structs are public because they are in this file instead
//!       of directly in py_hsgf
//!         - However, these are only structures used for the python binding anyway..
//!
//! Open Issues
//!     - The formatting via `cargo fmt` might sometimes need to be triggered manually because
//!       of the pyo3 annotations
//!
//! TODOs
//!
use crate::py::py_util::*;
use crate::{
    deg::{DEGParams, LID},
    efanna::EfannaParams,
    hsgf::HSGFParams,
    nssg::{NSSGInputDirLolGraph, NSSGInputGraphDataForHSGF, NSSGParams},
    rngg::{KeepEdges, RNGGParams},
};
use graphidxbaselines::{hnsw::HNSWParams, knn::BruteforceKNNParams, rnn::RNNParams};
use pyo3::prelude::*;

// ------------ Python Graph Builder Parameters ------------
// The parameters need to be accessible as own objects for the HSGF builder
// Additionally they allow nicer separation between the builder and its params

#[pyclass]
#[derive(Copy, Clone)]
pub struct PyDEGParams {
    pub params: DEGParams,
}
#[pymethods]
impl PyDEGParams {
    #[new]
    #[pyo3(signature = (edges_per_vertex=None,  max_build_heap_size=None,
        max_build_frontier_size=None, extend_k=None, improve_k=None, extend_eps=None,
        improve_eps=None, lid=None, swap_tries=None, additional_swap_tries=None,
        use_range_search=None,
    ))]
    fn new<'py>(
        edges_per_vertex: Option<usize>,
        max_build_heap_size: Option<usize>,
        max_build_frontier_size: Option<usize>,
        extend_k: Option<usize>,
        improve_k: Option<usize>,
        extend_eps: Option<f32>,
        improve_eps: Option<f32>,
        lid: Option<PyLID>,
        swap_tries: Option<usize>,
        additional_swap_tries: Option<usize>,
        use_range_search: Option<bool>,
    ) -> Self {
        // Note that we use LID::HIGH as a default which is an opinionated choice
        // based on observed faster construction times
        let lid = if lid.is_some() {
            match lid.unwrap() {
                PyLID::Unknown => Some(LID::Unknown),
                PyLID::Low => Some(LID::Low),
                PyLID::High => Some(LID::High),
            }
        } else {
            Some(LID::High)
        };
        let params = crate::deg::DEGParams::new()
            .maybe_with_edges_per_vertex(edges_per_vertex)
            .maybe_with_max_build_heap_size(max_build_heap_size)
            .with_max_build_frontier_size(max_build_frontier_size)
            .maybe_with_extend_k(extend_k)
            .maybe_with_improve_k(improve_k)
            .maybe_with_extend_eps(extend_eps)
            .maybe_with_improve_eps(improve_eps)
            .maybe_with_lid(lid)
            .maybe_with_additional_swap_tries(additional_swap_tries)
            .maybe_with_swap_tries(swap_tries)
            .maybe_with_use_range_search(use_range_search);
        PyDEGParams { params }
    }
    pub fn params_as_str(&self) -> String {
        let s = format!(
            "edges_per_vertex={}|max_build_heap_size={}|\
            max_build_frontier_size={:?}|extend_k={}|extend_eps={}|improve_k={}|\
            improve_eps={}|lid={:?}|additional_swap_tries={}|swap_tries={}|use_range_search={:?}",
            self.params.edges_per_vertex,
            self.params.max_build_heap_size,
            self.params.max_build_frontier_size,
            self.params.extend_k,
            self.params.extend_eps,
            self.params.improve_k,
            self.params.improve_eps,
            self.params.lid,
            self.params.additional_swap_tries,
            self.params.swap_tries,
            self.params.use_range_search,
        );
        s
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyNSSGParams {
    pub params: NSSGParams<F>,
    /* Using the NSSGInputGraphDataForHSGF as intermediate wrapper till unwrapped
    again in PyNSSG because pyo3 objects do not allow generics which would be needed
    for nssg::NSSGInputGraphData */
    pub input_graph: NSSGInputGraphDataForHSGF,
    pub input_graph_params: PyNSSGInputGraphDataForHSGF,
    pub input_graph_params_str: String,
}
#[pymethods]
impl PyNSSGParams {
    #[new]
    #[pyo3(signature = (input_graph, range=None, l=None, angle=None, n_try=None,
         derive_angle_from_dim=None))]
    fn new<'py>(
        input_graph: PyNSSGInputGraphDataForHSGF,
        range: Option<usize>,
        l: Option<usize>,
        angle: Option<F>,
        n_try: Option<usize>,
        derive_angle_from_dim: Option<bool>,
    ) -> Self {
        let params = crate::nssg::NSSGParams::new()
            .maybe_with_range(range)
            .maybe_with_l(l)
            .maybe_with_angle(angle)
            .maybe_with_n_try(n_try)
            .maybe_with_derive_angle_from_dim(derive_angle_from_dim);

        let mut input_graph_params_str = "".to_owned();

        let nssg_input_graph = match input_graph {
            PyNSSGInputGraphDataForHSGF::DEG(py_input_params) => {
                input_graph_params_str.push_str("DEG:");
                let params = if py_input_params.is_some() {
                    let py_params: PyDEGParams = py_input_params.unwrap();
                    input_graph_params_str.push_str(&py_params.params_as_str());
                    Some(py_params.params)
                } else {
                    // Just creating this to get a string of the default parameters
                    let py_params = PyDEGParams::new(
                        None, None, None, None, None, None, None, None, None, None, None,
                    );
                    input_graph_params_str.push_str(&py_params.params_as_str());
                    None
                };
                NSSGInputGraphDataForHSGF::DEG(params)
            }
            PyNSSGInputGraphDataForHSGF::EFANNA(py_input_params) => {
                input_graph_params_str.push_str("EFANNA:");
                let params = if py_input_params.is_some() {
                    let py_params: PyEFANNAParams = py_input_params.unwrap();
                    input_graph_params_str.push_str(&py_params.params_as_str());
                    Some(py_params.params)
                } else {
                    // Just creating this to get a string of the default parameters
                    let py_params = PyEFANNAParams::new(None, None, None, None, None);
                    input_graph_params_str.push_str(&py_params.params_as_str());
                    None
                };
                NSSGInputGraphDataForHSGF::EFANNA(params)
            }
            PyNSSGInputGraphDataForHSGF::GRAPH(ref graph) => {
                // not an elegant nor efficient solution
                // but currently it is also not an important functionality
                input_graph_params_str.push_str("Some Graph");
                let mut input_level_graph: NSSGInputDirLolGraph = NSSGInputDirLolGraph::new();
                input_level_graph.adjacency = graph.adjacency.clone();
                input_level_graph.n_edges = graph.n_edges;
                NSSGInputGraphDataForHSGF::GRAPH(input_level_graph)
            }
            PyNSSGInputGraphDataForHSGF::RANDOM(degree) => {
                input_graph_params_str
                    .push_str(&format!("Random-Initialization with degree:{degree}"));
                NSSGInputGraphDataForHSGF::RANDOM(degree)
            }
            PyNSSGInputGraphDataForHSGF::RNN(py_input_params) => {
                input_graph_params_str.push_str("RNN:");
                let params = if py_input_params.is_some() {
                    let py_params: PyRNNParams = py_input_params.unwrap();
                    input_graph_params_str.push_str(&py_params.params_as_str());
                    Some(py_params.params)
                } else {
                    // Just creating this to get a string of the default parameters
                    let py_params = PyRNNParams::new(None, None, None, None, None);
                    input_graph_params_str.push_str(&py_params.params_as_str());
                    None
                };
                NSSGInputGraphDataForHSGF::RNN(params)
            }
        };
        PyNSSGParams {
            params,
            input_graph: nssg_input_graph,
            input_graph_params: input_graph,
            input_graph_params_str,
        }
    }
    pub fn params_as_str(&self) -> String {
        let s = format!(
            "range={}|l={}|angle={}|n_try={}|derive_angle_from_dim:{}|Input-graph params:{}",
            self.params.range,
            self.params.l,
            self.params.angle,
            self.params.n_try,
            self.params.derive_angle_from_dim,
            self.input_graph_params_str.clone()
        );
        s
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct PyEFANNAParams {
    pub params: EfannaParams,
}
#[pymethods]
impl PyEFANNAParams {
    #[new]
    #[pyo3(signature = (k=None, l=None, iter=None, s=None, r=None))]
    fn new<'py>(
        k: Option<usize>,
        l: Option<usize>,
        iter: Option<usize>,
        s: Option<usize>,
        r: Option<usize>,
    ) -> Self {
        let params = crate::efanna::EfannaParams::new()
            .maybe_with_k(k)
            .maybe_with_l(l)
            .maybe_with_iter(iter)
            .maybe_with_s(s)
            .maybe_with_r(r);
        PyEFANNAParams { params }
    }
    pub fn params_as_str(&self) -> String {
        let s = format!(
            "k={}|l={}|iter={}|s={}|r={}",
            self.params.k, self.params.l, self.params.iter, self.params.s, self.params.r
        );
        s
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct PyRNGGParams {
    pub params: RNGGParams,
}
#[pymethods]
impl PyRNGGParams {
    #[new]
    #[pyo3(signature = (degree=None, degree_short_edges=None,
        prune_extend_degree=None, prune=None))]
    fn new<'py>(
        degree: Option<usize>,
        degree_short_edges: Option<usize>,
        prune_extend_degree: Option<usize>,
        prune: Option<PyKeepEdges>,
    ) -> Self {
        let prune = if prune.is_some() {
            match prune.unwrap() {
                PyKeepEdges::SHORT => Some(KeepEdges::SHORT),
                PyKeepEdges::MIDDLE => Some(KeepEdges::MIDDLE),
                PyKeepEdges::LONG => Some(KeepEdges::LONG),
                PyKeepEdges::SPECIAL => Some(KeepEdges::SPECIAL),
                PyKeepEdges::NONE => Some(KeepEdges::NONE),
            }
        } else {
            Some(KeepEdges::NONE)
        };
        let params = crate::rngg::RNGGParams::new()
            .maybe_with_degree(degree)
            .maybe_with_degree_short_edges(degree_short_edges)
            .maybe_with_prune_extend_degree(prune_extend_degree)
            .maybe_with_prune(prune);
        PyRNGGParams { params }
    }
    pub fn params_as_str(&self) -> String {
        let s = format!(
            "degree={}|degree_short_edges={}|prune_extend_degree={}|prune={:?}",
            self.params.degree,
            self.params.degree_short_edges,
            self.params.prune_extend_degree,
            self.params.prune
        );
        s
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct PyRNNParams {
    pub params: RNNParams,
}
#[pymethods]
impl PyRNNParams {
    #[new]
    #[pyo3(signature = (initial_degree=None, reduce_degree=None, n_outer_loops=None,
        n_inner_loops=None, concurrent_batch_size=None))]
    fn new<'py>(
        initial_degree: Option<usize>,
        reduce_degree: Option<usize>,
        n_outer_loops: Option<usize>,
        n_inner_loops: Option<usize>,
        concurrent_batch_size: Option<usize>,
    ) -> Self {
        let params = graphidxbaselines::rnn::RNNParams::new()
            .maybe_with_initial_degree(initial_degree)
            .maybe_with_reduce_degree(reduce_degree)
            .maybe_with_n_outer_loops(n_outer_loops)
            .maybe_with_n_inner_loops(n_inner_loops)
            .maybe_with_concurrent_batch_size(concurrent_batch_size);
        PyRNNParams { params }
    }
    pub fn params_as_str(&self) -> String {
        let s = format!(
            "initial_degree={}|reduce_degree={}|n_outer_loops={}|\
        n_inner_loops={}|concurrent_batch_size={}",
            self.params.initial_degree,
            self.params.reduce_degree,
            self.params.n_outer_loops,
            self.params.n_inner_loops,
            self.params.concurrent_batch_size
        );
        s
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct PyHNSWParams {
    pub params: HNSWParams<F>,
}
#[pymethods]
impl PyHNSWParams {
    #[new]
    #[pyo3(signature = (higher_max_degree=None, lowest_max_degree=None, max_layers=None,
         n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None,
         level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None,
         post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None,
         finetune_sen=None))]
    fn new<'py>(
        higher_max_degree: Option<usize>,
        lowest_max_degree: Option<usize>,
        max_layers: Option<usize>,
        n_parallel_burnin: Option<usize>,
        max_build_heap_size: Option<usize>,
        max_build_frontier_size: Option<usize>,
        level_norm_param_override: Option<f32>,
        insert_heuristic: Option<bool>,
        insert_heuristic_extend: Option<bool>,
        post_prune_heuristic: Option<bool>,
        insert_minibatch_size: Option<usize>,
        n_rounds: Option<usize>,
        finetune_rnn: Option<bool>,
        finetune_sen: Option<bool>,
    ) -> Self {
        let params = HNSWParams::new()
            .maybe_with_higher_max_degree(higher_max_degree)
            .maybe_with_lowest_max_degree(lowest_max_degree)
            .maybe_with_max_layers(max_layers)
            .maybe_with_n_parallel_burnin(n_parallel_burnin)
            .maybe_with_max_build_heap_size(max_build_heap_size)
            .with_max_build_frontier_size(max_build_frontier_size)
            .with_level_norm_param_override(level_norm_param_override)
            .maybe_with_insert_heuristic(insert_heuristic)
            .maybe_with_insert_heuristic_extend(insert_heuristic_extend)
            .maybe_with_post_prune_heuristic(post_prune_heuristic)
            .maybe_with_insert_minibatch_size(insert_minibatch_size)
            .maybe_with_n_rounds(n_rounds)
            .maybe_with_finetune_rnn(finetune_rnn)
            .maybe_with_finetune_sen(finetune_sen);
        PyHNSWParams { params: params }
    }
    pub fn params_as_str(&self) -> String {
        let s = format!(
            "higher_max_degree={}|lowest_max_degree={}|max_layers={}|\
        n_parallel_burnin={}|max_build_heap_size={}|max_build_frontier_size={:?}|\
        level_norm_param_override={:?}|insert_heuristic={}|insert_heuristic_extend={}|\
        post_prune_heuristic={}|insert_minibatch_size={}|n_rounds={}|finetune_rnn={}|\
        finetune_sen={}",
            self.params.higher_max_degree,
            self.params.lowest_max_degree,
            self.params.max_layers,
            self.params.n_parallel_burnin,
            self.params.max_build_heap_size,
            self.params.max_build_frontier_size,
            self.params.level_norm_param_override,
            self.params.insert_heuristic,
            self.params.insert_heuristic_extend,
            self.params.post_prune_heuristic,
            self.params.insert_minibatch_size,
            self.params.n_rounds,
            self.params.finetune_rnn,
            self.params.finetune_sen
        );
        s
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct PyHSGFParams {
    pub params: HSGFParams,
}
#[pymethods]
impl PyHSGFParams {
    #[new]
    #[pyo3(signature = (min_layers=None, max_layers=None, higher_max_degree=None,
        min_n_vertices_layer=None,level_norm_param_override=None,
        create_artificial_hub_nodes=None))]
    fn new<'py>(
        min_layers: Option<usize>,
        max_layers: Option<usize>,
        higher_max_degree: Option<usize>,
        min_n_vertices_layer: Option<usize>,
        level_norm_param_override: Option<f32>,
        create_artificial_hub_nodes: Option<usize>,
    ) -> Self {
        let params = HSGFParams::new()
            .maybe_with_min_layers(min_layers)
            .maybe_with_max_layers(max_layers)
            .maybe_with_higher_max_degree(higher_max_degree)
            .maybe_with_min_n_vertices_layer(min_n_vertices_layer)
            .with_level_norm_param_override(level_norm_param_override)
            .with_create_artificial_hub_nodes(create_artificial_hub_nodes);

        PyHSGFParams { params }
    }
    pub fn params_as_str(&self) -> String {
        let s = format!(
            "min_layers={}|max_layers={}|higher_max_degree={}|min_n_vertices_layer={}|\
            level_norm_param_override={:?}|create_artificial_hub_nodes={:?}",
            self.params.min_layers,
            self.params.max_layers,
            self.params.higher_max_degree,
            self.params.min_n_vertices_layer,
            self.params.level_norm_param_override,
            self.params.create_artificial_hub_nodes,
        );
        s
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct PyBruteforceKNNParams {
    pub params: BruteforceKNNParams,
}
#[pymethods]
impl PyBruteforceKNNParams {
    #[new]
    #[pyo3(signature = (degree=None, batch_size=None))]
    fn new<'py>(degree: Option<usize>, batch_size: Option<usize>) -> Self {
        let params = BruteforceKNNParams::new()
            .maybe_with_degree(degree)
            .maybe_with_batch_size(batch_size);

        PyBruteforceKNNParams { params }
    }
    pub fn params_as_str(&self) -> String {
        let s = format!(
            "degree={}|batch_size={}",
            self.params.degree, self.params.batch_size,
        );
        s
    }
}

// ------------ Helper Structs ------------

// Among others a PyIndex can now return a PyDirLoLGraph
// which then can be used as input to HSGF via ExistingGraph
#[pyclass]
#[derive(Clone)]
pub struct PyDirLoLGraph {
    pub adjacency: Vec<Vec<R>>,
    pub n_edges: usize,
}
#[pymethods]
impl PyDirLoLGraph {
    #[new]
    #[pyo3(signature = (adjacency, n_edges))]
    pub fn new<'py>(adjacency: Vec<Vec<R>>, n_edges: usize) -> Self {
        PyDirLoLGraph { adjacency, n_edges }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Copy, Clone, PartialEq)]
pub enum PyLID {
    Unknown,
    Low,
    High,
}

#[pyclass(eq, eq_int)]
#[derive(Copy, Clone, PartialEq)]
pub enum PyKeepEdges {
    SHORT,
    MIDDLE,
    LONG,
    SPECIAL,
    NONE,
}

#[pyclass]
#[derive(Clone)]
pub enum PyNSSGInputGraphDataForHSGF {
    DEG(Option<PyDEGParams>),
    EFANNA(Option<PyEFANNAParams>),
    GRAPH(PyDirLoLGraph), // currently bad memory usage; avoid for large datasets
    RANDOM(usize),
    RNN(Option<PyRNNParams>),
}

#[pyclass]
#[derive(Clone)]
pub enum PyHSGFLevelGraphParams {
    DEG(PyDEGParams),
    EFANNA(PyEFANNAParams),
    NSSG(PyNSSGParams),
    RNGG(PyRNGGParams),
    RNN(PyRNNParams),
    KNN(PyBruteforceKNNParams),
    ExistingGraph((PyDirLoLGraph, String)),
}

#[pyclass]
#[derive(Clone, Copy)]
pub enum PySubsetSelector {
    Flooding(usize),
    FloodingRepeat(usize),
    HubNodes(usize),
    Random(),
}
