//! Utils for py_hsgf
//!  
//! Big parts are copied from GraphIndexBaselines::python to have it local and avoid changing
//! everything to public there.
//!
//! Notes
//!
//! Open Issues
//!
//! TODOs
//!
use graphidx::{
    graphs::{DirLoLGraph, FatDirGraph, Graph},
    indices::{
        GraphIndex, GreedyCappedLayeredGraphIndex, GreedyCappedSingleGraphIndex,
        GreedyLayeredGraphIndex, GreedySingleGraphIndex,
    },
    measures::SquaredEuclideanDistance,
    types::UnsignedInteger,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
#[allow(unused)]
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

// specifying the "fixed" type at one place
pub type R = usize;
pub type F = f32;

/* Conversion code to handle different ndarray versions in this crate and numpy dependencies */
pub fn arr1_rust_to_py<T>(arr: Array1<T>) -> numpy::ndarray::Array1<T> {
    numpy::ndarray::Array1::from_vec(arr.into_raw_vec())
}
pub fn arr2_rust_to_py<T>(arr: Array2<T>) -> numpy::ndarray::Array2<T> {
    let shape = arr.shape();
    unsafe {
        numpy::ndarray::Array2::from_shape_vec_unchecked((shape[0], shape[1]), arr.into_raw_vec())
    }
}
pub fn arrview1_py_to_rust<T>(arr: numpy::ndarray::ArrayView1<T>) -> ArrayView1<'static, T> {
    let shape = arr.shape();
    unsafe { ArrayView1::from_shape_ptr((shape[0],), arr.as_ptr() as *const T) }
}
pub fn arrview2_py_to_rust<T>(arr: numpy::ndarray::ArrayView2<T>) -> ArrayView2<'static, T> {
    let shape = arr.shape();
    unsafe { ArrayView2::from_shape_ptr((shape[0], shape[1]), arr.as_ptr() as *const T) }
}

#[pyclass]
pub struct GraphStats {
    #[pyo3(get)]
    pub n_nodes: usize,
    #[pyo3(get)]
    pub n_edges: usize,
    #[pyo3(get)]
    pub max_degree: usize,
    #[pyo3(get)]
    pub min_degree: usize,
    #[pyo3(get)]
    pub avg_degree: f64,
    #[pyo3(get)]
    pub std_degree: f64,
}
impl GraphStats {
    pub fn from_graph<R: UnsignedInteger, G: Graph<R>>(g: &G) -> Self {
        let n_nodes = g.n_vertices();
        let (mut n_edges, mut max_degree, mut min_degree, mut ssq_degree) = (0, 0, usize::MAX, 0);
        (0..n_nodes).for_each(|i| {
            let degree = g.degree(R::from(i).unwrap());
            n_edges += degree;
            max_degree = max_degree.max(degree);
            min_degree = min_degree.min(degree);
            ssq_degree += degree * degree;
        });
        let mssq_degree = ssq_degree as f64 / n_nodes as f64;
        let avg_degree = n_edges as f64 / n_nodes as f64;
        let var_degree = mssq_degree - avg_degree * avg_degree;
        let std_degree = var_degree.sqrt();
        Self {
            n_nodes: n_nodes,
            n_edges: n_edges,
            max_degree: max_degree,
            min_degree: min_degree,
            avg_degree: avg_degree,
            std_degree: std_degree,
        }
    }
}

pub type GSIndex<M> =
    GreedySingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, DirLoLGraph<usize>>;
pub type GCSIndex<M> =
    GreedyCappedSingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, DirLoLGraph<usize>>;
// pub type FGSIndex<M> =
//  GreedySingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, FatDirGraph<usize>>;
// pub type FGCSIndex<M> =
//  GreedyCappedSingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, FatDirGraph<usize>>;
pub type GLIndex<M> =
    GreedyLayeredGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, DirLoLGraph<usize>>;
pub type GCLIndex<M> =
    GreedyCappedLayeredGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, DirLoLGraph<usize>>;
pub type FGLIndex<M> =
    GreedyLayeredGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, FatDirGraph<usize>>;
pub type FGCLIndex<M> =
    GreedyCappedLayeredGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, FatDirGraph<usize>>;

pub enum IndexOneOf<
    A: GraphIndex<R, F, SquaredEuclideanDistance<F>>,
    B: GraphIndex<R, F, SquaredEuclideanDistance<F>>,
> {
    A(A),
    B(B),
    None,
}
#[allow(dead_code)]
impl<
        A: GraphIndex<R, F, SquaredEuclideanDistance<F>>,
        B: GraphIndex<R, F, SquaredEuclideanDistance<F>>,
    > IndexOneOf<A, B>
{
    pub fn greedy_search(
        &self,
        queries: &ArrayView1<F>,
        k: R,
        max_heap_size: R,
    ) -> (Array1<R>, Array1<F>) {
        match self {
            IndexOneOf::A(a) => a.greedy_search(
                queries,
                k,
                max_heap_size,
                &mut a._new_search_cache(max_heap_size),
            ),
            IndexOneOf::B(b) => b.greedy_search(
                queries,
                k,
                max_heap_size,
                &mut b._new_search_cache(max_heap_size),
            ),
            IndexOneOf::None => panic!(),
        }
    }
    pub fn greedy_search_batch(
        &self,
        queries: &ArrayView2<F>,
        k: R,
        max_heap_size: R,
    ) -> (Array2<R>, Array2<F>) {
        match self {
            IndexOneOf::A(a) => a.greedy_search_batch(queries, k, max_heap_size),
            IndexOneOf::B(b) => b.greedy_search_batch(queries, k, max_heap_size),
            IndexOneOf::None => panic!(),
        }
    }
    pub fn as_a(&self) -> Option<&A> {
        match self {
            IndexOneOf::A(a) => Some(a),
            IndexOneOf::B(_) => None,
            IndexOneOf::None => None,
        }
    }
    pub fn as_b(&self) -> Option<&B> {
        match self {
            IndexOneOf::A(_) => None,
            IndexOneOf::B(b) => Some(b),
            IndexOneOf::None => None,
        }
    }
    pub fn as_a_mut(&mut self) -> Option<&mut A> {
        match self {
            IndexOneOf::A(a) => Some(a),
            IndexOneOf::B(_) => None,
            IndexOneOf::None => None,
        }
    }
    pub fn as_b_mut(&mut self) -> Option<&mut B> {
        match self {
            IndexOneOf::A(_) => None,
            IndexOneOf::B(b) => Some(b),
            IndexOneOf::None => None,
        }
    }
    pub fn is_a(&self) -> bool {
        match self {
            IndexOneOf::A(_) => true,
            IndexOneOf::B(_) => false,
            IndexOneOf::None => false,
        }
    }
    pub fn is_b(&self) -> bool {
        match self {
            IndexOneOf::A(_) => false,
            IndexOneOf::B(_) => true,
            IndexOneOf::None => false,
        }
    }
    pub fn into_a<F: FnOnce(B) -> A>(self, fun: F) -> Self {
        match self {
            IndexOneOf::A(_) => self,
            IndexOneOf::B(b) => IndexOneOf::A(fun(b)),
            IndexOneOf::None => self,
        }
    }
    pub fn into_b<F: FnOnce(A) -> B>(self, fun: F) -> Self {
        match self {
            IndexOneOf::A(a) => IndexOneOf::B(fun(a)),
            IndexOneOf::B(_) => self,
            IndexOneOf::None => self,
        }
    }
}

#[macro_export]
macro_rules! generic_graph_index_funs {
    ($type: ident) => {
        #[pymethods]
        impl $type {
            #[pyo3(signature = (query, k, max_heap_size=None))]
            pub fn knn_query<'py>(
                &self,
                py: Python<'py>,
                query: Bound<'py, PyArray1<F>>,
                k: usize,
                max_heap_size: Option<usize>,
            ) -> (Bound<'py, PyArray1<R>>, Bound<'py, PyArray1<F>>) {
                unsafe {
                    let (ids, dists) = self.index.greedy_search(
                        &arrview1_py_to_rust(query.as_array()),
                        k,
                        max_heap_size.unwrap_or(2 * k),
                    );
                    (
                        PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
                        PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
                    )
                }
            }
            #[pyo3(signature = (queries, k, max_heap_size=None))]
            pub fn knn_query_batch<'py>(
                &self,
                py: Python<'py>,
                queries: Bound<'py, PyArray2<F>>,
                k: usize,
                max_heap_size: Option<usize>,
            ) -> (Bound<'py, PyArray2<R>>, Bound<'py, PyArray2<F>>) {
                unsafe {
                    let (ids, dists) = self.index.greedy_search_batch(
                        &arrview2_py_to_rust(queries.as_array()),
                        k,
                        max_heap_size.unwrap_or(2 * k),
                    );
                    (
                        PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
                        PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
                    )
                }
            }
            #[getter]
            pub fn get_max_frontier_size(&self) -> Option<usize> {
                self.max_frontier_size
            }
            #[setter]
            pub fn set_max_frontier_size(&mut self, max_frontier_size: Option<usize>) {
                self.max_frontier_size = max_frontier_size;
                if self.max_frontier_size.is_none() {
                    if self.index.is_b() {
                        let mut index = IndexOneOf::None;
                        std::mem::swap(&mut self.index, &mut index);
                        let mut index = index.into_a(|index| index.into_uncapped());
                        std::mem::swap(&mut self.index, &mut index);
                    }
                } else {
                    if self.index.is_a() {
                        let mut index = IndexOneOf::None;
                        std::mem::swap(&mut self.index, &mut index);
                        let mut index =
                            index.into_b(|index| index.into_capped(max_frontier_size.unwrap()));
                        std::mem::swap(&mut self.index, &mut index);
                    } else {
                        self.index
                            .as_b_mut()
                            .unwrap()
                            .set_max_frontier_size(max_frontier_size.unwrap());
                    }
                }
            }
            pub fn get_index_stats(&self, reduced_stats: bool) -> PyIndexStats {
                match &self.index {
                    IndexOneOf::A(index) => {
                        let (_reported_edges, counted_edges_in_adj, _hashset_counted_edges) =
                            index.get_n_edges();

                        if !reduced_stats {
                            PyIndexStats {
                                n_vertices: index.get_graph_n_vertices(),
                                n_edges: counted_edges_in_adj,
                                avg_out_degrees: index.get_avg_out_degrees(),
                                min_max_out_degrees: index.get_min_max_out_degrees(),
                                total_avg_out_degree: index.get_total_avg_out_degree(),
                                avg_1nn_distance: index.get_avg_1nn_distance(),
                                avg_1nn_distances: index.get_avg_1nn_distances(),
                                min_max_1nn_distances: index.get_min_max_1nn_distances(),
                            }
                        } else {
                            PyIndexStats {
                                n_vertices: index.get_graph_n_vertices(),
                                n_edges: counted_edges_in_adj,
                                avg_out_degrees: index.get_avg_out_degrees(),
                                min_max_out_degrees: index.get_min_max_out_degrees(),
                                total_avg_out_degree: index.get_total_avg_out_degree(),
                                avg_1nn_distance: 0.0,
                                avg_1nn_distances: vec![],
                                min_max_1nn_distances: vec![],
                            }
                        }
                    }
                    IndexOneOf::B(index) => {
                        let (_reported_edges, counted_edges_in_adj, _hashset_counted_edges) =
                            index.get_n_edges();

                        if !reduced_stats {
                            PyIndexStats {
                                n_vertices: index.get_graph_n_vertices(),
                                n_edges: counted_edges_in_adj,
                                avg_out_degrees: index.get_avg_out_degrees(),
                                min_max_out_degrees: index.get_min_max_out_degrees(),
                                total_avg_out_degree: index.get_total_avg_out_degree(),
                                avg_1nn_distance: index.get_avg_1nn_distance(),
                                avg_1nn_distances: index.get_avg_1nn_distances(),
                                min_max_1nn_distances: index.get_min_max_1nn_distances(),
                            }
                        } else {
                            PyIndexStats {
                                n_vertices: index.get_graph_n_vertices(),
                                n_edges: counted_edges_in_adj,
                                avg_out_degrees: index.get_avg_out_degrees(),
                                min_max_out_degrees: index.get_min_max_out_degrees(),
                                total_avg_out_degree: index.get_total_avg_out_degree(),
                                avg_1nn_distance: 0.0,
                                avg_1nn_distances: vec![],
                                min_max_1nn_distances: vec![],
                            }
                        }
                    }
                    IndexOneOf::None => panic!(),
                }
            }
        }
    };
    (layered $type: ident) => {
        generic_graph_index_funs!($type);
        #[pymethods]
        impl $type {
            pub fn get_graph_stats(&self) -> Vec<GraphStats> {
                match &self.index {
                    IndexOneOf::A(index) => index
                        .graphs()
                        .iter()
                        .map(|g| GraphStats::from_graph(g))
                        .collect(),
                    IndexOneOf::B(index) => index
                        .graphs()
                        .iter()
                        .map(|g| GraphStats::from_graph(g))
                        .collect(),
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn get_neighbors(&self, layer: usize, node: R) -> Vec<R> {
                match &self.index {
                    IndexOneOf::A(index) => index.graphs()[layer].neighbors(node),
                    IndexOneOf::B(index) => index.graphs()[layer].neighbors(node),
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn get_next_layer_id(&self, layer: usize, node: R) -> R {
                if layer == 0 {
                    return node;
                }
                match &self.index {
                    IndexOneOf::A(index) => index.get_local_layer_ids(layer).unwrap()[node],
                    IndexOneOf::B(index) => index.get_local_layer_ids(layer).unwrap()[node],
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn get_global_id(&self, layer: usize, node: R) -> R {
                if layer == 0 {
                    return node;
                }
                match &self.index {
                    IndexOneOf::A(index) => index.get_global_layer_ids(layer).unwrap()[node],
                    IndexOneOf::B(index) => index.get_global_layer_ids(layer).unwrap()[node],
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn set_higher_level_max_heap_size(&mut self, higher_level_max_heap_size: usize) {
                match &mut self.index {
                    IndexOneOf::A(index) => {
                        index.set_higher_level_max_heap_size(higher_level_max_heap_size)
                    }
                    IndexOneOf::B(index) => {
                        index.set_higher_level_max_heap_size(higher_level_max_heap_size)
                    }
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn evaluate_level_graphs_performance<'py>(
                &self,
                data: Bound<'py, PyArray2<F>>,
                queries: Bound<'py, PyArray2<F>>,
                ground_truth: Bound<'py, PyArray2<R>>,
                k: usize,
                max_heap_size: usize,
                nq: usize,
                graph_name: &str,
                print_output: bool,
                only_top_level: bool,
                point_zero_entry: bool,
            ) -> Vec<PyQPSRecallStats> {
                unsafe {
                    match &self.index {
                        IndexOneOf::A(index) => {
                            let stats: Vec<QPSRecallStats> = hierarchy_graph_level_wise_performance(
                                &index,
                                arrview2_py_to_rust(data.as_array()),
                                arrview2_py_to_rust(queries.as_array()),
                                arrview2_py_to_rust(ground_truth.as_array()),
                                k,
                                max_heap_size,
                                nq,
                                graph_name,
                                print_output,
                                only_top_level,
                                point_zero_entry,
                            );
                            stats
                                .iter()
                                .map(|e| PyQPSRecallStats {
                                    layer_count: e.layer_count,
                                    k: e.k,
                                    max_heap_size: e.max_heap_size,
                                    nq: e.nq,
                                    query_time: e.query_time,
                                    qps: e.qps,
                                    recall: e.recall,
                                })
                                .collect()
                        }
                        IndexOneOf::B(index) => {
                            let stats: Vec<QPSRecallStats> =
                                hierarchy_graph_level_wise_performance_capped(
                                    &index,
                                    arrview2_py_to_rust(data.as_array()),
                                    arrview2_py_to_rust(queries.as_array()),
                                    arrview2_py_to_rust(ground_truth.as_array()),
                                    k,
                                    max_heap_size,
                                    nq,
                                    graph_name,
                                    print_output,
                                    only_top_level,
                                    point_zero_entry,
                                );
                            stats
                                .iter()
                                .map(|e| PyQPSRecallStats {
                                    layer_count: e.layer_count,
                                    k: e.k,
                                    max_heap_size: e.max_heap_size,
                                    nq: e.nq,
                                    query_time: e.query_time,
                                    qps: e.qps,
                                    recall: e.recall,
                                })
                                .collect()
                        }
                        IndexOneOf::None => panic!(),
                    }
                }
            }
            pub fn evaluate_level_graphs_performance_powerset<'py>(
                &self,
                data: Bound<'py, PyArray2<F>>,
                queries: Bound<'py, PyArray2<F>>,
                ground_truth: Bound<'py, PyArray2<R>>,
                k: usize,
                max_heap_size: usize,
                nq: usize,
                graph_name: &str,
                print_output: bool,
                point_zero_entry: bool,
            ) -> Vec<PyQPSRecallStats> {
                unsafe {
                    match &self.index {
                        IndexOneOf::A(index) => {
                            let stats: Vec<QPSRecallStats> =
                                hierarchy_graph_level_wise_performance_powerset(
                                    &index,
                                    arrview2_py_to_rust(data.as_array()),
                                    arrview2_py_to_rust(queries.as_array()),
                                    arrview2_py_to_rust(ground_truth.as_array()),
                                    k,
                                    max_heap_size,
                                    nq,
                                    graph_name,
                                    print_output,
                                    point_zero_entry,
                                );
                            stats
                                .iter()
                                .map(|e| PyQPSRecallStats {
                                    layer_count: e.layer_count,
                                    k: e.k,
                                    max_heap_size: e.max_heap_size,
                                    nq: e.nq,
                                    query_time: e.query_time,
                                    qps: e.qps,
                                    recall: e.recall,
                                })
                                .collect()
                        }
                        IndexOneOf::B(_index) => unimplemented!(""),
                        IndexOneOf::None => panic!(),
                    }
                }
            }
            pub fn get_layer_count(&self) -> usize {
                match &self.index {
                    IndexOneOf::A(index) => index.layer_count(),
                    IndexOneOf::B(index) => index.layer_count(),
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn get_local_layer_ids(&self, layer: usize) -> Vec<R> {
                match &self.index {
                    IndexOneOf::A(index) => index
                        .get_local_layer_ids(layer)
                        .unwrap_or(&Vec::new())
                        .to_vec(),
                    IndexOneOf::B(index) => index
                        .get_local_layer_ids(layer)
                        .unwrap_or(&Vec::new())
                        .to_vec(),
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn get_global_layer_ids(&self, layer: usize) -> Vec<R> {
                match &self.index {
                    IndexOneOf::A(index) => index
                        .get_global_layer_ids(layer)
                        .unwrap_or(&Vec::new())
                        .to_vec(),
                    IndexOneOf::B(index) => index
                        .get_global_layer_ids(layer)
                        .unwrap_or(&Vec::new())
                        .to_vec(),
                    IndexOneOf::None => panic!(),
                }
            }
        }
    };
    (single $type: ident) => {
        generic_graph_index_funs!($type);
        #[pymethods]
        impl $type {
            pub fn get_graph_stats(&self) -> GraphStats {
                match &self.index {
                    IndexOneOf::A(index) => GraphStats::from_graph(index.graph()),
                    IndexOneOf::B(index) => GraphStats::from_graph(index.graph()),
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn get_neighbors(&self, node: R) -> Vec<R> {
                match &self.index {
                    IndexOneOf::A(index) => index.graph().neighbors(node),
                    IndexOneOf::B(index) => index.graph().neighbors(node),
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn get_graph(&self) -> PyDirLoLGraph {
                match &self.index {
                    IndexOneOf::A(index) => {
                        let graph = index.graph().as_dir_lol_graph();
                        let n_vertices = graph.n_vertices();
                        let n_edges = graph.n_edges();
                        let mut adj: Vec<Vec<R>> = Vec::with_capacity(n_vertices);
                        (0..n_vertices).for_each(|v| adj.push(graph.neighbors(v)));
                        PyDirLoLGraph::new(adj, n_edges)
                    }
                    IndexOneOf::B(index) => {
                        let graph = index.graph().as_dir_lol_graph();
                        let n_vertices = graph.n_vertices();
                        let n_edges = graph.n_edges();
                        let mut adj: Vec<Vec<R>> = Vec::with_capacity(n_vertices);
                        (0..n_vertices).for_each(|v| adj.push(graph.neighbors(v)));
                        PyDirLoLGraph::new(adj, n_edges)
                    }
                    IndexOneOf::None => panic!(),
                }
            }
            pub fn evaluate_performance<'py>(
                &self,
                queries: Bound<'py, PyArray2<F>>,
                ground_truth: Bound<'py, PyArray2<R>>,
                k: usize,
                max_heap_size: usize,
                nq: usize,
                graph_name: &str,
                print_output: bool,
            ) -> PyQPSRecallStats {
                unsafe {
                    match &self.index {
                        IndexOneOf::A(index) => {
                            let stats: QPSRecallStats = evaluate_qps_recall(
                                &index,
                                arrview2_py_to_rust(queries.as_array()),
                                arrview2_py_to_rust(ground_truth.as_array()),
                                k,
                                max_heap_size,
                                nq,
                                graph_name,
                                print_output,
                            );
                            PyQPSRecallStats {
                                layer_count: stats.layer_count,
                                k: stats.k,
                                max_heap_size: stats.max_heap_size,
                                nq: stats.nq,
                                query_time: stats.query_time,
                                qps: stats.qps,
                                recall: stats.recall,
                            }
                        }
                        IndexOneOf::B(index) => {
                            let stats: QPSRecallStats = evaluate_qps_recall_capped(
                                &index,
                                arrview2_py_to_rust(queries.as_array()),
                                arrview2_py_to_rust(ground_truth.as_array()),
                                k,
                                max_heap_size,
                                nq,
                                graph_name,
                                print_output,
                            );
                            PyQPSRecallStats {
                                layer_count: stats.layer_count,
                                k: stats.k,
                                max_heap_size: stats.max_heap_size,
                                nq: stats.nq,
                                query_time: stats.query_time,
                                qps: stats.qps,
                                recall: stats.recall,
                            }
                        }
                        IndexOneOf::None => panic!(),
                    }
                }
            }
        }
    };
}

// ----

#[pyclass]
#[allow(unused)]
#[derive(Copy, Clone)]
pub struct PyQPSRecallStats {
    #[pyo3(get)]
    pub layer_count: usize,
    #[pyo3(get)]
    pub k: usize,
    #[pyo3(get)]
    pub max_heap_size: usize,
    #[pyo3(get)]
    pub nq: usize,
    #[pyo3(get)]
    pub query_time: f32,
    #[pyo3(get)]
    pub qps: f32,
    #[pyo3(get)]
    pub recall: f32,
}

#[pyclass]
#[allow(unused)]
#[derive(Clone)]
pub struct PyIndexStats {
    #[pyo3(get)]
    pub n_vertices: Vec<usize>,
    #[pyo3(get)]
    pub n_edges: usize,
    #[pyo3(get)]
    pub avg_out_degrees: Vec<f32>,
    #[pyo3(get)]
    pub min_max_out_degrees: Vec<(usize, usize)>,
    #[pyo3(get)]
    pub total_avg_out_degree: f32,
    #[pyo3(get)]
    pub avg_1nn_distance: f32,
    #[pyo3(get)]
    pub avg_1nn_distances: Vec<f32>,
    #[pyo3(get)]
    pub min_max_1nn_distances: Vec<(f32, f32)>,
}
