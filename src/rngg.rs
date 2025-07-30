//! RNGG graph crate
//!
//! Contains RNGGBuilder which creates a randomly connected graph with a fixed degree and
//! the option to prune its edges such that short/long/etc. edges are kept.
//!
//! Notes
//!
//! Open Issues
//!
//! TODOs
//! - [ ] (Unimportant feature) Add option to create graph with different degree for each node
//!         - Sample random number between min/max bounds of configured degree-range
//!
use graphidx::{
    data::MatrixDataSource,
    graphs::{DirLoLGraph, Graph, ViewableWeightedAdjGraph, WeightedGraph},
    indices::{GreedyCappedSingleGraphIndex, GreedySingleGraphIndex},
    measures::Distance,
    param_struct,
    types::{SyncFloat, SyncUnsignedInteger},
};
#[allow(unused)]
use graphidxbaselines::{
    knn::{BruteforceKNNGraphBuilder, BruteforceKNNParams},
    util::random_unique_usizes_except,
};
use rayon::{
    current_num_threads,
    iter::{ParallelBridge, ParallelIterator},
};

// -------------- UTIL --------------

#[inline(always)]
fn make_greedy_index<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    M: MatrixDataSource<F>,
    Dist: Distance<F>,
>(
    graph: RNGGBuildGraph<R, F>,
    mat: M,
    dist: Dist,
    entry_points: Option<Vec<R>>,
) -> GreedySingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
    GreedySingleGraphIndex::new(mat, graph.as_dir_lol_graph(), dist, entry_points)
}

#[inline(always)]
fn make_greedy_capped_index<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    M: MatrixDataSource<F>,
    Dist: Distance<F>,
>(
    graph: RNGGBuildGraph<R, F>,
    mat: M,
    dist: Dist,
    max_frontier_size: usize,
    entry_points: Option<Vec<R>>,
) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
    GreedyCappedSingleGraphIndex::new(
        mat,
        graph.as_dir_lol_graph(),
        dist,
        max_frontier_size,
        entry_points,
    )
}

// --------------

pub struct RNGGBuildGraph<R: SyncUnsignedInteger, F: SyncFloat> {
    adjacency: Vec<Vec<(F, R)>>,
    n_edges: usize,
}
impl<R: SyncUnsignedInteger, F: SyncFloat> RNGGBuildGraph<R, F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            adjacency: vec![],
            n_edges: 0,
        }
    }
}
impl<R: SyncUnsignedInteger, F: SyncFloat> Graph<R> for RNGGBuildGraph<R, F> {
    #[inline(always)]
    fn reserve(&mut self, n_vertices: usize) {
        self.adjacency.reserve(n_vertices);
    }
    #[inline(always)]
    fn n_vertices(&self) -> usize {
        self.adjacency.len()
    }
    #[inline(always)]
    fn n_edges(&self) -> usize {
        self.n_edges
    }
    #[inline(always)]
    fn neighbors(&self, vertex: R) -> Vec<R> {
        let vertex = unsafe { vertex.to_usize().unwrap_unchecked() };
        self.adjacency[vertex].iter().map(|&(_, v)| v).collect()
    }
    #[inline(always)]
    fn foreach_neighbor<Fun: FnMut(&R)>(&self, vertex: R, mut f: Fun) {
        self.adjacency[vertex.to_usize().unwrap()]
            .iter()
            .for_each(|v| f(&v.1));
    }
    #[inline(always)]
    fn foreach_neighbor_mut<Fun: FnMut(&mut R)>(&mut self, vertex: R, mut f: Fun) {
        self.adjacency[vertex.to_usize().unwrap()]
            .iter_mut()
            .for_each(|v| f(&mut v.1));
    }
    #[inline(always)]
    fn iter_neighbors<'a>(&'a self, vertex: R) -> impl Iterator<Item = &'a R> {
        unsafe {
            self.adjacency
                .get_unchecked(vertex.to_usize().unwrap_unchecked())
                .iter()
                .map(|v| &v.1)
        }
    }
    #[inline(always)]
    fn add_node(&mut self) {
        self.adjacency.push(Vec::new());
    }
    #[inline(always)]
    fn add_node_with_capacity(&mut self, capacity: usize) {
        self.adjacency.push(Vec::with_capacity(capacity));
    }
    #[inline(always)]
    fn add_edge(&mut self, _vertex1: R, _vertex2: R) {
        panic!("Cannot add edge without weight to a weighted graph");
    }
    #[inline(always)]
    fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
        let vertex = unsafe { vertex.to_usize().unwrap_unchecked() };
        self.adjacency[vertex].swap_remove(index);
        self.n_edges -= 1;
    }
}
impl<R: SyncUnsignedInteger, F: SyncFloat> WeightedGraph<R, F> for RNGGBuildGraph<R, F> {
    #[inline(always)]
    fn edge_weight(&self, vertex1: R, vertex2: R) -> F {
        let vertex1 = unsafe { vertex1.to_usize().unwrap_unchecked() };
        self.adjacency[vertex1]
            .iter()
            .find(|&&(_, v)| v == vertex2)
            .unwrap()
            .0
    }
    #[inline(always)]
    fn add_edge_with_weight(&mut self, vertex1: R, vertex2: R, weight: F) {
        let vertex1 = unsafe { vertex1.to_usize().unwrap_unchecked() };
        self.adjacency[vertex1].push((weight, vertex2));
        self.n_edges += 1;
    }
    #[inline(always)]
    fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>) {
        let mut neighbors = Vec::new();
        let mut weights = Vec::new();
        for &(w, v) in &self.adjacency[vertex.to_usize().unwrap()] {
            neighbors.push(v);
            weights.push(w);
        }
        (weights, neighbors)
    }
    #[inline(always)]
    fn neighbors_with_zipped_weights(&self, vertex: R) -> Vec<(F, R)> {
        self.adjacency[vertex.to_usize().unwrap()]
            .iter()
            .map(|&(v, w)| (v, w))
            .collect()
    }
    #[inline(always)]
    fn foreach_neighbor_with_zipped_weight<Fun: FnMut(&F, &R)>(&self, vertex: R, mut f: Fun) {
        self.adjacency[vertex.to_usize().unwrap()]
            .iter()
            .for_each(|v| f(&v.0, &v.1));
    }
    #[inline(always)]
    fn foreach_neighbor_with_zipped_weight_mut<Fun: FnMut(&mut F, &mut R)>(
        &mut self,
        vertex: R,
        mut f: Fun,
    ) {
        self.adjacency[vertex.to_usize().unwrap()]
            .iter_mut()
            .for_each(|v| f(&mut v.0, &mut v.1));
    }
}

pub trait RNGGStyleBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
where
    Self: Sized,
{
    type Params;
    type Graph: ViewableWeightedAdjGraph<R, F>;
    fn _graph(&self) -> &Self::Graph;
    fn _mut_graph(&mut self) -> &mut Self::Graph;
    fn _initial_degree(&self) -> usize;
    fn _dist(&self) -> &Dist;
    fn _into_graph_dist(self) -> (RNGGBuildGraph<R, F>, Dist);
    #[inline(always)]
    fn _get_dist<M: MatrixDataSource<F>>(&self, mat: &M, i: usize, j: usize) -> F {
        if M::SUPPORTS_ROW_VIEW {
            self._dist()
                .dist_slice(&mat.get_row_view(i), &mat.get_row_view(j))
        } else {
            self._dist().dist(&mat.get_row(i), &mat.get_row(j))
        }
    }
    fn new<M: MatrixDataSource<F> + Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self;
    fn train<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M);
    #[inline(always)]
    fn into_greedy<M: MatrixDataSource<F> + Sync>(
        self,
        mat: M,
    ) -> GreedySingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let (graph, dist) = self._into_graph_dist();
        make_greedy_index(graph, mat, dist, None)
    }
    #[inline(always)]
    fn into_greedy_capped<M: MatrixDataSource<F> + Sync>(
        self,
        mat: M,
        max_frontier_size: usize,
    ) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let (graph, dist) = self._into_graph_dist();
        make_greedy_capped_index(graph, mat, dist, max_frontier_size, None)
    }
    #[inline(always)]
    fn build<M: MatrixDataSource<F> + Sync>(
        mat: M,
        dist: Dist,
        params: Self::Params,
    ) -> GreedySingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let mut builder = Self::new(&mat, dist, params);
        builder.train(&mat);
        builder.into_greedy(mat)
    }
    #[inline(always)]
    fn build_capped<M: MatrixDataSource<F> + Sync>(
        mat: M,
        dist: Dist,
        params: Self::Params,
        max_frontier_size: usize,
    ) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let mut builder = Self::new(&mat, dist, params);
        builder.train(&mat);
        builder.into_greedy_capped(mat, max_frontier_size)
    }
}
impl<R: SyncUnsignedInteger, F: SyncFloat> ViewableWeightedAdjGraph<R, F> for RNGGBuildGraph<R, F> {
    #[inline(always)]
    fn view_neighbors(&self, vertex: R) -> &[(F, R)] {
        self.adjacency[vertex.to_usize().unwrap()].as_slice()
    }
    #[inline(always)]
    fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F, R)] {
        self.adjacency[vertex.to_usize().unwrap()].as_mut_slice()
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum KeepEdges {
    SHORT,
    MIDDLE,
    LONG,
    SPECIAL,
    NONE,
}

param_struct!(RNGGParams[Copy, Clone]{
    degree: usize = 30,
    degree_short_edges: usize = 30,
    // specifies how many points are randomly selected, and then sorted to keep degree-many edges
    prune_extend_degree: usize = 40,
    prune: KeepEdges = KeepEdges::NONE,
    // "max_heap_size" of the random build process
});

pub struct RNGGBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send> {
    _phantom: std::marker::PhantomData<F>,
    n_data: usize,
    params: RNGGParams,
    graph: RNGGBuildGraph<R, F>,
    dist: Dist,
}

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    RNGGBuilder<R, F, Dist>
{
    #[inline(always)]
    fn prune(&self, degree: usize, mut candidates_with_weights: Vec<(F, R)>) -> Vec<(F, R)> {
        let max_amount = degree.min(candidates_with_weights.len());

        match self.params.prune {
            KeepEdges::SHORT => {
                candidates_with_weights
                    .sort_by(|a, b| unsafe { a.0.partial_cmp(&b.0).unwrap_unchecked() });

                candidates_with_weights.as_slice()[0..max_amount].to_vec()
            }
            KeepEdges::MIDDLE => {
                candidates_with_weights
                    .sort_by(|a, b| unsafe { a.0.partial_cmp(&b.0).unwrap_unchecked() });
                let mid_point = self.params.prune_extend_degree / 2;

                candidates_with_weights.as_slice()
                    [(mid_point - (max_amount / 2))..(mid_point + (max_amount / 2))]
                    .to_vec()
            }
            KeepEdges::LONG => {
                candidates_with_weights
                    .sort_by(|a, b| unsafe { a.0.partial_cmp(&b.0).unwrap_unchecked() });
                candidates_with_weights.as_slice()[candidates_with_weights.len() - max_amount..]
                    .to_vec()
            }
            KeepEdges::SPECIAL => {
                // used for testing
                candidates_with_weights
                    .sort_by(|a, b| unsafe { a.0.partial_cmp(&b.0).unwrap_unchecked() });
                // let mut x =
                // candidates_with_weights.as_slice()[0..self.params.degree_short_edges].to_vec();
                // let mut y = candidates_with_weights.as_slice()
                //     [candidates_with_weights.len() - (degree - self.params.degree_short_edges)..]
                //     .to_vec();
                let mut x = candidates_with_weights.as_slice()[0..(degree / 2)].to_vec();
                let mut y =
                    candidates_with_weights.as_slice()[degree..(degree + (degree / 2))].to_vec();
                x.append(&mut y);
                x
                // candidates_with_weights.as_slice()[0..self.params.degree_short_edges].to_vec()
            }
            KeepEdges::NONE => candidates_with_weights.as_slice()[0..max_amount].to_vec(),
        }
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    RNGGStyleBuilder<R, F, Dist> for RNGGBuilder<R, F, Dist>
{
    type Params = RNGGParams;
    type Graph = RNGGBuildGraph<R, F>;
    fn _graph(&self) -> &Self::Graph {
        &self.graph
    }
    #[inline(always)]
    fn _mut_graph(&mut self) -> &mut Self::Graph {
        &mut self.graph
    }
    #[inline(always)]
    fn _dist(&self) -> &Dist {
        &self.dist
    }
    #[inline(always)]
    fn _initial_degree(&self) -> usize {
        self.params.degree
    }
    #[inline(always)]
    fn _into_graph_dist(self) -> (RNGGBuildGraph<R, F>, Dist) {
        (self.graph, self.dist)
    }
    fn new<M: MatrixDataSource<F> + Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
        let n_data = mat.n_rows();
        assert!(n_data < R::max_value().to_usize().unwrap());
        let graph = RNGGBuildGraph::new();
        Self {
            _phantom: std::marker::PhantomData,
            n_data,
            params,
            graph,
            dist,
        }
    }
    fn train<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
        let n_data = self.n_data;
        if n_data == 0 {
            return;
        }
        assert!(self.params.prune_extend_degree >= self.params.degree);
        let degree = self.params.degree;
        self.graph.reserve(n_data);
        (0..n_data).for_each(|_| self.graph.add_node_with_capacity(degree));

        let dist = self._dist();
        let n_threads = current_num_threads();
        let thread_chunk_size = (n_data + n_threads - 1) / n_threads;

        // adapted from make_random_adj_list() from GraphIndexBaselines
        unsafe {
            (0..n_data)
                .step_by(thread_chunk_size)
                .map(|start| (start, (start + thread_chunk_size).min(n_data)))
                .par_bridge()
                .for_each(|(start, end)| {
                    let unsafe_graph_ref =
                        std::ptr::addr_of!(self.graph) as *mut RNGGBuildGraph<R, F>;
                    (start..end).for_each(|i_usize| {
                        let candidates = random_unique_usizes_except(
                            n_data,
                            self.params.prune_extend_degree,
                            i_usize,
                        );
                        let candidates_with_weights: Vec<(F, R)> = candidates
                            .into_iter()
                            .map(|j_usize| {
                                let j = R::from_usize(j_usize).unwrap_unchecked();
                                let ij_dist = if M::SUPPORTS_ROW_VIEW {
                                    dist.dist_slice(
                                        &mat.get_row_view(i_usize),
                                        &mat.get_row_view(j_usize),
                                    )
                                } else {
                                    dist.dist(&mat.get_row(i_usize), &mat.get_row(j_usize))
                                };
                                (ij_dist, j)
                            })
                            .collect::<Vec<(F, R)>>();

                        (*unsafe_graph_ref).adjacency[i_usize] =
                            self.prune(degree, candidates_with_weights);
                    });
                });
            self.graph.n_edges += degree * n_data;
        }
    }
}

// -------------- TESTS --------------

#[cfg(test)]
mod tests {
    use crate::rngg::*;
    use crate::{
        print_index_build_info,
        utils::{
            eval::{calc_recall, get_test_data},
            index_stats::IndexGraphStats,
        },
    };
    use graphidx::{
        indices::{bruteforce_neighbors, GraphIndex},
        measures::SquaredEuclideanDistance,
    };
    use ndarray::{Array2, Axis, Slice};
    use ndarray_rand::{rand, rand_distr::Normal};
    use rand::prelude::Distribution;
    use std::time::Instant;
    type R = usize;
    type F = f32;
    type Dist = SquaredEuclideanDistance<F>;

    #[test]
    fn rngg_construction() {
        let (_nd, _nq, _d, _k, data) = get_test_data(true);

        let params = RNGGParams::new();

        let graph_time = Instant::now();
        let index = RNGGBuilder::<R, F, Dist>::build(data, Dist::new(), params);
        println!("RNGG Graph construction: {:.2?}", graph_time.elapsed());
        print_index_build_info!(index);
    }

    #[test]
    fn rngg_query() {
        let (_nd, nq, _d, k, data) = get_test_data(true);

        let queries = data.slice_axis(Axis(0), Slice::from(0..nq));

        // KNN (for comparison)
        let build_time = Instant::now();
        let dist = Dist::new();
        let params = BruteforceKNNParams::new().with_degree(30);
        type BuilderType2 = BruteforceKNNGraphBuilder<R, F, Dist>;
        let index_knn = BuilderType2::build(data.view(), dist, params);
        println!(
            "Graph construction ({:?}): {:.2?}",
            std::any::type_name::<BuilderType2>(),
            build_time.elapsed()
        );

        // Build and translate RNGG Graph
        let build_time = Instant::now();
        let dist = Dist::new();
        let params = RNGGParams::new();
        type BuilderType = RNGGBuilder<R, F, Dist>;
        let index1 = BuilderType::build(data.view(), dist, params);

        println!(
            "Graph construction ({:?}): {:.2?}",
            std::any::type_name::<BuilderType>(),
            build_time.elapsed()
        );
        print_index_build_info!(index1);

        let search_extended_k = 5 * k;
        // Brute force queries
        let bruteforce_time = Instant::now();
        let (bruteforce_ids, _) = bruteforce_neighbors(&data, &queries, &Dist::new(), k);
        println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());

        // KNN queries
        let knn_time = Instant::now();
        let (knn_ids1, _knn_dists1) = index_knn.greedy_search_batch(&queries, k, search_extended_k);
        println!("KNN queries 1: {:.2?}", knn_time.elapsed());

        // RNGG queries
        let rngg_time = Instant::now();
        let (rngg_ids1, _rngg_dists1) = index1.greedy_search_batch(&queries, k, search_extended_k);
        println!("RNGG queries 1: {:.2?}", rngg_time.elapsed());

        calc_recall(bruteforce_ids.view(), &knn_ids1, nq, k, "kNN", true);
        calc_recall(bruteforce_ids.view(), &rngg_ids1, nq, k, "RNGG", true);
    }

    #[test]
    fn rngg_query_all() {
        let (_nd, nq, d, k, data) = get_test_data(true);
        let search_extended_k = 20 * k;

        // let queries = data.slice_axis(Axis(0), Slice::from(0..nq));
        // unindex queries
        let rng = Normal::new(0.0, 1.0).unwrap();
        let queries: Array2<F> =
            Array2::from_shape_fn((nq, d), |_| rng.sample(&mut rand::thread_rng()));

        let bruteforce_time = Instant::now();
        let (bruteforce_ids, _) = bruteforce_neighbors(&data, &queries, &Dist::new(), k);
        println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());

        let prune_options: Vec<KeepEdges> = vec![
            KeepEdges::NONE,
            KeepEdges::SHORT,
            KeepEdges::SPECIAL,
            KeepEdges::MIDDLE,
            KeepEdges::LONG,
        ];
        let degree = 30;
        let degree_short_edges = 15;
        let prune_extend_degree = 1000;
        for opt in prune_options {
            let build_time = Instant::now();
            let dist = Dist::new();
            let params = RNGGParams::new()
                .with_degree(degree)
                .with_degree_short_edges(degree_short_edges)
                .with_prune_extend_degree(prune_extend_degree)
                .with_prune(opt);
            type BuilderType = RNGGBuilder<R, F, Dist>;
            let index1 = BuilderType::build(data.view(), dist, params);
            println!("--- Graph construction: {:.2?} ---", build_time.elapsed());

            let rngg_time = Instant::now();
            let (rngg_ids1, _rngg_dists1) =
                index1.greedy_search_batch(&queries, k, search_extended_k);
            let s = format!("RNGG-[{:?}]", opt);
            println!("{} queries: {:.2?}", s, rngg_time.elapsed());

            calc_recall(bruteforce_ids.view(), &rngg_ids1, nq, k, &s, true);
        }
    }
}
