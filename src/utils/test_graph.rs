//! Util
//!
//! Plain graph for testing *selectors*.
//! Easy wrapper to quickly get a test graph by specifying: nd, d and out_degree
//!
//! Notes
//!     - There (currently) exists only the BasicTestGraph<usize, f32> implementation
//!     - Includes helpers from graphindexbaselines to randomly initialize a graph
//!
//! Open Issues
//!
//! TODOs
//!
use graphidx::{
    data::MatrixDataSource,
    graphs::{Graph, WeightedGraph},
    measures::Distance,
    types::{SyncFloat, SyncUnsignedInteger},
};
use graphidxbaselines::util::random_unique_usizes_except;
use ndarray::Array2;
use ndarray_rand::{rand, rand_distr::Normal};
use rand::prelude::Distribution;
use rayon::iter::ParallelBridge;
use rayon::{current_num_threads, iter::ParallelIterator};
use std::{f32, usize};

pub struct BasicTestGraph<R: SyncUnsignedInteger, F: SyncFloat> {
    adjacency: Vec<Vec<(F, R)>>,
    n_edges: usize,
}

impl BasicTestGraph<usize, f32> {
    #[inline(always)]
    pub fn new<Dist: Distance<f32> + Sync + Send>(
        nd: usize,
        d: usize,
        out_degree: usize,
        dist: Dist,
    ) -> Self {
        let rng = Normal::new(0.0, 1.0).unwrap();
        let data = Array2::from_shape_fn((nd, d), |_| rng.sample(&mut rand::thread_rng()));

        Self {
            adjacency: make_random_adj_list(&data.view(), &dist, out_degree),
            n_edges: nd * out_degree,
        }
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat> Graph<R> for BasicTestGraph<R, F> {
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

impl<R: SyncUnsignedInteger, F: SyncFloat> WeightedGraph<R, F> for BasicTestGraph<R, F> {
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

// --------------------------- Helpers ---------------------------
// Copied from GraphIndexBaselines
fn make_random_adj_list<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
    M: MatrixDataSource<F> + Sync,
>(
    mat: &M,
    dist: &Dist,
    out_degree: usize,
) -> Vec<Vec<(F, R)>> {
    let n_data = mat.n_rows();
    let mut adjacency: Vec<Vec<(F, R)>> = Vec::with_capacity(n_data);
    (0..n_data).for_each(|_| adjacency.push(Vec::with_capacity(out_degree)));
    let n_threads = current_num_threads();
    let thread_chunk_size = (n_data + n_threads - 1) / n_threads;
    unsafe {
        (0..n_data)
            .step_by(thread_chunk_size)
            .map(|start| (start, (start + thread_chunk_size).min(n_data)))
            .par_bridge()
            .for_each(|(start, end)| {
                let unsafe_adjacency = std::ptr::addr_of!(adjacency) as *mut Vec<Vec<(F, R)>>;
                (start..end).for_each(|i_usize| {
                    let i = R::from_usize(i_usize).unwrap_unchecked();
                    let neighbors = random_unique_usizes_except(n_data, out_degree, i_usize);
                    (*unsafe_adjacency)[i.to_usize().unwrap_unchecked()].extend(
                        neighbors.into_iter().map(|j_usize| {
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
                        }),
                    );
                });
            });
    }
    adjacency
}
