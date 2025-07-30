//! NSSG graph crate
//! See [SSG/NSSG on GitHub](https://github.com/ZJULearning/SSG/tree/master)
//!
//! Contains
//!     - NSSGBuilder (single threaded)
//!     - NSSGParallelBuilder  
//!
//! Notes
//!     - Different builders which make use of heaps as the pool in link() have been tried but
//!       they performed worse than the current variant with a vec as pool, so that after
//!       commit 073c8617691435facbf020d68611a984b49056b3 these version have been removed for
//!       less code complexity
//!         - However, was this tested solely during development and after that any further
//!           optimizations focused therefore only on the NSSGParallelBuilder
//!     - The paper mentions NSSG as the practical variant but everything in the code
//!        only refers to SSG
//!     - The original code uses a dead function call `init_graph()` in the train/build function
//!     which is implemented only on the single-threaded NSSGBuilder. The function calls two
//!     other functions downstream which are therefore dead functions as well. Note that in the
//!     original C++ code the parameter `ep_` is set which, however, is unused (also marked as
//!     such in `index_ssg.h`)
//!         - It originally should have been at least in parts related to setting the entry point
//!
//! Open Issues
//!     - The `new()` function is a bit anti-pattern because it takes a another argument
//!     (input_graph: NSSGInputGraphData). This parameter can not (easily) be moved under
//!     NSSGParams because of lifetimes as well as missing implementations of the Copy and Clone
//!     traits. In addition would it collide with the HSGFBuilder where in the case of NSSG it
//!     makes only sense to use the reduced NSSGInputGraphDataForHSGF struct.
//!
//! TODOs
//!     - Test/Check if both loops in link() can be merged into one iteration
//!         - Small experiments have shown no major difference in performance
//!
//! --------------------------
//! LICENSE of source code of SSG
//! BSD 3-Clause License
//!
//! Copyright (c) 2019, Cong Fu
//! All rights reserved.
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//!
//! 1. Redistributions of source code must retain the above copyright notice, this
//!    list of conditions and the following disclaimer.
//!
//! 2. Redistributions in binary form must reproduce the above copyright notice,
//!    this list of conditions and the following disclaimer in the documentation
//!    and/or other materials provided with the distribution.
//!
//! 3. Neither the name of the copyright holder nor the names of its
//!    contributors may be used to endorse or promote products derived from
//!    this software without specific prior written permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//! --------------------------
//!
use crate::{
    deg::{DEGParallelBuilder, DEGParams, DEGStyleBuilder},
    efanna::{EfannaParallelMaxHeapBuilder, EfannaParams, EfannaStyleBuilder},
};
use core::f64;
use graphidx::{
    data::MatrixDataSource,
    graphs::{DirLoLGraph, Graph, ViewableAdjGraph, ViewableWeightedAdjGraph, WeightedGraph},
    indices::{GreedyCappedSingleGraphIndex, GreedySingleGraphIndex},
    measures::Distance,
    param_struct,
    random::random_unique_uint,
    sets::HashSetLike,
    types::{SyncFloat, SyncUnsignedInteger},
};
use graphidxbaselines::{
    rnn::{RNNDescentBuilder, RNNParams, RNNStyleBuilder},
    util::random_unique_usizes_except,
};
use rayon::{
    current_num_threads,
    iter::{ParallelBridge, ParallelIterator},
};
use std::{collections::VecDeque, sync::Mutex};

type HashSetBit<T> = graphidx::sets::BitSet<T>;

// -------------- UTIL --------------

#[inline(always)]
fn make_greedy_index<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    M: MatrixDataSource<F>,
    Dist: Distance<F>,
>(
    graph: NSSGBuildGraph<R, F>,
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
    graph: NSSGBuildGraph<R, F>,
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

// Source: GraphIndexBaselines
#[inline(always)]
fn make_random_graph<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
    M: MatrixDataSource<F> + Sync,
>(
    mat: &M,
    graph: &mut NSSGBuildGraph<R, F>,
    dist: &Dist,
    out_degree: usize,
) {
    let n_data = mat.n_rows();
    (0..n_data).for_each(|_| graph.add_node());
    let n_threads = current_num_threads();
    let thread_chunk_size = (n_data + n_threads - 1) / n_threads;
    unsafe {
        (0..n_data)
            .step_by(thread_chunk_size)
            .map(|start| (start, (start + thread_chunk_size).min(n_data)))
            .par_bridge()
            .for_each(|(start, end)| {
                let unsafe_graph_ref = std::ptr::addr_of!(*graph) as *mut NSSGBuildGraph<R, F>;
                (start..end).for_each(|i_usize| {
                    let i = R::from_usize(i_usize).unwrap_unchecked();
                    let neighbors = random_unique_usizes_except(n_data, out_degree, i_usize);
                    (*unsafe_graph_ref)
                        .get_adj_mut(i)
                        .extend(neighbors.into_iter().map(|j_usize| {
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
                        }));
                });
            });
        graph.n_edges += out_degree * n_data;
    }
}

// --------------

// Helper enum to organize the input graph/data
pub enum NSSGInputGraphData<R: SyncUnsignedInteger, F: SyncFloat> {
    GRAPH(DirLoLGraph<R>),
    ADJACENCY(Vec<Vec<(F, R)>>),
    DEG(Option<DEGParams>),
    EFANNA(Option<EfannaParams>),
    RNN(Option<RNNParams>),
    RANDOM(usize),
}

// Wrapper for DirLolGraph<usize> so that it can be used in the Python binding
// Defined here to avoid circular references
#[derive(Clone)]
pub struct NSSGInputDirLolGraph {
    pub adjacency: Vec<Vec<usize>>,
    pub n_edges: usize,
}
impl NSSGInputDirLolGraph {
    pub fn new() -> Self {
        NSSGInputDirLolGraph {
            adjacency: Vec::new(),
            n_edges: 0,
        }
    }
}

// Helper enum/wrapper for the HSGFLevelGraphStyleBuilder trait because NSSGInputGraphData does/can not implement the Clone trait
#[derive(Clone)]
pub enum NSSGInputGraphDataForHSGF {
    DEG(Option<DEGParams>),
    EFANNA(Option<EfannaParams>),
    GRAPH(NSSGInputDirLolGraph), // currently bad memory usage; avoid usage
    RANDOM(usize),
    RNN(Option<RNNParams>),
}

// --------------

pub struct NSSGBuildGraph<R: SyncUnsignedInteger, F: SyncFloat> {
    adjacency: Vec<Vec<(F, R)>>,
    n_edges: usize,
}
impl<R: SyncUnsignedInteger, F: SyncFloat> NSSGBuildGraph<R, F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            adjacency: vec![],
            n_edges: 0,
        }
    }
    #[inline(always)]
    fn get_adj_mut(&mut self, vertex: R) -> &mut Vec<(F, R)> {
        let vertex = unsafe { vertex.to_usize().unwrap_unchecked() };
        &mut self.adjacency[vertex]
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat> Graph<R> for NSSGBuildGraph<R, F> {
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

impl<R: SyncUnsignedInteger, F: SyncFloat> WeightedGraph<R, F> for NSSGBuildGraph<R, F> {
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
        debug_assert!(
            self.adjacency[vertex1][self.adjacency[vertex1].len() - 1] == (weight, vertex2)
        );
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
    #[inline(always)]
    fn as_viewable_weighted_adj_graph(&self) -> Option<&impl ViewableWeightedAdjGraph<R, F>> {
        Some(self)
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat> ViewableWeightedAdjGraph<R, F> for NSSGBuildGraph<R, F> {
    #[inline(always)]
    fn view_neighbors(&self, vertex: R) -> &[(F, R)] {
        self.adjacency[vertex.to_usize().unwrap()].as_slice()
    }
    #[inline(always)]
    fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F, R)] {
        self.adjacency[vertex.to_usize().unwrap()].as_mut_slice()
    }
}

pub trait NSSGStyleBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
where
    Self: Sized,
{
    type Params;
    type Graph: ViewableWeightedAdjGraph<R, F>;
    fn _graph(&self) -> &Self::Graph;
    fn _mut_graph(&mut self) -> &mut Self::Graph;
    fn _mut_graph_for_random_init(&mut self) -> &mut NSSGBuildGraph<R, F>;
    fn _initial_degree(&self) -> usize;
    fn _get_entry_points(&self) -> Option<Vec<R>>;
    fn _dist(&self) -> &Dist;
    fn _into_graph_dist(self) -> (NSSGBuildGraph<R, F>, Dist);
    #[inline(always)]
    fn _get_dist<M: MatrixDataSource<F>>(&self, mat: &M, i: usize, j: usize) -> F {
        if M::SUPPORTS_ROW_VIEW {
            self._dist()
                .dist_slice(&mat.get_row_view(i), &mat.get_row_view(j))
        } else {
            self._dist().dist(&mat.get_row(i), &mat.get_row(j))
        }
    }
    // unused because of dead code in og-code
    // #[inline(always)]
    // fn _get_dist_with_input_vec<M: MatrixDataSource<F>>(
    //     &self,
    //     mat: &M,
    //     i: usize,
    //     point: Vec<F>,
    // ) -> F {
    //     if M::SUPPORTS_ROW_VIEW {
    //         self._dist()
    //             .dist_slice(&mat.get_row_view(i), &point.as_slice())
    //     } else {
    //         self._dist().dist(&mat.get_row(i), &Array::from_vec(point))
    //     }
    // }
    fn new<M: MatrixDataSource<F> + Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self;
    fn train<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        input_graph: NSSGInputGraphData<R, F>,
    );
    #[inline(always)]
    fn into_greedy<M: MatrixDataSource<F> + Sync>(
        self,
        mat: M,
    ) -> GreedySingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let entry_points = self._get_entry_points();
        let (graph, dist) = self._into_graph_dist();
        make_greedy_index(graph, mat, dist, entry_points)
    }
    #[inline(always)]
    fn into_greedy_capped<M: MatrixDataSource<F> + Sync>(
        self,
        mat: M,
        max_frontier_size: usize,
    ) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let entry_points = self._get_entry_points();
        let (graph, dist) = self._into_graph_dist();
        make_greedy_capped_index(graph, mat, dist, max_frontier_size, entry_points)
    }
    #[inline(always)]
    fn build<M: MatrixDataSource<F> + Sync>(
        mat: M,
        dist: Dist,
        params: Self::Params,
        input_graph: NSSGInputGraphData<R, F>,
    ) -> GreedySingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let mut builder = Self::new(&mat, dist, params);
        builder.train(&mat, input_graph);
        builder.into_greedy(mat)
    }
    #[inline(always)]
    fn build_capped<M: MatrixDataSource<F> + Sync>(
        mat: M,
        dist: Dist,
        params: Self::Params,
        input_graph: NSSGInputGraphData<R, F>,
        max_frontier_size: usize,
    ) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let mut builder = Self::new(&mat, dist, params);
        builder.train(&mat, input_graph);
        builder.into_greedy_capped(mat, max_frontier_size)
    }
    // ----------------
    #[inline(always)]
    fn init_random<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M, degree: usize) {
        let dist = self._dist() as *const Dist;
        self._mut_graph_for_random_init()
            .adjacency
            .reserve(mat.n_rows());
        make_random_graph(
            mat,
            self._mut_graph_for_random_init(),
            unsafe { dist.as_ref().unwrap() },
            degree,
        );
    }
}

param_struct!(NSSGParams[Copy, Clone]<F: SyncFloat>{
    range: usize = 70,
    l: usize = 250,
    angle: F = F::from(60.0).unwrap(),
    n_try: usize = 10,
    derive_angle_from_dim: bool = false,
    /* not in use; only used in unimplemented strongly_connect()
    or optimize_graph() */
    // width: usize = 1,
});

/// IMPORTANT NOTE: init_random() actually uses parallel code
/// (but is only executed in the case of NSSGInputGraphData::NONE)
pub mod single_threaded {
    use crate::{
        deg::single_threaded::DEGBuilder,
        efanna::{single_threaded::EfannaBuilder, EfannaStyleBuilder},
        nssg::*,
    };

    // Most closely to the original implementation, while the NSSGParallelBuilders do not contain
    // the dead and unused functions still found on this struct (for completeness)
    pub struct NSSGBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send> {
        _phantom: std::marker::PhantomData<F>,
        n_data: usize,
        params: NSSGParams<F>,
        graph: NSSGBuildGraph<R, F>,
        tmp_graph: Vec<Vec<(F, R)>>,
        dist: Dist,
        entry_points: Vec<R>,
    }

    impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
        NSSGBuilder<R, F, Dist>
    {
        fn init_graph<M: MatrixDataSource<F> + Sync>(
            &mut self,
            input_graph: NSSGInputGraphData<R, F>,
            mat: &M,
        ) {
            match input_graph {
                NSSGInputGraphData::ADJACENCY(adj) => {
                    self.graph.adjacency = adj;
                }
                NSSGInputGraphData::DEG(params) => {
                    // uses the single thread builder!
                    let index = DEGBuilder::<R, F, Dist>::build(
                        mat,
                        self._dist().clone(),
                        params.unwrap_or(DEGParams::new()),
                    );
                    self.load_from_input_graph(mat, index.graph().as_dir_lol_graph());
                }
                NSSGInputGraphData::EFANNA(params) => {
                    // uses the single thread builder!
                    let index = EfannaBuilder::<R, F, Dist>::build(
                        mat,
                        self._dist().clone(),
                        params.unwrap_or(
                            EfannaParams::new()
                                .with_k(self.params.range)
                                .with_l(self.params.range),
                        ),
                    );
                    self.load_from_input_graph(mat, index.graph().as_dir_lol_graph());
                }
                NSSGInputGraphData::GRAPH(graph) => {
                    assert!(
                        mat.n_rows() == graph.n_vertices(),
                        "Input graph and input data not matching in number of rows/vertices"
                    );
                    self.load_from_input_graph(mat, graph);
                }
                NSSGInputGraphData::RANDOM(degree) => {
                    self.init_random(mat, degree);
                }
                NSSGInputGraphData::RNN(params) => {
                    // NOTE no single threaded version available
                    let index = RNNDescentBuilder::<R, F, Dist>::build(
                        mat,
                        self._dist().clone(),
                        params.unwrap_or(RNNParams::new()),
                    );
                    self.load_from_input_graph(mat, index.graph().as_dir_lol_graph());
                }
            }
        }
        /// Assumes the neighbors are sorted asc (based on their distance)
        /// which for Efanna is the case by default
        fn load_from_input_graph<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            input_graph: DirLoLGraph<R>,
        ) {
            let range = self.params.range;
            (0..self.n_data).for_each(|i| {
                let i_r = R::from(i).unwrap();
                self.graph.add_node();
                let mut cnt = 0;
                for &n in input_graph.view_neighbors(i_r) {
                    if cnt >= range {
                        break;
                    }
                    self.graph.add_edge_with_weight(
                        i_r,
                        n,
                        self._get_dist(mat, i, n.to_usize().unwrap()),
                    );
                    cnt += 1;
                }
            });
        }

        fn link<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
            let angle = unsafe { F::from(self.params.angle).unwrap_unchecked() };
            // threshold = cos(angle / 180 * PI)
            let threshold = <F as num::Float>::cos(
                angle / unsafe { F::from(180).unwrap_unchecked() }
                    * unsafe { F::from(f64::consts::PI).unwrap_unchecked() },
            );

            let mut pool: Vec<(F, R, bool)> = Vec::new();
            let mut flags: HashSetBit<R> = HashSetBit::new(self.n_data);
            // let mut tmp: Vec<(F, R, bool)> = Vec::new(); // seems to be unused
            for n in 0..self.n_data {
                pool.clear();
                flags.clear();
                // tmp.clear();
                self.get_neighbors_for_link(mat, n, &mut pool, &mut flags);
                self.sync_prune(mat, n, &mut pool, &mut flags, threshold);
            }

            for n in 0..self.n_data {
                self.inter_insert(mat, n, threshold);
            }
        }

        fn get_neighbors_for_link<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            q: usize,
            pool: &mut Vec<(F, R, bool)>,
            flags: &mut HashSetBit<R>,
        ) {
            let l = self.params.l;
            let q_r = unsafe { R::from(q).unwrap_unchecked() };
            flags.insert(q_r);

            for &(_, nid) in self.graph.view_neighbors(q_r) {
                for &(_, nnid) in self.graph.view_neighbors(nid) {
                    if flags.contains(&nnid) {
                        continue;
                    }
                    flags.insert(nnid);
                    let dist =
                        self._get_dist(mat, q, unsafe { nnid.to_usize().unwrap_unchecked() });
                    pool.push((dist, nnid, true));
                    if pool.len() >= l {
                        break;
                    }
                }
                if pool.len() >= l {
                    break;
                }
            }
        }

        fn sync_prune<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            q: usize,
            pool: &mut Vec<(F, R, bool)>,
            flags: &mut HashSetBit<R>,
            threshold: F,
        ) {
            let range = self.params.range;
            let q_r = unsafe { R::from(q).unwrap_unchecked() };

            for &(_, id) in self.graph.view_neighbors(q_r) {
                if flags.contains(&id) {
                    continue;
                }
                let dist: F = self._get_dist(mat, q, unsafe { id.to_usize().unwrap_unchecked() });
                pool.push((dist, id, true));
            }

            // sort asc
            pool.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut start = 0;
            let mut result: Vec<(F, R, bool)> = Vec::with_capacity(range);
            if pool[start].1 == q_r {
                start += 1;
            }
            result.push(pool[start]); // correct index of start?

            while result.len() < range && start < pool.len() {
                let p = &pool[start];
                start += 1;
                let mut occlude = false;
                for t in 0..result.len() {
                    if p.1 == result[t].1 {
                        occlude = true;
                        break;
                    }
                    let djk: F = self._get_dist(
                        mat,
                        unsafe { result[t].1.to_usize().unwrap_unchecked() },
                        unsafe { p.1.to_usize().unwrap_unchecked() },
                    );
                    let cos_ij = (p.0 + result[t].0 - djk)
                        / F::from(2).unwrap()
                        / <F as num::Float>::sqrt(p.0 * result[t].0);
                    if cos_ij > threshold {
                        occlude = true;
                        break;
                    }
                }
                if !occlude {
                    result.push(*p);
                }
            }

            let des_pool: &mut Vec<(F, R)> = &mut self.tmp_graph[q];
            for t in 0..result.len() {
                des_pool[t].0 = result[t].0;
                des_pool[t].1 = result[t].1;
            }
            if result.len() < range {
                des_pool[result.len()].0 = F::neg_infinity();
            }
        }

        fn inter_insert<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M, n: usize, threshold: F) {
            let range = self.params.range;
            let n_r = unsafe { R::from(n).unwrap_unchecked() };
            let src_pool = self.tmp_graph[n].clone();
            assert_eq!(range, src_pool.len(), "Len mismatch.");
            // so we can get the dist for ´djk´ from a mutable self
            let tmp_graph = &mut self.tmp_graph as *mut Vec<Vec<(F, R)>>;

            unsafe {
                for i in 0..range {
                    if src_pool[i].0 == F::neg_infinity() {
                        break;
                    }
                    let sn: (F, R) = (src_pool[i].0, n_r);
                    let des = src_pool[i].1.to_usize().unwrap_unchecked();
                    let des_pool = &mut (*tmp_graph)[des];

                    let mut tmp_pool: Vec<(F, R)> = Vec::with_capacity(range);
                    let mut dup = false;
                    for j in 0..range {
                        if des_pool[j].0 == F::neg_infinity() {
                            break;
                        }
                        if n_r == des_pool[j].1 {
                            dup = true;
                            break;
                        }
                        tmp_pool.push(des_pool[j]);
                    }
                    if dup {
                        continue;
                    }

                    tmp_pool.push(sn);

                    if tmp_pool.len() > range {
                        tmp_pool.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        let mut result: Vec<(F, R)> = Vec::with_capacity(range);
                        let mut start: usize = 0;
                        result.push(tmp_pool[start]);

                        start += 1;
                        while result.len() < range && start < tmp_pool.len() {
                            let p = tmp_pool[start];
                            start += 1;
                            let mut occlude = false;

                            for t in 0..result.len() {
                                if p.1 == result[t].1 {
                                    occlude = true;
                                    break;
                                }
                                let djk = self._get_dist(
                                    mat,
                                    result[t].1.to_usize().unwrap_unchecked(),
                                    p.1.to_usize().unwrap_unchecked(),
                                );
                                let cos_ij = (p.0 + result[t].0 - djk)
                                    / F::from(2).unwrap()
                                    / <F as num::Float>::sqrt(p.0 * result[t].0);

                                if cos_ij > threshold {
                                    occlude = true;
                                    break;
                                }
                            }
                            if !occlude {
                                result.push(p);
                            }
                        }

                        for t in 0..result.len() {
                            des_pool[t] = result[t];
                        }
                        if result.len() < range {
                            des_pool[result.len()].0 = F::neg_infinity();
                        }
                    } else {
                        for t in 0..range {
                            if des_pool[t].0 == F::neg_infinity() {
                                des_pool[t] = sn;
                                if t + 1 < range {
                                    des_pool[t + 1].0 = F::neg_infinity();
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        fn dfs_expand<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
            let n_try = self.params.n_try.min(self.n_data);
            let range = self.params.range;

            // select random entry points
            random_unique_uint(self.n_data, n_try)
                .iter()
                .for_each(|&e| self.entry_points.push(e));

            for i in 0..n_try {
                let rootid = self.entry_points[i];
                let mut flags: HashSetBit<R> = HashSetBit::new(self.n_data);
                let mut myqueue: VecDeque<R> = VecDeque::new();
                myqueue.push_back(rootid);
                flags.insert(rootid);

                // can't be empty, gets cleared first anyway
                let mut uncheck_set: Vec<R> = vec![R::zero()];

                while uncheck_set.len() > 0 {
                    while !myqueue.is_empty() {
                        let q_front = myqueue.pop_front().unwrap();
                        for &(_, child) in self.graph.view_neighbors(q_front) {
                            if flags.contains(&child) {
                                continue;
                            }
                            flags.insert(child);
                            myqueue.push_back(child);
                        }
                    }

                    uncheck_set.clear();
                    for j in 0..self.n_data {
                        let j_r = unsafe { R::from(j).unwrap_unchecked() };
                        if flags.contains(&j_r) {
                            continue;
                        }
                        uncheck_set.push(j_r);
                    }

                    if uncheck_set.len() > 0 {
                        for j in 0..self.n_data {
                            let j_r = unsafe { R::from(j).unwrap_unchecked() };
                            if flags.contains(&j_r) && self.graph.view_neighbors(j_r).len() < range
                            {
                                let dist = self._get_dist(mat, j, unsafe {
                                    uncheck_set[0].to_usize().unwrap_unchecked()
                                });
                                self.graph.add_edge_with_weight(j_r, uncheck_set[0], dist);
                                break;
                            }
                        }
                        myqueue.push_back(uncheck_set[0]);
                        flags.insert(uncheck_set[0]);
                    }
                }
            }
        }

        // DEAD (ORIGINAL) CODE just here for completeness

        // fn _init_graph<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
        //     let dimension_ = mat.n_cols();
        //     let mut center: Vec<F> = vec![F::zero(); dimension_];
        //     for i in 0..self.n_data {
        //         for j in 0..dimension_ {
        //             center[j] += *mat.get_row(i).get(j).unwrap();
        //         }
        //     }

        //     let nd_as_f: F = F::from(self.n_data).unwrap();
        //     for j in 0..dimension_ {
        //         center[j] /= nd_as_f;
        //     }
        //     let mut retset: Vec<(F, R, bool)> = Vec::with_capacity(self.params.l + 1);
        //     let mut fullset: Vec<(F, R, bool)> = Vec::with_capacity(self.params.range);
        //     self._get_neighbors(mat, center, &mut retset, &mut fullset);
        // }

        // fn _get_neighbors<M: MatrixDataSource<F> + Sync>(
        //     &mut self,
        //     mat: &M,
        //     query: Vec<F>,
        //     retset: &mut Vec<(F, R, bool)>,
        //     fullset: &mut Vec<(F, R, bool)>,
        // ) {
        //     let mut l = self.params.l;
        //     retset.resize(l + 1, (F::neg_infinity(), R::max_value(), false));
        //     let nd_r = unsafe { R::from(self.n_data).unwrap_unchecked() };
        //     let init_ids: Vec<R> = random_unique_uint(self.n_data, l);
        //     let mut flags: HashSetBit<R> = HashSetBit::new(self.n_data);
        //     l = 0;
        //     for i in 0..init_ids.len() {
        //         let id = init_ids[i];
        //         if id >= nd_r {
        //             continue;
        //         }
        //         let dist = self._get_dist_with_input_vec(
        //             mat,
        //             unsafe { id.to_usize().unwrap_unchecked() },
        //             query.clone(),
        //         );
        //         retset[i] = (dist, id, true);
        //         flags.insert(id);
        //         l += 1;
        //     }

        //     // std::sort(retset.begin(), retset.begin() + L); // ??
        //     // sorted asc
        //     retset.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        //     let mut k = 0; // originally int ?
        //     while k < l {
        //         let mut nk = l;
        //         if retset[k].2 {
        //             retset[k].2 = false;
        //             let n = retset[k].1;
        //             for &(_, id) in self.graph.view_neighbors(n) {
        //                 if flags.contains(&id) {
        //                     continue;
        //                 }
        //                 flags.insert(id);
        //                 let dist: F = self._get_dist_with_input_vec(
        //                     mat,
        //                     unsafe { id.to_usize().unwrap_unchecked() },
        //                     query.clone(),
        //                 );
        //                 let nn = (dist, id, true);
        //                 fullset.push(nn);
        //                 if dist >= retset[l - 1].0 {
        //                     continue;
        //                 }
        //                 let r = self._insert_into_pool(retset, l, nn);

        //                 if l + 1 < retset.len() {
        //                     l += 1;
        //                 }
        //                 if r < nk {
        //                     nk = r;
        //                 }
        //             }
        //         }
        //         if nk <= k {
        //             k = nk;
        //         } else {
        //             k += 1;
        //         }
        //     }
        // }

        // // Check correctness
        // // Currently swapped against random_unique_uint(nd_, n_try);
        // fn _gen_random_like_ssg(&self, size: usize, n: usize) -> Vec<R> {
        //     let mut ret: Vec<R> = Vec::with_capacity(n);
        //     let between = Uniform::from(0..i32::MAX); // correct?
        //     let mut rng = rand::thread_rng();
        //     for i in 0..size {
        //         ret[i] = R::from(between.sample(&mut rng) % (n - size) as i32).unwrap();
        //     }
        //     ret
        // }

        // Originally returns int ?
        // Basically a sorted insert
        // -> obsolete when using a DualHeap?
        // Find the location to insert
        // @Returns index of newly inserted element
        // fn _insert_into_pool(
        //     &self,
        //     pool: &mut Vec<(F, R, bool)>,
        //     k: usize,
        //     nn: (F, R, bool),
        // ) -> usize {
        //     let mut left: usize = 0;
        //     let mut right: usize = k - 1;

        //     if pool[left].0 > nn.0 {
        //         pool.insert(left, nn); // correct index?
        //         return left;
        //     }
        //     if pool[right].0 < nn.0 {
        //         pool[k] = nn;
        //         return k;
        //     }
        //     while left < right - 1 {
        //         let mid = (left + right) / 2;
        //         if pool[mid].0 > nn.0 {
        //             right = mid;
        //         } else {
        //             left = mid;
        //         }
        //     }
        //     // check equal ID
        //     while left > 0 {
        //         if pool[left].0 < nn.0 {
        //             break;
        //         }
        //         if pool[left].1 == nn.1 {
        //             return k + 1;
        //         }
        //         left -= 1;
        //     }
        //     if pool[left].1 == nn.1 || pool[right].1 == nn.1 {
        //         return k + 1;
        //     }
        //     pool.insert(right, nn); // correct index?
        //     return right;
        // }

        // fn _optimize_graph(&self) {
        //     unimplemented!("Not used during build and marked as 'use after build or load");
        //     /* Seems to optimize the memory layout and normalizes values but not any further
        //     optimization on the graph-architecture itself */
        // }
        // fn _strong_connect(&self) {
        //     unimplemented!(
        //         "Marked as buggy in original code, so skipping implementation of this\
        //      and following functions: dfs, check_edge, find_root."
        //     );
        // }
        // fn _dfs(&self) {
        //     unimplemented!("see _strong_connect()");
        // }
        // fn _check_edge(&self) {
        //     unimplemented!("see _strong_connect()");
        // }
        // fn _find_root(&self) {
        //     unimplemented!("see _strong_connect()");
        // }
    }

    impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
        NSSGStyleBuilder<R, F, Dist> for NSSGBuilder<R, F, Dist>
    {
        type Params = NSSGParams<F>;
        type Graph = NSSGBuildGraph<R, F>;
        fn _graph(&self) -> &NSSGBuildGraph<R, F> {
            &self.graph
        }
        #[inline(always)]
        fn _mut_graph(&mut self) -> &mut NSSGBuildGraph<R, F> {
            &mut self.graph
        }
        #[inline(always)]
        fn _mut_graph_for_random_init(&mut self) -> &mut NSSGBuildGraph<R, F> {
            &mut self.graph
        }
        #[inline(always)]
        fn _dist(&self) -> &Dist {
            &self.dist
        }
        #[inline(always)]
        fn _initial_degree(&self) -> usize {
            self.params.range
        }
        #[inline(always)]
        fn _get_entry_points(&self) -> Option<Vec<R>> {
            if self.entry_points.len() > 0 {
                Some(self.entry_points.clone())
            } else {
                None
            }
        }
        #[inline(always)]
        fn _into_graph_dist(self) -> (NSSGBuildGraph<R, F>, Dist) {
            (self.graph, self.dist)
        }
        fn new<M: MatrixDataSource<F> + Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
            let n_data = mat.n_rows();
            assert!(n_data < R::max_value().to_usize().unwrap());
            let graph = NSSGBuildGraph::new();
            let tmp_graph = Vec::new();
            let entry_points: Vec<R> = Vec::new();
            Self {
                _phantom: std::marker::PhantomData,
                n_data,
                params,
                graph,
                tmp_graph,
                dist,
                entry_points,
            }
        }

        /// Build the graph.
        fn train<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            input_graph: NSSGInputGraphData<R, F>,
        ) {
            // Data size sanity check; arbitrary threshold
            if self.n_data <= 30 {
                if self.n_data <= 1 {
                    if self.n_data == 1 {
                        self.graph.add_node();
                    }
                } else {
                    // Return a fully connected graph
                    for i in 0..self.n_data {
                        self.graph.add_node();
                        for j in 0..i {
                            let dist = self._get_dist(mat, i, j);
                            let i_r = unsafe { R::from(i).unwrap_unchecked() };
                            let j_r = unsafe { R::from(j).unwrap_unchecked() };
                            self.graph.add_edge_with_weight(i_r, j_r, dist);
                        }
                    }
                }
                return;
            }
            if self.params.range > self.n_data {
                self.params.range = self.n_data - 1;
            }
            if self.params.l > self.n_data {
                self.params.l = self.n_data - 1;
            }

            let range = self.params.range;

            // set angle as suggested by supervisor
            if self.params.derive_angle_from_dim {
                let d = mat.n_cols() as f64;
                let factor = 2.0 as f64; // 4.0;
                self.params.angle =
                    F::from(((d - factor) / (factor * d - factor)).acos().to_degrees()).unwrap();
            }

            self.graph.reserve(self.n_data);
            self.tmp_graph.resize(
                self.n_data,
                vec![(F::neg_infinity(), R::max_value()); range],
            );

            // init the graph with the provided input_graph
            self.init_graph(input_graph, mat);

            // self._init_graph(mat); // appears to be a dead function
            self.link(mat);

            // write from tmp_graph into final graph
            for i in 0..self.n_data {
                let pool: &Vec<(F, R)> = &self.tmp_graph[i];
                let mut pool_size = 0;
                assert_eq!(range, pool.len(),);
                for j in 0..range {
                    if pool[j].0 == F::neg_infinity() {
                        break;
                    }
                    pool_size = j;
                }
                pool_size += 1;
                let adj = self
                    .graph
                    .get_adj_mut(unsafe { R::from(i).unwrap_unchecked() });
                adj.resize(pool_size, (F::zero(), R::zero()));
                for j in 0..pool_size {
                    adj[j] = (pool[j].0, pool[j].1);
                }
            }
            std::mem::swap(&mut self.tmp_graph, &mut Vec::new());

            self.dfs_expand(mat);
            // self.strong_connect(); // originally commented as buggy
        }
    }
}

pub struct NSSGParallelBuilder<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
> {
    _phantom: std::marker::PhantomData<F>,
    n_data: usize,
    params: NSSGParams<F>,
    graph: NSSGBuildGraph<R, F>,
    tmp_graph: Vec<Vec<(F, R)>>,
    dist: Dist,
    entry_points: Vec<R>,
    n_threads: usize,
    node_locks: Vec<Mutex<()>>,
}

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    NSSGParallelBuilder<R, F, Dist>
{
    fn init_graph<M: MatrixDataSource<F> + Sync>(
        &mut self,
        input_graph: NSSGInputGraphData<R, F>,
        mat: &M,
    ) {
        match input_graph {
            NSSGInputGraphData::ADJACENCY(adj) => {
                self.graph.adjacency = adj;
            }
            NSSGInputGraphData::DEG(params) => {
                let index = DEGParallelBuilder::<R, F, Dist>::build(
                    mat,
                    self._dist().clone(),
                    params.unwrap_or(DEGParams::new()),
                );
                self.load_from_input_graph(mat, index.graph().as_dir_lol_graph());
            }
            NSSGInputGraphData::EFANNA(params) => {
                let index = EfannaParallelMaxHeapBuilder::<R, F, Dist>::build(
                    mat,
                    self._dist().clone(),
                    params.unwrap_or(
                        EfannaParams::new()
                            .with_k(self.params.range)
                            .with_l(self.params.range),
                    ),
                );
                self.load_from_input_graph(mat, index.graph().as_dir_lol_graph());
            }
            NSSGInputGraphData::GRAPH(graph) => {
                assert!(
                    mat.n_rows() == graph.n_vertices(),
                    "Input graph and input data not matching in number of rows/vertices"
                );
                self.load_from_input_graph(mat, graph);
            }
            NSSGInputGraphData::RANDOM(degree) => {
                self.init_random(mat, degree);
            }
            NSSGInputGraphData::RNN(params) => {
                let index = RNNDescentBuilder::<R, F, Dist>::build(
                    mat,
                    self._dist().clone(),
                    params.unwrap_or(RNNParams::new()),
                );
                self.load_from_input_graph(mat, index.graph().as_dir_lol_graph());
            }
        }
    }

    /// Assumes the neighbors are sorted asc
    /// which for Efanna is the case by default
    fn load_from_input_graph<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        input_graph: DirLoLGraph<R>,
    ) {
        let range = self.params.range;
        (0..self.n_data).for_each(|i| {
            let i_r = R::from(i).unwrap();
            self.graph.add_node();
            let mut cnt = 0;
            for &n in input_graph.view_neighbors(i_r) {
                if cnt >= range {
                    break;
                }
                self.graph.add_edge_with_weight(
                    i_r,
                    n,
                    self._get_dist(mat, i, n.to_usize().unwrap()),
                );
                cnt += 1;
            }
        });
    }

    fn link<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
        let angle = unsafe { F::from(self.params.angle).unwrap_unchecked() };
        // threshold = cos(angle / 180 * PI)
        let threshold = <F as num::Float>::cos(
            angle / unsafe { F::from(180).unwrap_unchecked() }
                * unsafe { F::from(f64::consts::PI).unwrap_unchecked() },
        );

        let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
        (0..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .par_bridge()
            .for_each(|chunk| unsafe {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                let mut pool: Vec<(F, R, bool)> = Vec::with_capacity(self.params.range);
                let mut flags: HashSetBit<R> = HashSetBit::new(self.n_data);
                chunk.for_each(|n| {
                    pool.clear();
                    flags.clear();
                    (*unsafe_self_ref).get_neighbors_for_link(mat, n, &mut pool, &mut flags);
                    flags.clear();
                    (*unsafe_self_ref).sync_prune(mat, n, &mut pool, &mut flags, threshold);
                });
            });

        (0..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .par_bridge()
            .for_each(|chunk| unsafe {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                chunk.for_each(|n| {
                    (*unsafe_self_ref).inter_insert(mat, n, threshold);
                });
            });
    }

    fn get_neighbors_for_link<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        q: usize,
        pool: &mut Vec<(F, R, bool)>,
        flags: &mut HashSetBit<R>,
    ) {
        let l = self.params.l;
        let q_r = unsafe { R::from(q).unwrap_unchecked() };
        flags.insert(q_r);

        for &(_, nid) in self.graph.view_neighbors(q_r) {
            for &(_, nnid) in self.graph.view_neighbors(nid) {
                if flags.contains(&nnid) {
                    continue;
                }
                flags.insert(nnid);
                let dist = self._get_dist(mat, q, unsafe { nnid.to_usize().unwrap_unchecked() });
                pool.push((dist, nnid, true));
                if pool.len() >= l {
                    break;
                }
            }
            if pool.len() >= l {
                break;
            }
        }
    }

    fn sync_prune<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        q: usize,
        pool: &mut Vec<(F, R, bool)>,
        flags: &mut HashSetBit<R>,
        threshold: F,
    ) {
        let range = self.params.range;
        let q_r = unsafe { R::from(q).unwrap_unchecked() };

        for i in 0..pool.len() {
            flags.insert(pool[i].1);
        }

        for &(_, id) in self.graph.view_neighbors(q_r) {
            if flags.contains(&id) {
                continue;
            }
            let dist: F = self._get_dist(mat, q, unsafe { id.to_usize().unwrap_unchecked() });
            pool.push((dist, id, true));
        }

        // sort asc
        pool.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut result: Vec<(F, R, bool)> = Vec::with_capacity(range);
        let mut start = 0;
        if pool[start].1 == q_r {
            start += 1;
        }
        result.push(pool[start]);

        // double check with (++start)
        while result.len() < range && start < pool.len() {
            start += 1;
            if start >= pool.len() {
                break;
            }
            let &(p_dist, p_id, p_bool) = &pool[start];
            let mut occlude = false;
            for &(dist_res, id_res, _) in result.as_slice() {
                if p_id == id_res {
                    occlude = true;
                    break;
                }
                let djk: F = self._get_dist(
                    mat,
                    unsafe { id_res.to_usize().unwrap_unchecked() },
                    unsafe { p_id.to_usize().unwrap_unchecked() },
                );
                let cos_ij = (p_dist + dist_res - djk)
                    / F::from(2).unwrap()
                    / <F as num::Float>::sqrt(p_dist * dist_res);
                if cos_ij > threshold {
                    occlude = true;
                    break;
                }
            }
            if !occlude {
                result.push((p_dist, p_id, p_bool));
            }
        }

        let des_pool: &mut Vec<(F, R)> = &mut self.tmp_graph[q];
        for t in 0..result.len() {
            des_pool[t].0 = result[t].0;
            des_pool[t].1 = result[t].1;
        }
        if result.len() < range {
            des_pool[result.len()].0 = F::neg_infinity();
        }
    }

    fn inter_insert<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M, n: usize, threshold: F) {
        let range = self.params.range;
        let n_r = unsafe { R::from(n).unwrap_unchecked() };
        let tmp_graph = &mut self.tmp_graph as *mut Vec<Vec<(F, R)>>;
        let src_pool = &self.tmp_graph[n];
        assert_eq!(range, src_pool.len(), "Len mismatch.");
        // so we can get the dist for ´djk´ from a mutable self

        unsafe {
            let mut tmp_pool: Vec<(F, R)> = Vec::with_capacity(range);
            for &(dist_src, id_src) in src_pool {
                if dist_src == F::neg_infinity() {
                    break;
                }
                let sn: (F, R) = (dist_src, n_r);
                let des = id_src.to_usize().unwrap_unchecked();
                let des_pool = (*tmp_graph)[des].as_mut_slice();

                tmp_pool.clear();
                let mut dup = false;
                let guard = self.node_locks[des].lock().unwrap();
                for &mut (dist_des, id_des) in &mut *des_pool {
                    if dist_des == F::neg_infinity() {
                        break;
                    }
                    if n_r == id_des {
                        dup = true;
                        break;
                    }
                    tmp_pool.push((dist_des, id_des));
                }
                drop(guard);

                if dup {
                    continue;
                }

                tmp_pool.push(sn);

                if tmp_pool.len() > range {
                    tmp_pool.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    let mut result: Vec<(F, R)> = Vec::with_capacity(range);
                    let mut start: usize = 0;
                    result.push(tmp_pool[start]);

                    while result.len() < range && start < tmp_pool.len() {
                        start += 1;
                        if start >= tmp_pool.len() {
                            break;
                        }
                        let p = tmp_pool[start];
                        let mut occlude = false;

                        for t in 0..result.len() {
                            if p.1 == result[t].1 {
                                occlude = true;
                                break;
                            }
                            let djk = self._get_dist(
                                mat,
                                result[t].1.to_usize().unwrap_unchecked(),
                                p.1.to_usize().unwrap_unchecked(),
                            );
                            let cos_ij = (p.0 + result[t].0 - djk)
                                / F::from(2).unwrap()
                                / <F as num::Float>::sqrt(p.0 * result[t].0);

                            if cos_ij > threshold {
                                occlude = true;
                                break;
                            }
                        }
                        if !occlude {
                            result.push(p);
                        }
                    }
                    let guard = self.node_locks[des].lock().unwrap();
                    for t in 0..result.len() {
                        des_pool[t] = result[t];
                    }
                    if result.len() < range {
                        des_pool[result.len()].0 = F::neg_infinity();
                    }
                    drop(guard);
                } else {
                    let guard = self.node_locks[des].lock().unwrap();
                    for t in 0..range {
                        if des_pool[t].0 == F::neg_infinity() {
                            des_pool[t] = sn;
                            if t + 1 < range {
                                des_pool[t + 1].0 = F::neg_infinity();
                            }
                            break;
                        }
                    }
                    drop(guard);
                }
            }
        }
    }

    fn dfs_expand<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
        let n_try = self.params.n_try.min(self.n_data);
        let range = self.params.range;

        // select random entry points
        random_unique_uint(self.n_data, n_try)
            .iter()
            .for_each(|&e| self.entry_points.push(e));

        let mut flags: HashSetBit<R> = HashSetBit::new(self.n_data);
        let mut myqueue: VecDeque<R> = VecDeque::with_capacity(range);
        for i in 0..n_try {
            flags.clear();
            myqueue.clear();
            let rootid = self.entry_points[i];
            myqueue.push_back(rootid);
            flags.insert(rootid);

            // can't be empty, gets cleared first anyway
            let mut uncheck_set: Vec<R> = vec![R::zero()];

            while uncheck_set.len() > 0 {
                while !myqueue.is_empty() {
                    let q_front = myqueue.pop_front().unwrap();
                    for &(_, child) in self.graph.view_neighbors(q_front) {
                        if flags.contains(&child) {
                            continue;
                        }
                        flags.insert(child);
                        myqueue.push_back(child);
                    }
                }

                uncheck_set.clear();
                for j in 0..self.n_data {
                    let j_r = unsafe { R::from(j).unwrap_unchecked() };
                    if flags.contains(&j_r) {
                        continue;
                    }
                    uncheck_set.push(j_r);
                }

                if uncheck_set.len() > 0 {
                    for j in 0..self.n_data {
                        let j_r = unsafe { R::from(j).unwrap_unchecked() };
                        if flags.contains(&j_r) && self.graph.view_neighbors(j_r).len() < range {
                            let dist = self._get_dist(mat, j, unsafe {
                                uncheck_set[0].to_usize().unwrap_unchecked()
                            });
                            self.graph.add_edge_with_weight(j_r, uncheck_set[0], dist);
                            break;
                        }
                    }
                    myqueue.push_back(uncheck_set[0]);
                    flags.insert(uncheck_set[0]);
                }
            }
        }
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    NSSGStyleBuilder<R, F, Dist> for NSSGParallelBuilder<R, F, Dist>
{
    type Params = NSSGParams<F>;
    type Graph = NSSGBuildGraph<R, F>;
    fn _graph(&self) -> &NSSGBuildGraph<R, F> {
        &self.graph
    }
    #[inline(always)]
    fn _mut_graph(&mut self) -> &mut NSSGBuildGraph<R, F> {
        &mut self.graph
    }
    #[inline(always)]
    fn _mut_graph_for_random_init(&mut self) -> &mut NSSGBuildGraph<R, F> {
        &mut self.graph
    }
    #[inline(always)]
    fn _dist(&self) -> &Dist {
        &self.dist
    }
    #[inline(always)]
    fn _initial_degree(&self) -> usize {
        self.params.range
    }
    #[inline(always)]
    fn _get_entry_points(&self) -> Option<Vec<R>> {
        if self.entry_points.len() > 0 {
            Some(self.entry_points.clone())
        } else {
            None
        }
    }
    #[inline(always)]
    fn _into_graph_dist(self) -> (NSSGBuildGraph<R, F>, Dist) {
        (self.graph, self.dist)
    }
    fn new<M: MatrixDataSource<F> + Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
        let n_data = mat.n_rows();
        assert!(n_data < R::max_value().to_usize().unwrap());
        let graph = NSSGBuildGraph::new();
        let tmp_graph = Vec::new();
        let entry_points: Vec<R> = Vec::new();
        Self {
            _phantom: std::marker::PhantomData,
            n_data,
            params,
            graph,
            tmp_graph,
            dist,
            entry_points,
            n_threads: rayon::current_num_threads(),
            node_locks: (0..n_data).map(|_| Mutex::new(())).collect(),
        }
    }
    /// Build the graph.
    fn train<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        input_graph: NSSGInputGraphData<R, F>,
    ) {
        // Data size sanity check; arbitrary threshold
        if self.n_data <= 30 {
            if self.n_data <= 1 {
                if self.n_data == 1 {
                    self.graph.add_node();
                }
            } else {
                // Return a fully connected graph
                for i in 0..self.n_data {
                    self.graph.add_node();
                    for j in 0..i {
                        let dist = self._get_dist(mat, i, j);
                        let i_r = unsafe { R::from(i).unwrap_unchecked() };
                        let j_r = unsafe { R::from(j).unwrap_unchecked() };
                        self.graph.add_edge_with_weight(i_r, j_r, dist);
                    }
                }
            }
            return;
        }
        if self.params.range > self.n_data {
            self.params.range = self.n_data - 1;
        }
        if self.params.l > self.n_data {
            self.params.l = self.n_data - 1;
        }

        let range = self.params.range;

        // set angle as suggested by supervisor
        if self.params.derive_angle_from_dim {
            let d = mat.n_cols() as f64;
            let factor = 2.0 as f64; // 4.0;
            self.params.angle =
                F::from(((d - factor) / (factor * d - factor)).acos().to_degrees()).unwrap();
        }

        self.graph.reserve(self.n_data);
        self.tmp_graph.resize(
            self.n_data,
            vec![(F::neg_infinity(), R::max_value()); range],
        );

        // init the graph with the provided input_graph
        self.init_graph(input_graph, mat);

        self.link(mat);

        // write from tmp_graph into final graph
        let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
        (0..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .par_bridge()
            .for_each(|chunk| {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                let unsafe_tmp_graph_ref =
                    std::ptr::addr_of!(self.tmp_graph) as *mut Vec<Vec<(F, R)>>;

                chunk.for_each(|i| unsafe {
                    let pool: &Vec<(F, R)> = &(*unsafe_tmp_graph_ref)[i];
                    let mut pool_size = 0;
                    assert_eq!(range, pool.len(), "Len of range and pool not matching.");
                    for j in 0..range {
                        if pool[j].0 == F::neg_infinity() {
                            break;
                        }
                        pool_size = j;
                    }
                    pool_size += 1;
                    let adj = (*unsafe_self_ref)
                        .graph
                        .get_adj_mut(R::from(i).unwrap_unchecked());
                    adj.resize(pool_size, (F::zero(), R::zero()));
                    for j in 0..pool_size {
                        adj[j] = (pool[j].0, pool[j].1);
                    }
                });
            });

        std::mem::swap(&mut self.tmp_graph, &mut Vec::new());

        self.dfs_expand(mat);
    }
}

// -------------- TESTS --------------

#[cfg(test)]
mod tests {
    use crate::{efanna::*, nssg::*};
    #[allow(unused)]
    use crate::{
        print_index_build_info, search_as_qps,
        utils::{
            eval::{calc_recall, get_test_data, init_with_dataset, EvalDataSet},
            index_stats::IndexGraphStats,
        },
    };
    #[allow(unused)]
    use graphidx::{
        indices::{bruteforce_neighbors, GraphIndex},
        measures::SquaredEuclideanDistance,
    };
    #[allow(unused)]
    use ndarray::{Array1, Array2, Axis, Slice};
    #[allow(unused)]
    use ndarray_rand::rand_distr::Normal;
    use std::time::Instant;

    type R = usize;
    type F = f32;
    type Dist = SquaredEuclideanDistance<F>;

    #[test]
    fn nssg_construction() {
        let (_nd, _nq, _d, _k, data) = get_test_data(true);

        let graph_time = Instant::now();
        type BuilderType = NSSGParallelBuilder<R, F, Dist>;
        // type BuilderType = crate::nssg::single_threaded::NSSGBuilder<R, F, Dist>;
        let params = NSSGParams::new();

        let index = BuilderType::build(
            data.clone(),
            Dist::new(),
            params,
            NSSGInputGraphData::EFANNA(None),
        );
        println!("NSSG-1 Graph construction: {:.2?}", graph_time.elapsed());
        print_index_build_info!(index);
    }

    // for fair comparison use the same input_graph
    #[test]
    fn nssg_compare_construction() {
        let (_nd, _nq, _d, _k, data) = get_test_data(true);
        let degree = 50;

        let params = NSSGParams::new();
        let graph_time = Instant::now();
        let input_index = EfannaParallelMaxHeapBuilder::<R, F, Dist>::build(
            data.clone(),
            Dist::new(),
            EfannaParams::new().with_k(2 * degree).with_l(2 * degree),
        );

        println!("Efanna Graph construction: {:.2?}", graph_time.elapsed());

        let graph_time = Instant::now();
        type BuilderType = crate::nssg::NSSGParallelBuilder<R, F, Dist>;
        // crate::nssg::single_threaded::NSSGBuilder::<R, _, _>;

        let index1 = BuilderType::build(
            data.clone(),
            Dist::new(),
            params,
            NSSGInputGraphData::GRAPH(input_index.graph().as_dir_lol_graph()),
        );
        println!("NSSG-1 Graph construction: {:.2?}", graph_time.elapsed());
        print_index_build_info!(index1);
    }

    #[test]
    fn nssg_query() {
        // let (_nd, nq, _d, k, data) = get_test_data(true); // normal data
        // let queries = data.slice_axis(Axis(0), Slice::from(0..nq));
        // un-indexed queries
        let (data, queries, ground_truth, _nd, nq, _d, k) = init_with_dataset(&EvalDataSet::AUDIO);
        let degree = 50;

        /* Build and translate NSSG Graph */
        let build_time = Instant::now();
        let dist = SquaredEuclideanDistance::new();
        let params = NSSGParams::new().with_range(degree);
        // .with_l(100)
        // .with_angle(60.0)
        // .with_max_build_frontier_size(Some(3 * k))
        // .with_max_build_heap_size(2 * 254)

        let index_input =
            DEGParallelBuilder::<R, F, Dist>::build(data.view(), dist.clone(), DEGParams::new());

        println!(
            "Graph construction Input-Index: {:.2?}",
            build_time.elapsed()
        );
        print_index_build_info!(index_input);

        let build_time = Instant::now();
        type BuilderType2 = NSSGParallelBuilder<R, F, Dist>;
        let index_par = BuilderType2::build(
            data.view(),
            SquaredEuclideanDistance::new(),
            params,
            // NSSGInputGraphData::EFANNA(None),
            NSSGInputGraphData::GRAPH(index_input.graph().as_dir_lol_graph()),
        );
        println!(
            "Graph construction ({:?}): {:.2?}",
            std::any::type_name::<BuilderType2>(),
            build_time.elapsed()
        );
        print_index_build_info!(index_par);

        // // Brute force queries
        // let bruteforce_time = Instant::now();
        // let (ground_truth, _) = bruteforce_neighbors(&data, &queries, &Dist::new(), k);
        // println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());

        let ssg_time = Instant::now();
        let (ssg_ids2, _ssg_dists2) = index_par.greedy_search_batch(&queries, k, 2 * k);
        println!("NSSG queries: {:.2?}", ssg_time.elapsed());

        let rnn_time = Instant::now();
        let (index_input_ids, _rnn_dists2) = index_input.greedy_search_batch(&queries, k, 2 * k);
        println!("Index-Input queries: {:.2?}", rnn_time.elapsed());

        // single threaded search
        // let graph_name = "SSG-Par";
        let ground_truth_view = ground_truth.view();
        // let (_query_time, _qps, _recall) = search_as_qps!(
        //     index_par,
        //     queries,
        //     ground_truth_view,
        //     k,
        //     2 * k,
        //     nq,
        //     graph_name,
        //     true
        // );

        calc_recall(
            ground_truth_view,
            &index_input_ids,
            nq,
            k,
            "Index-Input",
            true,
        );
        calc_recall(ground_truth_view, &ssg_ids2, nq, k, "SSG-2-Par", true);
    }

    /* Quick observations on short and small testing
    - No clear benefit in query speed or recall could be found, especially not for
    larger max_heap_sizes */
    #[test]
    fn nssg_query_compare_angles() {
        let (data, queries, ground_truth, _nd, nq, d, k) = init_with_dataset(&EvalDataSet::AUDIO);

        let d_ = d as f64;
        let angle = ((d_ - 2.0) / (2.0 * d_ - 2.0)).acos().to_degrees() as F;

        let _params_efanna = EfannaParams::new().with_l(100).with_k(100);
        let input_index = EfannaParallelMaxHeapBuilder::<R, F, Dist>::build(
            data.clone(),
            Dist::new(),
            _params_efanna,
        );

        let params = NSSGParams::new();
        type BuilderType = NSSGParallelBuilder<R, F, Dist>;

        let index1 = BuilderType::build(
            data.clone(),
            Dist::new(),
            params.clone().with_angle(60.0),
            NSSGInputGraphData::GRAPH(input_index.graph().as_dir_lol_graph()),
        );

        let index2 = BuilderType::build(
            data.clone(),
            Dist::new(),
            params.clone().with_angle(angle),
            NSSGInputGraphData::GRAPH(input_index.graph().as_dir_lol_graph()),
        );

        let angle2 = ((d_ - 4.0) / (4.0 * d_ - 4.0)).acos().to_degrees() as F;
        println!("Angle1: {}, Angle2: {}, d: {}", angle, angle2, d_);

        let index3 = BuilderType::build(
            data.clone(),
            Dist::new(),
            params.clone().with_angle(angle2),
            NSSGInputGraphData::GRAPH(input_index.graph().as_dir_lol_graph()),
        );

        let ground_truth_view = ground_truth.view();

        let max_heap_size = 1 * k;
        let ssg_time = Instant::now();
        let (ssg_ids1, _ssg_dists1) = index1.greedy_search_batch(&queries, k, max_heap_size);
        println!("SSG queries 1: {:.2?}", ssg_time.elapsed());

        let ssg_time = Instant::now();
        let (ssg_ids2, _ssg_dists2) = index2.greedy_search_batch(&queries, k, max_heap_size);
        println!("SSG queries 2: {:.2?}", ssg_time.elapsed());

        let ssg_time = Instant::now();
        let (ssg_ids3, _ssg_dists2) = index3.greedy_search_batch(&queries, k, max_heap_size);
        println!("SSG queries 2: {:.2?}", ssg_time.elapsed());

        calc_recall(ground_truth_view, &ssg_ids1, nq, k, "SSG-1", true);
        calc_recall(ground_truth_view, &ssg_ids2, nq, k, "SSG-2", true);
        calc_recall(ground_truth_view, &ssg_ids3, nq, k, "SSG-3", true);

        let max_heap_size = 3 * k;
        let ssg_time = Instant::now();
        let (ssg_ids1, _ssg_dists1) = index1.greedy_search_batch(&queries, k, max_heap_size);
        println!("SSG queries 1: {:.2?}", ssg_time.elapsed());

        let ssg_time = Instant::now();
        let (ssg_ids2, _ssg_dists2) = index2.greedy_search_batch(&queries, k, max_heap_size);
        println!("SSG queries 2: {:.2?}", ssg_time.elapsed());

        let ssg_time = Instant::now();
        let (ssg_ids3, _ssg_dists2) = index3.greedy_search_batch(&queries, k, max_heap_size);
        println!("SSG queries 2: {:.2?}", ssg_time.elapsed());

        calc_recall(ground_truth_view, &ssg_ids1, nq, k, "SSG-1", true);
        calc_recall(ground_truth_view, &ssg_ids2, nq, k, "SSG-2", true);
        calc_recall(ground_truth_view, &ssg_ids3, nq, k, "SSG-3", true);
    }
}
