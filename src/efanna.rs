//! Efanna graph crate
//! See [GitHub](https://github.com/ZJULearning/efanna_graph)
//!
//! Contains
//!     - EfannaBuilder (single-threaded)
//!         - Uses NHoodWithMaxHeap as pool
//!     - EfannaParallelMaxHeapBuilder
//!         - Uses NHoodWithMaxHeap as pool
//!
//! Notes
//!     - Builders which use a Vec or DualHeap as pool in NHood have been tried but the MaxHeap pool
//!       variant performed the best, so after commit 073c8617691435facbf020d68611a984b49056b3
//!       these version (and the corresponding NHoods) have been removed for less code complexity
//!         - However, was this tested solely during development and after that any further
//!           optimizations focused on the EfannaParallelMaxHeapBuilder were not necessarily tested
//!           for those version
//!             - As the Vec pool version in some cases performed closely to the MaxHeap one
//!     - Original code base of EFANNA seems to be very similar/related
//!       to [kGraph](https://github.com/aaalgo/kgraph/blob/master/kgraph.cpp)
//!     - It is very RAM intense which makes the mem::swap() operations actually critically
//!       important
//!     - The `k` and `l` parameters are directly influencing the memory usage during construction
//!       and therefore naturally the build time as well as the recall
//!
//! Open Issues
//!     - The original "GenRandom()" might be slightly different to
//!         graphidx::random::random_unique_uint (?)
//!     
//! TODOs
//!
//! --------------------------
//! LICENSE of source code of Efanna could not be found
//!
//! However, the source code is part of a public GitHub repository (see link above) as part of the
//! published paper [Arxiv](https://arxiv.org/abs/1609.07228).
//! --------------------------
//!
use graphidx::{
    data::MatrixDataSource,
    graphs::{DirLoLGraph, Graph, ViewableWeightedAdjGraph, WeightedGraph},
    heaps::MaxHeap,
    indices::{GreedyCappedSingleGraphIndex, GreedySingleGraphIndex},
    measures::Distance,
    param_struct,
    random::random_unique_uint,
    types::{SyncFloat, SyncUnsignedInteger},
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use rayon::{
    current_num_threads,
    iter::{ParallelBridge, ParallelIterator},
};
use std::{sync::Mutex, usize};

// -------------- UTIL --------------

#[inline(always)]
fn make_greedy_index<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    M: MatrixDataSource<F>,
    Dist: Distance<F>,
>(
    graph: EfannaBuildGraph<R, F>,
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
    graph: EfannaBuildGraph<R, F>,
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

pub struct EfannaBuildGraph<R: SyncUnsignedInteger, F: SyncFloat> {
    adjacency: Vec<Vec<(F, R)>>,
    n_edges: usize,
}
impl<R: SyncUnsignedInteger, F: SyncFloat> EfannaBuildGraph<R, F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            adjacency: vec![],
            n_edges: 0,
        }
    }
    #[inline(always)]
    fn _init_adj(&mut self, n_vertices: usize, k: usize) {
        self.adjacency.resize(n_vertices, Vec::with_capacity(k));
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat> Graph<R> for EfannaBuildGraph<R, F> {
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

impl<R: SyncUnsignedInteger, F: SyncFloat> WeightedGraph<R, F> for EfannaBuildGraph<R, F> {
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

impl<R: SyncUnsignedInteger, F: SyncFloat> ViewableWeightedAdjGraph<R, F>
    for EfannaBuildGraph<R, F>
{
    #[inline(always)]
    fn view_neighbors(&self, vertex: R) -> &[(F, R)] {
        self.adjacency[vertex.to_usize().unwrap()].as_slice()
    }
    #[inline(always)]
    fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F, R)] {
        self.adjacency[vertex.to_usize().unwrap()].as_mut_slice()
    }
}

pub trait EfannaStyleBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
where
    Self: Sized,
{
    type Params;
    type Graph: ViewableWeightedAdjGraph<R, F>;
    fn _graph(&self) -> &Self::Graph;
    fn _mut_graph(&mut self) -> &mut Self::Graph;
    fn _initial_degree(&self) -> usize;
    fn _dist(&self) -> &Dist;
    fn _into_graph_dist(self) -> (EfannaBuildGraph<R, F>, Dist);
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

/*  Helper struct from the original Efanna implementation
(for different flavors of NHood/pool see first comment under Notes at top of file)
`insert()` assumes that there is always at least one element in the pool
which for Efanna is always guaranteed because of `builder.initialize_graph()` */
pub struct NHoodWithMaxHeap<R: SyncUnsignedInteger, F: SyncFloat> {
    lock: Mutex<()>,
    m: usize,
    pool_capacity_l: usize,
    pool: MaxHeap<F, (R, bool)>,
    nn_old: Vec<R>,
    nn_new: Vec<R>,
    rnn_old: Vec<R>,
    rnn_new: Vec<R>,
}
// Not sure if initializing the Vecs with capacity makes such a difference or even sense here
impl<R: SyncUnsignedInteger, F: SyncFloat> NHoodWithMaxHeap<R, F> {
    fn new(l: usize, s: usize, n: usize) -> Self {
        Self {
            lock: Mutex::new(()),
            m: s,
            pool_capacity_l: l,
            pool: MaxHeap::with_capacity(l),
            nn_new: random_unique_uint(n, s * 2),
            nn_old: Vec::with_capacity(s),
            rnn_new: Vec::with_capacity(l),
            rnn_old: Vec::with_capacity(s),
        }
    }

    #[inline(always)]
    fn insert(&mut self, id: R, dist: F) {
        if dist > unsafe { self.pool.peek().unwrap_unchecked().0 } {
            return;
        }
        let guard = self.lock.lock().unwrap();
        let pool_len = self.pool.size();
        for (_, (id_, _)) in self.pool.as_slice() {
            if id == *id_ {
                drop(guard);
                return;
            }
        }
        if pool_len < self.pool_capacity_l {
            self.pool.push(dist, (id, true));
        } else {
            self.pool.push_pop(dist, (id, true));
        }
        drop(guard);
    }
}

param_struct!(EfannaParams[Copy, Clone]{
    k: usize = 50,
    l: usize = 50,
    iter: usize = 10,
    s: usize = 10,
    r: usize = 100,
});

pub mod single_threaded {
    use crate::efanna::*;

    pub struct EfannaBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send> {
        _phantom: std::marker::PhantomData<F>,
        n_data: usize,
        params: EfannaParams,
        graph: EfannaBuildGraph<R, F>,
        tmp_graph: Vec<NHoodWithMaxHeap<R, F>>,
        dist: Dist,
    }

    impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
        EfannaBuilder<R, F, Dist>
    {
        fn initialize_graph<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
            let mut tmp: Vec<R> = Vec::with_capacity(self.params.s);
            (0..self.n_data).for_each(|i| {
                self.tmp_graph.push(NHoodWithMaxHeap::new(
                    self.params.l,
                    self.params.l,
                    self.n_data,
                ));
                // this is correct because a IndexRandom is used in the og-code for "initializer"
                tmp = random_unique_uint(self.n_data, self.params.s);

                for j in 0..self.params.s {
                    let id = tmp[j];
                    if id == unsafe { R::from(i).unwrap_unchecked() } {
                        continue;
                    }
                    let dist: F = self._get_dist(mat, i, j);
                    self.tmp_graph[i].pool.push(dist, (id, true));
                }
            });
        }

        fn nn_descent<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
            for _it in 0..self.params.iter {
                self.join(mat);
                self.update();
            }
        }

        // moved the loops from the original code's nhood:join() here
        fn join<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
            (0..self.n_data).for_each(|n| unsafe {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                let curr_nhood = &self.tmp_graph[n];
                for i in curr_nhood.nn_new.as_slice() {
                    for j in curr_nhood.nn_new.as_slice() {
                        if i < j {
                            (*unsafe_self_ref).join_inner_fn(mat, i, j);
                        }
                    }
                    for j in curr_nhood.nn_old.as_slice() {
                        if i != j {
                            (*unsafe_self_ref).join_inner_fn(mat, i, j);
                        }
                    }
                }
            });
        }

        #[inline(always)]
        fn join_inner_fn<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M, i: &R, j: &R) {
            unsafe {
                let i_usize = i.to_usize().unwrap_unchecked();
                let j_usize = j.to_usize().unwrap_unchecked();
                let dist: F = self._get_dist(mat, i_usize, j_usize);
                self.tmp_graph[i_usize].insert(*j, dist);
                self.tmp_graph[j_usize].insert(*i, dist);
            }
        }

        fn update(&mut self) {
            let s = self.params.s;

            // swap to free memory
            (0..self.n_data).for_each(|n| {
                let nnhd = &mut self.tmp_graph[n];
                let nn_new = &mut nnhd.nn_new;
                let nn_old = &mut nnhd.nn_old;
                std::mem::swap(nn_new, &mut Vec::new());
                std::mem::swap(nn_old, &mut Vec::new());
            });

            (0..self.n_data).for_each(|n: usize| {
                let nn = &mut self.tmp_graph[n];
                // no sorting required here because of MaxHeap instead of Vec
                while nn.pool.size() > self.params.l {
                    nn.pool.pop();
                }
                let max_l = (nn.m + s).min(nn.pool.size());
                let (mut c, mut l) = (0, 0);
                // rev() because of the asc sort beforehand in og-code, correct?
                let mut nn_pool_iter = nn.pool.iter().rev();
                while l < max_l && c < self.params.s {
                    if nn_pool_iter.next().unwrap().1 .1 {
                        c += 1;
                    }
                    l += 1;
                }
                nn.m = l;
            });

            let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
            let mut rng = thread_rng();
            (0..self.n_data).for_each(|n| unsafe {
                let nnhd = &mut (*unsafe_self_ref).tmp_graph[n];
                let nn_new = &mut nnhd.nn_new;
                let nn_old = &mut nnhd.nn_old;
                let m = nnhd.m;
                let mut l = 0;
                for (nn_dist, (nn_id, nn_flag)) in nnhd.pool.as_mut_slice() {
                    if l >= m {
                        break;
                    }
                    l += 1;
                    // nn on the other side of the edge
                    let nhood_o =
                        &mut (*unsafe_self_ref).tmp_graph[nn_id.to_usize().unwrap_unchecked()];
                    if *nn_flag {
                        nn_new.push(*nn_id);
                        if *nn_dist > nhood_o.pool.peek().unwrap().0 {
                            // not guards needed in single-threaded version
                            if nhood_o.rnn_new.len() < self.params.r {
                                nhood_o.rnn_new.push(R::from(n).unwrap_unchecked());
                            } else {
                                let pos = rng.gen_range(0..nhood_o.rnn_new.len());
                                nhood_o.rnn_new[pos] = R::from(n).unwrap_unchecked();
                            }
                        }
                        *nn_flag = false;
                    } else {
                        nn_old.push(*nn_id);
                        if *nn_dist > nhood_o.pool.peek().unwrap().0 {
                            // not guards needed in the single-threaded version
                            if nhood_o.rnn_old.len() < self.params.r {
                                nhood_o.rnn_old.push(R::from(n).unwrap_unchecked());
                            } else {
                                let pos = rng.gen_range(0..nhood_o.rnn_old.len());
                                nhood_o.rnn_old[pos] = R::from(n).unwrap_unchecked();
                            }
                        }
                    }
                }
            });

            let mut rng = thread_rng();
            (0..self.n_data).for_each(|i| {
                let r = self.params.r;
                let nn = &mut self.tmp_graph[i];

                let nn_new = &mut nn.nn_new;
                let nn_old = &mut nn.nn_old;
                let rnn_new = &mut nn.rnn_new;
                let rnn_old = &mut nn.rnn_old;

                if rnn_new.len() > r {
                    rnn_new.shuffle(&mut rng);
                    rnn_new.truncate(r);
                }

                rnn_new.iter().for_each(|&e| nn_new.push(e));

                if rnn_old.len() > r {
                    rnn_old.shuffle(&mut rng);
                    rnn_old.truncate(r);
                }

                rnn_old.iter().for_each(|&e| nn_old.push(e));

                if nn_old.len() > r * 2 {
                    nn_old.truncate(r * 2);
                    // Skipping the reserve after resize
                }

                std::mem::swap(rnn_new, &mut Vec::new());
                std::mem::swap(rnn_old, &mut Vec::new());
            });
        }
    }

    impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
        EfannaStyleBuilder<R, F, Dist> for EfannaBuilder<R, F, Dist>
    {
        type Params = EfannaParams;
        type Graph = EfannaBuildGraph<R, F>;
        fn _graph(&self) -> &EfannaBuildGraph<R, F> {
            &self.graph
        }
        #[inline(always)]
        fn _mut_graph(&mut self) -> &mut EfannaBuildGraph<R, F> {
            &mut self.graph
        }
        #[inline(always)]
        fn _dist(&self) -> &Dist {
            &self.dist
        }
        #[inline(always)]
        fn _initial_degree(&self) -> usize {
            self.params.r
        }
        #[inline(always)]
        fn _into_graph_dist(self) -> (EfannaBuildGraph<R, F>, Dist) {
            (self.graph, self.dist)
        }
        fn new<M: MatrixDataSource<F> + Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
            let n_data = mat.n_rows();
            assert!(n_data < R::max_value().to_usize().unwrap());
            let graph = EfannaBuildGraph::new();
            Self {
                _phantom: std::marker::PhantomData,
                n_data,
                params,
                graph,
                tmp_graph: Vec::with_capacity(n_data),
                dist,
            }
        }
        fn train<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
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
            if self.params.s > self.n_data {
                // only relevant when n_data is really small
                self.params.s = self.n_data;
            }

            if self.params.k > self.n_data {
                // only relevant when n_data is really small
                self.params.k = self.n_data / 2;
            }
            let k = self.params.k;
            self.initialize_graph(mat);
            self.nn_descent(mat);

            // correctly resizing the graph so each index can be written to directly
            self.graph._init_adj(self.n_data, k);

            (0..self.n_data).for_each(|i| {
                #[cfg(debug_assertions)]
                {
                    assert!(
                        self.tmp_graph[i].pool.size() >= k,
                        "Pool sizes too small: {} < {}, {}",
                        self.tmp_graph[i].pool.size(),
                        k,
                        i
                    );
                }

                let tmp: Vec<(F, R)> = self.tmp_graph[i]
                    .pool
                    .iter()
                    .take(k)
                    .map(|&(dist, (id, _flag))| (dist, id))
                    .collect::<Vec<(F, R)>>();

                // In the og-code they actually swapped rnn_new twice
                std::mem::swap(&mut self.tmp_graph[i].pool, &mut MaxHeap::new());
                std::mem::swap(&mut self.tmp_graph[i].nn_new, &mut Vec::new());
                std::mem::swap(&mut self.tmp_graph[i].nn_old, &mut Vec::new());
                std::mem::swap(&mut self.tmp_graph[i].rnn_new, &mut Vec::new());
                std::mem::swap(&mut self.tmp_graph[i].rnn_old, &mut Vec::new());

                self.graph.adjacency[i] = tmp;
            });
            std::mem::swap(&mut self.tmp_graph, &mut Vec::new());
        }
    }
}

pub struct EfannaParallelMaxHeapBuilder<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
> {
    _phantom: std::marker::PhantomData<F>,
    n_data: usize,
    params: EfannaParams,
    graph: EfannaBuildGraph<R, F>,
    tmp_graph: Vec<NHoodWithMaxHeap<R, F>>,
    dist: Dist,
    n_threads: usize,
    // lock: Mutex<()>,
}

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    EfannaParallelMaxHeapBuilder<R, F, Dist>
{
    fn initialize_graph<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
        /* parallelizing this via set_len() and then assigning the new
        NHoodWithMaxHeap for each index did not work (it failed with a segmentation fault,
        however, only when the algorithm was executed twice in a row) - but it had a negative
        effect on the performance anyway */
        let mut tmp: Vec<R> = Vec::with_capacity(self.params.s);
        (0..self.n_data).for_each(|i| {
            self.tmp_graph.push(NHoodWithMaxHeap::new(
                self.params.l,
                self.params.l,
                self.n_data,
            ));
            // this is correct because an IndexRandom is used in the og-code for "initializer"
            tmp = random_unique_uint(self.n_data, self.params.s);

            for j in 0..self.params.s {
                let id = tmp[j];
                if id == unsafe { R::from(i).unwrap_unchecked() } {
                    continue;
                }
                let dist: F = self._get_dist(mat, i, j);
                self.tmp_graph[i].pool.push(dist, (id, true));
            }
        });
    }

    fn nn_descent<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
        for _it in 0..self.params.iter {
            self.join(mat);
            self.update();
        }
    }

    // moved the loops from the original code's nhood:join() here
    fn join<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
        let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
        (0..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .par_bridge()
            .for_each(|chunk| unsafe {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                chunk.for_each(|n: usize| {
                    let curr_nhood = &self.tmp_graph[n];
                    for i in curr_nhood.nn_new.as_slice() {
                        for j in curr_nhood.nn_new.as_slice() {
                            if i < j {
                                (*unsafe_self_ref).join_inner_fn(mat, i, j);
                            }
                        }
                        for j in curr_nhood.nn_old.as_slice() {
                            if i != j {
                                (*unsafe_self_ref).join_inner_fn(mat, i, j);
                            }
                        }
                    }
                })
            });
    }

    #[inline(always)]
    fn join_inner_fn<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M, i: &R, j: &R) {
        unsafe {
            let i_usize = i.to_usize().unwrap_unchecked();
            let j_usize = j.to_usize().unwrap_unchecked();
            let dist: F = self._get_dist(mat, i_usize, j_usize);
            self.tmp_graph[i_usize].insert(*j, dist);
            self.tmp_graph[j_usize].insert(*i, dist);
        }
    }

    fn update(&mut self) {
        let s = self.params.s;
        let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;

        // swap to free memory
        (0..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .par_bridge()
            .for_each(|chunk| {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                chunk.for_each(|n: usize| unsafe {
                    let nnhd = &mut (*unsafe_self_ref).tmp_graph[n];
                    let nn_new = &mut nnhd.nn_new;
                    let nn_old = &mut nnhd.nn_old;
                    std::mem::swap(nn_new, &mut Vec::new());
                    std::mem::swap(nn_old, &mut Vec::new());
                });
            });

        (0..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .par_bridge()
            .for_each(|chunk| {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                chunk.for_each(|n: usize| unsafe {
                    let nn = &mut (*unsafe_self_ref).tmp_graph[n];
                    // no sorting required here because of MaxHeap instead of Vec
                    while nn.pool.size() > self.params.l {
                        nn.pool.pop();
                    }
                    let max_l = (nn.m + s).min(nn.pool.size());
                    let (mut c, mut l) = (0, 0);
                    // rev() because of the asc sort beforehand in og-code, correct?
                    let mut nn_pool_iter = nn.pool.iter().rev();
                    while l < max_l && c < self.params.s {
                        if nn_pool_iter.next().unwrap().1 .1 {
                            c += 1;
                        }
                        l += 1;
                    }
                    nn.m = l;
                });
            });

        (0..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .par_bridge()
            .for_each(|chunk| {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                let mut rng = thread_rng();
                chunk.for_each(|n: usize| unsafe {
                    let nnhd = &mut (*unsafe_self_ref).tmp_graph[n];
                    let nn_new = &mut nnhd.nn_new;
                    let nn_old = &mut nnhd.nn_old;
                    let m = nnhd.m;
                    let mut l = 0;
                    for (nn_dist, (nn_id, nn_flag)) in nnhd.pool.as_mut_slice() {
                        if l >= m {
                            break;
                        }
                        l += 1;
                        let nhood_o =
                            &mut (*unsafe_self_ref).tmp_graph[nn_id.to_usize().unwrap_unchecked()];

                        if *nn_flag {
                            nn_new.push(*nn_id);
                            if *nn_dist > nhood_o.pool.peek().unwrap().0 {
                                let guard = nhood_o.lock.lock().unwrap();
                                if nhood_o.rnn_new.len() < self.params.r {
                                    nhood_o.rnn_new.push(R::from(n).unwrap_unchecked());
                                } else {
                                    let pos = rng.gen_range(0..nhood_o.rnn_new.len());
                                    nhood_o.rnn_new[pos] = R::from(n).unwrap_unchecked();
                                }
                                drop(guard);
                            }
                            *nn_flag = false;
                        } else {
                            nn_old.push(*nn_id);
                            if *nn_dist > nhood_o.pool.peek().unwrap().0 {
                                let guard = nhood_o.lock.lock().unwrap();
                                if nhood_o.rnn_old.len() < self.params.r {
                                    nhood_o.rnn_old.push(R::from(n).unwrap_unchecked());
                                } else {
                                    let pos = rng.gen_range(0..nhood_o.rnn_old.len());
                                    nhood_o.rnn_old[pos] = R::from(n).unwrap_unchecked();
                                }
                                drop(guard);
                            }
                        }
                    }
                });
            });

        (0..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .par_bridge()
            .for_each(|chunk| {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                let r = self.params.r;
                let mut rng = thread_rng();

                chunk.for_each(|i: usize| unsafe {
                    let nn = &mut (*unsafe_self_ref).tmp_graph[i];
                    let nn_new = &mut nn.nn_new;
                    let nn_old = &mut nn.nn_old;
                    let rnn_new = &mut nn.rnn_new;
                    let rnn_old = &mut nn.rnn_old;

                    if rnn_new.len() > r {
                        rnn_new.shuffle(&mut rng);
                        rnn_new.truncate(r);
                    }

                    rnn_new.iter().for_each(|&e| nn_new.push(e));

                    if rnn_old.len() > r {
                        rnn_old.shuffle(&mut rng);
                        rnn_old.truncate(r);
                    }

                    rnn_old.iter().for_each(|&e| nn_old.push(e));

                    if nn_old.len() > r * 2 {
                        nn_old.truncate(r * 2);
                    }
                    std::mem::swap(rnn_new, &mut Vec::new());
                    std::mem::swap(rnn_old, &mut Vec::new());
                });
            });
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    EfannaStyleBuilder<R, F, Dist> for EfannaParallelMaxHeapBuilder<R, F, Dist>
{
    type Params = EfannaParams;
    type Graph = EfannaBuildGraph<R, F>;
    fn _graph(&self) -> &EfannaBuildGraph<R, F> {
        &self.graph
    }
    #[inline(always)]
    fn _mut_graph(&mut self) -> &mut EfannaBuildGraph<R, F> {
        &mut self.graph
    }
    #[inline(always)]
    fn _dist(&self) -> &Dist {
        &self.dist
    }
    #[inline(always)]
    fn _initial_degree(&self) -> usize {
        self.params.r
    }
    #[inline(always)]
    fn _into_graph_dist(self) -> (EfannaBuildGraph<R, F>, Dist) {
        (self.graph, self.dist)
    }
    fn new<M: MatrixDataSource<F> + Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
        let n_data = mat.n_rows();
        assert!(n_data < R::max_value().to_usize().unwrap());
        let graph = EfannaBuildGraph::new();
        Self {
            _phantom: std::marker::PhantomData,
            n_data,
            params,
            graph,
            tmp_graph: Vec::with_capacity(n_data),
            dist,
            n_threads: current_num_threads(),
            // lock: Mutex::new(()),
        }
    }
    /// Build the graph.
    fn train<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
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
        if self.params.s > self.n_data {
            // only relevant when n_data is really small
            self.params.s = self.n_data;
        }

        if self.params.k > self.n_data {
            // only relevant when n_data is really small
            self.params.k = self.n_data / 2;
        }
        let k = self.params.k;

        self.initialize_graph(mat);
        self.nn_descent(mat);

        // correctly resizing the graph so each index can be written to directly
        self.graph._init_adj(self.n_data, k);

        let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
        (0..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .par_bridge()
            .for_each(|chunk| {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                chunk.for_each(|i: usize| unsafe {
                    #[cfg(debug_assertions)]
                    {
                        assert!(
                            self.tmp_graph[i].pool.size() >= k,
                            "Pool sizes too small: {} < {}, {}",
                            self.tmp_graph[i].pool.size(),
                            k,
                            i
                        );
                    }
                    let tmp: Vec<(F, R)> = self.tmp_graph[i]
                        .pool
                        .iter()
                        .rev() // reverse because of maxheap
                        .take(k)
                        .map(|&(dist, (id, _flag))| (dist, id))
                        .collect::<Vec<(F, R)>>();

                    std::mem::swap(
                        &mut (*unsafe_self_ref).tmp_graph[i].pool,
                        &mut MaxHeap::new(),
                    );
                    std::mem::swap(&mut (*unsafe_self_ref).tmp_graph[i].nn_new, &mut Vec::new());
                    std::mem::swap(&mut (*unsafe_self_ref).tmp_graph[i].nn_old, &mut Vec::new());
                    std::mem::swap(
                        &mut (*unsafe_self_ref).tmp_graph[i].rnn_new,
                        &mut Vec::new(),
                    );
                    std::mem::swap(
                        &mut (*unsafe_self_ref).tmp_graph[i].rnn_old,
                        &mut Vec::new(),
                    );

                    // no guards, as each index is written to only once and the graph is already correctly resized
                    (*unsafe_self_ref).graph.adjacency[i] = tmp;
                });
            });
        std::mem::swap(&mut self.tmp_graph, &mut Vec::new());
    }
}

// -------------- TESTS --------------

// Still contains now commented code of performance comparisons of other now removed builders
#[cfg(test)]
mod tests {
    use crate::efanna::*;
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
    use std::time::Instant;

    type R = usize;
    type F = f32;
    type Dist = SquaredEuclideanDistance<F>;

    #[test]
    fn efanna_construction() {
        // let (_nd, _nq, _d, _k, data) = get_test_data(true); // normal data
        let (data, _queries, _ground_truth, _nd, _nq, _d, _k) =
            init_with_dataset(&EvalDataSet::Normal100kIndexedQueries);

        let params = EfannaParams::new();

        let graph_time = Instant::now();
        // type BuilderType = crate::efanna::single_threaded::EfannaBuilder<R, F, Dist>;
        type BuilderType = EfannaParallelMaxHeapBuilder<R, F, Dist>;
        let index = BuilderType::build(data.view(), Dist::new(), params);

        println!("Efanna Graph construction: {:.2?}", graph_time.elapsed());
        print_index_build_info!(index);
    }

    #[test]
    fn efanna_query() {
        // let (_nd, nq, _d, k, data) = get_test_data(true); // normal data
        // let queries = data.slice_axis(Axis(0), Slice::from(0..nq));
        let (data, queries, ground_truth, _nd, nq, _d, k) = init_with_dataset(&EvalDataSet::AUDIO);
        let degree = 50;

        /* Build and translate Efanna Graph */
        let build_time = Instant::now();
        let dist = SquaredEuclideanDistance::new();
        let params = EfannaParams::new().with_r(degree);

        type BuilderType = EfannaParallelMaxHeapBuilder<R, F, Dist>;
        let index1 = BuilderType::build(data.view(), dist, params);

        println!(
            "Graph construction ({:?}): {:.2?}",
            std::any::type_name::<BuilderType>(),
            build_time.elapsed()
        );
        print_index_build_info!(index1);

        // // Brute force queries
        // let bruteforce_time = Instant::now();
        // let (ground_truth, _) = bruteforce_neighbors(&data, &queries, &Dist::new(), k);
        // println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());

        // Efanna queries
        let efanna_time = Instant::now();
        let (efanna_ids1, _efanna_dists1) = index1.greedy_search_batch(&queries, k, 5 * k);
        println!("Efanna queries 1: {:.2?}", efanna_time.elapsed());

        // single-threaded search
        let graph_name = "Efanna-1";
        let ground_truth_view = ground_truth.view();
        let (_query_time, _qps, _recall) = search_as_qps!(
            index1,
            queries,
            ground_truth_view,
            k,
            k,
            nq,
            graph_name,
            true
        );

        calc_recall(ground_truth_view, &efanna_ids1, nq, k, "Efanna-1", true);
    }

    // Trying to figure out where the duplicate edges are coming from
    #[test]
    fn efanna_compare_construction() {
        let (_nd, _nq, _d, _k, data) = get_test_data(true); // normal data

        let params = EfannaParams::new();
        let graph_time = Instant::now();
        let index1 = crate::efanna::single_threaded::EfannaBuilder::<R, F, Dist>::build(
            data.clone(),
            Dist::new(),
            params,
        );
        println!(
            "Efanna Graph construction - Single-Vec: {:.2?}",
            graph_time.elapsed()
        );

        // let graph_time = Instant::now();
        // let index2 =
        //     EfannaParallelBuilder::<R, F, Dist>::build(data.clone(), Dist::new(), params.clone());
        // println!(
        //     "Efanna Graph construction - Par-Vec: {:.2?}",
        //     graph_time.elapsed()
        // );

        let graph_time = Instant::now();
        let index3 = EfannaParallelMaxHeapBuilder::<R, F, Dist>::build(
            data.clone(),
            Dist::new(),
            params.clone(),
        );
        println!(
            "Efanna Graph construction - Par-MaxHeap: {:.2?}",
            graph_time.elapsed()
        );

        // let graph_time = Instant::now();
        // let index4 = EfannaParallelDualHeapBuilder::<R, F, Dist>::build(
        //     data.clone(),
        //     Dist::new(),
        //     params.clone(),
        // );
        // println!(
        //     "Efanna Graph construction - Par-DualHeap: {:.2?}",
        //     graph_time.elapsed()
        // );

        print_index_build_info!(index1);
        // print_index_build_info!(index2);
        print_index_build_info!(index3);
        // print_index_build_info!(index4);
    }
}
