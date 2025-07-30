//! DEG graph crate  
//! See [crEG/DEG on GitHub](https://github.com/Visual-Computing/DynamicExplorationGraph)
//!
//! Contains
//!     - DEGBuilder  
//!     - DEGParallelBuilder  
//!
//! Notes
//!     - While a version for `search_range` and `has_path_range` is implemented (following the
//!       graphindexbaselines implementation and the DEG paper), the original greedy `search`
//!       and a derived 'has_path' version are used as they have shown to be significantly faster
//!       than our range implementations  
//!     - Significant differences/changes:
//!         - The options to add or remove points to the graph after the main build process
//!             (which are part of the original code base) are currently not implemented
//!         - All vertices are initialized at once as in the beginning of `extend_graph()`
//!     - Entry points are in contrast to the paper (but apparently not the code base) not specified
//!          - Although, the original crEG/DEG paper suggested to take the median vector of the
//!             graph (whatever that actually is?)
//!     - Performance
//!         - Parameters
//!             - improve_k and swap_tries trigger `improve_edges()` which _can_ lead to
//!                 higher recall, but also significantly increases the build time
//!             - LID::Low generally leads to slightly lower recall than ::High
//!             - Increasing instead the search parameters like extend_k, max_build_(..) leads to
//!                 higher recall with a significantly smaller impact on the build time
//!
//! Open Issues
//!     - Performance
//!         - For some time this implementation was unable to really match the build speed of the
//!          original code (tested) via their Python module
//!             - UPDATE: Performance was increased by using swapping to the greedy search from the
//!               range search and dropping the locking during the parallel construction. While it
//!               is strange that it works, tests have shown that the resulting graphs are still
//!               stable and complete graphs (no loops, etc.)
//!                 - This might in big part work because all nodes are initialized at once,
//!                   in contrast to one-by-one as in the original implementation. Additionally, the
//!                   parallel processing in chunks might also help, although that should not make a
//!                   real difference here because the edges/nodes to be changed are still global
//!                 - It is still interesting though that no "race" conditions (seem to) appear
//!                     (see commented line of used lock/mutex for position in code)
//!         - NOTE: Using improve_edges_build with `params.additional_swap_tries` set to
//!         a value > 0 often leads to the construction getting stuck or building for a
//!         very long time due the counter in improve_edges_build() getting bigger and
//!         bigger as - apparently - edges can be improved without really hitting a limit
//!     - For LID::Unknown
//!         - The resulting graph's performance is on par with LID::High/Low, however,
//!           the build time is significantly slower (>2x; single thread only)
//!     - Furthermore, the graph in rare cases can end up with a few (<10) self loops
//!         - This is likely a bug but was not yet possible to detect where it is coming from, as
//!           the self-loops are part of the DEG node initialization as well as used during the
//!           construction phase
//!
//! TODOs
//!
//! --------------------------
//! LICENSE of source code of DEG
//! MIT License
//!
//! Copyright (c) 2024 Visual Computing Group
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.
//! --------------------------
//!
use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use graphidx::{
    data::MatrixDataSource,
    graphs::{DirLoLGraph, Graph, ViewableWeightedAdjGraph, WeightedGraph},
    heaps::{DualHeap, MaxHeap, MinHeap},
    indices::{GreedyCappedSingleGraphIndex, GreedySingleGraphIndex},
    measures::Distance,
    param_struct,
    sets::HashSetLike,
    types::{Float, SyncFloat, SyncUnsignedInteger},
};
use ndarray_rand::rand;
use rand::rngs::ThreadRng;
use rand::Rng;
use rayon::{
    current_num_threads,
    iter::{IntoParallelIterator, ParallelBridge, ParallelIterator},
};
use std::hash::{Hash, Hasher};
#[allow(unused)]
use std::sync::Mutex;

type HashSetBit<T> = graphidx::sets::BitSet<T>;

// -------------- UTIL --------------

// Helper struct from the original DEG implementation
#[derive(Clone)]
struct ReachableGroup<R: SyncUnsignedInteger> {
    vertex_index: R,
    missing_edges: HashSet<R>,
    reachable_vertices: HashSet<R>,
}
impl<R: SyncUnsignedInteger> ReachableGroup<R> {
    fn new(vertex_index: R, expected_size: usize) -> Self {
        let mut missing_edges: HashSet<R> = foldhash::HashSet::with_capacity(expected_size);
        missing_edges.insert(vertex_index);
        let mut reachable_vertices: HashSet<R> = foldhash::HashSet::with_capacity(expected_size);
        reachable_vertices.insert(vertex_index);
        Self {
            vertex_index,
            missing_edges,
            reachable_vertices,
        }
    }
    #[inline(always)]
    /// Remove the element from the list of vertices with missing edges
    fn has_edge(&mut self, elem: R) {
        self.missing_edges.remove(&elem);
    }
    #[inline(always)]
    /// Return the vertex associated with this group
    fn get_vertex_index(&self) -> R {
        self.vertex_index
    }
    #[inline(always)]
    /// Get the number of vertices which can be reached by this group
    fn size(&self) -> usize {
        self.reachable_vertices.len()
    }
    #[inline(always)]
    /// Get the number of vertices in this group which are missing an edge
    fn get_missing_edge_size(&self) -> usize {
        self.missing_edges.len()
    }
    #[inline(always)]
    /// Get the vertices which are missing an edges
    fn get_missing_edges(&self) -> &HashSet<R> {
        &self.missing_edges
    }
    #[inline(always)]
    /// Copy the data from the other group to this group
    fn copy_from(&mut self, other_group: &ReachableGroup<R>) {
        // skip if both are the same object
        if self.vertex_index == other_group.get_vertex_index() {
            return;
        } else {
            other_group.missing_edges.iter().for_each(|&e| {
                _ = self.missing_edges.insert(e);
            });
            other_group.reachable_vertices.iter().for_each(|&e| {
                _ = self.reachable_vertices.insert(e);
            });
        }
    }
}
// So ReachableGroup can be hashed
impl<R: SyncUnsignedInteger> PartialEq for ReachableGroup<R> {
    fn eq(&self, other: &Self) -> bool {
        self.vertex_index == other.vertex_index
    }
}
impl<R: SyncUnsignedInteger> Eq for ReachableGroup<R> {}
impl<R: SyncUnsignedInteger> Hash for ReachableGroup<R> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_index.hash(state);
    }
}

// --------------

// Helper struct from the original DEG implementation
struct UnionFindDeg<R: SyncUnsignedInteger> {
    default_value: R,
    parents: HashMap<R, R>,
}
impl<R: SyncUnsignedInteger> UnionFindDeg<R> {
    #[inline(always)]
    fn new(expected_size: usize) -> Self {
        let parents: HashMap<R, R> = HashMap::with_capacity(expected_size);
        Self {
            parents,
            default_value: R::max_value(),
        }
    }
    #[inline(always)]
    // Get the default value if an element is not in the union
    fn get_default_value(&self) -> R {
        self.default_value
    }
    #[inline(always)]
    fn find(&self, l: R) -> R {
        let entry = self.parents.get(&l);
        if entry.is_none() {
            return self.default_value;
        }
        let entry: R = *unsafe { entry.unwrap_unchecked() };
        if entry == l {
            // if l is root
            return l;
        }
        return self.find(entry); // recurs for parent till we find root
    }
    #[inline(always)]
    // Perform Union of two subsets element1 and element2
    fn _union(&mut self, m: R, n: R) {
        let x: R = self.find(m);
        let y: R = self.find(n);
        self.update(x, y);
    }
    #[inline(always)]
    // If the parents are known via find this method can be called instead of union
    fn update(&mut self, element: R, parent: R) {
        self.parents.insert(element, parent);
    }
}

// --------------

/// Wrapper from original DEG implementation
#[derive(Clone, Copy)]
struct BuilderChange<R: SyncUnsignedInteger, F: Float> {
    internal_index: R,
    from_neighbor_index: R,
    from_neighbor_weight: F,
    to_neighbor_index: R,
    _to_neighbor_weight: F,
}
impl<R: SyncUnsignedInteger, F: Float> BuilderChange<R, F> {
    fn new(
        internal_index: R,
        from_neighbor_index: R,
        from_neighbor_weight: F,
        to_neighbor_index: R,
        _to_neighbor_weight: F,
    ) -> Self {
        Self {
            internal_index,
            from_neighbor_index,
            from_neighbor_weight,
            to_neighbor_index,
            _to_neighbor_weight,
        }
    }
}

// -------------- UTIL --------------

#[inline(always)]
fn make_greedy_index<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    M: MatrixDataSource<F>,
    Dist: Distance<F>,
>(
    graph: DEGBuildGraph<R, F>,
    mat: M,
    dist: Dist,
) -> GreedySingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
    GreedySingleGraphIndex::new(mat, graph.as_dir_lol_graph(), dist, None)
}

#[inline(always)]
fn make_greedy_capped_index<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    M: MatrixDataSource<F>,
    Dist: Distance<F>,
>(
    graph: DEGBuildGraph<R, F>,
    mat: M,
    dist: Dist,
    max_frontier_size: usize,
) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
    GreedyCappedSingleGraphIndex::new(mat, graph.as_dir_lol_graph(), dist, max_frontier_size, None)
}

// ----------------------------

/// DEGBuildGraph  
pub struct DEGBuildGraph<R: SyncUnsignedInteger, F: SyncFloat> {
    adjacency: Vec<Vec<(F, R)>>,
    n_edges: usize,
}
impl<R: SyncUnsignedInteger, F: SyncFloat> DEGBuildGraph<R, F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            adjacency: vec![],
            n_edges: 0,
        }
    }
    // Only used in the single-threaded builder
    /// Add a new vertex to graph, initialized with edges_per_vertex-times many edges to itself
    #[inline(always)]
    fn add_vertex(&mut self, edges_per_vertex: usize) -> R {
        let new_idx_r: R = unsafe { R::from(self.n_vertices()).unwrap_unchecked() };
        self.adjacency
            .push(vec![(F::zero(), new_idx_r); edges_per_vertex]);
        self.n_edges += edges_per_vertex;
        new_idx_r
    }
    // Note: uses par_iter() but also not used in the single-threaded builder
    #[inline(always)]
    fn init_adj(&mut self, edges_per_vertex: usize, n_data: usize) {
        self.adjacency
            .resize(n_data, Vec::with_capacity(edges_per_vertex));
        self.adjacency = (0..n_data)
            .into_par_iter()
            .map(|i| vec![(F::zero(), unsafe { R::from(i).unwrap_unchecked() }); edges_per_vertex])
            .collect();
        self.n_edges = n_data * edges_per_vertex;
    }
    /// Changes edge from v1->v2 to v1->v3, if v1->v2 exists
    /// Might add a (even another) self-loop when v1 == v3
    // This is a highly used function during construction
    #[inline(always)]
    fn change_edge(&mut self, v1: R, v2: R, v3: R, weight_v1v3: F) {
        if v1 != v3 && self.has_edge(v1, v3) {
            // return when edge between v1 and v3 already exists
            // but allow (multiple, temporary) self-loops
            return;
        }
        let index_v1v2 = self.find_edge(v1, v2);
        if index_v1v2.is_some() {
            unsafe {
                self.adjacency[v1.to_usize().unwrap_unchecked()][index_v1v2.unwrap_unchecked()] =
                    (weight_v1v3, v3);
            }
        }
    }
    // Might be only marginally faster than the find_edge()->Option<usize>
    // Most time is saved on the avoided .is_some() and .is_none() checks
    fn has_edge(&self, v1: R, v2: R) -> bool {
        self.view_neighbors(v1).iter().any(|&(_, n)| n == v2)
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat> Graph<R> for DEGBuildGraph<R, F> {
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
    fn find_edge(&self, vertex1: R, vertex2: R) -> Option<usize> {
        // Note the change to view_neighbors() here compared to the default
        self.view_neighbors(vertex1)
            .iter()
            .position(|&(_, v)| v == vertex2)
    }
    #[inline(always)]
    fn neighbors(&self, vertex: R) -> Vec<R> {
        self.adjacency[unsafe { vertex.to_usize().unwrap_unchecked() }]
            .iter()
            .map(|&(_, v)| v)
            .collect()
    }
    #[inline(always)]
    fn foreach_neighbor<Fun: FnMut(&R)>(&self, vertex: R, mut f: Fun) {
        self.adjacency[unsafe { vertex.to_usize().unwrap_unchecked() }]
            .iter()
            .for_each(|v| f(&v.1));
    }
    #[inline(always)]
    fn foreach_neighbor_mut<Fun: FnMut(&mut R)>(&mut self, vertex: R, mut f: Fun) {
        self.adjacency[unsafe { vertex.to_usize().unwrap_unchecked() }]
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
    fn remove_edge(&mut self, vertex1: R, vertex2: R) {
        let index = self.find_edge(vertex1, vertex2);
        if index.is_some() {
            self.remove_edge_by_index(vertex1, unsafe { index.unwrap_unchecked() });
        }
    }
    #[inline(always)]
    fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
        self.adjacency[unsafe { vertex.to_usize().unwrap_unchecked() }].swap_remove(index);
        let neighbor = self.adjacency[unsafe { vertex.to_usize().unwrap_unchecked() }][index];
        let neighbor_index = unsafe {
            self.adjacency[neighbor.1.to_usize().unwrap_unchecked()]
                .iter()
                .position(|&v| v.1 == vertex)
                .unwrap_unchecked()
        };
        self.adjacency[unsafe { neighbor.1.to_usize().unwrap_unchecked() }]
            .swap_remove(neighbor_index);
        self.n_edges -= 1;
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat> WeightedGraph<R, F> for DEGBuildGraph<R, F> {
    #[inline(always)]
    fn edge_weight(&self, vertex1: R, vertex2: R) -> F {
        self.adjacency[vertex1.to_usize().unwrap()]
            .iter()
            .find(|(_, v)| v == &vertex2)
            .unwrap()
            .0
    }
    #[inline(always)]
    fn add_edge_with_weight(&mut self, _v1: R, _v2: R, _w: F) {
        unimplemented!(
            "Adding an edge does not make sense in a fixed degree graph. See change_edge()."
        )
    }
    #[inline(always)]
    fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>) {
        let mut neighbors = Vec::new();
        let mut weights = Vec::new();
        for &(w, v) in self.adjacency[unsafe { vertex.to_usize().unwrap_unchecked() }].iter() {
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
        self.adjacency[unsafe { vertex.to_usize().unwrap_unchecked() }]
            .iter()
            .for_each(|&v| f(&v.0, &v.1));
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

impl<R: SyncUnsignedInteger, F: SyncFloat> ViewableWeightedAdjGraph<R, F> for DEGBuildGraph<R, F> {
    #[inline(always)]
    fn view_neighbors(&self, vertex: R) -> &[(F, R)] {
        self.adjacency[vertex.to_usize().unwrap()].as_slice()
    }
    #[inline(always)]
    fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F, R)] {
        self.adjacency[vertex.to_usize().unwrap()].as_mut_slice()
    }
}

/// Contains `_search()` and `_has_path()` as classical greedy search versions and
/// `_search_range()` and `_has_path_range()` as "range" search versions following the DEG paper
pub trait DEGStyleBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
where
    Self: Sized,
{
    type Params;
    type Graph: ViewableWeightedAdjGraph<R, F>;
    fn _graph(&self) -> &Self::Graph;
    fn _mut_graph(&mut self) -> &mut Self::Graph;
    fn _max_build_heap_size(&self) -> usize;
    fn _max_build_frontier_size(&self) -> Option<usize>;
    fn _dist(&self) -> &Dist;
    fn _into_graph_dist(self) -> (DEGBuildGraph<R, F>, Dist);
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
        make_greedy_index(graph, mat, dist)
    }
    #[inline(always)]
    fn into_greedy_capped<M: MatrixDataSource<F> + Sync>(
        self,
        mat: M,
        max_frontier_size: usize,
    ) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let (graph, dist) = self._into_graph_dist();
        make_greedy_capped_index(graph, mat, dist, max_frontier_size)
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
    // ----------------
    // Copied and slightly adapted from graphidxbaselines::hnsw
    // Final result is contained in `entry_points`
    fn _search<M: MatrixDataSource<F> + Sync>(
        &self,
        mat: &M,
        q: usize,
        visited_set: &mut HashSetBit<R>,
        search_maxheap: &mut MaxHeap<F, R>,
        frontier_minheap: &mut MinHeap<F, R>,
        frontier_dualheap: &mut DualHeap<F, R>,
        entry_points: &mut Vec<(F, R)>,
        max_heap_size_override: Option<usize>,
    ) {
        visited_set.clear();
        let max_heap_size = max_heap_size_override.unwrap_or(self._max_build_heap_size());
        if max_heap_size > 1 {
            search_maxheap.clear();
            entry_points.into_iter().for_each(|&mut (d, i)| {
                if search_maxheap.size() < max_heap_size {
                    search_maxheap.push(d, i);
                } else {
                    search_maxheap.push_pop(d, i);
                }
                visited_set.insert(i);
            });
            if self._max_build_frontier_size().is_none() {
                frontier_minheap.clear();
                entry_points
                    .into_iter()
                    .for_each(|&mut (d, i)| frontier_minheap.push(d, i));
                while let Some((d, v)) = frontier_minheap.pop() {
                    if d > search_maxheap.peek().unwrap().0 {
                        break;
                    }
                    for &(_, j) in self._graph().view_neighbors(v) {
                        if visited_set.insert(j) {
                            let neighbor_dist =
                                self._get_dist(mat, q, unsafe { j.to_usize().unwrap_unchecked() });
                            if search_maxheap.size() < max_heap_size {
                                search_maxheap.push(neighbor_dist, j);
                            } else {
                                search_maxheap.push_pop(neighbor_dist, j);
                            }
                            frontier_minheap.push(neighbor_dist, j);
                        }
                    }
                }
            } else {
                frontier_dualheap.clear();
                let max_frontier_size =
                    unsafe { self._max_build_frontier_size().unwrap_unchecked() };
                entry_points.into_iter().for_each(|&mut (d, i)| {
                    if frontier_dualheap.size() < max_frontier_size {
                        frontier_dualheap.push(d, i);
                    } else {
                        frontier_dualheap.push_pop::<false>(d, i);
                    }
                });
                while let Some((d, v)) = frontier_dualheap.pop::<true>() {
                    if d > search_maxheap.peek().unwrap().0 {
                        break;
                    }
                    for &(_, j) in self._graph().view_neighbors(v) {
                        if visited_set.insert(j) {
                            let neighbor_dist =
                                self._get_dist(mat, q, unsafe { j.to_usize().unwrap_unchecked() });
                            if search_maxheap.size() < max_heap_size {
                                search_maxheap.push(neighbor_dist, j);
                            } else {
                                search_maxheap.push_pop(neighbor_dist, j);
                            }
                            if frontier_dualheap.size() < max_frontier_size {
                                frontier_dualheap.push(neighbor_dist, j);
                            } else {
                                frontier_dualheap.push_pop::<false>(neighbor_dist, j);
                            }
                        }
                    }
                }
            }
            entry_points.clear();
            entry_points.reserve(search_maxheap.size());
            unsafe {
                entry_points.set_len(search_maxheap.size());
            }
            entry_points
                .iter_mut()
                .rev()
                .zip(search_maxheap.sorted_iter())
                .for_each(|(x, y)| (x.0, x.1) = y);
        } else {
            let (mut min_dist, mut min_idx) = (F::infinity(), R::zero());
            entry_points.into_iter().for_each(|&mut (d, i)| {
                if min_dist > d {
                    (min_dist, min_idx) = (d, i);
                }
                visited_set.insert(i);
            });
            if self._max_build_frontier_size().is_none() {
                frontier_minheap.clear();
                entry_points
                    .into_iter()
                    .for_each(|&mut (d, i)| frontier_minheap.push(d, i));
                while let Some((d, v)) = frontier_minheap.pop() {
                    if d > min_dist {
                        break;
                    }
                    for &(_, j) in self._graph().view_neighbors(v) {
                        if visited_set.insert(j) {
                            let neighbor_dist =
                                self._get_dist(mat, q, unsafe { j.to_usize().unwrap_unchecked() });
                            if min_dist > neighbor_dist {
                                (min_dist, min_idx) = (neighbor_dist, j);
                            }
                            frontier_minheap.push(neighbor_dist, j);
                        }
                    }
                }
            } else {
                frontier_dualheap.clear();
                let max_frontier_size =
                    unsafe { self._max_build_frontier_size().unwrap_unchecked() };
                entry_points.into_iter().for_each(|&mut (d, i)| {
                    if frontier_dualheap.size() < max_frontier_size {
                        frontier_dualheap.push(d, i);
                    } else {
                        frontier_dualheap.push_pop::<false>(d, i);
                    }
                });
                while let Some((d, v)) = frontier_dualheap.pop::<true>() {
                    if d > min_dist {
                        break;
                    }
                    for &(_, j) in self._graph().view_neighbors(v) {
                        if visited_set.insert(j) {
                            let neighbor_dist =
                                self._get_dist(mat, q, unsafe { j.to_usize().unwrap_unchecked() });
                            if min_dist > neighbor_dist {
                                (min_dist, min_idx) = (neighbor_dist, j);
                            }
                            if frontier_dualheap.size() < max_frontier_size {
                                frontier_dualheap.push(neighbor_dist, j);
                            } else {
                                frontier_dualheap.push_pop::<false>(neighbor_dist, j);
                            }
                        }
                    }
                }
            }
            entry_points.clear();
            entry_points.push((min_dist, min_idx));
        }
    }

    /// Performs a search but returns true once `to_vertex` was found, else false.
    fn _has_path<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        q: R,
        visited_set: &mut HashSetBit<R>,
        search_maxheap: &mut MaxHeap<F, R>,
        frontier_minheap: &mut MinHeap<F, R>,
        frontier_dualheap: &mut DualHeap<F, R>,
        entry_points: &mut Vec<(F, R)>,
        max_heap_size_override: Option<usize>,
    ) -> bool {
        // assumes max_heap_size is always > 1
        visited_set.clear();
        let max_heap_size = max_heap_size_override.unwrap_or(self._max_build_heap_size());
        search_maxheap.clear();
        /* Populate heap with entry points */
        entry_points.into_iter().for_each(|&mut (d, i)| {
            if search_maxheap.size() < max_heap_size {
                search_maxheap.push(d, i);
            } else {
                search_maxheap.push_pop(d, i);
            }
            visited_set.insert(i);
        });
        if self._max_build_frontier_size().is_none() {
            frontier_minheap.clear();
            entry_points
                .into_iter()
                .for_each(|&mut (d, i)| frontier_minheap.push(d, i));
            while let Some((d, v)) = frontier_minheap.pop() {
                if d > search_maxheap.peek().unwrap().0 {
                    break;
                }
                for &(_, j) in self._graph().view_neighbors(v) {
                    if j == q {
                        return true;
                    }
                    if visited_set.insert(j) {
                        let neighbor_dist = unsafe {
                            self._get_dist(
                                mat,
                                q.to_usize().unwrap_unchecked(),
                                j.to_usize().unwrap_unchecked(),
                            )
                        };
                        if search_maxheap.size() < max_heap_size {
                            search_maxheap.push(neighbor_dist, j);
                        } else {
                            search_maxheap.push_pop(neighbor_dist, j);
                        }
                        frontier_minheap.push(neighbor_dist, j);
                    }
                }
            }
        } else {
            frontier_dualheap.clear();
            let max_frontier_size = unsafe { self._max_build_frontier_size().unwrap_unchecked() };
            entry_points.into_iter().for_each(|&mut (d, i)| {
                if frontier_dualheap.size() < max_frontier_size {
                    frontier_dualheap.push(d, i);
                } else {
                    frontier_dualheap.push_pop::<false>(d, i);
                }
            });
            while let Some((d, v)) = frontier_dualheap.pop::<true>() {
                if d > search_maxheap.peek().unwrap().0 {
                    break;
                }
                for &(_, j) in self._graph().view_neighbors(v) {
                    if j == q {
                        return true;
                    }
                    if visited_set.insert(j) {
                        let neighbor_dist = unsafe {
                            self._get_dist(
                                mat,
                                q.to_usize().unwrap_unchecked(),
                                j.to_usize().unwrap_unchecked(),
                            )
                        };
                        if search_maxheap.size() < max_heap_size {
                            search_maxheap.push(neighbor_dist, j);
                        } else {
                            search_maxheap.push_pop(neighbor_dist, j);
                        }
                        if frontier_dualheap.size() < max_frontier_size {
                            frontier_dualheap.push(neighbor_dist, j);
                        } else {
                            frontier_dualheap.push_pop::<false>(neighbor_dist, j);
                        }
                    }
                }
            }
        }

        false
    }

    // Adapted from graphidxbaselines::hnsw and following the pseudo-code from the DEG paper
    // Final result is contained in `entry_points`
    fn _search_range<M: MatrixDataSource<F> + Sync>(
        &self,
        mat: &M,
        q: usize,
        k: usize,
        extend_eps: F,
        visited_set: &mut HashSetBit<R>,
        search_maxheap: &mut MaxHeap<F, R>, // serves has the result heap
        frontier_minheap: &mut MinHeap<F, R>,
        frontier_dualheap: &mut DualHeap<F, R>,
        entry_points: &mut Vec<(F, R)>,
    ) {
        // no max_heap_size is used in the range search
        let mut radius = F::infinity();
        let mut exploration_radius = radius;
        let extend_eps_plus_1 = F::one() + extend_eps;

        visited_set.clear();
        search_maxheap.clear();

        entry_points.into_iter().for_each(|&mut (d, i)| {
            if search_maxheap.size() < k {
                search_maxheap.push(d, i);
            } else {
                search_maxheap.push_pop(d, i);
            }
            visited_set.insert(i);
        });

        if self._max_build_frontier_size().is_none() {
            frontier_minheap.clear();
            entry_points
                .into_iter()
                .for_each(|&mut (d, i)| frontier_minheap.push(d, i));
            while let Some((d, v)) = frontier_minheap.pop() {
                if d > exploration_radius {
                    break;
                }
                for &(_, j) in self._graph().view_neighbors(v) {
                    if visited_set.insert(j) {
                        let neighbor_dist =
                            self._get_dist(mat, q, unsafe { j.to_usize().unwrap_unchecked() });
                        if neighbor_dist <= exploration_radius {
                            frontier_minheap.push(neighbor_dist, j);
                            if neighbor_dist < radius {
                                if search_maxheap.size() < k {
                                    search_maxheap.push(neighbor_dist, j);
                                } else {
                                    // also first pop and then push is possible here
                                    search_maxheap.push_pop(neighbor_dist, j);
                                    // search_maxheap.push_pop(neighbor_dist, j);
                                    radius = search_maxheap.peek().unwrap().0;
                                    exploration_radius = radius * extend_eps_plus_1;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            frontier_dualheap.clear();
            let max_frontier_size = unsafe { self._max_build_frontier_size().unwrap_unchecked() };
            entry_points.into_iter().for_each(|&mut (d, i)| {
                if frontier_dualheap.size() < max_frontier_size {
                    frontier_dualheap.push(d, i);
                } else {
                    frontier_dualheap.push_pop::<false>(d, i);
                }
            });
            while let Some((d, v)) = frontier_dualheap.pop::<true>() {
                if d > exploration_radius {
                    break;
                }
                for &(_, j) in self._graph().view_neighbors(v) {
                    if visited_set.insert(j) {
                        let neighbor_dist =
                            self._get_dist(mat, q, unsafe { j.to_usize().unwrap_unchecked() });
                        if neighbor_dist <= exploration_radius {
                            frontier_dualheap.push(neighbor_dist, j);
                            if neighbor_dist < radius {
                                if search_maxheap.size() < k {
                                    search_maxheap.push(neighbor_dist, j);
                                } else {
                                    // also first pop and then push is possible here
                                    search_maxheap.push_pop(neighbor_dist, j);
                                    // search_maxheap.push_pop(neighbor_dist, j);
                                    radius = search_maxheap.peek().unwrap().0;
                                    exploration_radius = radius * extend_eps_plus_1;
                                }

                                if frontier_dualheap.size() < max_frontier_size {
                                    frontier_dualheap.push(neighbor_dist, j);
                                } else {
                                    frontier_dualheap.push_pop::<false>(neighbor_dist, j);
                                }
                            }
                        }
                    }
                }
            }
        }
        entry_points.clear();
        entry_points.reserve(search_maxheap.size());
        unsafe {
            entry_points.set_len(search_maxheap.size());
        }
        entry_points
            .iter_mut()
            .rev()
            .zip(search_maxheap.sorted_iter())
            .for_each(|(x, y)| (x.0, x.1) = y);
    }

    // Adapted further from `_search_range()` and hasPath() in graph.h of DEG
    /// Performs a search but stops and returns true once `to_vertex` was found, else false.
    fn _has_path_range<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        q: R,
        k: usize,
        extend_eps: F,
        visited_set: &mut HashSetBit<R>,
        search_maxheap: &mut MaxHeap<F, R>,
        frontier_minheap: &mut MinHeap<F, R>,
        frontier_dualheap: &mut DualHeap<F, R>,
        entry_points: &mut Vec<(F, R)>,
    ) -> bool {
        // no max_heap_size is used in the range search
        let mut radius = F::infinity();
        let mut exploration_radius = radius;
        let extend_eps_plus_1 = F::one() + extend_eps;

        visited_set.clear();
        search_maxheap.clear();

        entry_points.into_iter().for_each(|&mut (d, i)| {
            if search_maxheap.size() < k {
                search_maxheap.push(d, i);
            } else {
                search_maxheap.push_pop(d, i);
            }
            visited_set.insert(i);
        });

        if self._max_build_frontier_size().is_none() {
            frontier_minheap.clear();
            entry_points.into_iter().for_each(|&mut (d, i)| {
                frontier_minheap.push(d, i);
            });
            while let Some((d, v)) = frontier_minheap.pop() {
                if d > exploration_radius {
                    break;
                }
                for &(_, j) in self._graph().view_neighbors(v) {
                    if visited_set.insert(j) {
                        if j == q {
                            return true;
                        }
                        let neighbor_dist = unsafe {
                            self._get_dist(
                                mat,
                                q.to_usize().unwrap_unchecked(),
                                j.to_usize().unwrap_unchecked(),
                            )
                        };
                        if neighbor_dist <= exploration_radius {
                            frontier_minheap.push(neighbor_dist, j);
                            if neighbor_dist < radius {
                                if search_maxheap.size() < k {
                                    search_maxheap.push(neighbor_dist, j);
                                } else {
                                    // also first pop and then push is possible here
                                    search_maxheap.push_pop(neighbor_dist, j);
                                    radius = search_maxheap.peek().unwrap().0;
                                    exploration_radius = radius * extend_eps_plus_1;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            frontier_dualheap.clear();
            let max_frontier_size = unsafe { self._max_build_frontier_size().unwrap_unchecked() };
            entry_points.into_iter().for_each(|&mut (d, i)| {
                if frontier_dualheap.size() < max_frontier_size {
                    frontier_dualheap.push(d, i);
                } else {
                    frontier_dualheap.push_pop::<false>(d, i);
                }
            });
            while let Some((d, v)) = frontier_dualheap.pop::<true>() {
                if d > exploration_radius {
                    break;
                }
                for &(_, j) in self._graph().view_neighbors(v) {
                    if visited_set.insert(j) {
                        if j == q {
                            return true;
                        }
                        let neighbor_dist = unsafe {
                            self._get_dist(
                                mat,
                                q.to_usize().unwrap_unchecked(),
                                j.to_usize().unwrap_unchecked(),
                            )
                        };
                        if neighbor_dist <= exploration_radius {
                            frontier_dualheap.push(neighbor_dist, j);
                            if neighbor_dist < radius {
                                if search_maxheap.size() < k {
                                    search_maxheap.push(neighbor_dist, j);
                                } else {
                                    // also first pop and then push is possible here
                                    search_maxheap.push_pop(neighbor_dist, j);
                                    radius = search_maxheap.peek().unwrap().0;
                                    exploration_radius = radius * extend_eps_plus_1;
                                }

                                if frontier_dualheap.size() < k {
                                    frontier_dualheap.push(neighbor_dist, j);
                                } else {
                                    frontier_dualheap.push_pop::<false>(neighbor_dist, j);
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }
}

/// Local intrinsic dimension (LID)
/// Specifies which functions to use when constructing the DEG  
/// See DEG paper or [Estimating the intrinsic dimension of datasets]
/// (https://www.nature.com/articles/s41598-017-11873-y)
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum LID {
    Unknown,
    High,
    Low,
}

// Original default values can be found here: https://github.com/Visual-Computing/
// DynamicExplorationGraph/blob/3ebb0996a4cc192524fe66e57a48a7cd9a49321f/python/
// src/deglib/builder.py#L32
// Here are different values though, and the optimization is turned off by default
// Also, uses LID::High by default
param_struct!(DEGParams[Copy, Clone] {
    edges_per_vertex: usize = 40,
    // Parameters for the search and has_path function (not the range variant) during build:
    max_build_heap_size: usize = 60, // should at least be as big edges_per_vertex
    max_build_frontier_size: Option<usize> = None,
    // extend_k/_eps is only relevant when using the range_search()
    // for simplicity the defaults for these are zero
    extend_k: usize = 0, // 30,
    extend_eps: f32  = 0.0, // 0.2,
    // if set to zero improve_() won't run -> significantly faster construction time
    improve_k: usize = 0,
    improve_eps: f32 = 0.0, // 0.001,
    lid: LID = LID::High,
    swap_tries: usize = 0, //  3,
    additional_swap_tries: usize = 0,
    use_range_search: bool = false,
    max_path_length: usize = 0, // 10,
});

// Adapted from graphindexbaselines::hnsw
// Also used as wrapper in the single-threaded version
struct DEGThreadCache<R: SyncUnsignedInteger, F: SyncFloat> {
    search_hashset: HashSetBit<R>,
    search_maxheap: MaxHeap<F, R>,
    frontier_minheap: MinHeap<F, R>,
    frontier_dualheap: DualHeap<F, R>,
}
impl<R: SyncUnsignedInteger, F: SyncFloat> DEGThreadCache<R, F> {
    fn new(
        n_data: usize,
        max_build_heap_size: usize,
        lowest_max_degree: usize,
        max_build_frontier_size: Option<usize>,
    ) -> Self {
        let mut search_hashset = <HashSetBit<R> as HashSetLike<R>>::new(n_data);
        search_hashset.reserve(max_build_heap_size * 2);
        let mut heuristic_hashset = <HashSetBit<R> as HashSetLike<R>>::new(n_data);
        heuristic_hashset.reserve(lowest_max_degree * lowest_max_degree);
        let search_maxheap = MaxHeap::with_capacity(max_build_heap_size);
        let frontier_minheap = MinHeap::with_capacity(if max_build_frontier_size.is_some() {
            0
        } else {
            max_build_heap_size
        });
        let frontier_dualheap = DualHeap::with_capacity(max_build_frontier_size.unwrap_or(0));

        DEGThreadCache {
            search_hashset,
            search_maxheap,
            frontier_minheap,
            frontier_dualheap,
        }
    }
}

pub mod single_threaded {
    use crate::deg::*;

    // This implementation most closely resembles the original code (in an un-parallelized version)
    // and contains virtually all original comments.
    // The (other) DEGParallelBuilder does not have the original comments but is otherwise exactly
    // the same implementation as the parallelization happens in `extend_graph()`. Furthermore,
    // DEGParallelBuilder has few improved variables names and statement order.
    // This implementation compared to the original misses: the option to remove vertices, the
    // option to continue training
    /// Following the implementation of the paper's authors:
    /// [GitHub](https://github.com/Visual-Computing/DynamicExplorationGraph/
    /// tree/main/cpp/deglib/include)
    pub struct DEGBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send> {
        _phantom: std::marker::PhantomData<F>,
        n_data: usize,
        params: DEGParams,
        graph: DEGBuildGraph<R, F>,
        dist: Dist,
    }

    impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send> DEGBuilder<R, F, Dist> {
        fn extend_graph<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            search_cache: &mut DEGThreadCache<R, F>,
        ) {
            let edges_per_vertex = self.params.edges_per_vertex;

            // initialize graph with all vertices already
            self.graph.reserve(self.n_data);
            (0..self.n_data).for_each(|_| {
                self.graph.add_vertex(edges_per_vertex);
            });

            let split_index = (edges_per_vertex + 1).min(self.n_data);
            // fully connect the first split_index many vertices
            for i in 0..split_index {
                for j in 0..i {
                    let dist = self._get_dist(mat, i, j);
                    let i_r = unsafe { R::from(i).unwrap_unchecked() };
                    let j_r = unsafe { R::from(j).unwrap_unchecked() };
                    self.graph.change_edge(i_r, i_r, j_r, dist);
                    self.graph.change_edge(j_r, j_r, i_r, dist);
                }
            }

            if self.n_data <= split_index {
                return;
            }

            // Add the remaining points to the graph
            if self.params.lid == LID::Unknown {
                (split_index..self.n_data).for_each(|index| {
                    self.extend_graph_unknown_lid(
                        mat,
                        unsafe { R::from(index).unwrap_unchecked() },
                        search_cache,
                    );
                });
            } else {
                (split_index..self.n_data).for_each(|index| {
                    self.extend_graph_known_lid(
                        mat,
                        unsafe { R::from(index).unwrap_unchecked() },
                        search_cache,
                    );
                });
            }
        }

        /// The LID of the dataset is unknown, add the new data one-by-one single threaded.
        fn extend_graph_unknown_lid<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            internal_index: R,
            search_cache: &mut DEGThreadCache<R, F>,
        ) {
            let edges_per_vertex = self.params.edges_per_vertex;

            // find good neighbor candidates for the new vertex
            // Skipping sampling random entry_point, as it is also not done
            // in extend_graph_known_lid() (see original code)
            let number_candidates = if self.params.use_range_search {
                self.params.extend_k.max(edges_per_vertex * 2)
            } else {
                self.params.max_build_heap_size
            };
            let mut candidates: Vec<(F, R)> = Vec::with_capacity(number_candidates);
            candidates.push((
                unsafe { self._get_dist(mat, 0, internal_index.to_usize().unwrap_unchecked()) },
                R::zero(),
            ));
            if self.params.use_range_search {
                self._search_range(
                    mat,
                    unsafe { internal_index.to_usize().unwrap_unchecked() },
                    number_candidates,
                    F::from(self.params.extend_eps).unwrap(),
                    &mut search_cache.search_hashset,
                    &mut search_cache.search_maxheap,
                    &mut search_cache.frontier_minheap,
                    &mut search_cache.frontier_dualheap,
                    &mut candidates,
                );
            } else {
                self._search(
                    mat,
                    unsafe { internal_index.to_usize().unwrap_unchecked() },
                    &mut search_cache.search_hashset,
                    &mut search_cache.search_maxheap,
                    &mut search_cache.frontier_minheap,
                    &mut search_cache.frontier_dualheap,
                    &mut candidates,
                    Some(number_candidates),
                );
            }

            /* there should always be enough neighbors (search candidates),
            otherwise the graph would be broken */
            assert!(
                candidates.len() >= edges_per_vertex,
                "Broken graph - found not enough neighbors during the search.\
             No_Candidates: {}, No_edges_per_vertex {}",
                candidates.len(),
                edges_per_vertex
            );

            // add an empty vertex to the graph (no neighbor information yet)
            // let internal_index = self.graph.add_vertex(edges_per_vertex);

            /* adding neighbors happens in two phases, the first tries to retain RNG, the second
            adds them without checking */
            let mut check_rng_phase: bool = true; // true = activated, false = deactivated

            // list of potential isolates vertices
            // arbitrary capacity
            let mut isolated_vertices: Vec<R> = Vec::with_capacity(edges_per_vertex);
            isolated_vertices.push(internal_index); // self loop needed for restore phase

            // remove an edge of a good neighbor candidate and connect
            // the candidate with the new vertex
            // the new vertex will get an additional neighbor during the restore phase
            let mut slots: usize = edges_per_vertex - 1;
            while slots > 0 {
                for &(candidate_weight, candidate_index) in candidates.as_slice() {
                    if slots == 0 {
                        break;
                    }

                    /* check if the vertex is already in the edge list of the new vertex
                    (added during a previous loop-run) since all edges are undirected and the edge
                    information of the new vertex does not yet exist, we search the other way around.
                    OR does the candidate has a neighbor which is connected to the new vertex and
                    has a lower distance? */
                    if self.graph.has_edge(candidate_index, internal_index)
                        || (check_rng_phase
                            && self.check_rng(candidate_index, internal_index, candidate_weight)
                                == false)
                    {
                        continue;
                    }

                    /* the vertex is already missing an edge (one of its longer edges was removed
                    during a previous iteration),just add an new edge between the candidate and
                    the new vertex */
                    if self.graph.has_edge(candidate_index, candidate_index) {
                        self.graph.change_edge(
                            candidate_index,
                            candidate_index,
                            internal_index,
                            candidate_weight,
                        );
                        self.graph.change_edge(
                            internal_index,
                            internal_index,
                            candidate_index,
                            candidate_weight,
                        );
                        slots -= 1;
                        continue;
                    }

                    /* This version is good for high LID datasets or small graphs with
                    low distance count limit during ANNS */
                    let mut new_neighbor_index: R = R::zero();
                    {
                        // find the worst edge of the new neighbor
                        let mut new_neighbor_weight = F::neg_infinity();
                        for &(neighbor_weight, neighbor_index) in
                            self.graph.view_neighbors(candidate_index)
                        {
                            // the suggested neighbor might already be in the edge
                            // list of the new vertex OR
                            // is the neighbor already missing an edge?
                            if self.graph.has_edge(neighbor_index, internal_index)
                                || self.graph.has_edge(neighbor_index, neighbor_index)
                            {
                                continue;
                            }

                            // find highest weighted neighbor
                            if neighbor_weight > new_neighbor_weight {
                                new_neighbor_weight = neighbor_weight;
                                new_neighbor_index = neighbor_index;
                            }
                        }
                        /* this should not be possible, otherwise the new vertex is connected to
                        every vertex in the neighbor-list of the candidate-vertex and still
                        has space for more */
                        if new_neighbor_weight == F::neg_infinity() {
                            continue;
                        }
                    }

                    /* place the new vertex in the edge list of the candidate_index and
                    the new vertex internal_index */
                    self.graph.change_edge(
                        candidate_index,
                        new_neighbor_index,
                        internal_index,
                        candidate_weight,
                    );
                    self.graph.change_edge(
                        internal_index,
                        internal_index,
                        candidate_index,
                        candidate_weight,
                    );
                    slots -= 1;

                    /* replace the edge to the candidate_index from the edge list of
                    new_neighbor_index with a self-reference */
                    self.graph.change_edge(
                        new_neighbor_index,
                        candidate_index,
                        new_neighbor_index,
                        F::zero(),
                    );
                    isolated_vertices
                        .push(unsafe { R::from(new_neighbor_index).unwrap_unchecked() });
                }
                check_rng_phase = false;
            }

            // get all vertices which are missing an edge (aka have an edge to themselves)
            isolated_vertices.retain(|&e| self.graph.has_edge(e, e));

            // restore the potential disconnected graph components
            self.restore_graph(&mat, isolated_vertices, false, search_cache);
        }

        /// The LID of the dataset is known and defined
        fn extend_graph_known_lid<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            internal_index: R,
            search_cache: &mut DEGThreadCache<R, F>,
        ) {
            let edges_per_vertex = self.params.edges_per_vertex;

            // find good neighbors for the new vertex
            // apparently no entry point sampling?
            let mut entry_points: Vec<(F, R)> = vec![(
                unsafe { self._get_dist(mat, 0, internal_index.to_usize().unwrap_unchecked()) },
                R::zero(),
            )];

            if self.params.use_range_search {
                self._search_range(
                    mat,
                    unsafe { internal_index.to_usize().unwrap_unchecked() },
                    self.params.extend_k.max(edges_per_vertex),
                    F::from(self.params.extend_eps).unwrap(),
                    &mut search_cache.search_hashset,
                    &mut search_cache.search_maxheap,
                    &mut search_cache.frontier_minheap,
                    &mut search_cache.frontier_dualheap,
                    &mut entry_points,
                );
            } else {
                self._search(
                    mat,
                    unsafe { internal_index.to_usize().unwrap_unchecked() },
                    &mut search_cache.search_hashset,
                    &mut search_cache.search_maxheap,
                    &mut search_cache.frontier_minheap,
                    &mut search_cache.frontier_dualheap,
                    &mut entry_points,
                    Some(self.params.max_build_heap_size),
                    // F::from(self.params.extend_eps).unwrap(),
                );
            }
            let candidates = entry_points;

            /* there should always be enough neighbors (search results), otherwise
            the graph would be broken */
            assert!(
                candidates.len() >= edges_per_vertex,
                "Broken graph - found not enough neighbors during the search.\
                No_Candidates: {}, No_edges_per_vertex {}",
                candidates.len(),
                edges_per_vertex
            );

            /* adding neighbors happens in two phases, the first tries to retain RNG, the
            second adds them without checking */
            let mut check_rng_phase = true;

            // remove an edge of the good neighbors and connect them with this new vertex
            let mut new_neighbors: Vec<(F, R)> = Vec::with_capacity(edges_per_vertex);
            while new_neighbors.len() < edges_per_vertex {
                for &(candidate_weight, candidate_index) in candidates.as_slice() {
                    if new_neighbors.len() >= edges_per_vertex {
                        break;
                    }

                    /* check if the vertex is already in the edge list of the new vertex
                    (added during a previous loop-run) since all edges are undirected and the edge
                    information of the new vertex does not yet exist, we search the other way
                    around. OR does the candidate have a neighbor which is connected to the new
                    vertex and have a lower distance? */
                    if self.graph.has_edge(candidate_index, internal_index)
                        || (check_rng_phase
                            && self.check_rng(candidate_index, internal_index, candidate_weight)
                                == false)
                    {
                        continue;
                    }

                    /* Schema C: This version is good for high LID datasets or small graphs with
                    low distance count limit during ANNS */
                    let mut new_neighbor_index = R::zero();
                    let mut new_neighbor_distance = F::neg_infinity();
                    if self.params.lid == LID::High {
                        // find the worst edge of the new neighbor
                        let mut new_neighbor_weight = F::neg_infinity();
                        for &(neighbor_weight, neighbor_index) in
                            self.graph.view_neighbors(candidate_index)
                        {
                            /* if another thread is building the candidate_index at the moment, than
                            its neighbor list contains self references OR the suggested neighbor
                            might already be in the edge list of the new vertex */
                            if candidate_index == neighbor_index
                                || self.graph.has_edge(neighbor_index, internal_index)
                            {
                                continue;
                            }

                            /* the weight of the neighbor might not be worst than the current
                            worst one */

                            if neighbor_weight > new_neighbor_weight {
                                new_neighbor_weight = neighbor_weight;
                                new_neighbor_index = neighbor_index;
                            }
                        }

                        /* should not be possible, otherwise the new vertex is connected to every
                        vertex in the neighbor-list of the result-vertex and still has space
                        for more */
                        if new_neighbor_weight == F::neg_infinity() {
                            continue;
                        }
                        new_neighbor_distance = self._get_dist(
                            mat,
                            unsafe { internal_index.to_usize().unwrap_unchecked() },
                            unsafe { new_neighbor_index.to_usize().unwrap_unchecked() },
                        )
                    } else {
                        /* find the edge which improves the distortion the most: (
                        distance_new_edge1 + distance_new_edge2) - distance_removed_edge */
                        let mut best_distortion = F::infinity();
                        for &(neighbor_weight, neighbor_index) in
                            self.graph.view_neighbors(candidate_index)
                        {
                            /* if another thread is building the candidate_index at the moment, than
                            its neighbor list contains self references OR the suggested neighbor
                            might already be in the edge list of the new vertex */
                            if candidate_index == neighbor_index
                                || self.graph.has_edge(neighbor_index, internal_index)
                            {
                                continue;
                            }

                            /* take the neighbor with the best distance to the new vertex, which
                            might already be in its edge list */
                            let neighbor_distance = self._get_dist(
                                mat,
                                unsafe { internal_index.to_usize().unwrap_unchecked() },
                                unsafe { neighbor_index.to_usize().unwrap_unchecked() },
                            );
                            let distortion =
                                // version D in the paper
                                (candidate_weight + neighbor_distance) - neighbor_weight;
                            if distortion < best_distortion {
                                best_distortion = distortion;
                                new_neighbor_index = neighbor_index;
                                new_neighbor_distance = neighbor_distance;
                            }
                        }
                    }

                    /* this should not be possible, otherwise the new vertex is connected to every
                    vertex in the neighbor-list of the result-vertex and still has space for more */
                    if new_neighbor_distance == F::neg_infinity() {
                        continue;
                    }
                    /* update all edges
                    Skipping edge-existing checks in single-threaded mode: other threads might
                    have already changed the edges of the new_neighbor_index update edge list of
                     the new vertex */
                    self.graph.change_edge(
                        internal_index,
                        internal_index,
                        candidate_index,
                        candidate_weight,
                    );
                    self.graph.change_edge(
                        internal_index,
                        internal_index,
                        new_neighbor_index,
                        new_neighbor_distance,
                    );
                    new_neighbors.push((candidate_weight, candidate_index));
                    new_neighbors.push((new_neighbor_distance, new_neighbor_index));

                    // place the new vertex in the edge list of the result-vertex
                    self.graph.change_edge(
                        candidate_index,
                        new_neighbor_index,
                        internal_index,
                        candidate_weight,
                    );

                    // place the new vertex in the edge list of the best edge neighbor
                    self.graph.change_edge(
                        new_neighbor_index,
                        candidate_index,
                        internal_index,
                        new_neighbor_distance,
                    );
                }

                check_rng_phase = false;
            }
            assert!(new_neighbors.len() >= edges_per_vertex);
        }

        // helper function to check if we need to find more connected components
        #[inline(always)]
        fn is_enough_free_connections(
            &self,
            vertices: &[R],
            paths: &mut UnionFindDeg<R>,
            groups: &HashMap<R, ReachableGroup<R>>,
        ) -> bool {
            let mut isolated_vertex_counter: usize = 0;
            let mut available_connections_counter: usize = 0;
            for &involved_vertex in vertices {
                let reachable_vertex = paths.find(involved_vertex);
                if involved_vertex == reachable_vertex {
                    let group = unsafe { groups.get(&reachable_vertex).unwrap_unchecked() };
                    if group.size() == 1 {
                        isolated_vertex_counter += 1;
                    } else if group.get_missing_edge_size() > 2 {
                        available_connections_counter += group.get_missing_edge_size() - 2;
                    }
                }
            }
            available_connections_counter < isolated_vertex_counter
        }

        /// Reconnect the vertices indicated in the list of involved_indices.
        /// All these vertices are missing an edge.
        fn restore_graph<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            involved_indices: Vec<R>,
            improve_edges: bool,
            search_cache: &mut DEGThreadCache<R, F>,
        ) {
            let edges_per_vertex: usize = self.params.edges_per_vertex;
            // 2 find pairs or groups of vertices which can reach each other
            // using a Vec instead of a Set for easier handling
            let mut unique_groups: Vec<ReachableGroup<R>> =
                Vec::with_capacity(involved_indices.len());
            {
                let mut path_map: UnionFindDeg<R> = UnionFindDeg::new(edges_per_vertex);
                let mut reachable_groups: HashMap<R, ReachableGroup<R>> = HashMap::new();
                reachable_groups.reserve(edges_per_vertex);

                involved_indices.iter().for_each(|&involved_index| {
                    reachable_groups.insert(
                        involved_index,
                        ReachableGroup::new(involved_index, edges_per_vertex),
                    );
                    path_map.update(involved_index, involved_index);
                });

                /* helper function `is_enough_free_connections` is defined on self and
                not here directly */

                // 2.1 start with checking the adjacent neighbors
                let mut neighbor_check_depth: usize = 0;
                let mut check: HashSet<R> =
                    foldhash::HashSet::with_capacity(involved_indices.len());
                involved_indices.iter().for_each(|&e| {
                    check.insert(e);
                });
                let mut check_next: HashSet<R> = foldhash::HashSet::with_capacity(check.len());
                while self.is_enough_free_connections(
                    involved_indices.as_slice(),
                    &mut path_map,
                    &reachable_groups,
                ) {
                    check.iter().for_each(|&check_vertex| {
                        let involved_vertex = path_map.find(check_vertex);
                        /* check only involved vertices and vertices which can only reach 1 involved
                        vertex no need for big groups to find other groups at the expense of
                        processing power */
                        if neighbor_check_depth > 0
                            && unsafe {
                                reachable_groups
                                    .get(&involved_vertex)
                                    .unwrap_unchecked()
                                    .size()
                            } > 1
                        {
                            // continue
                        } else {
                            /* check the neighbors of check_vertex if they can reach
                            another reachableGroup */
                            for &(_, neighbor_index) in self.graph.view_neighbors(check_vertex) {
                                // skip self references (loops)
                                if neighbor_index == check_vertex {
                                    continue;
                                }

                                // which other involved vertex can be reached by this neighbor
                                let other_involved_vertex = path_map.find(neighbor_index);

                                // neighbor is not yet in the union find
                                if other_involved_vertex == path_map.get_default_value() {
                                    path_map.update(neighbor_index, involved_vertex);
                                    check_next.insert(neighbor_index);
                                } else if other_involved_vertex != involved_vertex {
                                    // the neighbor can reach another involved vertex
                                    path_map.update(other_involved_vertex, involved_vertex);
                                    let other_group_binding = unsafe {
                                        reachable_groups
                                            .get(&other_involved_vertex)
                                            .unwrap_unchecked()
                                            .clone()
                                    };
                                    let reachable_group = unsafe {
                                        reachable_groups
                                            .get_mut(&involved_vertex)
                                            .unwrap_unchecked()
                                    };
                                    reachable_group.copy_from(&other_group_binding);
                                }
                            }
                        }
                    });

                    // prepare for the next iteration
                    std::mem::swap(&mut check, &mut check_next);
                    check_next.clear();
                    neighbor_check_depth += 1;
                }

                // copy the unique groups
                involved_indices.iter().for_each(|&involved_index| unsafe {
                    let group = reachable_groups
                        .get(&path_map.find(involved_index))
                        .unwrap_unchecked();
                    if !unique_groups.contains(group) {
                        unique_groups.push(group.to_owned());
                    }
                });
            }

            // 2.2 get all isolated vertices
            // Vec because unique groups should be unique already
            let mut isolated_groups: Vec<ReachableGroup<R>> = Vec::new();
            unique_groups.iter().for_each(|group| {
                if group.size() == 1 {
                    isolated_groups.push(group.clone());
                }
            });

            /* 2.3 find for every isolated vertex the best other involved vertex which is part
            of a unique group */
            let mut new_edges: Vec<(R, R, F)> = Vec::with_capacity(isolated_groups.len());
            isolated_groups.iter_mut().for_each(|isolated_group| {
                // are you still isolated?
                if isolated_group.size() > 1 {
                    // continue
                } else {
                    let isolated_vertex = isolated_group.get_vertex_index();

                    /* check the reachable groups for good candidates which can connect to
                    the isolated vertex */
                    let mut best_candidate_index: R = R::zero();
                    let mut best_candidate_distance: F = F::infinity();
                    // using index access element outside/after closure of for_each()
                    let mut best_candidate_group_index: usize = 0;

                    unique_groups
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, candidate_group)| {
                            // skip all groups which do not have enough vertices missing an edge
                            let missing_edges = candidate_group.get_missing_edges();
                            if missing_edges.len() <= 2 {
                                // continue
                            } else {
                                // find the candidate with the best distance to the isolated vertex
                                missing_edges.iter().for_each(|&candidate| unsafe {
                                    let dist = self._get_dist(
                                        mat,
                                        isolated_vertex.to_usize().unwrap_unchecked(),
                                        candidate.to_usize().unwrap_unchecked(),
                                    );
                                    if dist < best_candidate_distance {
                                        best_candidate_distance = dist;
                                        best_candidate_index = candidate;
                                        best_candidate_group_index = i;
                                    }
                                });
                            }
                        });

                    assert!(unique_groups.len() > 0);
                    let best_candidate_group = &mut unique_groups[best_candidate_group_index];

                    /* found a good candidate, add the isolated vertex to its reachable group and
                    an edge between them */
                    self.graph.change_edge(
                        isolated_vertex,
                        isolated_vertex,
                        best_candidate_index,
                        best_candidate_distance,
                    );
                    self.graph.change_edge(
                        best_candidate_index,
                        best_candidate_index,
                        isolated_vertex,
                        best_candidate_distance,
                    );
                    new_edges.push((
                        isolated_vertex,
                        best_candidate_index,
                        best_candidate_distance,
                    ));

                    // merge groups
                    best_candidate_group.has_edge(best_candidate_index);
                    isolated_group.has_edge(isolated_vertex);
                    best_candidate_group.copy_from(&isolated_group);

                    // remove isolated_group from unique_groups
                    unique_groups
                        .retain(|e| e.get_vertex_index() != isolated_group.get_vertex_index());
                }
            });

            // 3 reconnect the groups
            // unique_groups is already a vector so just move/rename
            let mut reachable_groups: Vec<ReachableGroup<R>> = unique_groups;

            // Sort the groups by size in ascending order
            reachable_groups
                .sort_unstable_by(|a, b| a.get_missing_edge_size().cmp(&b.get_missing_edge_size()));

            /* 3.1 Find the biggest group and one of its vertices to one vertex of a smaller group.
            Repeat until only one group is left */
            while reachable_groups.len() >= 2 {
                let l = reachable_groups.len();
                // Splitting to have two 'separate' mutable references
                let (rgs_front, rgs_back) = reachable_groups.split_at_mut(l - 1);
                let idx = rgs_front.len() - 1;
                let reachable_group = &mut rgs_front[idx];
                let other_group = &mut rgs_back[0];
                let reachable_vertices = reachable_group.get_missing_edges();
                let other_vertices = other_group.get_missing_edges();

                let mut best_other_it = R::zero();
                let mut best_reachable_it = R::zero();
                let mut best_other_distance = F::infinity();
                /* assertions to avoid extracting first element from best_other_it and
                best_reachable_it, because if assertion are passed then they will
                definitely be reassigned */
                assert!(reachable_vertices.len() > 0);
                assert!(other_vertices.len() > 0);

                // iterate over all its entries to find a vertex which is still missing an edge
                reachable_vertices.iter().for_each(|&reachable_it| {
                    /* find another vertex in a smaller group, also missing an edge
                    the other vertex and reachable_index can not share an edge yet, otherwise
                    they would be in the same group due to step 2.1 */
                    other_vertices.iter().for_each(|&other_it| {
                        let candidate_dist: F = self._get_dist(
                            mat,
                            unsafe { reachable_it.to_usize().unwrap_unchecked() },
                            unsafe { other_it.to_usize().unwrap_unchecked() },
                        );
                        if candidate_dist < best_other_distance {
                            best_other_it = other_it;
                            best_reachable_it = reachable_it;
                            best_other_distance = candidate_dist;
                        }
                    });
                });

                // connect reachable_index and other_index
                let reachable_index = best_reachable_it;
                let other_index = best_other_it;
                self.graph.change_edge(
                    reachable_index,
                    reachable_index,
                    other_index,
                    best_other_distance,
                );
                self.graph.change_edge(
                    other_index,
                    other_index,
                    reachable_index,
                    best_other_distance,
                );

                // move the element from the list of missing edges
                reachable_group.has_edge(reachable_index);
                other_group.has_edge(other_index);

                // merge both groups
                other_group.copy_from(&reachable_group);

                // remove the current group from the list of group since its merged
                reachable_groups.pop();
            }

            /* 3.4 now all groups are reachable but still some vertices are missing edge, try
            to connect them to each other */
            let remaining_indices: Vec<R> = if reachable_groups.len() > 0 {
                let mut v = Vec::with_capacity(reachable_groups[0].get_missing_edge_size());
                reachable_groups[0]
                    .get_missing_edges()
                    .iter()
                    .for_each(|&e| v.push(e));
                v
            } else {
                Vec::new()
            };

            for (i, &index_a) in remaining_indices.iter().enumerate() {
                // still missing an edge?
                if self.graph.has_edge(index_a, index_a) {
                    // find a index_B with the smallest distance to index_A
                    let mut best_index_b = R::zero();
                    let mut best_distance_ab = F::infinity();
                    /* using this instead of the original comparison below when
                    connect vertexA and vertexB */
                    let mut best_index_b_changed = false;

                    for &index_b in &remaining_indices[i + 1..] {
                        if self.graph.has_edge(index_b, index_b)
                            && !self.graph.has_edge(index_a, index_b)
                        {
                            let new_neighbor_dist = self._get_dist(
                                mat,
                                unsafe { index_a.to_usize().unwrap_unchecked() },
                                unsafe { index_b.to_usize().unwrap_unchecked() },
                            );
                            if new_neighbor_dist < best_distance_ab {
                                best_distance_ab = new_neighbor_dist;
                                best_index_b = index_b;
                                best_index_b_changed = true;
                            }
                        }
                    }

                    // connect vertexA and vertexB
                    if best_index_b_changed {
                        self.graph
                            .change_edge(index_a, index_a, best_index_b, best_distance_ab);
                        self.graph.change_edge(
                            best_index_b,
                            best_index_b,
                            index_a,
                            best_distance_ab,
                        );
                    }
                }
            }

            /* 3.5 the remaining vertices can not be connected to any of the other involved
            vertices, because they already have an edge to all of them. */
            for (i, &index_a) in remaining_indices.iter().enumerate() {
                // still missing an edge?
                if self.graph.has_edge(index_a, index_a) {
                    /* scan the neighbors of the adjacent vertices of A and find a vertex B
                    with the smallest distance to A */
                    let mut best_index_b = R::zero();
                    let mut best_distance_ab = F::infinity();
                    for &(_, neighbor_id) in self.graph.view_neighbors(index_a) {
                        for &(_, index_b_id) in self.graph.view_neighbors(neighbor_id) {
                            if index_a != index_b_id && !self.graph.has_edge(index_a, index_b_id) {
                                let new_neighbor_dist = self._get_dist(
                                    mat,
                                    unsafe { index_a.to_usize().unwrap_unchecked() },
                                    unsafe { index_b_id.to_usize().unwrap_unchecked() },
                                );
                                if new_neighbor_dist < best_distance_ab {
                                    best_distance_ab = new_neighbor_dist;
                                    best_index_b = index_b_id;
                                }
                            }
                        }
                    }

                    /* Get another vertex missing an edge called C and at this point sharing an
                    edge with A (by definition of 3.2) */
                    for &index_c in &remaining_indices[i + 1..] {
                        // still missing an edge?
                        if self.graph.has_edge(index_c, index_c) {
                            /* check the neighborhood of B to find a vertex D not yet adjacent to
                            C but with the smallest possible distance to C */
                            let mut best_index_d = R::zero();
                            let mut best_distance_cd = F::infinity();

                            for &(_, index_d) in self.graph.view_neighbors(best_index_b) {
                                if index_a != index_d
                                    && best_index_b != index_d
                                    && !self.graph.has_edge(index_c, index_d)
                                {
                                    let new_neighbor_dist = self._get_dist(
                                        mat,
                                        unsafe { index_c.to_usize().unwrap_unchecked() },
                                        unsafe { index_d.to_usize().unwrap_unchecked() },
                                    );
                                    if new_neighbor_dist < best_distance_cd {
                                        best_distance_cd = new_neighbor_dist;
                                        best_index_d = index_d;
                                    }
                                }
                            }

                            /* replace edge between B and D, with one between A and B as well as
                            C and D */
                            self.graph.change_edge(
                                best_index_b,
                                best_index_d,
                                index_a,
                                best_distance_ab,
                            );
                            self.graph.change_edge(
                                index_a,
                                index_a,
                                best_index_b,
                                best_distance_ab,
                            );
                            self.graph.change_edge(
                                best_index_d,
                                best_index_b,
                                index_c,
                                best_distance_cd,
                            );
                            self.graph.change_edge(
                                index_c,
                                index_c,
                                best_index_d,
                                best_distance_cd,
                            );

                            break;
                        }
                    }
                }
            }

            // improve some of the new edges which are not so good
            if improve_edges && self.params.improve_k > 0 {
                /* Define a custom comparison function based on the size of the sets
                Sort the groups by size in ascending order
                Note ascending like written in original code although they use a
                descending sorter */
                new_edges.sort_unstable_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap());

                // 4 try to improve some of the new edges
                for i in 0..new_edges.len() {
                    let edge = new_edges[i];
                    if self.graph.has_edge(edge.0, edge.1) {
                        self.improve_edge(mat, edge.0, edge.1, edge.2, search_cache);
                    }
                }
            }
        }

        // In original paper also called optimizeEdges?
        /// Try to improve the edge of a random vertex to its worst neighbor
        /// @return true if a change could be made otherwise false
        fn improve_edges_build<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            rng: &mut ThreadRng,
            search_cache: &mut DEGThreadCache<R, F>,
        ) -> bool {
            let mut success = false;

            // 1.1 select a random vertex
            let vertex1 = unsafe {
                R::from((*rng).gen_range(0..(self.graph.n_vertices()))).unwrap_unchecked()
            };

            // 1.2 find the worst edge of this vertex
            unsafe {
                let graph = &mut self.graph as *mut DEGBuildGraph<R, F>;
                for &(weight_v1v2, vertex2) in (*graph).view_neighbors(vertex1) {
                    if self.check_rng(vertex2, vertex1, weight_v1v2) == false
                    // && (*graph).has_edge(vertex1, vertex2).is_some() // kinda redundant check if going through neighbors, no? - dropped it therefore
                    {
                        success |=
                            self.improve_edge(mat, vertex1, vertex2, weight_v1v2, search_cache);
                    }
                }
            }
            success
        }

        /// Try to improve the existing edge between the two vertices
        /// @return true if a change could be made otherwise false
        fn improve_edge<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            vertex1: R,
            vertex2: R,
            weight_v1v2: F,
            search_cache: &mut DEGThreadCache<R, F>,
        ) -> bool {
            // improving edges is disabled
            if self.params.improve_k <= 0 {
                return false;
            }

            // remove the edge between vertex 1 and vertex 2 (add temporary self-loops)
            let mut changes: Vec<BuilderChange<R, F>> =
                Vec::with_capacity(self.params.edges_per_vertex * 2);
            self.graph.change_edge(vertex1, vertex2, vertex1, F::zero());
            changes.push(BuilderChange::new(
                vertex1,
                vertex2,
                weight_v1v2,
                vertex1,
                F::zero(),
            ));
            self.graph.change_edge(vertex2, vertex1, vertex2, F::zero());
            changes.push(BuilderChange::new(
                vertex2,
                vertex1,
                weight_v1v2,
                vertex2,
                F::zero(),
            ));

            if self.improve_edges(
                mat,
                &mut changes,
                vertex1,
                vertex2,
                vertex1,
                vertex1,
                weight_v1v2,
                0,
                search_cache,
            ) == false
            {
                // undo all changes, in reverse order
                let size = changes.len();
                for i in 0..size {
                    let c = changes[(size - 1) - i];
                    self.graph.change_edge(
                        c.internal_index,
                        c.to_neighbor_index,
                        c.from_neighbor_index,
                        c.from_neighbor_weight,
                    );
                }
                return false;
            }
            true
        }

        /**
        Do not call this method directly instead call improve() to improve the graph.
        This is the extended part of the optimization process.
        The method takes an array where all graph changes will be documented.
        Vertex1 and vertex2 might be in a separate subgraph than vertex3 and vertex4.
        Thru a series of edges swaps both subgraphs should be reconnected..
        If those changes improve the graph this method returns true otherwise false.
        @return true if a good sequences of changes has been found
        **/
        fn improve_edges<M: MatrixDataSource<F> + Sync>(
            &mut self,
            mat: &M,
            changes: &mut Vec<BuilderChange<R, F>>,
            mut vertex1: R,
            vertex2: R,
            mut vertex3: R,
            mut vertex4: R,
            mut total_gain: F,
            steps: usize,
            search_cache: &mut DEGThreadCache<R, F>,
        ) -> bool {
            let vertex1_usize = unsafe { vertex1.to_usize().unwrap_unchecked() };
            let vertex2_usize = unsafe { vertex2.to_usize().unwrap_unchecked() };
            let vertex3_usize = unsafe { vertex3.to_usize().unwrap_unchecked() };
            let vertex4_usize = unsafe { vertex4.to_usize().unwrap_unchecked() };
            /* 1. Find an edge for vertex2 which connects to the subgraph of vertex3
            and vertex4.
            Consider only vertices of the approximate nearest neighbor search. Since the
            search started from vertex3 and vertex4 all vertices in the result list are in
            their subgraph and would therefore connect the two potential subgraphs. */
            let number_candidates = self.params.improve_k;
            let mut top_list: Vec<(F, R)> = Vec::with_capacity(number_candidates);
            top_list.push((self._get_dist(mat, vertex2_usize, vertex3_usize), vertex3));
            top_list.push((self._get_dist(mat, vertex2_usize, vertex4_usize), vertex4));

            if self.params.use_range_search {
                self._search_range(
                    mat,
                    vertex2_usize,
                    number_candidates,
                    F::from(self.params.improve_eps).unwrap(),
                    &mut search_cache.search_hashset,
                    &mut search_cache.search_maxheap,
                    &mut search_cache.frontier_minheap,
                    &mut search_cache.frontier_dualheap,
                    &mut top_list,
                );
            } else {
                self._search(
                    mat,
                    vertex2_usize,
                    &mut search_cache.search_hashset,
                    &mut search_cache.search_maxheap,
                    &mut search_cache.frontier_minheap,
                    &mut search_cache.frontier_dualheap,
                    &mut top_list,
                    Some(number_candidates),
                    // F::from(self.params.improve_eps).unwrap(),
                );
            }
            // find a good new vertex3
            let mut best_gain = total_gain;
            let mut dist23 = F::neg_infinity();
            let mut dist34 = F::neg_infinity();

            /* We use the descending order to find the worst swap combination with the best gain
            Sometimes the gain between the two best combinations is the same, its better to use
            one with the bad edges to make later improvements easier */
            top_list.sort_unstable_by(|(a, _), &(b, _)| b.partial_cmp(a).unwrap()); // sort desc
            top_list.iter().for_each(|&result| {
                let new_vertex3 = result.1;
                // vertex1 and vertex2 got tested in the recursive call before and
                // vertex4 got just disconnected from vertex2
                if vertex1 != new_vertex3
                    && vertex2 != new_vertex3
                    && !self.graph.has_edge(vertex2, new_vertex3)
                {
                    /* 1.1 When vertex2 and the new vertex 3 gets connected, the full graph
                    connectivity is assured again, but the subgraph between vertex1/vertex2
                    and vertex3/vertex4 might just have one edge(vertex2, vertex3).
                    Furthermore Vertex 3 has now to many edges, find an good edge to remove
                    to improve the overall graph distortion. FYI: If the just selected
                    vertex3 is the same as the old vertex3, this process might cut its
                    connection to vertex4 again. This will be fixed in the next step or
                    until the recursion reaches max_path_length. */
                    for &(new_vertex4_weight, new_vertex4) in self.graph.view_neighbors(new_vertex3)
                    {
                        /* compute the gain of the graph distortion if this change would
                        be applied */
                        let gain: F = total_gain - result.0 + new_vertex4_weight;
                        // do not remove the edge which was just added
                        if new_vertex4 != vertex2 && best_gain < gain {
                            best_gain = gain;
                            vertex3 = new_vertex3;
                            vertex4 = new_vertex4;
                            dist23 = result.0;
                            dist34 = new_vertex4_weight;
                        }
                    }
                }
            });

            // no new vertex3 was found
            if dist23 == F::neg_infinity() {
                return false;
            }

            // replace the temporary self-loop of vertex2 with a connection to vertex3.
            total_gain = (total_gain - dist23) + dist34;
            self.graph.change_edge(vertex2, vertex2, vertex3, dist23);
            changes.push(BuilderChange::new(
                vertex2,
                vertex2,
                F::zero(),
                vertex3,
                dist23,
            ));

            /* 1.2 Remove the worst edge of vertex3 to vertex4 and replace it with the
            connection to vertex2. Add a temporary self-loop for vertex4 for the missing
            edge to vertex3 */
            self.graph.change_edge(vertex3, vertex4, vertex2, dist23);
            changes.push(BuilderChange::new(
                vertex3, vertex4, dist34, vertex2, dist23,
            ));
            self.graph.change_edge(vertex4, vertex3, vertex4, F::zero());
            changes.push(BuilderChange::new(
                vertex4,
                vertex3,
                dist34,
                vertex4,
                F::zero(),
            ));

            // 2. Try to connect vertex1 with vertex4
            /* 2.1a Vertex1 and vertex4 might be the same. This is quite the rare case, but
            would mean there are two edges missing. Proceed like extending the graph:
            Search for a good vertex to connect to, remove its worst edge and connect
            both vertices of the worst edge to the vertex4. Skip the edge any of the two
            two vertices are already connected to vertex4. */
            if vertex1 == vertex4 {
                // find a good (not yet connected) vertex for vertex1/vertex4
                let number_candidates = self.params.improve_k;
                let mut top_list: Vec<(F, R)> = Vec::with_capacity(number_candidates);
                top_list.push((self._get_dist(mat, vertex4_usize, vertex2_usize), vertex2));
                top_list.push((self._get_dist(mat, vertex4_usize, vertex3_usize), vertex3));

                if self.params.use_range_search {
                    self._search_range(
                        mat,
                        vertex4_usize,
                        number_candidates,
                        F::from(self.params.improve_eps).unwrap(),
                        &mut search_cache.search_hashset,
                        &mut search_cache.search_maxheap,
                        &mut search_cache.frontier_minheap,
                        &mut search_cache.frontier_dualheap,
                        &mut top_list,
                    );
                } else {
                    self._search(
                        mat,
                        vertex4_usize,
                        &mut search_cache.search_hashset,
                        &mut search_cache.search_maxheap,
                        &mut search_cache.frontier_minheap,
                        &mut search_cache.frontier_dualheap,
                        &mut top_list,
                        Some(number_candidates),
                    );
                }
                let mut best_gain = F::zero();
                let mut best_selected_neighbor = R::zero();
                let mut best_old_neighbor_dist = F::zero();
                let mut best_new_neighbor_dist = F::zero();
                let mut best_good_vertex = R::zero();
                let mut best_good_vertex_dist = F::zero();
                top_list.sort_unstable_by(|(a, _), &(b, _)| a.partial_cmp(&b).unwrap()); // sort asc
                top_list.iter().for_each(|&result| {
                    let good_vertex = result.1;
                    // the new vertex should not be connected to vertex4 yet
                    if vertex4 != good_vertex && !self.graph.has_edge(vertex4, good_vertex) {
                        let good_vertex_dist = result.0;

                        /* select any edge of the good vertex which improves the graph quality
                        when replaced with a connection to vertex 4 */
                        for &(old_neighbor_dist, selected_neighbor) in
                            self.graph.view_neighbors(good_vertex)
                        {
                            /* ignore edges where the second vertex is already connect
                            to vertex4 */
                            if vertex4 != selected_neighbor
                                && !self.graph.has_edge(vertex4, selected_neighbor)
                            {
                                // skipping pointless factor = 1
                                let new_neighbor_dist =
                                    self._get_dist(mat, vertex4_usize, unsafe {
                                        selected_neighbor.to_usize().unwrap_unchecked()
                                    });

                                // do all the changes improve the graph?
                                let new_gain: F = (total_gain + old_neighbor_dist)
                                    - (good_vertex_dist + new_neighbor_dist);
                                if best_gain < new_gain {
                                    best_gain = new_gain;
                                    best_selected_neighbor = selected_neighbor;
                                    best_old_neighbor_dist = old_neighbor_dist;
                                    best_new_neighbor_dist = new_neighbor_dist;
                                    best_good_vertex = good_vertex;
                                    best_good_vertex_dist = good_vertex_dist;
                                }
                            }
                        }
                    }
                });

                if best_gain > F::zero() {
                    /* replace the two self-loops of vertex4/vertex1 with a connection to the
                    good vertex and its selected neighbor */
                    self.graph.change_edge(
                        vertex4,
                        vertex4,
                        best_good_vertex,
                        best_good_vertex_dist,
                    );
                    changes.push(BuilderChange::new(
                        vertex4,
                        vertex4,
                        F::zero(),
                        best_good_vertex,
                        best_good_vertex_dist,
                    ));
                    self.graph.change_edge(
                        vertex4,
                        vertex4,
                        best_selected_neighbor,
                        best_new_neighbor_dist,
                    );
                    changes.push(BuilderChange::new(
                        vertex4,
                        vertex4,
                        F::zero(),
                        best_selected_neighbor,
                        best_new_neighbor_dist,
                    ));

                    /* replace from good vertex the connection to the selected neighbor
                    with one to vertex4 */
                    self.graph.change_edge(
                        best_good_vertex,
                        best_selected_neighbor,
                        vertex4,
                        best_good_vertex_dist,
                    );
                    changes.push(BuilderChange::new(
                        best_good_vertex,
                        best_selected_neighbor,
                        best_old_neighbor_dist,
                        vertex4,
                        best_good_vertex_dist,
                    ));

                    /* replace from the selected neighbor the connection to the
                    good vertex with one to vertex4 */
                    self.graph.change_edge(
                        best_selected_neighbor,
                        best_good_vertex,
                        vertex4,
                        best_new_neighbor_dist,
                    );
                    changes.push(BuilderChange::new(
                        best_selected_neighbor,
                        best_good_vertex,
                        best_old_neighbor_dist,
                        vertex4,
                        best_new_neighbor_dist,
                    ));

                    return true;
                }
            } else {
                /* 2.1b If there is a way from vertex2 or vertex3, to vertex1 or vertex4 then..
                Try to connect vertex1 with vertex4
                Much more likely than 2.1a */
                if !self.graph.has_edge(vertex1, vertex4) {
                    // Is the total of all changes still beneficial?
                    let dist14 = self._get_dist(mat, vertex1_usize, vertex4_usize);

                    if (total_gain - dist14) > F::zero() {
                        let mut entry_points_v1: Vec<(F, R)> = vec![
                            (self._get_dist(mat, vertex1_usize, vertex2_usize), vertex2),
                            (self._get_dist(mat, vertex1_usize, vertex3_usize), vertex3),
                        ];

                        let mut entry_points_v4: Vec<(F, R)> = vec![
                            (self._get_dist(mat, vertex4_usize, vertex2_usize), vertex2),
                            (self._get_dist(mat, vertex4_usize, vertex3_usize), vertex3),
                        ];

                        let has_either_path: bool = if self.params.use_range_search {
                            self._has_path_range(
                                mat,
                                vertex1,
                                self.params.improve_k,
                                F::from(self.params.improve_eps).unwrap(),
                                &mut search_cache.search_hashset,
                                &mut search_cache.search_maxheap,
                                &mut search_cache.frontier_minheap,
                                &mut search_cache.frontier_dualheap,
                                &mut entry_points_v1,
                            ) || self._has_path_range(
                                mat,
                                vertex4,
                                self.params.improve_k,
                                F::from(self.params.improve_eps).unwrap(),
                                &mut search_cache.search_hashset,
                                &mut search_cache.search_maxheap,
                                &mut search_cache.frontier_minheap,
                                &mut search_cache.frontier_dualheap,
                                &mut entry_points_v4,
                            )
                        } else {
                            self._has_path(
                                mat,
                                vertex1,
                                &mut search_cache.search_hashset,
                                &mut search_cache.search_maxheap,
                                &mut search_cache.frontier_minheap,
                                &mut search_cache.frontier_dualheap,
                                &mut entry_points_v1,
                                Some(self.params.improve_k),
                            ) || self._has_path(
                                mat,
                                vertex4,
                                &mut search_cache.search_hashset,
                                &mut search_cache.search_maxheap,
                                &mut search_cache.frontier_minheap,
                                &mut search_cache.frontier_dualheap,
                                &mut entry_points_v4,
                                Some(self.params.improve_k),
                            )
                        };

                        if has_either_path {
                            /* replace the the self-loops of vertex1 with a connection
                            to the vertex4 */
                            self.graph.change_edge(vertex1, vertex1, vertex4, dist14);
                            changes.push(BuilderChange::new(
                                vertex1,
                                vertex1,
                                F::zero(),
                                vertex4,
                                dist14,
                            ));
                            /* replace the the self-loops of vertex4 with a connection
                            to the vertex1 */
                            self.graph.change_edge(vertex4, vertex4, vertex1, dist14);
                            changes.push(BuilderChange::new(
                                vertex4,
                                vertex4,
                                F::zero(),
                                vertex1,
                                dist14,
                            ));
                            return true;
                        }
                    }
                }
            }

            // 3. Maximum path length
            if steps >= self.params.max_path_length {
                return false;
            }

            // 4. swap vertex1 and vertex4 every second round, to give each a fair chance
            if steps % 2 == 1 {
                let b = vertex1;
                vertex1 = vertex4;
                vertex4 = b;
            }

            // 5. early stop
            if total_gain < F::zero() {
                return false;
            }

            return self.improve_edges(
                mat,
                changes,
                vertex1,
                vertex4,
                vertex2,
                vertex3,
                total_gain,
                steps + 1,
                search_cache,
            );
        }

        // Helper
        #[inline(always)]
        fn e_weight(&self, vertex1: R, vertex2: R) -> F {
            let edge = self
                .graph
                .view_neighbors(vertex1)
                .iter()
                .find(|(_, v)| v == &vertex2);
            if edge.is_some() {
                return unsafe { edge.unwrap_unchecked().0 };
            } else {
                F::neg_infinity()
            }
        }

        /// Is the vertex_index a RNG conform neighbor if it gets connected to target_vertex?  
        /// Does vertex have a neighbor which is connected to the target_vertex and has a lower weight?
        fn check_rng(&self, vertex: R, target_vertex: R, vertex_target_weight: F) -> bool {
            for &(weight, index) in self.graph.view_neighbors(vertex) {
                let neighbor_target_weight = self.e_weight(index, target_vertex);
                if neighbor_target_weight >= F::zero()
                    && vertex_target_weight > weight.max(neighbor_target_weight)
                {
                    return false;
                }
            }
            true
        }

        /// Remove a vertex from the graph.
        fn _reduce_graph<M: MatrixDataSource<F> + Sync>(&mut self) {
            unimplemented!("not the main focus of this work.");
        }
    }

    impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
        DEGStyleBuilder<R, F, Dist> for DEGBuilder<R, F, Dist>
    {
        type Params = DEGParams;
        type Graph = DEGBuildGraph<R, F>;
        #[inline(always)]
        fn _graph(&self) -> &DEGBuildGraph<R, F> {
            &self.graph
        }
        #[inline(always)]
        fn _mut_graph(&mut self) -> &mut DEGBuildGraph<R, F> {
            &mut self.graph
        }
        #[inline(always)]
        fn _max_build_frontier_size(&self) -> Option<usize> {
            self.params.max_build_frontier_size
        }
        #[inline(always)]
        fn _max_build_heap_size(&self) -> usize {
            self.params.max_build_heap_size
        }
        #[inline(always)]
        fn _dist(&self) -> &Dist {
            &self.dist
        }
        #[inline(always)]
        fn _into_graph_dist(self) -> (DEGBuildGraph<R, F>, Dist) {
            (self.graph, self.dist)
        }
        fn new<M: MatrixDataSource<F> + Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
            let n_data = mat.n_rows();
            assert!(n_data < R::max_value().to_usize().unwrap());
            let graph = DEGBuildGraph::new();
            Self {
                _phantom: std::marker::PhantomData,
                n_data,
                params,
                graph,
                dist,
            }
        }
        // Ignoring original options 'stop_building' and 'infinite'
        // Skimmed version because of missing remove and continue training options
        fn train<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
            if self.n_data == 0 {
                return;
            }
            if self.params.max_build_heap_size < self.params.edges_per_vertex {
                self.params.max_build_heap_size = self.params.edges_per_vertex;
            }

            let mut search_thread_thread_cache: DEGThreadCache<R, F> = DEGThreadCache::new(
                self.n_data,
                self.params.max_build_heap_size,
                self.params.edges_per_vertex,
                self.params.max_build_frontier_size,
            );

            self.extend_graph(mat, &mut search_thread_thread_cache);

            // try to improve the graph
            if self.graph.n_vertices() > self.params.edges_per_vertex && self.params.improve_k > 0 {
                let mut swap_try: i32 = 0;
                let mut rng = rand::thread_rng();
                while swap_try < self.params.swap_tries as i32 {
                    if self.improve_edges_build(mat, &mut rng, &mut search_thread_thread_cache) {
                        swap_try -= self.params.additional_swap_tries as i32;
                    }
                    swap_try += 1;
                }
            }
        }
    }
}

// Most original code comments have been removed
pub struct DEGParallelBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
{
    _phantom: std::marker::PhantomData<F>,
    n_data: usize,
    params: DEGParams,
    graph: DEGBuildGraph<R, F>,
    dist: Dist,
    // lock: Mutex<()>,
    n_threads: usize,
}

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    DEGParallelBuilder<R, F, Dist>
{
    fn extend_graph<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
        let edges_per_vertex = self.params.edges_per_vertex;

        // add all vertices already to the graph which are initialized with self-loops
        // which helps us to avoid locking
        self.graph.init_adj(edges_per_vertex, self.n_data);

        let split_index = (edges_per_vertex + 1).min(self.n_data);
        // fully connect the first split_index many vertices
        for i in 0..split_index {
            for j in 0..i + 1 {
                if i != j {
                    let dist = self._get_dist(mat, i, j);
                    let i_r = unsafe { R::from(i).unwrap_unchecked() };
                    let j_r = unsafe { R::from(j).unwrap_unchecked() };
                    self.graph.change_edge(i_r, i_r, j_r, dist);
                    self.graph.change_edge(j_r, j_r, i_r, dist);
                }
            }
        }
        if self.n_data <= split_index {
            return;
        }

        let mut thread_caches: Vec<DEGThreadCache<R, F>> = (0..self.n_threads)
            .map(|_| {
                DEGThreadCache::new(
                    self.n_data,
                    self.params.max_build_heap_size,
                    self.params.edges_per_vertex,
                    self.params.max_build_frontier_size,
                )
            })
            .collect();

        // Add the remaining points to the graph
        let chunk_size = (self.n_data - split_index + self.n_threads - 1) / self.n_threads;
        (split_index..self.n_data)
            .step_by(chunk_size)
            .map(|start| start..(start + chunk_size).min(self.n_data))
            .zip(thread_caches.iter_mut())
            .par_bridge()
            .for_each(|(chunk, thread_cache)| {
                let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
                unsafe {
                    chunk.for_each(|index: usize| {
                        (*unsafe_self_ref).extend_graph_known_lid(
                            mat,
                            R::from(index).unwrap_unchecked(),
                            thread_cache,
                        );
                    });
                }
            });
    }

    fn extend_graph_known_lid<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        internal_index: R,
        thread_cache: &mut DEGThreadCache<R, F>,
    ) {
        let edges_per_vertex = self.params.edges_per_vertex;

        let number_candidates = if self.params.use_range_search {
            self.params.extend_k.max(edges_per_vertex)
        } else {
            self.params.max_build_heap_size
        };
        let mut candidates: Vec<(F, R)> = Vec::with_capacity(number_candidates);
        candidates.push((
            unsafe { self._get_dist(mat, 0, internal_index.to_usize().unwrap_unchecked()) },
            R::zero(),
        ));

        if self.params.use_range_search {
            self._search_range(
                mat,
                unsafe { internal_index.to_usize().unwrap_unchecked() },
                number_candidates,
                F::from(self.params.extend_eps).unwrap(),
                &mut thread_cache.search_hashset,
                &mut thread_cache.search_maxheap,
                &mut thread_cache.frontier_minheap,
                &mut thread_cache.frontier_dualheap,
                &mut candidates,
            );
        } else {
            self._search(
                mat,
                unsafe { internal_index.to_usize().unwrap_unchecked() },
                &mut thread_cache.search_hashset,
                &mut thread_cache.search_maxheap,
                &mut thread_cache.frontier_minheap,
                &mut thread_cache.frontier_dualheap,
                &mut candidates,
                Some(number_candidates),
            );
        }
        #[cfg(debug_assertions)]
        {
            assert!(
                candidates.len() >= edges_per_vertex,
                "Found not enough neighbors during the search."
            );
        }
        let mut check_rng_phase = true;
        let mut new_neighbors: Vec<(F, R)> = Vec::with_capacity(edges_per_vertex);

        while new_neighbors.len() < edges_per_vertex {
            for &(candidate_weight, candidate_index) in candidates.as_slice() {
                if new_neighbors.len() >= edges_per_vertex {
                    break;
                }

                if self.graph.has_edge(candidate_index, internal_index)
                    || (check_rng_phase
                        && self.check_rng(candidate_index, internal_index, candidate_weight)
                            == false)
                {
                    continue;
                }

                let mut new_neighbor_index = R::zero();
                let mut new_neighbor_distance = F::neg_infinity();
                if self.params.lid == LID::High {
                    let mut new_neighbor_weight = F::neg_infinity();
                    for &(neighbor_weight, neighbor_index) in
                        self.graph.view_neighbors(candidate_index)
                    {
                        if candidate_index == neighbor_index
                            || self.graph.has_edge(neighbor_index, internal_index)
                        {
                            continue;
                        }

                        if neighbor_weight > new_neighbor_weight {
                            new_neighbor_weight = neighbor_weight;
                            new_neighbor_index = neighbor_index;
                        }
                    }

                    if new_neighbor_weight == F::neg_infinity() {
                        continue;
                    }
                    new_neighbor_distance = self._get_dist(
                        mat,
                        unsafe { internal_index.to_usize().unwrap_unchecked() },
                        unsafe { new_neighbor_index.to_usize().unwrap_unchecked() },
                    )
                } else {
                    let mut best_distortion = F::infinity();
                    for &(neighbor_weight, neighbor_index) in
                        self.graph.view_neighbors(candidate_index)
                    {
                        if candidate_index == neighbor_index
                            || self.graph.has_edge(neighbor_index, internal_index)
                        {
                            continue;
                        }

                        let neighbor_distance = self._get_dist(
                            mat,
                            unsafe { internal_index.to_usize().unwrap_unchecked() },
                            unsafe { neighbor_index.to_usize().unwrap_unchecked() },
                        );
                        let distortion = (candidate_weight + neighbor_distance) - neighbor_weight;
                        if distortion < best_distortion {
                            best_distortion = distortion;
                            new_neighbor_index = neighbor_index;
                            new_neighbor_distance = neighbor_distance;
                        }
                    }
                }

                if new_neighbor_distance == F::neg_infinity() {
                    continue;
                }
                {
                    // update all edges
                    // let guard = self.lock.lock().unwrap();
                    if self.graph.has_edge(candidate_index, new_neighbor_index)
                        && self.graph.has_edge(new_neighbor_index, candidate_index)
                        && !self.graph.has_edge(internal_index, candidate_index)
                        && !self.graph.has_edge(candidate_index, internal_index)
                        && !self.graph.has_edge(internal_index, new_neighbor_index)
                        && !self.graph.has_edge(new_neighbor_index, internal_index)
                    {
                        self.graph.change_edge(
                            internal_index,
                            internal_index,
                            candidate_index,
                            candidate_weight,
                        );
                        self.graph.change_edge(
                            internal_index,
                            internal_index,
                            new_neighbor_index,
                            new_neighbor_distance,
                        );
                        new_neighbors.push((candidate_weight, candidate_index));
                        new_neighbors.push((new_neighbor_distance, new_neighbor_index));
                        self.graph.change_edge(
                            candidate_index,
                            new_neighbor_index,
                            internal_index,
                            candidate_weight,
                        );
                        self.graph.change_edge(
                            new_neighbor_index,
                            candidate_index,
                            internal_index,
                            new_neighbor_distance,
                        );
                    }
                    // drop(guard);
                }
            }

            check_rng_phase = false;
        }
        #[cfg(debug_assertions)]
        {
            assert!(new_neighbors.len() >= edges_per_vertex);
        }
    }

    // ------ Optional optimization functions:

    /// @return true if a change could be made otherwise false
    fn improve_edges_build<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        rng: &mut ThreadRng,
        thread_cache: &mut DEGThreadCache<R, F>,
    ) -> bool {
        let mut success = false;

        let vertex1 =
            unsafe { R::from((*rng).gen_range(0..(self.graph.n_vertices()))).unwrap_unchecked() };

        let graph = &mut self.graph as *mut DEGBuildGraph<R, F>;
        unsafe {
            for &(weight_v1v2, vertex2) in (*graph).view_neighbors(vertex1) {
                if self.check_rng(vertex2, vertex1, weight_v1v2) == false {
                    success |= self.improve_edge(mat, vertex1, vertex2, weight_v1v2, thread_cache);
                }
            }
        }
        success
    }

    /// Try to improve the existing edge between the two vertices
    /// @return true if a change could be made otherwise false
    fn improve_edge<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        vertex1: R,
        vertex2: R,
        weight_v1v2: F,
        thread_cache: &mut DEGThreadCache<R, F>,
    ) -> bool {
        if self.params.improve_k <= 0 {
            return false;
        }

        // remove the edge between vertex 1 and vertex 2 (add temporary self-loops)
        let mut changes: Vec<BuilderChange<R, F>> =
            Vec::with_capacity(self.params.edges_per_vertex * 2);
        self.graph.change_edge(vertex1, vertex2, vertex1, F::zero());
        changes.push(BuilderChange::new(
            vertex1,
            vertex2,
            weight_v1v2,
            vertex1,
            F::zero(),
        ));
        self.graph.change_edge(vertex2, vertex1, vertex2, F::zero());
        changes.push(BuilderChange::new(
            vertex2,
            vertex1,
            weight_v1v2,
            vertex2,
            F::zero(),
        ));

        if self.improve_edges(
            mat,
            &mut changes,
            vertex1,
            vertex2,
            vertex1,
            vertex1,
            weight_v1v2,
            0,
            thread_cache,
        ) == false
        {
            // undo all changes, in reverse order
            let size = changes.len();
            for i in 0..size {
                let c = changes[(size - 1) - i];
                self.graph.change_edge(
                    c.internal_index,
                    c.to_neighbor_index,
                    c.from_neighbor_index,
                    c.from_neighbor_weight,
                );
            }
            return false;
        }
        true
    }

    /// Vertex1 and vertex2 might be in a separate subgraph than vertex3 and vertex4.
    /// Through a series of edges swaps both subgraphs should be reconnected..
    /// If those changes improve the graph this method returns true otherwise false.
    /// @return true if a good sequences of changes has been found
    fn improve_edges<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        changes: &mut Vec<BuilderChange<R, F>>,
        mut vertex1: R,
        vertex2: R,
        mut vertex3: R,
        mut vertex4: R,
        mut total_gain: F,
        steps: usize,
        thread_cache: &mut DEGThreadCache<R, F>,
    ) -> bool {
        let vertex1_usize = unsafe { vertex1.to_usize().unwrap_unchecked() };
        let vertex2_usize = unsafe { vertex2.to_usize().unwrap_unchecked() };
        let vertex3_usize = unsafe { vertex3.to_usize().unwrap_unchecked() };
        let vertex4_usize = unsafe { vertex4.to_usize().unwrap_unchecked() };

        let number_candidates = self.params.improve_k;
        let mut top_list: Vec<(F, R)> = Vec::with_capacity(number_candidates);
        top_list.push((self._get_dist(mat, vertex2_usize, vertex3_usize), vertex3));
        top_list.push((self._get_dist(mat, vertex2_usize, vertex4_usize), vertex4));

        if self.params.use_range_search {
            self._search_range(
                mat,
                vertex2_usize,
                number_candidates,
                F::from(self.params.improve_eps).unwrap(),
                &mut thread_cache.search_hashset,
                &mut thread_cache.search_maxheap,
                &mut thread_cache.frontier_minheap,
                &mut thread_cache.frontier_dualheap,
                &mut top_list,
            );
        } else {
            self._search(
                mat,
                vertex2_usize,
                &mut thread_cache.search_hashset,
                &mut thread_cache.search_maxheap,
                &mut thread_cache.frontier_minheap,
                &mut thread_cache.frontier_dualheap,
                &mut top_list,
                Some(number_candidates),
            );
        }
        let mut best_gain = total_gain;
        let mut dist23 = F::neg_infinity();
        let mut dist34 = F::neg_infinity();

        // reverse to desc sort
        for result in top_list.as_slice().into_iter().rev() {
            let new_vertex3 = result.1;
            if vertex1 != new_vertex3
                && vertex2 != new_vertex3
                && !self.graph.has_edge(vertex2, new_vertex3)
            {
                for &(new_vertex4_weight, new_vertex4) in self.graph.view_neighbors(new_vertex3) {
                    let gain: F = total_gain - result.0 + new_vertex4_weight;
                    if new_vertex4 != vertex2 && best_gain < gain {
                        best_gain = gain;
                        vertex3 = new_vertex3;
                        vertex4 = new_vertex4;
                        dist23 = result.0;
                        dist34 = new_vertex4_weight;
                    }
                }
            }
        }

        if dist23 == F::neg_infinity() {
            return false;
        }

        total_gain = (total_gain - dist23) + dist34;
        self.graph.change_edge(vertex2, vertex2, vertex3, dist23);
        changes.push(BuilderChange::new(
            vertex2,
            vertex2,
            F::zero(),
            vertex3,
            dist23,
        ));

        self.graph.change_edge(vertex3, vertex4, vertex2, dist23);
        changes.push(BuilderChange::new(
            vertex3, vertex4, dist34, vertex2, dist23,
        ));
        self.graph.change_edge(vertex4, vertex3, vertex4, F::zero());
        changes.push(BuilderChange::new(
            vertex4,
            vertex3,
            dist34,
            vertex4,
            F::zero(),
        ));

        if vertex1 == vertex4 {
            let number_candidates = self.params.improve_k;
            let mut top_list: Vec<(F, R)> = Vec::with_capacity(number_candidates);
            top_list.push((self._get_dist(mat, vertex2_usize, vertex4_usize), vertex2));
            top_list.push((self._get_dist(mat, vertex3_usize, vertex4_usize), vertex3));

            if self.params.use_range_search {
                self._search_range(
                    mat,
                    vertex4_usize,
                    number_candidates,
                    F::from(self.params.extend_eps).unwrap(),
                    &mut thread_cache.search_hashset,
                    &mut thread_cache.search_maxheap,
                    &mut thread_cache.frontier_minheap,
                    &mut thread_cache.frontier_dualheap,
                    &mut top_list,
                );
            } else {
                self._search(
                    mat,
                    vertex4_usize,
                    &mut thread_cache.search_hashset,
                    &mut thread_cache.search_maxheap,
                    &mut thread_cache.frontier_minheap,
                    &mut thread_cache.frontier_dualheap,
                    &mut top_list,
                    Some(number_candidates),
                );
            }
            let mut best_gain = F::zero();
            let mut best_selected_neighbor = R::zero();
            let mut best_old_neighbor_dist = F::zero();
            let mut best_new_neighbor_dist = F::zero();
            let mut best_good_vertex = R::zero();
            let mut best_good_vertex_dist = F::zero();

            // here the top_list needs to stay asc sorted
            for result in top_list.as_slice() {
                let good_vertex = result.1;
                if vertex4 != good_vertex && !self.graph.has_edge(vertex4, good_vertex) {
                    let good_vertex_dist = result.0;

                    for &(old_neighbor_dist, selected_neighbor) in
                        self.graph.view_neighbors(good_vertex)
                    {
                        if vertex4 != selected_neighbor
                            && !self.graph.has_edge(vertex4, selected_neighbor)
                        {
                            let new_neighbor_dist = self._get_dist(mat, vertex4_usize, unsafe {
                                selected_neighbor.to_usize().unwrap_unchecked()
                            });

                            let new_gain: F = (total_gain + old_neighbor_dist)
                                - (good_vertex_dist + new_neighbor_dist);
                            if best_gain < new_gain {
                                best_gain = new_gain;
                                best_selected_neighbor = selected_neighbor;
                                best_old_neighbor_dist = old_neighbor_dist;
                                best_new_neighbor_dist = new_neighbor_dist;
                                best_good_vertex = good_vertex;
                                best_good_vertex_dist = good_vertex_dist;
                            }
                        }
                    }
                }
            }

            if best_gain > F::zero() {
                self.graph
                    .change_edge(vertex4, vertex4, best_good_vertex, best_good_vertex_dist);
                changes.push(BuilderChange::new(
                    vertex4,
                    vertex4,
                    F::zero(),
                    best_good_vertex,
                    best_good_vertex_dist,
                ));
                self.graph.change_edge(
                    vertex4,
                    vertex4,
                    best_selected_neighbor,
                    best_new_neighbor_dist,
                );
                changes.push(BuilderChange::new(
                    vertex4,
                    vertex4,
                    F::zero(),
                    best_selected_neighbor,
                    best_new_neighbor_dist,
                ));
                self.graph.change_edge(
                    best_good_vertex,
                    best_selected_neighbor,
                    vertex4,
                    best_good_vertex_dist,
                );
                changes.push(BuilderChange::new(
                    best_good_vertex,
                    best_selected_neighbor,
                    best_old_neighbor_dist,
                    vertex4,
                    best_good_vertex_dist,
                ));
                self.graph.change_edge(
                    best_selected_neighbor,
                    best_good_vertex,
                    vertex4,
                    best_new_neighbor_dist,
                );
                changes.push(BuilderChange::new(
                    best_selected_neighbor,
                    best_good_vertex,
                    best_old_neighbor_dist,
                    vertex4,
                    best_new_neighbor_dist,
                ));

                return true;
            }
        } else {
            if !self.graph.has_edge(vertex1, vertex4) {
                let dist14 = self._get_dist(mat, vertex1_usize, vertex4_usize);

                if (total_gain - dist14) > F::zero() {
                    let mut entry_points_v1: Vec<(F, R)> = vec![
                        (self._get_dist(mat, vertex1_usize, vertex2_usize), vertex2),
                        (self._get_dist(mat, vertex1_usize, vertex3_usize), vertex3),
                    ];

                    let mut entry_points_v4: Vec<(F, R)> = vec![
                        (self._get_dist(mat, vertex4_usize, vertex2_usize), vertex2),
                        (self._get_dist(mat, vertex4_usize, vertex3_usize), vertex3),
                    ];

                    let has_either_path: bool = if self.params.use_range_search {
                        self._has_path_range(
                            mat,
                            vertex1,
                            self.params.improve_k,
                            F::from(self.params.improve_eps).unwrap(),
                            &mut thread_cache.search_hashset,
                            &mut thread_cache.search_maxheap,
                            &mut thread_cache.frontier_minheap,
                            &mut thread_cache.frontier_dualheap,
                            &mut entry_points_v1,
                        ) || self._has_path_range(
                            mat,
                            vertex4,
                            self.params.improve_k,
                            F::from(self.params.improve_eps).unwrap(),
                            &mut thread_cache.search_hashset,
                            &mut thread_cache.search_maxheap,
                            &mut thread_cache.frontier_minheap,
                            &mut thread_cache.frontier_dualheap,
                            &mut entry_points_v4,
                        )
                    } else {
                        self._has_path(
                            mat,
                            vertex1,
                            &mut thread_cache.search_hashset,
                            &mut thread_cache.search_maxheap,
                            &mut thread_cache.frontier_minheap,
                            &mut thread_cache.frontier_dualheap,
                            &mut entry_points_v1,
                            Some(self.params.improve_k),
                            // F::from(self.params.improve_eps).unwrap(),
                        ) || self._has_path(
                            mat,
                            vertex4,
                            &mut thread_cache.search_hashset,
                            &mut thread_cache.search_maxheap,
                            &mut thread_cache.frontier_minheap,
                            &mut thread_cache.frontier_dualheap,
                            &mut entry_points_v4,
                            Some(self.params.improve_k),
                            // F::from(self.params.improve_eps).unwrap(),
                        )
                    };

                    if has_either_path {
                        self.graph.change_edge(vertex1, vertex1, vertex4, dist14);
                        changes.push(BuilderChange::new(
                            vertex1,
                            vertex1,
                            F::zero(),
                            vertex4,
                            dist14,
                        ));
                        self.graph.change_edge(vertex4, vertex4, vertex1, dist14);
                        changes.push(BuilderChange::new(
                            vertex4,
                            vertex4,
                            F::zero(),
                            vertex1,
                            dist14,
                        ));
                        return true;
                    }
                }
            }
        }

        if steps >= self.params.max_path_length {
            return false;
        }

        if steps % 2 == 1 {
            let b = vertex1;
            vertex1 = vertex4;
            vertex4 = b;
        }

        if total_gain < F::zero() {
            return false;
        }

        return self.improve_edges(
            mat,
            changes,
            vertex1,
            vertex4,
            vertex2,
            vertex3,
            total_gain,
            steps + 1,
            thread_cache,
        );
    }

    // Helper
    #[inline(always)]
    fn e_weight(&self, vertex1: R, vertex2: R) -> F {
        let edge = self
            .graph
            .view_neighbors(vertex1)
            .iter()
            .find(|(_, v)| v == &vertex2);
        if edge.is_some() {
            return unsafe { edge.unwrap_unchecked().0 };
        } else {
            F::neg_infinity()
        }
    }

    /// Is the vertex_index a RNG conform neighbor if it gets connected to target_vertex?  
    /// Does vertex have a neighbor which is connected to the target_vertex and has a lower weight?
    fn check_rng(&self, vertex: R, target_vertex: R, vertex_target_weight: F) -> bool {
        for (weight, index) in self.graph.view_neighbors(vertex) {
            let neighbor_target_weight = self.e_weight(*index, target_vertex);
            if neighbor_target_weight >= F::zero()
                && vertex_target_weight > weight.max(neighbor_target_weight)
            {
                return false;
            }
        }
        true
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    DEGStyleBuilder<R, F, Dist> for DEGParallelBuilder<R, F, Dist>
where
    Self: Sized,
{
    type Params = DEGParams;
    type Graph = DEGBuildGraph<R, F>;
    #[inline(always)]
    fn _graph(&self) -> &DEGBuildGraph<R, F> {
        &self.graph
    }
    #[inline(always)]
    fn _mut_graph(&mut self) -> &mut DEGBuildGraph<R, F> {
        &mut self.graph
    }
    #[inline(always)]
    fn _max_build_frontier_size(&self) -> Option<usize> {
        self.params.max_build_frontier_size
    }
    #[inline(always)]
    fn _max_build_heap_size(&self) -> usize {
        self.params.max_build_heap_size
    }
    #[inline(always)]
    fn _dist(&self) -> &Dist {
        &self.dist
    }
    #[inline(always)]
    fn _into_graph_dist(self) -> (DEGBuildGraph<R, F>, Dist) {
        (self.graph, self.dist)
    }
    fn new<M: MatrixDataSource<F> + Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
        let n_data = mat.n_rows();
        assert!(n_data < R::max_value().to_usize().unwrap());
        let graph = DEGBuildGraph::new();

        Self {
            _phantom: std::marker::PhantomData,
            n_data,
            params,
            graph,
            dist,
            // lock: Mutex::new(()),
            n_threads: current_num_threads(),
        }
    }
    // Not implemented for LID::Unknown as it points need to inserted one by one
    fn train<M: MatrixDataSource<F> + Sync>(&mut self, mat: &M) {
        if self.n_data == 0 {
            return;
        }
        if self.params.lid == LID::Unknown {
            println!("For LID::Unknown use the single threaded version.");
            return;
        }
        if self.params.max_build_heap_size < self.params.edges_per_vertex {
            self.params.max_build_heap_size = self.params.edges_per_vertex;
        }

        self.graph.reserve(self.n_data);

        self.extend_graph(mat);

        // Try to improve_edges
        if self.graph.n_vertices() > self.params.edges_per_vertex && self.params.improve_k > 0 {
            let mut swap_try: i32 = 0;
            let mut search_thread_thread_cache: DEGThreadCache<R, F> = DEGThreadCache::new(
                self.n_data,
                self.params.max_build_heap_size,
                self.params.edges_per_vertex,
                self.params.max_build_frontier_size,
            );
            let mut rng = rand::thread_rng();
            while swap_try < self.params.swap_tries as i32 {
                if self.improve_edges_build(mat, &mut rng, &mut search_thread_thread_cache) {
                    swap_try -= self.params.additional_swap_tries as i32;
                }
                swap_try += 1;
            }
        }
    }
}

// -------------- TESTS --------------

#[cfg(test)]
mod tests {
    use crate::deg::*;
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
    use std::time::Instant;

    type R = usize;
    type F = f32;
    type Dist = SquaredEuclideanDistance<F>;

    #[test]
    fn deg_construction() {
        // let (_nd, _nq, _d, _k, data) = get_test_data(true);
        let (data, _queries, _ground_truth, _nd, _nq, _d, _k) =
            init_with_dataset(&EvalDataSet::SIFTSmall);
        let edges_per_vertex: usize = 30;

        let params = DEGParams::new()
            .with_edges_per_vertex(edges_per_vertex)
            .with_use_range_search(false)
            .with_lid(LID::High)
            .with_max_build_heap_size(edges_per_vertex);
        // .with_improve_k(20)
        // .with_swap_tries(3)
        // .with_additional_swap_tries(3)
        // .with_extend_k(edges_per_vertex)
        // .with_max_build_heap_size(edges_per_vertex);

        let build_time = Instant::now();
        type BuilderType = DEGParallelBuilder<R, F, Dist>;
        // type BuilderType = crate::deg::single_threaded::DEGBuilder<R, F, Dist>;
        let index = BuilderType::build(data.view(), Dist::new(), params.clone()); // LID::Unknown, LID::Low

        println!(
            "Graph construction ({:?}): {:.2?}",
            std::any::type_name::<BuilderType>(),
            build_time.elapsed()
        );
        print_index_build_info!(index);

        // let build_time = Instant::now();
        // // type BuilderType = crate::deg::single_threaded::DEGBuilder<R, F, Dist>;
        // let index = BuilderType::build(data.view(), Dist::new(), params); // LID::Unknown, LID::Low

        // println!(
        //     "Graph construction ({:?}): {:.2?}",
        //     std::any::type_name::<BuilderType>(),
        //     build_time.elapsed()
        // );
        // print_index_build_info!(index);
    }

    #[test]
    fn deg_query() {
        // let (_nd, nq, _d, k, data) = get_test_data(true);
        // let queries = data.slice_axis(Axis(0), Slice::from(0..nq));
        let (data, queries, ground_truth, _nd, nq, _d, k) = init_with_dataset(&EvalDataSet::AUDIO);
        let edges_per_vertex: usize = 30;

        let params = DEGParams::new()
            .with_lid(LID::Unknown)
            .with_use_range_search(true)
            .with_edges_per_vertex(edges_per_vertex);
        // .with_improve_k(30)
        // .with_swap_tries(5)
        // .with_additional_swap_tries(5)
        // .with_extend_k(edges_per_vertex * 5)
        // .with_max_build_heap_size(edges_per_vertex * 5);

        // let build_time = Instant::now();
        // type BuilderType1 = crate::deg::single_threaded::DEGBuilder<R, F, Dist>;
        // let index1 = BuilderType1::build(data.view(), Dist::new(), params);
        // println!(
        //     "Graph construction ({:?}): {:.2?}",
        //     std::any::type_name::<BuilderType1>(),
        //     build_time.elapsed()
        // );
        // print_index_build_info!(index1);

        let build_time = Instant::now();
        type BuilderType2 = DEGParallelBuilder<R, F, Dist>;
        let index_lid_high = BuilderType2::build(
            data.view(),
            Dist::new(),
            // params,
            params.with_lid(LID::High),
        );

        println!(
            "Graph construction ({:?}): {:.2?}",
            std::any::type_name::<BuilderType2>(),
            build_time.elapsed()
        );
        print_index_build_info!(index_lid_high);

        // Brute force queries
        // let bruteforce_time = Instant::now();
        // let (ground_truth, _) = bruteforce_neighbors(&data, &queries, &Dist::new(), k);
        // println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());

        // DEG queries
        // let deg_time = Instant::now();
        // let (deg_ids1, _deg_dists1) = index1.greedy_search_batch(&queries, k, 2 * k);
        // println!("DEG queries 1: {:.2?}", deg_time.elapsed());

        let index_lid_high_capped = GreedyCappedSingleGraphIndex::new(
            data.view(),
            index_lid_high.graph().as_dir_lol_graph(),
            Dist::new(),
            3 * k,
            None,
        );

        let deg_time = Instant::now();
        let (deg_ids2, _deg_dists2) = index_lid_high_capped.greedy_search_batch(&queries, k, 5 * k);
        println!("DEG queries 2: {:.2?}", deg_time.elapsed());

        let deg_time = Instant::now();
        let (deg_ids3, _deg_dists3) = index_lid_high.greedy_search_batch(&queries, k, k);
        println!("DEG queries 3: {:.2?}", deg_time.elapsed());

        // single threaded search
        let graph_name = "DEG-3-QPS";
        let ground_truth_view = ground_truth.view();
        let (_query_time, _qps, _recall) = search_as_qps!(
            index_lid_high,
            queries,
            ground_truth_view,
            k,
            k,
            nq,
            graph_name,
            true
        );

        // calc_recall(bruteforce_ids.view(), &deg_ids1, nq, k, "DEG-1", true);
        calc_recall(ground_truth_view, &deg_ids2, nq, k, "DEG-2", true);
        calc_recall(ground_truth_view, &deg_ids3, nq, k, "DEG-3", true);
    }
}
