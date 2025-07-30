//! Selectors crate    
//!  
//! Contains SubsetSelector for selecting points from a graph (DirLolGraph)
//! - RandomSelector
//! - HubNodesSelector
//!     - Select the nodes with the highest out-degree
//!     - For graphs with a constant out-degree this is basically a random selection
//! - FloodingSelectorParallelChunked
//!     - Best performance
//! - FloodingSelectorParallelChunkedRepeat
//!     - Significantly slower than all others for large subset_size
//!     - Read local description
//! - FloodingSelector
//!     - Only kept for completeness
//! - FloodingSelectorParallelLocking
//!     - Only kept for completeness
//!
//! Notes
//! - The flooding approach
//!     - Works by selecting a point, adding it to the returned subset, and "flooding"
//!       neighbors of that point up to a depth of flooding_range many hops. Flooded points will
//!       not be selected for the subset.
//!     - The resulting subset of the flooding heuristic depends highly on the order
//!       of visited points
//!     - It basically is a "opinionated random selection", as it deselects the neighbors
//!       of the selected points
//!
//! - "sort-ids" is a debug feature that sorts the returned ids ascendingly
//!
//! Open Issues
//!
//! TODOs
//!
use crate::hsgf::HSGFLevelGraph;
use graphidx::{
    graphs::{Graph, ViewableAdjGraph},
    random::{random_unique_uint, RandomPermutationGenerator},
    sets::HashSetLike,
    types::{Float, SyncFloat, SyncUnsignedInteger},
};
use rayon::{
    iter::{IntoParallelIterator, ParallelBridge, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::sync::Mutex;

type HashSetBit<T> = graphidx::sets::BitSet<T>;

/// Return a subset of points (> 0) from the points in the graph
pub trait SubsetSelector<R: SyncUnsignedInteger, F: Float> {
    fn select_subset(
        &self,
        graph: &HSGFLevelGraph<R>,
        subset_size: Option<usize>,
        ensure_point_zero_on_all_levels: bool,
    ) -> Vec<R>;
}

/// Does what it says  
/// Consider specifying `subset_size`, otherwise a random subset with the length of half the
/// number of input points will be returned
pub struct RandomSelector<R: SyncUnsignedInteger, F: Float> {
    _phantom: std::marker::PhantomData<F>,
    _phantom2: std::marker::PhantomData<R>,
}

impl<R: SyncUnsignedInteger, F: Float> RandomSelector<R, F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
            _phantom2: std::marker::PhantomData,
        }
    }
}

impl<R: SyncUnsignedInteger, F: Float> SubsetSelector<R, F> for RandomSelector<R, F> {
    /// If subset_size is None, than n_vertices/2 many points will be returned
    /// If larger than n_vertices
    fn select_subset(
        &self,
        graph: &HSGFLevelGraph<R>,
        subset_size: Option<usize>,
        ensure_point_zero_on_all_levels: bool,
    ) -> Vec<R> {
        let n_vertices = graph.n_vertices();
        let mut size = subset_size.unwrap_or(n_vertices / 2);
        if size >= n_vertices {
            size = n_vertices / 2;
        }
        let mut res: Vec<R> = random_unique_uint(n_vertices, size);
        if ensure_point_zero_on_all_levels {
            if !res.contains(&R::zero()) {
                res[0] = R::zero();
            }
        }
        #[cfg(feature = "sort-ids")]
        res.par_sort_unstable();
        res
    }
}

/// Select nodes with the highest out-degree
/// percentage_low_degree_nodes is an experimental feature to potentially balance the subset
///     - Or maybe choose randomly instead of low out-degree nodes?
pub struct HubNodesSelector<R: SyncUnsignedInteger, F: Float> {
    _phantom: std::marker::PhantomData<F>,
    _phantom2: std::marker::PhantomData<R>,
    percentage_low_degree_nodes: usize,
}

impl<R: SyncUnsignedInteger, F: Float> HubNodesSelector<R, F> {
    #[inline(always)]
    pub fn new(percentage_low_degree_nodes: usize) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
            _phantom2: std::marker::PhantomData,
            percentage_low_degree_nodes,
        }
    }
}

impl<R: SyncUnsignedInteger, F: Float> SubsetSelector<R, F> for HubNodesSelector<R, F> {
    /// If subset_size is None, than n_vertices/2 many points will be returned
    fn select_subset(
        &self,
        graph: &HSGFLevelGraph<R>,
        subset_size: Option<usize>,
        ensure_point_zero_on_all_levels: bool,
    ) -> Vec<R> {
        let n_vertices = graph.n_vertices();
        // calculate return size
        let mut percentage_low_degree_nodes = self.percentage_low_degree_nodes;
        if percentage_low_degree_nodes > 100 {
            println!(
                "Warning: percentage_low_degree_nodes was\
             set to zero as it was larger than {} > 100.",
                percentage_low_degree_nodes
            );
            percentage_low_degree_nodes = 0;
        }
        let mut size = subset_size.unwrap_or(n_vertices / 2);
        if size >= n_vertices {
            size = n_vertices / 2;
        }
        let mut size_low_degree_nodes: usize = 0;
        if percentage_low_degree_nodes > 0 {
            size_low_degree_nodes =
                (size as f32 * (percentage_low_degree_nodes as f32 / 100.0)) as usize;
            size -= size_low_degree_nodes;
        }
        unsafe {
            // zip the out-degree to each point
            let mut nodes_with_neighbors_count: Vec<(R, usize)> = Vec::with_capacity(n_vertices);
            nodes_with_neighbors_count.set_len(n_vertices);
            (0..n_vertices).par_bridge().for_each(|n| {
                let n_r: R = R::from(n).unwrap_unchecked();
                let nodes = std::ptr::addr_of!(nodes_with_neighbors_count) as *mut Vec<(R, usize)>;
                (*nodes)[n] = (n_r, graph.view_neighbors(n_r).len());
            });

            // sort asc on the out-degree
            nodes_with_neighbors_count
                .par_sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_unchecked());

            // take size many points from the start and size_low_degree_nodes from the end
            let mut res: Vec<R> = nodes_with_neighbors_count
                .iter()
                .take(size)
                .map(|(a, _)| *a)
                .collect();

            if size_low_degree_nodes > 0 {
                nodes_with_neighbors_count
                    .iter()
                    .rev()
                    .take(size_low_degree_nodes)
                    .map(|(a, _)| a)
                    .for_each(|a| res.push(*a));
            }

            if ensure_point_zero_on_all_levels {
                if !res.contains(&R::zero()) {
                    res[0] = R::zero();
                }
            }
            #[cfg(feature = "sort-ids")]
            res.par_sort_unstable();
            res
        }
    }
}

/// Creates chunks of the to be iterated points and returns a partial result Vec<R> for each
/// chunk/thread which are then aggregated at the end, thus avoiding locks (as no shared
/// result vector between threads exists)  
/// Uses RandomPermutationGenerator to select the next flooding point
pub struct FloodingSelectorParallelChunked<R: SyncUnsignedInteger, F: SyncFloat> {
    _phantom: std::marker::PhantomData<F>,
    _phantom2: std::marker::PhantomData<R>,
    flooding_range: usize,
}

impl<R: SyncUnsignedInteger, F: SyncFloat> FloodingSelectorParallelChunked<R, F> {
    #[inline(always)]
    pub fn new(flooding_range: usize) -> Self {
        assert!(flooding_range > 0);
        Self {
            _phantom: std::marker::PhantomData,
            _phantom2: std::marker::PhantomData,
            flooding_range,
        }
    }
    #[inline(always)]
    fn flood_neighbors(
        &self,
        graph: &HSGFLevelGraph<R>,
        idx: R,
        flooded: *mut HashSetBit<R>,
        remaining_flooding_range: usize,
    ) {
        // flood neighbors of idx cascadingly till remaining_flooding_range is zero
        unsafe {
            for &neighbor in graph.view_neighbors(idx) {
                (*flooded).insert(neighbor);
                if (remaining_flooding_range - 1) > 0 {
                    self.flood_neighbors(graph, neighbor, flooded, remaining_flooding_range - 1);
                }
            }
        }
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat> SubsetSelector<R, F>
    for FloodingSelectorParallelChunked<R, F>
{
    #[inline(always)]
    fn select_subset(
        &self,
        graph: &HSGFLevelGraph<R>,
        subset_size: Option<usize>,
        ensure_point_zero_on_all_levels: bool,
    ) -> Vec<R> {
        let n_vertices = graph.n_vertices();
        let mut flooded: HashSetBit<R> = HashSetBit::new(n_vertices);

        let perm = RandomPermutationGenerator::new(n_vertices, 4);

        let n_threads = rayon::current_num_threads();
        let chunk_size = (n_vertices - 1 + n_threads - 1) / n_threads;

        if ensure_point_zero_on_all_levels {
            // make sure r_zero does not get selected
            flooded.insert(R::zero());
        }

        let mut flood_points: Vec<R> = (0..n_threads)
            .into_par_iter()
            .map(|i_chunk| {
                let start = (chunk_size * i_chunk).max(1);
                let end = (start + chunk_size).min(n_vertices);
                let mut selected_idxs = Vec::with_capacity(chunk_size);
                (start..end).for_each(|i| {
                    let i_r = unsafe { R::from(perm.apply_rounds(i)).unwrap_unchecked() };
                    if !flooded.contains(&i_r) {
                        // select point if not yet marked as flooded
                        selected_idxs.push(i_r);
                        let unsafe_flooded = std::ptr::addr_of!(flooded) as *mut HashSetBit<R>;
                        // flood its neighbors
                        self.flood_neighbors(&graph, i_r, unsafe_flooded, self.flooding_range);
                    }
                });
                selected_idxs
            })
            .collect::<Vec<Vec<R>>>()
            .into_iter()
            .map(|v| v.into_iter())
            .flatten()
            .collect();

        // ensure that there are no duplicates
        flood_points = foldhash::HashSet::from_iter(flood_points.into_iter())
            .into_iter()
            .collect();

        // ensure_point_zero_on_all_levels and reduce to subset_size if applicable
        // add r_zero manually here after potentially reducing the len/size
        flood_points = if subset_size.is_some() && flood_points.len() > subset_size.unwrap() {
            if ensure_point_zero_on_all_levels {
                flood_points = flood_points
                    .into_iter()
                    .take(subset_size.unwrap() - 1)
                    .collect::<Vec<R>>();
                flood_points.push(R::zero());
                flood_points
            } else {
                flood_points
                    .into_iter()
                    .take(subset_size.unwrap())
                    .collect::<Vec<R>>()
            }
        } else {
            if ensure_point_zero_on_all_levels {
                flood_points.push(R::zero());
            }
            flood_points
        };

        #[cfg(feature = "sort-ids")]
        flood_points.par_sort_unstable();

        flood_points
    }
}

/* In contrast to FloodingSelectorParallelChunked this fills up the return subset till it has
subset_size many points by repeating the flooding process on the flooded-points without the already
selected (flood-)points.
It basically allows to select additional points following the flooding principle.

Therefore, it is meant to be used with a subset_size and designed for
**larger flooding_ranges** >=2.

Consequently, it is slower than FloodingSelectorParallelChunked and version which would stop
once the subset size threshold is passed will probably be faster. But than again, the point
selection is not the performance bottleneck in HSGF
*/
pub struct FloodingSelectorParallelChunkedRepeat<R: SyncUnsignedInteger, F: SyncFloat> {
    _phantom: std::marker::PhantomData<F>,
    _phantom2: std::marker::PhantomData<R>,
    flooding_range: usize,
}

impl<R: SyncUnsignedInteger, F: SyncFloat> FloodingSelectorParallelChunkedRepeat<R, F> {
    #[inline(always)]
    pub fn new(flooding_range: usize) -> Self {
        assert!(flooding_range > 0);
        Self {
            _phantom: std::marker::PhantomData,
            _phantom2: std::marker::PhantomData,
            flooding_range,
        }
    }
    #[inline(always)]
    fn flood_neighbors(
        &self,
        graph: &HSGFLevelGraph<R>,
        idx: R,
        flooded: *mut HashSetBit<R>,
        remaining_flooding_range: usize,
    ) {
        unsafe {
            for &neighbor in graph.view_neighbors(idx) {
                (*flooded).insert(neighbor);
                if (remaining_flooding_range - 1) > 0 {
                    self.flood_neighbors(graph, neighbor, flooded, remaining_flooding_range - 1);
                }
            }
        }
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat> SubsetSelector<R, F>
    for FloodingSelectorParallelChunkedRepeat<R, F>
{
    #[inline(always)]
    fn select_subset(
        &self,
        graph: &HSGFLevelGraph<R>,
        subset_size: Option<usize>,
        ensure_point_zero_on_all_levels: bool,
    ) -> Vec<R> {
        let n_vertices = graph.n_vertices();
        let mut flooded: HashSetBit<R> = HashSetBit::new(n_vertices);

        let perm = RandomPermutationGenerator::new(n_vertices, 4);

        let n_threads = rayon::current_num_threads();
        let chunk_size = (n_vertices - 1 + n_threads - 1) / n_threads;

        if ensure_point_zero_on_all_levels {
            // make sure r_zero does not get selected
            flooded.insert(R::zero());
        }
        let size = subset_size.unwrap_or(1);
        let mut flood_points: Vec<R> = Vec::with_capacity(size);
        flood_points.push(R::zero());

        // select points
        while flood_points.len() < size {
            flooded.clear();
            for p in &flood_points {
                // avoid duplicate selection
                // only the already selected points are now marked as flooded
                // (not any of their neighbors)
                flooded.insert(*p);
            }
            let mut add_flood_points: Vec<R> = (0..n_threads)
                .into_par_iter()
                .map(|i_chunk| {
                    let start = (chunk_size * i_chunk).max(1);
                    let end = (start + chunk_size).min(n_vertices);
                    let mut selected_idxs = Vec::with_capacity(chunk_size);
                    (start..end).for_each(|i| {
                        let i_r = unsafe { R::from(perm.apply_rounds(i)).unwrap_unchecked() };
                        if !flooded.contains(&i_r) {
                            selected_idxs.push(i_r);
                            let unsafe_flooded = std::ptr::addr_of!(flooded) as *mut HashSetBit<R>;
                            self.flood_neighbors(&graph, i_r, unsafe_flooded, self.flooding_range);
                        }
                    });
                    selected_idxs
                })
                .collect::<Vec<Vec<R>>>()
                .into_iter()
                .map(|v| v.into_iter())
                .flatten()
                .collect();
            flood_points.append(&mut add_flood_points);
        }

        flood_points = foldhash::HashSet::from_iter(flood_points.into_iter())
            .into_iter()
            .collect();

        // Reduce to subset_size if applicable
        // ensure_point_zero_on_all_levels is ensured as take() starts at the front of the Vec
        if flood_points.len() > size {
            flood_points = flood_points.into_iter().take(size).collect::<Vec<R>>();
        }

        #[cfg(feature = "sort-ids")]
        flood_points.par_sort_unstable();

        flood_points
    }
}

/// Base impl of the flooding approach for selecting points on a graph  
/// See also other FloodingSelectors for better performance (parallel)  
/// Only here for comparison
pub struct FloodingSelector<R: SyncUnsignedInteger, F: Float> {
    _phantom: std::marker::PhantomData<F>,
    _phantom2: std::marker::PhantomData<R>,
    flooding_range: usize,
}

impl<R: SyncUnsignedInteger, F: Float> FloodingSelector<R, F> {
    #[inline(always)]
    pub fn new(flooding_range: usize) -> Self {
        assert!(flooding_range > 0);
        Self {
            _phantom: std::marker::PhantomData,
            _phantom2: std::marker::PhantomData,
            flooding_range,
        }
    }
    #[inline(always)]
    fn flood_neighbors(
        &self,
        graph: &HSGFLevelGraph<R>,
        idx: R,
        flooded: &mut HashSetBit<R>,
        remaining_flooding_range: usize,
    ) {
        for &neighbor in graph.view_neighbors(idx) {
            // for undirected graphs this will add the flood_point to the flooded_points
            flooded.insert(neighbor);
            if (remaining_flooding_range - 1) > 0 {
                self.flood_neighbors(graph, neighbor, flooded, remaining_flooding_range - 1);
            }
        }
    }
}

impl<R: SyncUnsignedInteger, F: Float> SubsetSelector<R, F> for FloodingSelector<R, F> {
    #[inline(always)]
    fn select_subset(
        &self,
        graph: &HSGFLevelGraph<R>,
        subset_size: Option<usize>,
        ensure_point_zero_on_all_levels: bool,
    ) -> Vec<R> {
        let n_vertices = graph.n_vertices();
        let mut flooded: HashSetBit<R> = HashSetBit::new(n_vertices);
        let mut flood_points: Vec<R> = Vec::new();
        let perm = RandomPermutationGenerator::new(n_vertices, 4);

        if ensure_point_zero_on_all_levels {
            flooded.insert(R::zero());
            flood_points.push(R::zero());
        }

        (0..n_vertices).for_each(|i| {
            let j = unsafe { R::from(perm.apply_rounds(i)).unwrap_unchecked() };
            if !flooded.contains(&j) {
                flood_points.push(j);
                self.flood_neighbors(&graph, j, &mut flooded, self.flooding_range);
            }
        });
        if subset_size.is_some() && flood_points.len() > subset_size.unwrap() {
            flood_points = flood_points
                .into_iter()
                .take(subset_size.unwrap())
                .collect::<Vec<R>>();
        }

        #[cfg(feature = "sort-ids")]
        flood_points.par_sort_unstable();

        flood_points
    }
}

/// Uses par_iter to iterate the points.   
/// Uses a lock to write to the shared return subset `flood_points`  
/// Only here for comparison
pub struct FloodingSelectorParallelLocking<R: SyncUnsignedInteger, F: SyncFloat> {
    _phantom: std::marker::PhantomData<F>,
    _phantom2: std::marker::PhantomData<R>,
    flooding_range: usize,
}

impl<R: SyncUnsignedInteger, F: SyncFloat> FloodingSelectorParallelLocking<R, F> {
    #[inline(always)]
    pub fn new(flooding_range: usize) -> Self {
        assert!(flooding_range > 0);
        Self {
            _phantom: std::marker::PhantomData,
            _phantom2: std::marker::PhantomData,
            flooding_range,
        }
    }
    #[inline(always)]
    fn flood_neighbors(
        &self,
        graph: &HSGFLevelGraph<R>,
        idx: R,
        flooded: *mut HashSetBit<R>,
        remaining_flooding_range: usize,
    ) {
        unsafe {
            for &neighbor in graph.view_neighbors(idx) {
                (*flooded).insert(neighbor);
                if (remaining_flooding_range - 1) > 0 {
                    self.flood_neighbors(graph, neighbor, flooded, remaining_flooding_range - 1);
                }
            }
        }
    }
}

impl<R: SyncUnsignedInteger, F: SyncFloat> SubsetSelector<R, F>
    for FloodingSelectorParallelLocking<R, F>
{
    #[inline(always)]
    fn select_subset(
        &self,
        graph: &HSGFLevelGraph<R>,
        subset_size: Option<usize>,
        ensure_point_zero_on_all_levels: bool,
    ) -> Vec<R> {
        let n_vertices = graph.n_vertices();
        let mut flooded: HashSetBit<R> = HashSetBit::new(n_vertices);
        let mut flood_points: Vec<R> = Vec::new();
        let perm = RandomPermutationGenerator::new(n_vertices, 4);

        if ensure_point_zero_on_all_levels {
            flooded.insert(R::zero());
            flood_points.push(R::zero());
        }

        let lock = Mutex::new(());
        (0..n_vertices).into_par_iter().for_each(|i| unsafe {
            let i = R::from(perm.apply_rounds(i)).unwrap_unchecked();
            if !flooded.contains(&i) {
                let unsafe_flooding_points = std::ptr::addr_of!(flood_points) as *mut Vec<R>;
                let guard = lock.lock().unwrap();
                (*unsafe_flooding_points).push(i);
                drop(guard);
                let unsafe_flooded = std::ptr::addr_of!(flooded) as *mut HashSetBit<R>;
                self.flood_neighbors(&graph, i, unsafe_flooded, self.flooding_range);
            }
        });
        if subset_size.is_some() && flood_points.len() > subset_size.unwrap() {
            flood_points = flood_points
                .into_iter()
                .take(subset_size.unwrap())
                .collect::<Vec<R>>();
        }
        #[cfg(feature = "sort-ids")]
        flood_points.par_sort_unstable();

        flood_points
    }
}

// -------------- TESTS --------------

#[cfg(test)]
mod tests {
    use crate::selectors::*;
    use crate::utils::test_graph::BasicTestGraph;
    use foldhash::HashSet;
    use graphidx::graphs::Graph;
    use graphidx::measures::SquaredEuclideanDistance;
    use graphidx::types::SyncUnsignedInteger;
    use std::time::{Duration, Instant};

    // Basic unit test for the RandomSelector
    #[test]
    fn test_random_and_hub_selector() {
        let (nd, d, out_degree) = (10_000, 30, 10);
        type R = usize;
        type F = f32;
        let graph = BasicTestGraph::new(nd, d, out_degree, SquaredEuclideanDistance::new())
            .as_dir_lol_graph();

        type Selector = RandomSelector<R, F>;
        type Selector2 = HubNodesSelector<R, F>;

        let prev_subset: Vec<R> = (0..nd).collect();
        let subset_selector = Selector::new();
        let subset: Vec<R> = subset_selector.select_subset(&graph, None, true);

        assert!(subset.len() == prev_subset.len() / 2);
        subset.iter().for_each(|e| assert!(prev_subset.contains(e)));

        let size: Option<usize> = Some(239);
        let subset2: Vec<R> = subset_selector.select_subset(&graph, size, true);

        assert!(subset2.len() == size.unwrap());
        assert!(subset2.contains(&0));
        subset2
            .iter()
            .for_each(|e| assert!(prev_subset.contains(e)));

        let subset_selector3 = Selector2::new(10);
        let subset3: Vec<R> = subset_selector3.select_subset(&graph, size, true);
        assert!(subset3.len() == size.unwrap());
        assert!(subset3.contains(&0));
    }

    // Basic unit tests for the different flooding selecting implementations
    #[test]
    fn test_flooding_selectors() {
        let (nd, d, out_degree, max_tested_flooding_range) = (100_000, 30, 20, 3);
        type R = usize;
        type F = f32;
        let graph = BasicTestGraph::new(nd, d, out_degree, SquaredEuclideanDistance::new())
            .as_dir_lol_graph();

        type Selector1 = FloodingSelector<R, F>;
        type Selector2 = FloodingSelectorParallelLocking<R, F>;
        type Selector3 = FloodingSelectorParallelChunked<R, F>;
        type Selector4 = FloodingSelectorParallelChunkedRepeat<R, F>;

        let prev_subset: Vec<R> = (0..nd).collect();
        let _skip_flooding_percentage: usize = 25;
        let _size: Option<usize> = Some(25);
        let ensure_point_zero_on_all_levels = false;
        for _ in 0..2 {
            for fr in 1..max_tested_flooding_range {
                println!("{}", std::any::type_name::<Selector1>());
                let init_time = Instant::now();
                let subset_selector1 = Selector1::new(fr);
                let subset1: Vec<R> = subset_selector1.select_subset(
                    &graph,
                    Some(30_000),
                    ensure_point_zero_on_all_levels,
                );
                print_assert_flooding_info(
                    fr,
                    nd,
                    subset1,
                    &prev_subset,
                    init_time.elapsed(),
                    ensure_point_zero_on_all_levels,
                );
                // ---------------
                println!("{}", std::any::type_name::<Selector2>());
                let init_time = Instant::now();
                // let size: Option<usize> = Some(25);
                let subset_selector2 = Selector2::new(fr);
                let subset2: Vec<R> =
                    subset_selector2.select_subset(&graph, None, ensure_point_zero_on_all_levels);
                print_assert_flooding_info(
                    fr,
                    nd,
                    subset2,
                    &prev_subset,
                    init_time.elapsed(),
                    ensure_point_zero_on_all_levels,
                );
                // ---------------
                println!("{}", std::any::type_name::<Selector3>());
                let init_time = Instant::now();
                let subset_selector3 = Selector3::new(fr);
                let subset3: Vec<R> =
                    subset_selector3.select_subset(&graph, None, ensure_point_zero_on_all_levels);
                print_assert_flooding_info(
                    fr,
                    nd,
                    subset3,
                    &prev_subset,
                    init_time.elapsed(),
                    ensure_point_zero_on_all_levels,
                );
                // ---------------
                println!("{}", std::any::type_name::<Selector4>());
                let init_time = Instant::now();
                let target_subset_size: usize = nd / 4;
                let subset_selector4 = Selector4::new(fr);
                let subset4: Vec<R> = subset_selector4.select_subset(
                    &graph,
                    Some(target_subset_size),
                    ensure_point_zero_on_all_levels,
                );
                assert!(subset4.len() == target_subset_size);
                print_assert_flooding_info(
                    fr,
                    nd,
                    subset4,
                    &prev_subset,
                    init_time.elapsed(),
                    ensure_point_zero_on_all_levels,
                );

                println!("###############");
            }
        }
    }

    // Helper
    fn print_assert_flooding_info<R: SyncUnsignedInteger>(
        fr: usize,
        nd: usize,
        subset: Vec<R>,
        prev_subset: &Vec<R>,
        time_passed: Duration,
        ensure_point_zero_on_all_levels: bool,
    ) {
        assert!(
            subset.len() == HashSet::from_iter(subset.iter()).len(),
            "Duplicate points selected. {} != {}",
            subset.len(),
            HashSet::from_iter(subset.iter()).len()
        );
        if ensure_point_zero_on_all_levels {
            assert!(subset.contains(&R::zero()));
        }
        subset.iter().for_each(|e| {
            assert!(
                prev_subset.contains(e),
                "Point did not exist in prev_subset."
            )
        });

        println!("Flooding range: {}", fr);
        println!("Total points: {}", nd);
        println!("Len of subset: {}", subset.len(),);
        println!("Time: {:.2?}", time_passed);
        println!("---");
    }
}
