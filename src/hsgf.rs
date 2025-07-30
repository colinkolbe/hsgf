//! HSGF crate   
//!   
//! Contains HSGF graph builders implementing the HSGFStyleBuilder trait
//!     - HSGFEnumBuilder
//!         - Uses a predefined enum for the level-graph-builders and point-selectors arguments
//!         - Used in all experiments for this work
//!     - HSGFClosureBuilder
//!         - Uses closures for the level-graph-builders and point-selectors arguments
//!         - This in principle should allow more flexibility to the user but for our work this
//!           builder goes mostly unused beyond testing as the EnumBuilder is simpler to
//!           handle and port to Python
//!         - See also second bullet point under Notes
//!
//! Notes  
//! - Info about dyn trait objects as input:
//!   See https://antoinerr.github.io/blog-website/2022/07/23/rust-polymorphism.html#trait-objects
//!   or https://doc.rust-lang.org/reference/items/traits.html#dyn-compatibility
//!  
//! - Info about data input type: <M: MatrixDataSource<F> + Sync> cannot be guaranteed beyond the
//!   first level as a type because .get_rows()->Array<F> is/needs to be used, so that for any level
//!   besides the first (bottom) layer M cannot be guaranteed as a type but only Array<F>. This
//!   would be conflicting with the higher_level_builders input argument which (specifically in the
//!   HSGFClosureBuilder case) needs to have a trait bound on the receiving function but once that
//!   functions accepts that argument the type is fixed on M which would not be equal to Array<F>.
//!    
//! - Info about layered graphs and specifically `layer_ids` with this API:  
//!     - layer_ids.len() = graphs.len() - 1  
//!     - Mapping local_ids[layer_id] to global_ids[layer_id-1] (bottom layer: layer_id = 0)
//!     - For more see: see graphidx::indices::GeneralIndex  
//!
//! - "measure-print-hsgf-level-times" is a flag to measure and print the time needed for each
//!     level's subset selection and graph construction (is a default flag)
//!
//! Open Issues
//!     - The ExistingGraph inputs are very memory intense as they clone the graph
//!     - Potentially even larger memory usage for large datasets, given the duplicate (subset-)data
//!        during construction for higher levels
//!         - A possible solution/mitigation could be to load in the whole dataset and then drop
//!         that data once the (mat_)subset has been selected. After all levels have been build,
//!         only a reference to the graphs is returned and the overall data needs to be loaded and
//!         passed in again to finally create the actual GraphIndex (?)
//!
//! TODOs
//!     - Feature: Level build times - Write to file (.csv) directly instead of printing out (?)
//!     - Feature: is_closure_builder(), is_enum_builder() - currently not needed though (?)
//!
use crate::selectors::SubsetSelector;
use crate::{
    deg::{DEGParallelBuilder, DEGParams, DEGStyleBuilder},
    efanna::{EfannaParallelMaxHeapBuilder, EfannaParams, EfannaStyleBuilder},
    nssg::{
        NSSGInputGraphData, NSSGInputGraphDataForHSGF, NSSGParallelBuilder, NSSGParams,
        NSSGStyleBuilder,
    },
    rngg::{RNGGBuilder, RNGGParams, RNGGStyleBuilder},
};
use graphidx::{
    data::MatrixDataSource,
    graphs::{DirLoLGraph, Graph, ViewableAdjGraph},
    indices::{GreedyCappedLayeredGraphIndex, GreedyLayeredGraphIndex},
    measures::Distance,
    param_struct,
    random::random_unique_uint,
    types::{SyncFloat, SyncUnsignedInteger},
};
use graphidxbaselines::{
    knn::{BruteforceKNNGraphBuilder, BruteforceKNNParams},
    rnn::{RNNDescentBuilder, RNNParams, RNNStyleBuilder},
};
use ndarray::Array2;
#[allow(unused)]
use std::time::Instant;

pub type HSGFLevelGraph<R> = DirLoLGraph<R>;

// -------------- UTIL --------------

#[inline(always)]
fn make_greedy_index<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    M: MatrixDataSource<F>,
    Dist: Distance<F>,
    G: Graph<R>,
>(
    graphs: Vec<G>,
    local_layer_ids: Vec<Vec<R>>,
    global_layer_ids: Vec<Vec<R>>,
    mat: M,
    dist: Dist,
    higher_level_max_heap_size: usize,
    top_entry_points: Option<Vec<R>>,
) -> GreedyLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
    GreedyLayeredGraphIndex::new(
        mat,
        graphs.iter().map(|g| g.as_dir_lol_graph()).collect(),
        local_layer_ids,
        global_layer_ids,
        dist,
        higher_level_max_heap_size,
        top_entry_points,
    )
}

#[inline(always)]
fn make_greedy_capped_index<
    R: SyncUnsignedInteger,
    F: SyncFloat,
    M: MatrixDataSource<F>,
    Dist: Distance<F>,
    G: Graph<R>,
>(
    graphs: Vec<G>,
    local_layer_ids: Vec<Vec<R>>,
    global_layer_ids: Vec<Vec<R>>,
    mat: M,
    dist: Dist,
    higher_level_max_heap_size: usize,
    max_frontier_size: usize,
    top_entry_points: Option<Vec<R>>,
) -> GreedyCappedLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
    GreedyCappedLayeredGraphIndex::new(
        mat,
        graphs.iter().map(|g| g.as_dir_lol_graph()).collect(),
        local_layer_ids,
        global_layer_ids,
        dist,
        higher_level_max_heap_size,
        max_frontier_size,
        top_entry_points,
    )
}

// --------------

pub trait HSGFStyleBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
where
    Self: Sized,
{
    type Params;
    type Graph: ViewableAdjGraph<R>;
    type LevelGraphBuilders;
    type BottomLevelGraphBuilder<M>; // only needed for the ClosureBuilder
    fn _graphs(&self) -> &Vec<Self::Graph>;
    fn _mut_graphs(&mut self) -> &mut Vec<Self::Graph>;
    fn _global_layer_ids(&self) -> &Vec<Vec<R>>;
    fn _dist(&self) -> &Dist;
    fn _into_parts(self) -> (Vec<Self::Graph>, Vec<Vec<R>>, Vec<Vec<R>>, Dist);
    #[inline(always)]
    fn _get_dist<M: MatrixDataSource<F>>(&self, mat: &M, i: usize, j: usize) -> F {
        if M::SUPPORTS_ROW_VIEW {
            self._dist()
                .dist_slice(&mat.get_row_view(i), &mat.get_row_view(j))
        } else {
            self._dist().dist(&mat.get_row(i), &mat.get_row(j))
        }
    }
    /* Selects not directly the points themselves but the indices for the points of the
    vector on the top level in the graph */
    #[inline(always)]
    fn _get_top_entry_points(
        local_layer_ids: &Vec<Vec<R>>,
        higher_level_max_heap_size: usize,
    ) -> Option<Vec<R>> {
        if local_layer_ids.len() > 0 {
            Some(
                (0..local_layer_ids
                    .last()
                    .unwrap()
                    .len()
                    .min(higher_level_max_heap_size))
                    .map(|e| R::from(e).unwrap())
                    .collect(),
            )
        } else {
            Some(vec![R::zero()])
        }
    }
    fn _is_enum_builder() -> bool;
    fn new<M: MatrixDataSource<F> + Sync>(
        mat: &M,
        dist: Dist,
        params: Self::Params,
        level_subset_sizes: Option<Vec<Option<usize>>>,
    ) -> Self;
    fn train<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        level_builders: Vec<Self::LevelGraphBuilders>,
    );
    #[inline(always)]
    fn into_greedy<M: MatrixDataSource<F> + Sync>(
        self,
        mat: M,
        top_entry_points: Option<Vec<R>>,
        higher_level_max_heap_size: usize,
    ) -> GreedyLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let (graphs, local_layer_ids, global_layer_ids, dist) = self._into_parts();
        let top_entry_points = if top_entry_points.is_some() {
            top_entry_points
        } else {
            Self::_get_top_entry_points(&local_layer_ids, higher_level_max_heap_size)
        };
        make_greedy_index(
            graphs,
            local_layer_ids,
            global_layer_ids,
            mat,
            dist,
            higher_level_max_heap_size,
            top_entry_points,
        )
    }
    #[inline(always)]
    fn into_greedy_capped<M: MatrixDataSource<F> + Sync>(
        self,
        mat: M,
        top_entry_points: Option<Vec<R>>,
        higher_level_max_heap_size: usize,
        max_frontier_size: usize,
    ) -> GreedyCappedLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let (graphs, local_layer_ids, global_layer_ids, dist) = self._into_parts();
        let top_entry_points = if top_entry_points.is_some() {
            top_entry_points
        } else {
            Self::_get_top_entry_points(&local_layer_ids, higher_level_max_heap_size)
        };
        make_greedy_capped_index(
            graphs,
            local_layer_ids,
            global_layer_ids,
            mat,
            dist,
            higher_level_max_heap_size,
            max_frontier_size,
            top_entry_points,
        )
    }
    fn build<M: MatrixDataSource<F> + Sync>(
        mat: M,
        dist: Dist,
        params: Self::Params,
        level_builders: Vec<Self::LevelGraphBuilders>,
        level_subset_sizes: Option<Vec<Option<usize>>>,
        top_entry_points: Option<Vec<R>>,
        higher_level_max_heap_size: usize,
    ) -> GreedyLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let lb_len = level_builders.len();
        let n_data = mat.n_rows();
        let mut builder = Self::new(&mat, dist, params, level_subset_sizes);
        if lb_len > 0 {
            builder.train(&mat, level_builders);
        } else {
            println!(
                "Nothing was build. \
                Supply level_builders {} or check that the input data is not empty {}.",
                lb_len, n_data
            );
        }
        builder.into_greedy(mat, top_entry_points, higher_level_max_heap_size)
    }
    #[inline(always)]
    fn build_capped<M: MatrixDataSource<F> + Sync>(
        mat: M,
        dist: Dist,
        params: Self::Params,
        level_builders: Vec<Self::LevelGraphBuilders>,
        level_subset_sizes: Option<Vec<Option<usize>>>,
        top_entry_points: Option<Vec<R>>,
        higher_level_max_heap_size: usize,
        max_frontier_size: usize,
    ) -> GreedyCappedLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let lb_len = level_builders.len();
        let n_data = mat.n_rows();
        let mut builder = Self::new(&mat, dist, params, level_subset_sizes);
        if lb_len > 0 {
            builder.train(&mat, level_builders);
        } else {
            println!(
                "Nothing was build. \
                Supply level_builders {} or check that the input data is not empty {}.",
                lb_len, n_data
            );
        }
        builder.into_greedy_capped(
            mat,
            top_entry_points,
            higher_level_max_heap_size,
            max_frontier_size,
        )
    }
    // -------- Closure builder only --------
    fn train_closure<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        bottom_level_builder: Self::BottomLevelGraphBuilder<M>,
        higher_level_builders: Vec<Self::LevelGraphBuilders>,
    );
    fn build_closure<M: MatrixDataSource<F> + Sync>(
        mat: M,
        dist: Dist,
        params: Self::Params,
        bottom_level_builder: Self::BottomLevelGraphBuilder<M>,
        higher_level_builders: Vec<Self::LevelGraphBuilders>,
        level_subset_sizes: Option<Vec<Option<usize>>>,
        top_entry_points: Option<Vec<R>>,
        higher_level_max_heap_size: usize,
    ) -> GreedyLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let hlb_len = higher_level_builders.len();
        let n_data = mat.n_rows();
        let mut builder = Self::new(&mat, dist, params, level_subset_sizes);
        if hlb_len > 0 {
            builder.train_closure(&mat, bottom_level_builder, higher_level_builders);
        } else {
            println!(
                "Supply higher_level_builders {} or check that the input data is not empty {}.",
                hlb_len, n_data
            );
        }
        builder.into_greedy(mat, top_entry_points, higher_level_max_heap_size)
    }
    #[inline(always)]
    fn build_closure_capped<M: MatrixDataSource<F> + Sync>(
        mat: M,
        dist: Dist,
        params: Self::Params,
        bottom_level_builder: Self::BottomLevelGraphBuilder<M>,
        higher_level_builders: Vec<Self::LevelGraphBuilders>,
        level_subset_sizes: Option<Vec<Option<usize>>>,
        top_entry_points: Option<Vec<R>>,
        higher_level_max_heap_size: usize,
        max_frontier_size: usize,
    ) -> GreedyCappedLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
        let hlb_len = higher_level_builders.len();
        let n_data = mat.n_rows();
        let mut builder = Self::new(&mat, dist, params, level_subset_sizes);
        if hlb_len > 0 {
            builder.train_closure(&mat, bottom_level_builder, higher_level_builders);
        } else {
            println!(
                "Supply higher_level_builders {} or check that the input data is not empty {}.",
                hlb_len, n_data
            );
        }

        builder.into_greedy_capped(
            mat,
            top_entry_points,
            higher_level_max_heap_size,
            max_frontier_size,
        )
    }
}

// --------------

/// Defining the available level-graph-builders.
/// Currently supporting:
///     - DEG, EFANNA, ExistingGraph (input is a pre-build graph), NSSG, RNGG, RNN
///
/// Note that ExistingGraph (currently) only works for the bottom level and if the exact input data
/// (as well as the same order of points, because of the subset selection) was used, otherwise
/// the HSGF graph will be broken or panic during construction. It is meant mainly for fair testing
/// by using the same bottom level graph for multiple (otherwise different) HSGF graphs.
///
/// Additionally, it allows choosing different SubsetSelector on each level.   
pub enum HSGFLevelGraphBuilder<R: SyncUnsignedInteger, F: SyncFloat> {
    // Example: GRAPH((GraphParams, SubsetSelector))
    DEG((DEGParams, Box<dyn SubsetSelector<R, F>>)),
    EFANNA((EfannaParams, Box<dyn SubsetSelector<R, F>>)),
    ExistingGraph((DirLoLGraph<R>, Box<dyn SubsetSelector<R, F>>)),
    KNN((BruteforceKNNParams, Box<dyn SubsetSelector<R, F>>)),
    NSSG(
        (
            NSSGParams<F>,
            Box<dyn SubsetSelector<R, F>>,
            NSSGInputGraphDataForHSGF,
        ),
    ),
    RNGG((RNGGParams, Box<dyn SubsetSelector<R, F>>)),
    RNN((RNNParams, Box<dyn SubsetSelector<R, F>>)),
}

impl<R: SyncUnsignedInteger, F: SyncFloat> HSGFLevelGraphBuilder<R, F> {
    pub fn new_deg(params: DEGParams, selector: Box<dyn SubsetSelector<R, F>>) -> Self {
        HSGFLevelGraphBuilder::DEG((params, selector))
    }
    pub fn new_efanna(params: EfannaParams, selector: Box<dyn SubsetSelector<R, F>>) -> Self {
        HSGFLevelGraphBuilder::EFANNA((params, selector))
    }
    pub fn new_knn(params: BruteforceKNNParams, selector: Box<dyn SubsetSelector<R, F>>) -> Self {
        HSGFLevelGraphBuilder::KNN((params, selector))
    }
    pub fn new_nssg(
        params: NSSGParams<F>,
        selector: Box<dyn SubsetSelector<R, F>>,
        input_graph: NSSGInputGraphDataForHSGF,
    ) -> Self {
        HSGFLevelGraphBuilder::NSSG((params, selector, input_graph))
    }
    pub fn new_rngg(params: RNGGParams, selector: Box<dyn SubsetSelector<R, F>>) -> Self {
        HSGFLevelGraphBuilder::RNGG((params, selector))
    }
    pub fn new_rnn(params: RNNParams, selector: Box<dyn SubsetSelector<R, F>>) -> Self {
        HSGFLevelGraphBuilder::RNN((params, selector))
    }
    /** This is a special variant as it allows to insert a already existing graph to a level.
    However, because of the local_ and global_id mappings this only really works (at least
    currently) on the bottom level. */
    pub fn new_existing_input_graph(
        input_graph: DirLoLGraph<R>,
        selector: Box<dyn SubsetSelector<R, F>>,
    ) -> Self {
        HSGFLevelGraphBuilder::ExistingGraph((input_graph, selector))
    }
    /// Constructs the graph corresponding to the input params with `mat_subset` and returns
    /// the graph as well as the specified subset selector
    #[inline(always)]
    pub fn get_level_graph_and_selector<
        'a,
        Dist: Distance<F> + Sync + Send,
        M: MatrixDataSource<F> + Sync,
    >(
        builder: &'a mut HSGFLevelGraphBuilder<R, F>,
        mat_subset: &M,
        dist: Dist,
    ) -> (DirLoLGraph<R>, &'a mut Box<(dyn SubsetSelector<R, F>)>) {
        let (level_graph, selector) = match builder {
            HSGFLevelGraphBuilder::DEG((params, selector)) => {
                let level_graph = DEGParallelBuilder::build(mat_subset, dist, params.clone())
                    .graph()
                    .as_dir_lol_graph();
                (level_graph, selector)
            }
            HSGFLevelGraphBuilder::EFANNA((params, selector)) => {
                let level_graph =
                    EfannaParallelMaxHeapBuilder::build(mat_subset, dist, params.clone())
                        .graph()
                        .as_dir_lol_graph();
                (level_graph, selector)
            }
            HSGFLevelGraphBuilder::ExistingGraph((input_level_graph, selector)) => {
                /* TODO inefficient: creates a clone of the graph without freeing the old memory
                which likely won't be released until the end of the build process due to being
                contained in the level_builders Vec<HSGFLevelGraphBuilder>
                The neighbors can be cleared but that probably leaves the space of the neighbor
                vectors occupied thus not freeing up space, and direct access to the adjacency is
                not possible/allowed */
                (input_level_graph.as_dir_lol_graph(), selector)
            }
            HSGFLevelGraphBuilder::KNN((params, selector)) => {
                if params.degree >= mat_subset.n_rows() {
                    // degree cannot be more than available points -1
                    *params = params.with_degree(mat_subset.n_rows() - 1);
                }
                // needs to transform mat_subset compared to the other builders
                let all_rows: Vec<usize> = (0..mat_subset.n_rows()).collect();
                let level_graph = BruteforceKNNGraphBuilder::build(
                    mat_subset.get_rows(&all_rows),
                    dist,
                    params.clone(),
                )
                .graph()
                .as_dir_lol_graph();
                (level_graph, selector)
            }
            HSGFLevelGraphBuilder::NSSG((params, selector, input_graph_params)) => {
                let input_graph: NSSGInputGraphData<R, F> = match input_graph_params {
                    NSSGInputGraphDataForHSGF::DEG(params) => {
                        NSSGInputGraphData::DEG(params.clone())
                    }
                    NSSGInputGraphDataForHSGF::EFANNA(params) => {
                        NSSGInputGraphData::EFANNA(params.clone())
                    }
                    NSSGInputGraphDataForHSGF::GRAPH(graph) => {
                        // This should be slow; Could maybe be parallelized as the
                        // data-write would always be to a unique index?
                        // Same issue as in py::py_hsgf
                        // for PyHSGFLevelGraphParams::ExistingGraph
                        let mut input_level_graph: DirLoLGraph<R> = DirLoLGraph::new();
                        let n_vertices = graph.adjacency.len();
                        for v in 0..n_vertices {
                            input_level_graph.add_node_with_capacity(graph.adjacency[v].len());
                        }
                        for v in 0..n_vertices {
                            for n in graph.adjacency[v].iter() {
                                unsafe {
                                    input_level_graph.add_edge(
                                        R::from(v).unwrap_unchecked(),
                                        R::from(*n).unwrap_unchecked(),
                                    );
                                }
                            }
                        }
                        // try to help by freeing up memory after copying all the data
                        graph.adjacency = Vec::new();
                        graph.n_edges = 0;
                        NSSGInputGraphData::GRAPH(input_level_graph)
                    }
                    NSSGInputGraphDataForHSGF::RANDOM(degree) => {
                        NSSGInputGraphData::RANDOM(*degree)
                    }
                    NSSGInputGraphDataForHSGF::RNN(params) => {
                        NSSGInputGraphData::RNN(params.clone())
                    }
                };
                let level_graph =
                    NSSGParallelBuilder::build(mat_subset, dist, params.clone(), input_graph)
                        .graph()
                        .as_dir_lol_graph();
                (level_graph, selector)
            }
            HSGFLevelGraphBuilder::RNGG((params, selector)) => {
                let level_graph = RNGGBuilder::build(mat_subset, dist, params.clone())
                    .graph()
                    .as_dir_lol_graph();
                (level_graph, selector)
            }
            HSGFLevelGraphBuilder::RNN((params, selector)) => {
                let level_graph = RNNDescentBuilder::build(mat_subset, dist, params.clone())
                    .graph()
                    .as_dir_lol_graph();
                (level_graph, selector)
            }
        };
        (level_graph, selector)
    }
}

param_struct!(HSGFParams[Copy, Clone] {
    min_layers: usize = 1,
    max_layers: usize = 6,
    min_n_vertices_layer: usize = 50, // threshold for the last level
    higher_max_degree: usize = 40, // used for calculating the level_subset_sizes if that is not set
    level_norm_param_override: Option<f32> = None,
    create_artificial_hub_nodes: Option<usize> = None, // experimental feature (not promising..)
});

/// Uses the enum HSGFLevelGraphBuilders to construct a HSGF graph
/// self::train uses the last element in level_builders if level_builders.len() is
/// less than n_layers many level_builders are supplied  
///  
/// level_builders and level_subset_sizes cannot (easily) be moved under (Py)HSGFParams because
/// the Copy trait cannot be implemented for Vec
pub struct HSGFEnumBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send> {
    _phantom: std::marker::PhantomData<F>,
    n_data: usize,
    params: HSGFParams,
    dist: Dist,
    graphs: Vec<HSGFLevelGraph<R>>,
    n_layers: usize,
    // in case of standard flooding level_subset_sizes serves only as a upper but not lower bound
    level_subset_sizes: Vec<Option<usize>>,
    local_layer_ids: Vec<Vec<R>>,
    global_layer_ids: Vec<Vec<R>>,
}

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    HSGFStyleBuilder<R, F, Dist> for HSGFEnumBuilder<R, F, Dist>
{
    type Params = HSGFParams;
    type Graph = HSGFLevelGraph<R>;
    type LevelGraphBuilders = HSGFLevelGraphBuilder<R, F>;
    // only needed in the case of being a ClosureBuilder
    type BottomLevelGraphBuilder<M> = HSGFLevelGraphBuilder<R, F>;
    #[inline(always)]
    fn _graphs(&self) -> &Vec<Self::Graph> {
        &self.graphs
    }
    #[inline(always)]
    fn _mut_graphs(&mut self) -> &mut Vec<Self::Graph> {
        &mut self.graphs
    }
    #[inline(always)]
    fn _global_layer_ids(&self) -> &Vec<Vec<R>> {
        &self.global_layer_ids
    }
    #[inline(always)]
    fn _dist(&self) -> &Dist {
        &self.dist
    }
    #[inline(always)]
    fn _into_parts(self) -> (Vec<Self::Graph>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
        (
            self.graphs,
            self.local_layer_ids,
            self.global_layer_ids,
            self.dist,
        )
    }
    fn new<M: MatrixDataSource<F> + Sync>(
        mat: &M,
        dist: Dist,
        params: Self::Params,
        // if set to None to automatically calculate level_subset_sizes
        level_subset_sizes: Option<Vec<Option<usize>>>,
    ) -> Self {
        let n_data = mat.n_rows();
        assert!(n_data < R::max_value().to_usize().unwrap());

        let level_norm_param = params
            .level_norm_param_override
            .unwrap_or(1.0 / (params.higher_max_degree as f32).ln());
        let mut n_layers = (((n_data as f32).ln() * level_norm_param).floor() as usize + 1)
            .min(params.max_layers)
            .max(params.min_layers);

        let level_subset_sizes = if level_subset_sizes.is_some() {
            level_subset_sizes.unwrap()
        } else {
            // Calculate level sizes if not set by parameter
            let mut level_subset_sizes = Vec::with_capacity(n_layers);
            for i in 1..n_layers {
                let mut expected_size = n_data as f32;
                (0..i).for_each(|_| expected_size /= (1.0 / level_norm_param).exp());
                if i > 0 {
                    expected_size *= 1.2;
                }
                let expected_size =
                    (expected_size.floor() as usize).max(params.min_n_vertices_layer);
                level_subset_sizes.push(Some(expected_size));
                if expected_size == params.min_n_vertices_layer {
                    if i < (n_layers - 1) {
                        n_layers -= 1;
                        if n_layers < params.min_layers {
                            println!(
                                "N_layer count was lowered by 1 \
                            because of level subset sizes heuristic."
                            )
                        }
                    }
                    break;
                }
            }
            level_subset_sizes
        };

        Self {
            _phantom: std::marker::PhantomData,
            n_data,
            params,
            dist,
            graphs: Vec::with_capacity(n_layers),
            n_layers,
            local_layer_ids: (0..n_layers - 1).map(|_| Vec::new()).collect(),
            global_layer_ids: (0..n_layers - 1).map(|_| Vec::new()).collect(),
            level_subset_sizes,
        }
    }

    fn train<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        mut level_builders: Vec<Self::LevelGraphBuilders>,
    ) {
        // Data size sanity check, otherwise create one fully connected graph for the bottom level
        if self.n_data <= self.params.min_n_vertices_layer {
            let mut graph = DirLoLGraph::<R>::new();
            if self.n_data > 0 {
                if self.n_data == 1 {
                    graph.add_node();
                } else {
                    for i in 0..self.n_data {
                        graph.add_node();
                        for j in 0..i {
                            let i_r = unsafe { R::from(i).unwrap_unchecked() };
                            let j_r = unsafe { R::from(j).unwrap_unchecked() };
                            graph.add_edge(i_r, j_r);
                        }
                    }
                }
            }
            self.n_layers = 1;
            self.local_layer_ids = Vec::new();
            self.global_layer_ids = Vec::new();
            self.graphs.push(graph);
            return;
        }

        // --- Build bottom level
        #[cfg(feature = "measure-print-hsgf-level-times")]
        let time = Instant::now();

        let (mut level_graph, selector) = HSGFLevelGraphBuilder::get_level_graph_and_selector(
            &mut level_builders[0],
            &mat,
            self.dist.clone(),
        );

        #[cfg(feature = "measure-print-hsgf-level-times")]
        println!("Graph 0: {:.2?}", time.elapsed());
        #[cfg(feature = "measure-print-hsgf-level-times")]
        let time = Instant::now();

        // Select the points for the next level
        let mut subset = selector.select_subset(&level_graph, self.level_subset_sizes[0], true);
        #[cfg(feature = "measure-print-hsgf-level-times")]
        println!("Subset 0: {:.2?}", time.elapsed());
        let subset_usize = subset
            .iter()
            .map(|e| unsafe { e.to_usize().unwrap_unchecked() })
            .collect::<Vec<usize>>();
        let mut mat_subset: Array2<F> = mat.get_rows(&subset_usize);
        // At first prev_subset contains all points
        let mut prev_subset: Vec<R> = (0..self.n_data)
            .map(|e| unsafe { R::from(e).unwrap_unchecked() })
            .collect();

        // Optional hub-nodes creation in current level_graph
        if self.params.create_artificial_hub_nodes.is_some() {
            _create_artificial_hub_nodes(
                &mut level_graph,
                &subset_usize,
                self.params.create_artificial_hub_nodes.unwrap(),
            );
        }

        // Add level-graph to hsgf
        self.graphs.push(level_graph);

        // --- Build higher levels
        // let mut break_early_next_loop = false;
        let mut subset_usize: Vec<usize>;

        // Building level-graphs
        for i in 1..self.n_layers {
            let idx = (i).min(level_builders.len() - 1);
            #[cfg(feature = "measure-print-hsgf-level-times")]
            let time = Instant::now();
            let (mut level_graph, selector) = HSGFLevelGraphBuilder::get_level_graph_and_selector(
                &mut level_builders[idx],
                &mat_subset,
                self.dist.clone(),
            );
            #[cfg(feature = "measure-print-hsgf-level-times")]
            println!("Graph {}: {:.2?}", i, time.elapsed());

            // Compute and store local & global ids
            let local_ids: Vec<R> = subset
                .iter()
                .map(|&e| {
                    let j = prev_subset
                        .iter()
                        .position(|&prev_id| prev_id == e)
                        .unwrap();
                    R::from(j).unwrap()
                })
                .collect();
            self.local_layer_ids[i - 1] = local_ids;
            // todo how to get around the expensive(?) cloning?
            self.global_layer_ids[i - 1] = subset.clone();
            prev_subset = subset;

            if i == self.n_layers - 1
            // || break_early_next_loop
            {
                // No further subset selection needed anymore so break here
                self.graphs.push(level_graph);
                return;
            }

            #[cfg(feature = "measure-print-hsgf-level-times")]
            let time = Instant::now();
            // Select subset for next (higher) level
            subset = selector.select_subset(
                &level_graph,
                self.level_subset_sizes[i.min(self.level_subset_sizes.len() - 1)],
                true,
            );
            #[cfg(feature = "measure-print-hsgf-level-times")]
            println!("Subset {}: {:.2?}", i, time.elapsed());
            subset_usize = subset
                .iter()
                .map(|e| unsafe { e.to_usize().unwrap_unchecked() })
                .collect();
            mat_subset = mat_subset.get_rows(&subset_usize);

            // Optional hub-nodes creation in current level_graph
            if self.params.create_artificial_hub_nodes.is_some() {
                _create_artificial_hub_nodes(
                    &mut level_graph,
                    &subset_usize,
                    self.params.create_artificial_hub_nodes.unwrap(),
                );
            }

            // Add level-graph to hsgf
            self.graphs.push(level_graph);

            // Map selected subset back to global_ids by using previous global_layer_ids map
            let global_id_map = self.global_layer_ids[i - 1].as_slice();
            subset = subset
                .into_iter()
                .map(|e| {
                    let e_usize = unsafe { e.to_usize().unwrap_unchecked() };
                    global_id_map[e_usize]
                })
                .collect();

            if subset.len() < self.params.min_n_vertices_layer {
                // if subset.len() <= 1 {
                return;
                // }
                // Make sure to only build the (remaining) final/top layer
                // break_early_next_loop = true;
            }
        }
    }
    fn train_closure<M: MatrixDataSource<F> + Sync>(
        &mut self,
        _mat: &M,
        _bottom_level_builder: Self::BottomLevelGraphBuilder<M>,
        _higher_level_builders: Vec<Self::LevelGraphBuilders>,
    ) {
        unimplemented!("This is an EnumStyle not a ClosureStyle builder.")
    }
    fn _is_enum_builder() -> bool {
        true
    }
}

/// Uses closures as input for the level_builders for each level to construct a HSGF graph
/// self::train uses the last element in higher_level_builders if higher_level_builders.len()
/// less than n_layers many are supplied.
///
/// The closure gets the exact subset of the data for the corresponding level and must return
/// a DirLoLGraph and SubsetSelector.
/// See tests::hsgf_with_closure_construction() for examples.
/// NOTE that for the closure builder the level_builder input is a vector which
/// needs to be passed directly into the its call and cannot be assigned to any variable
/// beforehand because otherwise the types get locked in.
///
/// level_builders need to (in contrast to the EnumBuilder) be split
/// into bottom_level_builder and higher_level_builders because once the first mat_subset is
/// selected the datatype is not the input dynamic 'M' anymore but "definitely" ArrayBase
pub struct HSGFClosureBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
{
    _phantom: std::marker::PhantomData<F>,
    n_data: usize,
    params: HSGFParams,
    dist: Dist,
    graphs: Vec<HSGFLevelGraph<R>>,
    n_layers: usize,
    // in case of flooding this is ignored and serves only as a upper but not lower bound
    level_subset_sizes: Vec<Option<usize>>,
    local_layer_ids: Vec<Vec<R>>,
    global_layer_ids: Vec<Vec<R>>,
}

// impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
//     HSGFClosureBuilder<R, F, Dist>
// {
// }

impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F> + Sync + Send>
    HSGFStyleBuilder<R, F, Dist> for HSGFClosureBuilder<R, F, Dist>
{
    type Params = HSGFParams;
    type Graph = HSGFLevelGraph<R>;
    type BottomLevelGraphBuilder<M> =
        Box<dyn Fn(&M) -> (DirLoLGraph<R>, Box<dyn SubsetSelector<R, F>>)>;
    type LevelGraphBuilders =
        Box<dyn Fn(&Array2<F>) -> (DirLoLGraph<R>, Box<dyn SubsetSelector<R, F>>)>;
    #[inline(always)]
    fn _graphs(&self) -> &Vec<Self::Graph> {
        &self.graphs
    }
    #[inline(always)]
    fn _mut_graphs(&mut self) -> &mut Vec<Self::Graph> {
        &mut self.graphs
    }
    #[inline(always)]
    fn _global_layer_ids(&self) -> &Vec<Vec<R>> {
        &self.global_layer_ids
    }
    #[inline(always)]
    fn _dist(&self) -> &Dist {
        &self.dist
    }
    #[inline(always)]
    fn _into_parts(self) -> (Vec<Self::Graph>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
        (
            self.graphs,
            self.local_layer_ids,
            self.global_layer_ids,
            self.dist,
        )
    }
    fn new<M: MatrixDataSource<F> + Sync>(
        mat: &M,
        dist: Dist,
        params: Self::Params,
        level_subset_sizes: Option<Vec<Option<usize>>>,
    ) -> Self {
        let n_data = mat.n_rows();
        assert!(n_data < R::max_value().to_usize().unwrap());

        let level_norm_param = params
            .level_norm_param_override
            .unwrap_or(1.0 / (params.higher_max_degree as f32).ln());
        let mut n_layers = (((n_data as f32).ln() * level_norm_param).floor() as usize + 1)
            .min(params.max_layers)
            .max(params.min_layers);

        let level_subset_sizes = if level_subset_sizes.is_some() {
            level_subset_sizes.unwrap()
        } else {
            // Calculate level sizes if not set by parameter
            let mut level_subset_sizes = Vec::with_capacity(n_layers);
            for i in 1..n_layers {
                let mut expected_size = n_data as f32;
                (0..i).for_each(|_| expected_size /= (1.0 / level_norm_param).exp());
                if i > 0 {
                    expected_size *= 1.2;
                }
                let expected_size =
                    (expected_size.floor() as usize).max(params.min_n_vertices_layer);
                level_subset_sizes.push(Some(expected_size));
                if expected_size == params.min_n_vertices_layer {
                    if i < (n_layers - 1) {
                        n_layers -= 1;
                        if n_layers < params.min_layers {
                            println!(
                                "N_layer count was lowered by 1 \
                            because of level subset sizes heuristic."
                            )
                        }
                    }
                    break;
                }
            }
            level_subset_sizes
        };

        Self {
            _phantom: std::marker::PhantomData,
            n_data,
            params,
            dist,
            graphs: Vec::with_capacity(n_layers),
            n_layers,
            local_layer_ids: (0..n_layers - 1).map(|_| Vec::new()).collect(),
            global_layer_ids: (0..n_layers - 1).map(|_| Vec::new()).collect(),
            level_subset_sizes,
        }
    }

    /// Uses last element in higher_level_builders if less than n_layers many are supplied
    fn train_closure<M: MatrixDataSource<F> + Sync>(
        &mut self,
        mat: &M,
        bottom_level_builder: Box<dyn Fn(&M) -> (DirLoLGraph<R>, Box<dyn SubsetSelector<R, F>>)>,
        higher_level_builders: Vec<
            Box<dyn Fn(&Array2<F>) -> (DirLoLGraph<R>, Box<dyn SubsetSelector<R, F>>)>,
        >,
    ) {
        // Data size sanity check, otherwise create one fully connected graph for the bottom level
        if self.n_data <= self.params.min_n_vertices_layer {
            let mut graph = DirLoLGraph::<R>::new();
            if self.n_data > 0 {
                if self.n_data == 1 {
                    graph.add_node();
                } else {
                    for i in 0..self.n_data {
                        graph.add_node();
                        for j in 0..i {
                            let i_r = unsafe { R::from(i).unwrap_unchecked() };
                            let j_r = unsafe { R::from(j).unwrap_unchecked() };
                            graph.add_edge(i_r, j_r);
                        }
                    }
                }
            }
            self.n_layers = 1;
            self.local_layer_ids = Vec::new();
            self.global_layer_ids = Vec::new();
            self.graphs.push(graph);
            return;
        }

        // --- Build bottom level
        let (mut level_graph, selector) = bottom_level_builder(&mat);
        // Select the points for the next level
        let mut subset = selector.select_subset(&level_graph, self.level_subset_sizes[0], true);
        let subset_usize = subset
            .iter()
            .map(|e| unsafe { e.to_usize().unwrap_unchecked() })
            .collect();
        let mut mat_subset: Array2<F> = mat.get_rows(&subset_usize);
        // At first prev_subset contains all points
        let mut prev_subset: Vec<R> = (0..self.n_data)
            .map(|e| unsafe { R::from(e).unwrap_unchecked() })
            .collect();

        // Optional hub-nodes creation in current level_graph
        if self.params.create_artificial_hub_nodes.is_some() {
            _create_artificial_hub_nodes(
                &mut level_graph,
                &subset_usize,
                self.params.create_artificial_hub_nodes.unwrap(),
            );
        }

        // Add level-graph to hsgf
        self.graphs.push(level_graph);

        // --- Build higher levels
        // let mut break_early_next_loop = false;
        let mut subset_usize: Vec<usize>;

        // Building level-graphs
        for i in 1..self.n_layers {
            let idx = (i).min(higher_level_builders.len() - 1);
            let (mut level_graph, selector) = higher_level_builders[idx](&mat_subset);

            // Compute and store local & global ids
            let local_ids: Vec<R> = subset
                .iter()
                .map(|&e| {
                    let idx = prev_subset
                        .iter()
                        .position(|&prev_id| prev_id == e)
                        .unwrap();
                    R::from(idx).unwrap()
                })
                .collect();
            self.local_layer_ids[i - 1] = local_ids;
            self.global_layer_ids[i - 1] = subset.clone();
            // todo how to get around the expensive(?) cloning?
            prev_subset = subset;

            if i == self.n_layers - 1
            //  || break_early_next_loop
            {
                // No further subset selection needed anymore so break here
                self.graphs.push(level_graph);
                return;
            }

            // Select subset for next (higher) level
            subset = selector.select_subset(
                &level_graph,
                self.level_subset_sizes[i.min(self.level_subset_sizes.len() - 1)],
                true,
            );
            subset_usize = subset
                .iter()
                .map(|e| unsafe { e.to_usize().unwrap_unchecked() })
                .collect();
            mat_subset = mat_subset.get_rows(&subset_usize);

            // Optional hub-nodes creation in current level_graph
            if self.params.create_artificial_hub_nodes.is_some() {
                _create_artificial_hub_nodes(
                    &mut level_graph,
                    &subset_usize,
                    self.params.create_artificial_hub_nodes.unwrap(),
                );
            }

            // Add level-graph to hsgf
            self.graphs.push(level_graph);

            // Map selected subset back to global_ids by using previous global_layer_ids map
            let global_id_map = self.global_layer_ids[i - 1].as_slice();
            subset = subset
                .into_iter()
                .map(|e| {
                    let e_usize = unsafe { e.to_usize().unwrap_unchecked() };
                    global_id_map[e_usize]
                })
                .collect();

            if subset.len() < self.params.min_n_vertices_layer {
                // if subset.len() <= 1 {
                return;
                // }
                // Make sure to only build the (remaining) final/top layer
                // break_early_next_loop = true;
            }
        }
    }
    fn train<M: MatrixDataSource<F> + Sync>(
        &mut self,
        _mat: &M,
        _level_builders: Vec<Self::LevelGraphBuilders>,
    ) {
        unimplemented!("This is an ClosureStyle not an EnumStyle builder.")
    }
    fn _is_enum_builder() -> bool {
        false
    }
}

/* Tries to increase the out-degree of the nodes in the selected subset by randomly selecting other
nodes as additional neighbors from the current level_graph
A different heuristic than random selection is possible
NOTE: consequently this does not uphold the original graph's properties but simply tries to
increase the degree with the goal of better reachability */
fn _create_artificial_hub_nodes<R: SyncUnsignedInteger>(
    level_graph: &mut DirLoLGraph<R>,
    selected_subset: &Vec<usize>,
    add_node_count: usize,
) {
    let n_vertices = level_graph.n_vertices();
    for vertex in selected_subset {
        let vertex_r = R::from(*vertex).unwrap();
        let curr_neighbors = level_graph.neighbors(vertex_r);
        let potential_new_neighbors: Vec<R> = random_unique_uint(n_vertices, add_node_count);
        for new_neighbor in potential_new_neighbors {
            if new_neighbor != vertex_r && !curr_neighbors.contains(&new_neighbor) {
                level_graph.add_edge(vertex_r, new_neighbor);
            }
        }
    }
}

pub mod hierarchy_merger {
    use graphidx::{
        indices::{GraphIndex, GreedyLayeredGraphIndex, GreedySingleGraphIndex},
        measures::SquaredEuclideanDistance,
    };
    use graphidxbaselines::hnsw::{HNSWParallelHeapBuilder, HNSWParams, HNSWStyleBuilder};

    use crate::hsgf::*;

    /// Creates a new hierarchy graph by merging a single-layer graph with a hierarchy graph
    /// Basically swapping the bottom layer of the hierarchy graph with the single-layer graph
    /// Both graphs must be build on the same data
    /// Works as the bottom layer is not directly concerned with the id_maps
    pub fn merge_hnsw_with_new_bottom_level<
        R: SyncUnsignedInteger,
        F: SyncFloat,
        Dist: Distance<F> + Sync + Send,
        M: MatrixDataSource<F> + Sync,
        G: Graph<R> + Sync,
    >(
        data: M,
        index_bottom: GreedySingleGraphIndex<R, F, Dist, M, G>,
        hnsw_params: HNSWParams<F>,
        bottom_subset_selector: Box<dyn SubsetSelector<R, F>>,
        higher_level_max_heap_size: usize,
        level_subset_size: Option<usize>,
    ) -> GreedyLayeredGraphIndex<R, F, SquaredEuclideanDistance<F>, M, DirLoLGraph<R>> {
        let bottom_graph = index_bottom.graph().as_dir_lol_graph();
        // Select the points for the next level
        let subset = bottom_subset_selector.select_subset(&bottom_graph, level_subset_size, true);
        let subset_usize = subset
            .iter()
            .map(|e| unsafe { e.to_usize().unwrap_unchecked() })
            .collect::<Vec<usize>>();
        let mat_subset: Array2<F> = data.get_rows(&subset_usize);
        let prev_subset: Vec<R> = (0..data.n_rows())
            .map(|e| unsafe { R::from(e).unwrap_unchecked() })
            .collect();

        let local_ids: Vec<R> = subset
            .iter()
            .map(|&e| {
                let j = prev_subset
                    .iter()
                    .position(|&prev_id| prev_id == e)
                    .unwrap();
                R::from(j).unwrap()
            })
            .collect();
        let mut local_layer_ids = vec![local_ids];
        // todo how to get around the expensive(?) cloning?
        let mut global_layer_ids = vec![subset.clone()];

        let index_hnsw = HNSWParallelHeapBuilder::build(
            &mat_subset,
            SquaredEuclideanDistance::new(),
            hnsw_params,
            higher_level_max_heap_size,
        );

        let mut graphs = vec![bottom_graph];
        // add all but the bottom graph of the hierarchy_index to the new index's graphs
        index_hnsw
            .graphs()
            .iter()
            .for_each(|g| graphs.push(g.as_dir_lol_graph()));

        (1..index_hnsw.graphs().len()).for_each(|i| {
            local_layer_ids.push(index_hnsw.get_local_layer_ids(i).unwrap().clone());
        });
        (1..index_hnsw.graphs().len()).for_each(|i| {
            global_layer_ids.push(index_hnsw.get_global_layer_ids(i).unwrap().clone());
        });

        GreedyLayeredGraphIndex::new(
            data,
            graphs,
            local_layer_ids,
            global_layer_ids,
            SquaredEuclideanDistance::new(),
            higher_level_max_heap_size,
            None,
        )
    }
}

// -------------- TESTS --------------

#[cfg(test)]
mod tests {
    use crate::hsgf::*;
    use crate::selectors::*;
    #[allow(unused)]
    use crate::{
        print_index_build_info, search_as_qps,
        utils::{
            eval::{calc_recall, get_test_data, init_with_dataset, EvalDataSet},
            index_stats::IndexGraphStats,
        },
    };
    use graphidx::{graphs::Graph, indices::GreedyCappedLayeredGraphIndex};
    #[allow(unused)]
    use graphidx::{
        indices::{bruteforce_neighbors, GraphIndex},
        measures::SquaredEuclideanDistance,
    };
    #[allow(unused)]
    use ndarray::{Array1, Array2, Axis, Slice};

    type R = usize;
    type F = f32;
    type Dist = SquaredEuclideanDistance<F>;

    #[test]
    fn hsgf_with_enum_construction() {
        // let (_nd, _nq, _d, _k, data) = get_test_data(true);
        let (data, _queries, _ground_truth, _nd, _nq, _d, _k) =
            init_with_dataset(&EvalDataSet::SIFTSmall);

        // type SRand = RandomSelector<R, F>;
        type SFlood = FloodingSelectorParallelChunked<R, F>;
        let dist = Dist::new();

        let params = HSGFParams::new().with_min_layers(3);
        let level_subset_sizes = None;
        let graph_time = Instant::now();

        // let input_graph_rnn =
        //     RNNDescentBuilder::<R, F, Dist>::build(data.view(), Dist::new(), RNNParams::new())
        //         .graph()
        //         .as_dir_lol_graph();

        // tests all available HSGFLevelGraphBuilders
        let level_builders = vec![HSGFLevelGraphBuilder::new_deg(
            DEGParams::new(),
            Box::new(SFlood::new(1)),
        )];

        type BuilderType = HSGFEnumBuilder<R, F, Dist>;
        let index = BuilderType::build(
            &data,
            dist,
            params,
            level_builders,
            level_subset_sizes,
            None,
            1,
        );
        println!("HSGFWithEnum construction: {:.2?}", graph_time.elapsed());
        print_index_build_info!(index);
    }

    /// Unit test for HSGFClosureBuilder
    #[test]
    fn hsgf_with_closure_construction() {
        // let (_nd, _nq, _d, _k, data) = get_test_data(true);
        let (data, _queries, _ground_truth, _nd, _nq, _d, _k) =
            init_with_dataset(&EvalDataSet::SIFTSmall);

        type SRand = RandomSelector<R, F>;
        type SFlood = FloodingSelectorParallelChunked<R, F>;
        let dist = Dist::new();

        let params = HSGFParams::new().with_min_layers(3);
        let level_subset_sizes = None;
        let graph_time = Instant::now();

        type BuilderType = HSGFClosureBuilder<R, F, Dist>;
        let index = BuilderType::build_closure(
            &data,
            dist,
            params,
            Box::new(|x| {
                let level_graph = RNNDescentBuilder::build(x, Dist::new(), RNNParams::new())
                    .graph()
                    .as_dir_lol_graph();
                (level_graph, Box::new(SRand::new()))
            }),
            vec![Box::new(|x| {
                let level_graph = DEGParallelBuilder::build(x, Dist::new(), DEGParams::new())
                    .graph()
                    .as_dir_lol_graph();
                (level_graph, Box::new(SFlood::new(1)))
            })],
            level_subset_sizes,
            None,
            1,
        );
        println!("HSGFWithClosure construction: {:.2?}", graph_time.elapsed());
        print_index_build_info!(index);
    }

    #[test]
    fn hsgf_query() {
        // let (_nd, nq, _d, k, data) = get_test_data(true); // normal data
        // let queries = data.slice_axis(Axis(0), Slice::from(0..nq));
        let (data, queries, ground_truth, _nd, nq, _d, k) =
            init_with_dataset(&EvalDataSet::SIFTSmall);

        let extend_search_k = 5 * k;
        let _flooding_range = 1;

        const USE_ENUM_BUILDER: bool = true;

        // type P = RNGGParams;
        type P2 = RNNParams;
        type S = RandomSelector<R, F>;
        type S2 = FloodingSelectorParallelChunked<R, F>;
        let dist = Dist::new();

        let build_time = Instant::now();

        let params = HSGFParams::new().with_min_layers(4);
        let higher_level_max_heap_size = 1;
        let level_subset_sizes = None;
        // Some(vec![Some(400), Some(300)]);

        let index1 = if USE_ENUM_BUILDER {
            type BuilderType = HSGFEnumBuilder<R, F, Dist>;
            print!(
                "Graph construction ({:?}):",
                std::any::type_name::<BuilderType>(),
            );
            let level_builders = vec![
                HSGFLevelGraphBuilder::new_rnn(P2::new(), Box::new(S::new())),
                // HSGFLevelGraphBuilder::new_rngg(P::new(), Box::new(S2::new(2))),
                // HSGFLevelGraphBuilder::new_rnn(P2::new(), Box::new(S::new())),
            ];

            BuilderType::build(
                data.view(),
                dist,
                params,
                level_builders,
                level_subset_sizes,
                None,
                higher_level_max_heap_size,
            )
        } else {
            type BuilderType = HSGFClosureBuilder<R, F, Dist>;
            print!(
                "Graph construction ({:?}):",
                std::any::type_name::<BuilderType>(),
            );
            BuilderType::build_closure(
                data.view(),
                dist,
                params,
                Box::new(|x| {
                    let level_graph = RNNDescentBuilder::build(x, Dist::new(), RNNParams::new())
                        .graph()
                        .as_dir_lol_graph();
                    (level_graph, Box::new(S::new()))
                }),
                vec![
                    Box::new(|x| {
                        let level_graph = RNGGBuilder::build(x, Dist::new(), RNGGParams::new())
                            .graph()
                            .as_dir_lol_graph();
                        (level_graph, Box::new(S2::new(2)))
                    }),
                    Box::new(|x| {
                        let level_graph =
                            RNNDescentBuilder::build(x, Dist::new(), RNNParams::new())
                                .graph()
                                .as_dir_lol_graph();
                        (level_graph, Box::new(S::new()))
                    }),
                ],
                level_subset_sizes,
                None,
                higher_level_max_heap_size,
            )
        };

        println!("{:.2?}", build_time.elapsed());
        print_index_build_info!(index1);

        let index2 = GreedyCappedLayeredGraphIndex::new(
            data.view(),
            index1
                .graphs()
                .iter()
                .map(|g| g.as_dir_lol_graph())
                .collect(),
            (1..index1.graphs().len())
                .map(|i| index1.get_local_layer_ids(i).unwrap().clone())
                .collect(),
            (1..index1.graphs().len())
                .map(|i| index1.get_global_layer_ids(i).unwrap().clone())
                .collect(),
            Dist::new(),
            index1.higher_level_max_heap_size(),
            3 * k,
            None,
        );

        // let bruteforce_time = Instant::now();
        // let (ground_truth, _) = bruteforce_neighbors(&data, &queries, &Dist::new(), k);
        // println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());
        let ground_truth_view = ground_truth.view();

        /* HSGF queries */
        let hsgf_time = Instant::now();
        let (hsgf_ids1, _hsgf_dists1) = index1.greedy_search_batch(&queries, k, extend_search_k);
        println!("HSGF queries 1: {:.2?}", hsgf_time.elapsed());
        let hsgf_time = Instant::now();
        let (hsgf_ids2, _hsgf_dists2) = index2.greedy_search_batch(&queries, k, extend_search_k);
        println!("HSGF queries 2: {:.2?}", hsgf_time.elapsed());

        calc_recall(ground_truth_view, &hsgf_ids1, nq, k, "HSGF-1", true);
        calc_recall(ground_truth_view, &hsgf_ids2, nq, k, "HSGF-2", true);
    }

    #[test]
    fn hsgf_query_compare() {
        // let (_nd, nq, _d, k, data) = get_test_data(true); // normal data
        // let queries = data.slice_axis(Axis(0), Slice::from(0..nq));
        let (data, queries, ground_truth, _nd, nq, _d, k) =
            init_with_dataset(&EvalDataSet::SIFTSmall);

        let extend_search_k = 5 * k;
        let _flooding_range = 1;

        // type P = RNGGParams;
        type P2 = RNNParams;
        // type P2 = DEGParams;
        type S = RandomSelector<R, F>;
        let dist = Dist::new();

        let build_time = Instant::now();

        let params = HSGFParams::new()
            .with_min_layers(4)
            .with_create_artificial_hub_nodes(None); // Some(50)
        let higher_level_max_heap_size = 1;
        let level_subset_sizes = None;
        // Some(vec![Some(400), Some(300)]);

        type BuilderType = HSGFEnumBuilder<R, F, Dist>;
        print!(
            "Graph construction ({:?}):",
            std::any::type_name::<BuilderType>(),
        );

        let index1 = BuilderType::build(
            data.view(),
            dist.clone(),
            params,
            vec![HSGFLevelGraphBuilder::new_rnn(
                P2::new(),
                Box::new(S::new()),
            )],
            level_subset_sizes.clone(),
            None,
            higher_level_max_heap_size,
        );

        println!("{:.2?}", build_time.elapsed());
        print_index_build_info!(index1);

        let build_time = Instant::now();
        let index2 = BuilderType::build(
            data.view(),
            dist,
            params.with_create_artificial_hub_nodes(Some(10)),
            vec![HSGFLevelGraphBuilder::new_rnn(
                P2::new(),
                Box::new(S::new()),
            )],
            level_subset_sizes,
            None,
            higher_level_max_heap_size,
        );

        println!("{:.2?}", build_time.elapsed());
        print_index_build_info!(index2);

        // let bruteforce_time = Instant::now();
        // let (ground_truth, _) = bruteforce_neighbors(&data, &queries, &Dist::new(), k);
        // println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());
        let ground_truth_view = ground_truth.view();

        /* HSGF queries */
        let hsgf_time = Instant::now();
        let (hsgf_ids1, _hsgf_dists1) = index1.greedy_search_batch(&queries, k, extend_search_k);
        println!("HSGF queries 1: {:.2?}", hsgf_time.elapsed());
        let hsgf_time = Instant::now();
        let (hsgf_ids2, _hsgf_dists2) = index2.greedy_search_batch(&queries, k, extend_search_k);
        println!("HSGF queries 2: {:.2?}", hsgf_time.elapsed());

        calc_recall(ground_truth_view, &hsgf_ids1, nq, k, "HSGF-1", true);
        calc_recall(ground_truth_view, &hsgf_ids2, nq, k, "HSGF-2", true);
    }

    #[test]
    fn test_merge_indices() {
        use crate::hsgf::hierarchy_merger::merge_hnsw_with_new_bottom_level;
        use graphidxbaselines::hnsw::*;

        // let (_nd, nq, _d, k, data) = get_test_data(true);
        // let queries = data.slice_axis(Axis(0), Slice::from(0..nq));
        let (data, queries, ground_truth, _nd, nq, _d, k) =
            init_with_dataset(&EvalDataSet::SIFTSmall);

        type SFlood = FloodingSelectorParallelChunked<R, F>;

        // Bottom graph
        let build_time = Instant::now();
        let bottom_graph_params = DEGParams::new()
            .with_edges_per_vertex(40)
            .with_extend_k(100);

        type BuilderTypeBottom = DEGParallelBuilder<R, F, Dist>;
        let index_bottom = BuilderTypeBottom::build(data.view(), Dist::new(), bottom_graph_params);

        println!(
            "Graph construction ({:?}): {:.2?}",
            std::any::type_name::<BuilderTypeBottom>(),
            build_time.elapsed()
        );
        print_index_build_info!(index_bottom);

        // Merge
        let higher_level_max_heap_size = 1;
        let index_merged = merge_hnsw_with_new_bottom_level(
            data.view(),
            index_bottom,
            HNSWParams::new(),
            Box::new(SFlood::new(1)),
            higher_level_max_heap_size,
            None,
        );

        // Eval
        // single threaded search
        let max_heap_size = 1 * k;
        let ground_truth_view = ground_truth.view();

        let graph_name = "Merged-Graph";
        let (_query_time, _qps, _recall) = search_as_qps!(
            index_merged,
            queries,
            ground_truth_view,
            k,
            max_heap_size,
            nq,
            graph_name,
            true
        );
    }
}
