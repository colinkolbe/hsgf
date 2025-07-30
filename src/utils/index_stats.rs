//! Index Analyzer
//!
//! Notes
//!
//! Open Issues
//!
//! TODOs
//!     - [ ] Improve performance
//!
use foldhash::HashSet;
use graphidx::{
    data::MatrixDataSource,
    graphs::Graph,
    indices::{
        GraphIndex, GreedyCappedLayeredGraphIndex, GreedyCappedSingleGraphIndex,
        GreedyLayeredGraphIndex, GreedySingleGraphIndex, IndexedDistance,
    },
    measures::Distance,
    types::{SyncFloat, SyncUnsignedInteger},
};

#[allow(unused)]
pub trait IndexGraphStats<F: SyncFloat> {
    fn get_graph_n_vertices(&self) -> Vec<usize>;
    /// Returns the following tuple: (reported_edges, counted_edges_in_adj, hashset_counted_edges)
    fn get_n_edges(&self) -> (usize, usize, i32);
    fn get_avg_out_degrees(&self) -> Vec<f32>;
    fn get_min_max_out_degrees(&self) -> Vec<(usize, usize)>;
    fn get_total_avg_out_degree(&self) -> f32;
    fn get_avg_1nn_distance(&self) -> F;
    fn get_avg_1nn_distances(&self) -> Vec<F>;
    fn get_min_max_1nn_distances(&self) -> Vec<(F, F)>;
    fn get_duplicate_neighbor_counts(&self) -> Vec<usize>;
    fn get_duplicate_neighbor_total_count(&self) -> usize;
    fn has_loops(&self) -> bool;
    fn has_escaping_edges(&self) -> bool;
}

impl<
        R: SyncUnsignedInteger,
        F: SyncFloat,
        Dist: Distance<F> + Sync + Send,
        M: MatrixDataSource<F>,
        G: Graph<R>,
    > IndexGraphStats<F> for GreedySingleGraphIndex<R, F, Dist, M, G>
{
    fn get_graph_n_vertices(&self) -> Vec<usize> {
        vec![self.graph().n_vertices()]
    }
    fn get_n_edges(&self) -> (usize, usize, i32) {
        let nd = self.graph().n_vertices();
        let counted_edges_in_adj = (0..nd)
            .map(|v| self.graph().neighbors(R::from(v).unwrap()).len())
            .collect::<Vec<usize>>()
            .iter()
            .sum::<usize>();
        let hashset_counted_edges = (0..nd)
            .map(|v| {
                let neighbors = self.graph().neighbors(R::from(v).unwrap());
                neighbors.len() as i32 - neighbors.iter().collect::<HashSet<_>>().len() as i32
            })
            .collect::<Vec<i32>>()
            .iter()
            .sum::<i32>();

        (
            self.graph().n_edges(),
            counted_edges_in_adj,
            hashset_counted_edges,
        )
    }
    fn get_avg_out_degrees(&self) -> Vec<f32> {
        let counted_edges_in_adj = self.get_n_edges().1;
        vec![counted_edges_in_adj as f32 / self.graph().n_vertices() as f32]
    }
    fn get_min_max_out_degrees(&self) -> Vec<(usize, usize)> {
        let nd = self.graph().n_vertices();
        let mut min_degree: usize = std::usize::MAX;
        let mut max_degree: usize = 0;
        (0..nd).for_each(|v| {
            let n_len = self.graph().neighbors(R::from(v).unwrap()).len();
            if n_len < min_degree {
                min_degree = n_len;
            }
            if n_len > max_degree {
                max_degree = n_len;
            }
        });
        vec![(min_degree, max_degree)]
    }
    fn get_total_avg_out_degree(&self) -> f32 {
        self.get_avg_out_degrees()[0]
    }
    fn get_avg_1nn_distances(&self) -> Vec<F> {
        let nd = self.graph().n_vertices();
        let mut edge_counter = 0;
        let mut dist_sum: F = F::zero();
        (0..nd).for_each(|v| {
            let neighbors = &self.graph().neighbors(R::from(v).unwrap());
            edge_counter += neighbors.len();
            neighbors.iter().for_each(|e| {
                dist_sum += self.indexed_distance(R::from(v).unwrap(), R::from(*e).unwrap());
            });
        });
        vec![dist_sum / F::from(edge_counter).unwrap()]
    }
    fn get_avg_1nn_distance(&self) -> F {
        self.get_avg_1nn_distances()[0]
    }
    fn get_min_max_1nn_distances(&self) -> Vec<(F, F)> {
        unimplemented!()
    }
    fn get_duplicate_neighbor_counts(&self) -> Vec<usize> {
        let graph = &self.graph();
        let sum: usize = (0..graph.n_vertices())
            .map(|i| {
                let neighbors = graph.neighbors(R::from(i).unwrap());
                (neighbors.len() as i32
                    - neighbors
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len() as i32) as usize
            })
            .sum();
        vec![sum]
    }
    fn get_duplicate_neighbor_total_count(&self) -> usize {
        self.get_duplicate_neighbor_counts()[0]
    }
    fn has_loops(&self) -> bool {
        let graph = self.graph();
        for i in 0..graph.n_vertices() {
            for j in graph.neighbors(R::from(i).unwrap()) {
                if j.to_usize().unwrap() == i {
                    return true;
                }
            }
        }

        false
    }
    fn has_escaping_edges(&self) -> bool {
        let graph = self.graph();
        for i in 0..graph.n_vertices() {
            for j in graph.neighbors(R::from(i).unwrap()) {
                if j.to_usize().unwrap() >= graph.n_vertices() {
                    return true;
                }
            }
        }

        false
    }
}

impl<
        R: SyncUnsignedInteger,
        F: SyncFloat,
        Dist: Distance<F> + Sync + Send,
        M: MatrixDataSource<F> + Sync,
        G: Graph<R> + Sync,
    > IndexGraphStats<F> for GreedyLayeredGraphIndex<R, F, Dist, M, G>
{
    fn get_graph_n_vertices(&self) -> Vec<usize> {
        (0..self.layer_count())
            .map(|i_layer| {
                let graph = &self.graphs()[i_layer];
                graph.n_vertices()
            })
            .collect::<Vec<_>>()
    }
    fn get_n_edges(&self) -> (usize, usize, i32) {
        let mut edges_reported = 0;
        let mut hashset_counted_edges = 0;
        let mut counted_edges_in_adj = 0;
        (0..self.layer_count()).for_each(|i_layer| {
            let nd = self.graphs()[i_layer].n_vertices();
            edges_reported += &self.graphs()[i_layer].n_edges();
            counted_edges_in_adj += (0..nd)
                .map(|v| self.graphs()[i_layer].neighbors(R::from(v).unwrap()).len())
                .collect::<Vec<usize>>()
                .iter()
                .sum::<usize>();
            hashset_counted_edges += (0..nd)
                .map(|v| {
                    let neighbors = self.graphs()[i_layer].neighbors(R::from(v).unwrap());
                    neighbors.len() as i32 - neighbors.iter().collect::<HashSet<_>>().len() as i32
                })
                .collect::<Vec<i32>>()
                .iter()
                .sum::<i32>();
        });

        (edges_reported, counted_edges_in_adj, hashset_counted_edges)
    }
    fn get_avg_out_degrees(&self) -> Vec<f32> {
        (0..self.layer_count())
            .map(|i_layer| {
                let graph = &self.graphs()[i_layer];
                graph.n_edges() as f32 / (graph.n_vertices() as f32)
            })
            .collect::<Vec<_>>()
    }
    fn get_min_max_out_degrees(&self) -> Vec<(usize, usize)> {
        let mut res: Vec<(usize, usize)> = Vec::with_capacity(self.layer_count());

        (0..self.layer_count()).for_each(|i_layer| {
            let nd = self.graphs()[i_layer].n_vertices();
            let mut min_degree: usize = std::usize::MAX;
            let mut max_degree: usize = 0;
            (0..nd).for_each(|v| {
                let n_len = self.graphs()[i_layer].neighbors(R::from(v).unwrap()).len();
                if n_len < min_degree {
                    min_degree = n_len;
                }
                if n_len > max_degree {
                    max_degree = n_len;
                }
            });
            res.push((min_degree, max_degree));
        });
        res
    }
    fn get_total_avg_out_degree(&self) -> f32 {
        let out_degrees = self.get_avg_out_degrees();
        let len: f32 = out_degrees.len() as f32;
        out_degrees.into_iter().sum::<f32>() / len
    }
    fn get_avg_1nn_distances(&self) -> Vec<F> {
        let mut res: Vec<F> = Vec::with_capacity(self.layer_count());
        (0..self.layer_count()).for_each(|i_layer| {
            let nd = self.graphs()[i_layer].n_vertices();
            let mut edge_counter = 0;
            let mut dist_sum: F = F::zero();
            (0..nd).for_each(|v| {
                let neighbors = self.graphs()[i_layer].neighbors(R::from(v).unwrap());
                edge_counter += neighbors.len();
                neighbors.iter().for_each(|e| {
                    dist_sum += self.indexed_distance(R::from(v).unwrap(), R::from(*e).unwrap());
                });
            });
            res.push(dist_sum / F::from(edge_counter).unwrap());
        });
        res
    }
    fn get_avg_1nn_distance(&self) -> F {
        self.get_avg_1nn_distances().into_iter().sum()
    }
    fn get_min_max_1nn_distances(&self) -> Vec<(F, F)> {
        unimplemented!()
    }
    fn get_duplicate_neighbor_counts(&self) -> Vec<usize> {
        (0..self.layer_count())
            .map(|i_layer| {
                let graph = &self.graphs()[i_layer];
                (0..graph.n_vertices())
                    .map(|i| {
                        let neighbors = graph.neighbors(R::from(i).unwrap());
                        (neighbors.len() as i32
                            - neighbors
                                .iter()
                                .collect::<std::collections::HashSet<_>>()
                                .len() as i32) as usize
                    })
                    .sum::<usize>()
            })
            .collect::<Vec<usize>>()
    }
    fn get_duplicate_neighbor_total_count(&self) -> usize {
        self.get_duplicate_neighbor_counts().into_iter().sum()
    }
    fn has_loops(&self) -> bool {
        for i_layer in 0..self.layer_count() {
            let graph = &self.graphs()[i_layer];
            for i in 0..graph.n_vertices() {
                for j in graph.neighbors(R::from(i).unwrap()) {
                    if j.to_usize().unwrap() == i {
                        return true;
                    }
                }
            }
        }
        false
    }
    fn has_escaping_edges(&self) -> bool {
        for i_layer in 0..self.layer_count() {
            let graph = &self.graphs()[i_layer];
            for i in 0..graph.n_vertices() {
                for j in graph.neighbors(R::from(i).unwrap()) {
                    if j.to_usize().unwrap() >= graph.n_vertices() {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl<
        R: SyncUnsignedInteger,
        F: SyncFloat,
        Dist: Distance<F> + Sync + Send,
        M: MatrixDataSource<F>,
        G: Graph<R>,
    > IndexGraphStats<F> for GreedyCappedSingleGraphIndex<R, F, Dist, M, G>
{
    fn get_graph_n_vertices(&self) -> Vec<usize> {
        vec![self.graph().n_vertices()]
    }
    fn get_n_edges(&self) -> (usize, usize, i32) {
        let nd = self.graph().n_vertices();
        let counted_edges_in_adj = (0..nd)
            .map(|v| self.graph().neighbors(R::from(v).unwrap()).len())
            .collect::<Vec<usize>>()
            .iter()
            .sum::<usize>();
        let hashset_counted_edges = (0..nd)
            .map(|v| {
                let neighbors = self.graph().neighbors(R::from(v).unwrap());
                neighbors.len() as i32 - neighbors.iter().collect::<HashSet<_>>().len() as i32
            })
            .collect::<Vec<i32>>()
            .iter()
            .sum::<i32>();

        (
            self.graph().n_edges(),
            counted_edges_in_adj,
            hashset_counted_edges,
        )
    }
    fn get_avg_out_degrees(&self) -> Vec<f32> {
        let counted_edges_in_adj = self.get_n_edges().1;
        vec![counted_edges_in_adj as f32 / self.graph().n_vertices() as f32]
    }
    fn get_min_max_out_degrees(&self) -> Vec<(usize, usize)> {
        let nd = self.graph().n_vertices();
        let mut min_degree: usize = std::usize::MAX;
        let mut max_degree: usize = 0;
        (0..nd).for_each(|v| {
            let n_len = self.graph().neighbors(R::from(v).unwrap()).len();
            if n_len < min_degree {
                min_degree = n_len;
            }
            if n_len > max_degree {
                max_degree = n_len;
            }
        });
        vec![(min_degree, max_degree)]
    }
    fn get_total_avg_out_degree(&self) -> f32 {
        self.get_avg_out_degrees()[0]
    }
    fn get_avg_1nn_distances(&self) -> Vec<F> {
        let nd = self.graph().n_vertices();
        let mut edge_counter = 0;
        let mut dist_sum: F = F::zero();
        (0..nd).for_each(|v| {
            let neighbors = &self.graph().neighbors(R::from(v).unwrap());
            edge_counter += neighbors.len();
            neighbors.iter().for_each(|e| {
                dist_sum += self.indexed_distance(R::from(v).unwrap(), R::from(*e).unwrap());
            });
        });
        vec![dist_sum / F::from(edge_counter).unwrap()]
    }
    fn get_avg_1nn_distance(&self) -> F {
        self.get_avg_1nn_distances()[0]
    }
    fn get_min_max_1nn_distances(&self) -> Vec<(F, F)> {
        unimplemented!()
    }
    fn get_duplicate_neighbor_counts(&self) -> Vec<usize> {
        let graph = &self.graph();
        let sum: usize = (0..graph.n_vertices())
            .map(|i| {
                let neighbors = graph.neighbors(R::from(i).unwrap());
                (neighbors.len() as i32
                    - neighbors
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len() as i32) as usize
            })
            .sum();
        vec![sum]
    }
    fn get_duplicate_neighbor_total_count(&self) -> usize {
        self.get_duplicate_neighbor_counts()[0]
    }
    fn has_loops(&self) -> bool {
        let graph = self.graph();
        for i in 0..graph.n_vertices() {
            for j in graph.neighbors(R::from(i).unwrap()) {
                if j.to_usize().unwrap() == i {
                    return true;
                }
            }
        }

        false
    }
    fn has_escaping_edges(&self) -> bool {
        let graph = self.graph();
        for i in 0..graph.n_vertices() {
            for j in graph.neighbors(R::from(i).unwrap()) {
                if j.to_usize().unwrap() >= graph.n_vertices() {
                    return true;
                }
            }
        }

        false
    }
}

impl<
        R: SyncUnsignedInteger,
        F: SyncFloat,
        Dist: Distance<F> + Sync + Send,
        M: MatrixDataSource<F> + Sync,
        G: Graph<R> + Sync,
    > IndexGraphStats<F> for GreedyCappedLayeredGraphIndex<R, F, Dist, M, G>
{
    fn get_graph_n_vertices(&self) -> Vec<usize> {
        (0..self.layer_count())
            .map(|i_layer| {
                let graph = &self.graphs()[i_layer];
                graph.n_vertices()
            })
            .collect::<Vec<_>>()
    }
    fn get_n_edges(&self) -> (usize, usize, i32) {
        let mut edges_reported = 0;
        let mut hashset_counted_edges = 0;
        let mut counted_edges_in_adj = 0;
        (0..self.layer_count()).for_each(|i_layer| {
            let nd = self.graphs()[i_layer].n_vertices();
            edges_reported += &self.graphs()[i_layer].n_edges();
            counted_edges_in_adj += (0..nd)
                .map(|v| self.graphs()[i_layer].neighbors(R::from(v).unwrap()).len())
                .collect::<Vec<usize>>()
                .iter()
                .sum::<usize>();
            hashset_counted_edges += (0..nd)
                .map(|v| {
                    let neighbors = self.graphs()[i_layer].neighbors(R::from(v).unwrap());
                    neighbors.len() as i32 - neighbors.iter().collect::<HashSet<_>>().len() as i32
                })
                .collect::<Vec<i32>>()
                .iter()
                .sum::<i32>();
        });

        (edges_reported, counted_edges_in_adj, hashset_counted_edges)
    }
    fn get_avg_out_degrees(&self) -> Vec<f32> {
        (0..self.layer_count())
            .map(|i_layer| {
                let graph = &self.graphs()[i_layer];
                graph.n_edges() as f32 / (graph.n_vertices() as f32)
            })
            .collect::<Vec<_>>()
    }
    fn get_min_max_out_degrees(&self) -> Vec<(usize, usize)> {
        let mut res: Vec<(usize, usize)> = Vec::with_capacity(self.layer_count());

        (0..self.layer_count()).for_each(|i_layer| {
            let nd = self.graphs()[i_layer].n_vertices();
            let mut min_degree: usize = std::usize::MAX;
            let mut max_degree: usize = 0;
            (0..nd).for_each(|v| {
                let n_len = self.graphs()[i_layer].neighbors(R::from(v).unwrap()).len();
                if n_len < min_degree {
                    min_degree = n_len;
                }
                if n_len > max_degree {
                    max_degree = n_len;
                }
            });
            res.push((min_degree, max_degree));
        });
        res
    }
    fn get_total_avg_out_degree(&self) -> f32 {
        let out_degrees = self.get_avg_out_degrees();
        let len: f32 = out_degrees.len() as f32;
        out_degrees.into_iter().sum::<f32>() / len
    }
    fn get_avg_1nn_distances(&self) -> Vec<F> {
        let mut res: Vec<F> = Vec::with_capacity(self.layer_count());
        (0..self.layer_count()).for_each(|i_layer| {
            let nd = self.graphs()[i_layer].n_vertices();
            let mut edge_counter = 0;
            let mut dist_sum: F = F::zero();
            (0..nd).for_each(|v| {
                let neighbors = self.graphs()[i_layer].neighbors(R::from(v).unwrap());
                edge_counter += neighbors.len();
                neighbors.iter().for_each(|e| {
                    dist_sum += self.indexed_distance(R::from(v).unwrap(), R::from(*e).unwrap());
                });
            });
            res.push(dist_sum / F::from(edge_counter).unwrap());
        });
        res
    }
    fn get_avg_1nn_distance(&self) -> F {
        self.get_avg_1nn_distances().into_iter().sum()
    }
    fn get_min_max_1nn_distances(&self) -> Vec<(F, F)> {
        unimplemented!()
    }
    fn get_duplicate_neighbor_counts(&self) -> Vec<usize> {
        (0..self.layer_count())
            .map(|i_layer| {
                let graph = &self.graphs()[i_layer];
                (0..graph.n_vertices())
                    .map(|i| {
                        let neighbors = graph.neighbors(R::from(i).unwrap());
                        (neighbors.len() as i32
                            - neighbors
                                .iter()
                                .collect::<std::collections::HashSet<_>>()
                                .len() as i32) as usize
                    })
                    .sum::<usize>()
            })
            .collect::<Vec<usize>>()
    }
    fn get_duplicate_neighbor_total_count(&self) -> usize {
        self.get_duplicate_neighbor_counts().into_iter().sum()
    }
    fn has_loops(&self) -> bool {
        for i_layer in 0..self.layer_count() {
            let graph = &self.graphs()[i_layer];
            for i in 0..graph.n_vertices() {
                for j in graph.neighbors(R::from(i).unwrap()) {
                    if j.to_usize().unwrap() == i {
                        return true;
                    }
                }
            }
        }
        false
    }
    fn has_escaping_edges(&self) -> bool {
        for i_layer in 0..self.layer_count() {
            let graph = &self.graphs()[i_layer];
            for i in 0..graph.n_vertices() {
                for j in graph.neighbors(R::from(i).unwrap()) {
                    if j.to_usize().unwrap() >= graph.n_vertices() {
                        return true;
                    }
                }
            }
        }
        false
    }
}
