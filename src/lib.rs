//! # HSGF - Graphs
//!
//! This lib contains several graphs builders:
//! - DEG *
//! - NSSG *
//! - EFANNA *
//!
//! - HSGF - The main contribution of this crate/work
//!     - A hierarchical graph builder which uses existing graph builder (like the above)
//!      to build each layer of a graph
//!         - The graph builder and point selection are the two primary arguments/parameters of HSGF
//! - Selectors
//!     - The different level selectors which are available for and used in HSGF
//!
//! - RNGG - A helper to create a placeholder randomly initialized graph
//!
//! * Re-implementation of the corresponding original work (see files for source and license),
//!   translated from C++ to Rust including the framework that are the GraphIndexAPI and
//!   GraphIndexBaselines libraries which are both provided by Erik Thordsen (TU Dortmund)
//!
//! Notes
//!
//! Open Issues
//!
//! TODOs
//!

pub mod deg;
pub mod efanna;
pub mod hsgf;
pub mod nssg;
pub mod rngg;
pub mod selectors;
pub mod utils;

#[cfg(feature = "python")]
pub mod py;
