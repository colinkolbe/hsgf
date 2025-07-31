# Hierarchical Search Graph Framework (HSGF)

The Hierarchical Search Graph Framework (HSGF) is the main contribution of a master thesis at the Data Mining Group at TU Dortmund (2025). HSGF allows to extend existing approximate nearest neighbor search (ANNS) graphs with a hierarchy architecture and corresponding redundancy to improve their search performance in terms of recall and queries-per-second (QPS).

This project is build on the framework that is the GraphIndexAPI crate and follows closely the patterns of GraphIndexBaselines crate and can, therefore, be seen as an extension to the latter.

## Contents

- `/src` 
    - hsgf code
    - See more information further down in this README
- `/GraphIndexAPI`
    - Uses a fork of the external library [GitHub](https://github.com/eth42/GraphIndexAPI)
        - Current diff is virtually zero
    - Some useful explainers
        - WUnDirLoLGraph: Weighted Un-Directed List-of-List Graph
- `/GraphIndexBaselines`
    - Uses a fork of the external library [GitHub](https://github.com/eth42/GraphIndexBaselines)
        - Current diff is virtually zero
- `/evaluation`
    - Evaluation related files (see local README)

## Installation

- Pyo3 with [maturin](https://www.maturin.rs/) to build the _hsgf_ Python module/binding
```bash
# in ./src
python3 -m venv .env
source .env/bin/activate
pip install -r py_requirements.txt
maturin develop
# In one line:
# python3 -m venv .env; source .env/bin/activate; pip install -r py_requirements.txt; maturin develop
```

## HSGF - Crate

- About hierarchy (ANNS) graphs..
    - A hierarchy graphs consist of multiple (>1) graphs which are stacked vertically on each other and each level-graph contains a subset of the previous layer. With the bottom layer/level/graph containing all points of the training data
    - The HSGF builds hierarchy graphs which are similar to [HNSW](https://ieeexplore.ieee.org/document/8594636) but HSGF uses existing graphs and constructs hierarchy graphs from the bottom-level sequentially up
    - The HSGF adds relatively minimal overhead due to the significantly smaller (level-)subset-sizes on levels above the bottom level (0.5-5% of the training data is only contained on the second level already)

- General
    - Each graph builder
        - Has a single and parallel version
        - Where the single one is closer to the original reimplemented code base, while the parallel one has fewer or none of the original code comments and is generally more optimized 
    - Building hierarchy graphs from the bottom up
        - Allows to specify (available) graph builders and subset selectors for each level
    - One trait definition for the HSGFStyleBuilder, which can be either an Enum- or a ClosureStyleBuilder
    
- DEG
    - Reimplementation of [GitHub](https://github.com/Visual-Computing/DynamicExplorationGraph/)
    - Based on the main branch, specifically commit `305e121`
        - Specifically relevant for the re-implementation is [builder.h](https://github.com/Visual-Computing/DynamicExplorationGraph/blob/main/cpp/deglib/include/builder.h)
        - Important, this corresponds to the latest version of the DEG by the authors, and not the later renamed (and original) crEG implementation. However, we expect the latest version to also be the most performant.
- (N)SSG
    - Reimplementation of [GitHub](https://github.com/ZJULearning/SSG/)
    - Based on the master branch, specifically commit `f573041`
        - Specifically relevant for the re-implementation is [index_ssg.cpp](https://github.com/ZJULearning/SSG/blob/master/src/index_ssg.cpp)
- Efanna 
    - Reimplementation of [GitHub](https://github.com/ZJULearning/efanna_graph)
    - Based on the master branch, specifically commit `50c4445`
        - Specifically relevant for the re-implementation is [index_graph.cpp](https://github.com/ZJULearning/efanna_graph/blob/master/src/index_graph.cpp)
- RNGG
    - Builds a random graph that allows for fast and basic testing and evaluation
- Selectors 
    - Random, Flooding, FloodingRepeat, Hubs
    - See file for more information on the different selectors and their relevance
- `src/utils`
    - Utility, primarily for evaluation
- `src/py`
    - Collective folder and exports to generate the Python binding *hsgf*
    - /benches/bench.py can be seen as a unit tester for the Python binding *hsgf*


## Dev Notes
- `cargo test` can not be successfully run if the default features include pyo3 (not the case here)
    - If so, use `cargo test --no-default-features` instead 
        - For reference see [pyo3 FAQ](https://pyo3.rs/v0.23.4/faq.html#i-cant-run-cargo-test-my-crate-cannot-be-found-for-tests-in-tests-directory)

- The formatting via `cargo fmt` needs to be triggered manually sometimes in /src/py/* because of the pyo3 annotations

- VS Code
    - The following makes the rust-analyzer in VS Code analyze files that are behind a particular feature flag
        - Settings: "rust-analyzer.cargo.features": ["python"]
    - Nice highlighting for TOML files
        - "[toml]": {"editor.defaultFormatter": "tamasfe.even-better-toml"}

- Useful aliases and functions
```bash

alias senv="source .env/bin/activate"
alias mdev="maturin develop"

alias carc="cargo check"
alias carbu="cargo build"
alias cart="cargo test"
alias carbe="cargo bench -- --nocapture"

function carunit {
    # cargo test --package CRATE_NAME --lib -- FILE_NAME::tests::UNIT_TEST_NAME --exact --show-output --nocapture
    # Example (when in /hsgf): carunit deg deg_construction
    cargo test --no-default-features --package hsgf --lib -- $1::tests::$2 --exact --show-output
}

function carflame {
    timestamp=$(($(date +%s%N)/1000000))
    filename="flamegraph_$1_${timestamp}.svg"
    # cargo flamegraph --root --unit-test -- UNIT_TEST_NAME --nocapture
    cargo flamegraph -o $filename --root --unit-test -- $1 --nocapture --no-default-features; open -a "Brave Browser" $filename
}
```

## Project Information
- Author: Colin Kolbe
- Year: 2025