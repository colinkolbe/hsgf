# Hierarchical Search Graph Framework (HSGF)

The Hierarchical Search Graph Framework (HSGF) is the main contribution of a master thesis at the Data Mining Group at TU Dortmund (2025). HSGF allows to extend existing approximate nearest neighbor search (ANNS) graphs with a hierarchy architecture and corresponding redundancy to improve their search performance in terms of recall and queries-per-second (QPS).

This project is build on the framework that is the GraphIndexAPI crate and follows closely the patterns of GraphIndexBaselines crate and can, therefore, be seen as an extension to the latter.

## Contents
- `/src` 
    - hsgf Rust code
- `/GraphIndexAPI`
    - External library [GitHub](https://github.com/eth42/GraphIndexAPI)
    - Uses its own hsgf branch, which has integrated the latest changes of main
        - Current diff is virtually zero
    - Some useful explainers
        - WUnDirLoLGraph: Weighted Un-Directed List-of-List Graph
- `/GraphIndexBaselines`
    - External library [GitHub](https://github.com/eth42/GraphIndexBaselines)
    - Uses its own hsgf branch, which has integrated the latest changes of master
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