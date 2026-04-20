# Titanic MLE

Minimal starter repository for Titanic notebook work where Codex controls JupyterLab through MCP.

This repository keeps the scope narrow:

- run a local JupyterLab server with the MCP-compatible collaboration stack
- expose a stable endpoint for Codex or another MCP-capable agent
- provide a Titanic-oriented starter notebook and a place for `train.csv` and `test.csv`

It is intentionally not a full ML framework. It is the smallest reusable base for local Titanic experimentation with agent-driven notebooks.

## What You Get

After setup, you have:

- JupyterLab bound to `127.0.0.1:8894`
- token-based authentication with a repo-specific default token
- `jupyter-collaboration` enabled for stable notebook editing
- a dedicated `titanic-mle` kernel
- a starter notebook at `notebooks/titanic_starter.ipynb`
- a `data/` directory convention for Titanic CSV files

## Stability Choices

This repo keeps the working integration path:

- `jupyterlab`
- `jupyter-collaboration`
- `jupyter-mcp-server`
- `jupyter-mcp-tools`
- `pycrdt<0.12.50`

The `pycrdt` pin is intentional. Relaxing it without retesting the notebook edit flow is a bad trade.

## Quick Start

Prerequisites:

- Python `3.10+`
- `uv`
- an MCP-capable agent environment with Jupyter MCP support

Install:

```bash
uv sync --dev
./scripts/install_kernel.sh
```

Start JupyterLab:

```bash
./scripts/start_jupyter_lab.sh
```

Default endpoint:

- URL: `http://127.0.0.1:8894`
- token: `titanic-mle-local-token`

Override the token if needed:

```bash
TITANIC_MLE_JUPYTER_TOKEN=my-token ./scripts/start_jupyter_lab.sh
```

## Connect Codex

Point Codex or another MCP client to:

- URL: `http://localhost:8894`
- token: `titanic-mle-local-token`

The first smoke test is:

1. Open `notebooks/titanic_starter.ipynb`.
2. Insert a markdown or code cell.
3. Execute a cell.
4. Confirm that the notebook saves back to disk.

## Titanic Data Convention

This repo does not bundle Kaggle Titanic data. Put local copies here:

- `data/train.csv`
- `data/test.csv`

The starter notebook checks whether those files exist before you build any analysis or model cells on top.

## Repository Layout

```text
.
├── data/
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── codex_jupyter_workflow.md
│   └── troubleshooting.md
├── notebooks/
│   └── titanic_starter.ipynb
├── scripts/
│   ├── install_kernel.sh
│   └── start_jupyter_lab.sh
├── tests/
│   └── test_repo_layout.py
├── pyproject.toml
└── README.md
```

Key files:

- [`pyproject.toml`](./pyproject.toml): pinned dependency set
- [`scripts/start_jupyter_lab.sh`](./scripts/start_jupyter_lab.sh): strict Jupyter launcher with a fixed default port
- [`scripts/install_kernel.sh`](./scripts/install_kernel.sh): installs the repo kernel
- [`notebooks/titanic_starter.ipynb`](./notebooks/titanic_starter.ipynb): starter notebook for MCP smoke tests
- [`data/README.md`](./data/README.md): expected local data placement

## Common Commands

```bash
uv sync --dev
./scripts/install_kernel.sh
./scripts/start_jupyter_lab.sh
uv run pytest
```

## Verification

Run:

```bash
uv run pytest
```

That validates the repository layout and the dependency pins. It does not replace a live end-to-end Jupyter plus MCP smoke test.

## Troubleshooting

See [`docs/troubleshooting.md`](./docs/troubleshooting.md) for the usual failure modes:

- port collision
- token mismatch
- missing kernel
- unstable collaboration dependency versions

## Scope

This repository is for:

- local Titanic notebook work
- Codex-driven notebook editing through MCP
- small, reproducible experimentation environments

It is not yet a training pipeline, feature store, deployment stack, or competition submission framework.
