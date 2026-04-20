# Architecture

This repository is a small local substrate for Titanic notebook work through Codex and MCP.

## Goal

Provide a stable JupyterLab setup where an MCP-capable agent can open, edit, and execute Titanic notebooks without manual notebook UI steps.

## Components

### JupyterLab

`jupyterlab` provides the notebook server and browser UI.

### Jupyter Collaboration

`jupyter-collaboration` provides the collaboration model that the MCP notebook editing flow depends on.

### Jupyter MCP Server

`jupyter-mcp-server` exposes notebook operations to the agent through MCP.

### Jupyter MCP Tools

`jupyter-mcp-tools` supports the Jupyter-side MCP integration path used in practice here.

### Repo Kernel

The dedicated `titanic-mle` kernel keeps notebook execution tied to the repo environment.

### Starter Assets

- `notebooks/titanic_starter.ipynb` is the first smoke-test notebook.
- `data/` is the expected location for local Titanic CSV files.

## Runtime Flow

1. `uv` creates the local environment.
2. `install_kernel.sh` registers the `titanic-mle` kernel.
3. `start_jupyter_lab.sh` launches JupyterLab on `127.0.0.1:8894` with token auth.
4. Codex connects to that endpoint over MCP.
5. Notebook edits and cell execution requests flow through MCP into JupyterLab.

## Design Constraints

The repo deliberately enforces:

- fixed default port: the MCP client needs a predictable address
- token auth enabled: required for the stable collaboration path
- `pycrdt<0.12.50`: avoids a known incompatible release in this stack
- local-only binding: intended for local development, not public exposure

## Non-Goals

This repository does not yet provide:

- model training orchestration
- experiment tracking
- remote notebook hosting
- deployment automation

It is the notebook-control layer and a light Titanic project scaffold, nothing more.
