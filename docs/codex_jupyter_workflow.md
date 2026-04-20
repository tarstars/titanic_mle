# Codex And Jupyter Workflow

This repo owns the Jupyter side of the integration. Codex connects to the running JupyterLab instance through MCP.

## Expected Local Endpoint

- URL: `http://localhost:8894`
- token: `titanic-mle-local-token`

Those defaults come from `scripts/start_jupyter_lab.sh`.

## Stable Assumptions

- JupyterLab runs with token auth enabled.
- `jupyter-collaboration` is installed.
- `jupyter-mcp-server` is available in the Codex MCP environment.
- `pycrdt<0.12.50` remains pinned.
- The server stays bound to `localhost` for local development.
- The launcher does not auto-retry ports, because Codex depends on a predictable URL.

## Typical Codex Session

1. Start JupyterLab from this repo.
2. Connect Codex to the endpoint above.
3. Open `notebooks/titanic_starter.ipynb` or create another notebook in `notebooks/`.
4. Use MCP tools to add cells, execute them, and inspect outputs.
5. Save changes back into the repository.

## Repository Layout

- `scripts/start_jupyter_lab.sh`: launches token-authenticated JupyterLab on port `8894`
- `scripts/install_kernel.sh`: registers the `titanic-mle` kernel
- `notebooks/titanic_starter.ipynb`: starter notebook for smoke testing
- `data/`: expected local placement for `train.csv` and `test.csv`

## Notes

This repo does not bundle Codex itself. It provides the Jupyter environment and the endpoint contract that Codex can reliably control through MCP.
