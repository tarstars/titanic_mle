# Troubleshooting

## JupyterLab does not start because the port is busy

The launcher uses port `8894` by default and does not auto-retry other ports.

That is intentional. An MCP client needs a stable endpoint.

Options:

- stop the process already using `8894`
- start on another port explicitly: `./scripts/start_jupyter_lab.sh 8895`
- point your MCP client to the same alternate port

## MCP client cannot authenticate

Verify:

- the token passed to the MCP client matches the Jupyter token
- the server was started with the token you expect
- you did not override `TITANIC_MLE_JUPYTER_TOKEN` in another shell

Default token:

- `titanic-mle-local-token`

## The agent can connect but notebook editing is unstable

Check package versions first. The stable combination in this repo depends on:

- `jupyter-mcp-server`
- `jupyter-collaboration`
- `jupyter-mcp-tools`
- `pycrdt<0.12.50`

If you upgrade these blindly, regression risk is high.

## Kernel is missing in JupyterLab

Run:

```bash
./scripts/install_kernel.sh
```

Then restart JupyterLab.

## Tests pass but the agent still fails to operate notebooks

The included tests validate repository structure and dependency intent. They do not replace a live MCP integration test.

For a real smoke test:

1. start JupyterLab
2. connect the MCP client
3. open `notebooks/titanic_starter.ipynb`
4. insert a cell
5. execute the cell

If that works, the end-to-end path is healthy.
