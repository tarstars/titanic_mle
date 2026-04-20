from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_files_exist() -> None:
    expected = [
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "data" / "README.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "codex_jupyter_workflow.md",
        ROOT / "docs" / "troubleshooting.md",
        ROOT / "scripts" / "install_kernel.sh",
        ROOT / "scripts" / "start_jupyter_lab.sh",
        ROOT / "notebooks" / "titanic_starter.ipynb",
    ]
    for path in expected:
        assert path.exists(), f"missing expected file: {path}"


def test_pyproject_contains_stable_pins() -> None:
    content = (ROOT / "pyproject.toml").read_text()
    assert 'name = "titanic-mle"' in content
    assert "jupyter-mcp-server" in content
    assert "jupyter-collaboration" in content
    assert "pycrdt<0.12.50" in content
