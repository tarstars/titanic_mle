"""Simple tree primitives for organizing the Titanic MLE approach."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MLETreeNode:
    """Minimal node representation for the project-specific mle tree."""

    name: str
    description: str
    children: list["MLETreeNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_child(self, node: "MLETreeNode") -> None:
        self.children.append(node)


def bootstrap_mle_tree() -> MLETreeNode:
    """Return the first skeleton tree for the project."""

    root = MLETreeNode(
        name="titanic_mle_root",
        description="Top-level tree for solving Titanic with the custom mle(x, y) function.",
    )
    root.add_child(
        MLETreeNode(
            name="function_knowledge",
            description="Properties, constraints, and transformations related to mle(x, y).",
        )
    )
    root.add_child(
        MLETreeNode(
            name="dataset_knowledge",
            description="Facts about Titanic columns, targets, and raw-data limitations.",
        )
    )
    root.add_child(
        MLETreeNode(
            name="modeling_context",
            description="Current mapping ideas from Titanic features to the mle tree structure.",
        )
    )
    return root
