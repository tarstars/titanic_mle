from titanic_mle.sexpr_trees import build_exact_tree_cache, tree_shape_summary, tree_to_sexpr


def test_exact_tree_counts_match_known_values() -> None:
    cache = build_exact_tree_cache(3)

    assert len(cache[0]) == 2
    assert len(cache[1]) == 4
    assert len(cache[2]) == 32
    assert len(cache[3]) == 1408


def test_tree_to_sexpr_serializes_unbalanced_tree() -> None:
    tree = (("x", "1"), "1")
    assert tree_to_sexpr(tree) == "((x 1) 1)"


def test_tree_shape_summary_counts_nodes() -> None:
    tree = (("x", "1"), "1")
    summary = tree_shape_summary(tree)

    assert summary.height == 2
    assert summary.leaf_count == 3
    assert summary.internal_node_count == 2
    assert summary.total_node_count == 5
