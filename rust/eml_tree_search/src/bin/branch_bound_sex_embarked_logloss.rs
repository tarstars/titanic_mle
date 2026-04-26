use std::cmp::Ordering;
use std::collections::HashMap;

use eml_tree_search::{ValueSignature, sex_embarked_domain};

const NEGATIVE_COUNTS: [u32; 6] = [66, 38, 364, 9, 9, 63];
const POSITIVE_COUNTS: [u32; 6] = [29, 3, 77, 64, 27, 142];
const LEAF_SIZE: usize = 32;

#[derive(Clone, Debug)]
struct Candidate {
    signature: ValueSignature,
    expr: String,
}

#[derive(Clone, Debug)]
struct BoundingBox {
    min: [f64; 6],
    max: [f64; 6],
}

#[derive(Clone, Debug)]
enum KdNode {
    Leaf {
        bbox: BoundingBox,
        points: Vec<Candidate>,
    },
    Split {
        bbox: BoundingBox,
        left: Box<KdNode>,
        right: Box<KdNode>,
    },
}

#[derive(Clone, Debug)]
struct SearchResult {
    logloss: f64,
    auc: f64,
    expr: String,
}

fn main() {
    let exact = build_exact_with_exprs(4);
    let union_le_3 = build_union(&exact, 3);
    let union_le_4 = build_union(&exact, 4);

    let left_exact_4 = collect_candidates(&exact[4], |signature| signature.is_left_safe());
    let right_le_4 = collect_candidates(&union_le_4, |signature| signature.is_right_safe());
    let left_le_3 = collect_candidates(&union_le_3, |signature| signature.is_left_safe());
    let right_exact_4 = collect_candidates(&exact[4], |signature| signature.is_right_safe());

    let tree_right_le_4 = build_kd_tree(right_le_4);
    let tree_right_exact_4 = build_kd_tree(right_exact_4);

    let exact_le_4_best = best_logloss_in_union(&union_le_4);

    let case_a = search_case(&left_exact_4, &tree_right_le_4, exact_le_4_best.clone());
    let case_b = search_case(&left_le_3, &tree_right_exact_4, case_a.clone());
    let best = if compare_results(&case_a, &case_b) == Ordering::Less {
        case_a
    } else {
        case_b
    };

    println!("Exact best logloss for height <= 4:");
    println!(
        "  logloss={:.16}, auc={:.16}, expr={}",
        exact_le_4_best.logloss, exact_le_4_best.auc, exact_le_4_best.expr
    );
    println!();
    println!("Exact branch-and-bound best logloss for height <= 5:");
    println!(
        "  logloss={:.16}, auc={:.16}, expr={}",
        best.logloss, best.auc, best.expr
    );
}

fn collect_candidates<F>(
    source: &HashMap<ValueSignature, String>,
    predicate: F,
) -> Vec<Candidate>
where
    F: Fn(&ValueSignature) -> bool,
{
    source
        .iter()
        .filter(|(signature, _)| predicate(signature))
        .map(|(signature, expr)| Candidate {
            signature: signature.clone(),
            expr: expr.clone(),
        })
        .collect()
}

fn compare_results(left: &SearchResult, right: &SearchResult) -> Ordering {
    left.logloss
        .total_cmp(&right.logloss)
        .then_with(|| right.auc.total_cmp(&left.auc))
        .then_with(|| left.expr.cmp(&right.expr))
}

fn build_exact_with_exprs(max_height: usize) -> Vec<HashMap<ValueSignature, String>> {
    let domain = sex_embarked_domain();
    let mut exact = Vec::new();

    let mut base = HashMap::new();
    base.insert(ValueSignature::one(), "1".to_string());
    base.insert(ValueSignature::x0(&domain), "x0".to_string());
    base.insert(ValueSignature::x1(&domain), "x1".to_string());
    exact.push(base);

    for height in 1..=max_height {
        let mut lower = HashMap::new();
        for counter in exact.iter().take(height) {
            for (signature, expr) in counter {
                lower.entry(signature.clone()).or_insert_with(|| expr.clone());
            }
        }

        let lower_entries: Vec<(ValueSignature, String)> = lower.into_iter().collect();
        let previous_exact = &exact[height - 1];
        let mut current = HashMap::new();

        for (left_signature, left_expr) in &lower_entries {
            let left_is_previous = previous_exact.contains_key(left_signature);
            for (right_signature, right_expr) in &lower_entries {
                if !(left_is_previous || previous_exact.contains_key(right_signature)) {
                    continue;
                }
                if let Some(output_signature) = left_signature.eml(right_signature) {
                    current
                        .entry(output_signature)
                        .or_insert_with(|| format!("({left_expr} {right_expr})"));
                }
            }
        }

        exact.push(current);
    }

    exact
}

fn build_union(
    exact: &[HashMap<ValueSignature, String>],
    max_height: usize,
) -> HashMap<ValueSignature, String> {
    let mut out = HashMap::new();
    for counter in exact.iter().take(max_height + 1) {
        for (signature, expr) in counter {
            out.entry(signature.clone()).or_insert_with(|| expr.clone());
        }
    }
    out
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let ez = (-z).exp();
        1.0 / (1.0 + ez)
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn auc(signature: &ValueSignature) -> f64 {
    let mut wins = 0.0;
    let values = signature.values();
    let positives: u32 = POSITIVE_COUNTS.iter().sum();
    let negatives: u32 = NEGATIVE_COUNTS.iter().sum();

    for i in 0..values.len() {
        for j in 0..values.len() {
            let pair_count = (POSITIVE_COUNTS[i] * NEGATIVE_COUNTS[j]) as f64;
            if values[i] > values[j] {
                wins += pair_count;
            } else if values[i].to_bits() == values[j].to_bits() {
                wins += 0.5 * pair_count;
            }
        }
    }

    wins / ((positives * negatives) as f64)
}

fn logloss(signature: &ValueSignature) -> f64 {
    let values = signature.values();
    let total: u32 = POSITIVE_COUNTS
        .iter()
        .zip(NEGATIVE_COUNTS.iter())
        .map(|(positive, negative)| positive + negative)
        .sum();

    let mut loss = 0.0;
    for index in 0..values.len() {
        let probability = sigmoid(values[index]).clamp(1e-15, 1.0 - 1e-15);
        loss += -(POSITIVE_COUNTS[index] as f64) * probability.ln();
        loss += -(NEGATIVE_COUNTS[index] as f64) * (1.0 - probability).ln();
    }

    loss / (total as f64)
}

fn grouped_best_score(index: usize) -> f64 {
    ((POSITIVE_COUNTS[index] as f64) / (NEGATIVE_COUNTS[index] as f64)).ln()
}

fn grouped_interval_best_loss(index: usize, lower: f64, upper: f64) -> f64 {
    let target = grouped_best_score(index);
    let score = target.clamp(lower, upper);
    let probability = sigmoid(score).clamp(1e-15, 1.0 - 1e-15);
    -(POSITIVE_COUNTS[index] as f64) * probability.ln()
        - (NEGATIVE_COUNTS[index] as f64) * (1.0 - probability).ln()
}

fn bbox_from_candidates(points: &[Candidate]) -> BoundingBox {
    let mut min = [f64::INFINITY; 6];
    let mut max = [f64::NEG_INFINITY; 6];

    for point in points {
        for index in 0..6 {
            min[index] = min[index].min(point.signature.values()[index]);
            max[index] = max[index].max(point.signature.values()[index]);
        }
    }

    BoundingBox { min, max }
}

fn build_kd_tree(points: Vec<Candidate>) -> KdNode {
    let bbox = bbox_from_candidates(&points);
    if points.len() <= LEAF_SIZE {
        return KdNode::Leaf { bbox, points };
    }

    let split_dim = (0..6)
        .max_by(|left, right| {
            let left_span = bbox.max[*left] - bbox.min[*left];
            let right_span = bbox.max[*right] - bbox.min[*right];
            left_span.total_cmp(&right_span)
        })
        .unwrap_or(0);

    let mut points = points;
    points.sort_by(|left, right| {
        left.signature.values()[split_dim].total_cmp(&right.signature.values()[split_dim])
    });

    let right_points = points.split_off(points.len() / 2);
    let left_points = points;

    KdNode::Split {
        bbox,
        left: Box::new(build_kd_tree(left_points)),
        right: Box::new(build_kd_tree(right_points)),
    }
}

fn optimistic_lower_bound(left: &ValueSignature, bbox: &BoundingBox) -> f64 {
    let mut total = 0.0;
    for index in 0..6 {
        let lower = left.values()[index].exp() - bbox.max[index].ln();
        let upper = left.values()[index].exp() - bbox.min[index].ln();
        total += grouped_interval_best_loss(index, lower, upper);
    }

    let all_count: u32 = POSITIVE_COUNTS
        .iter()
        .zip(NEGATIVE_COUNTS.iter())
        .map(|(positive, negative)| positive + negative)
        .sum();
    total / (all_count as f64)
}

fn best_logloss_in_union(source: &HashMap<ValueSignature, String>) -> SearchResult {
    let mut best: Option<SearchResult> = None;

    for (signature, expr) in source {
        let candidate = SearchResult {
            logloss: logloss(signature),
            auc: auc(signature),
            expr: expr.clone(),
        };
        if best
            .as_ref()
            .map(|current| compare_results(&candidate, current) == Ordering::Less)
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }

    best.expect("union should not be empty")
}

fn search_case(left_candidates: &[Candidate], right_tree: &KdNode, seed: SearchResult) -> SearchResult {
    let whole_bbox = match right_tree {
        KdNode::Leaf { bbox, .. } | KdNode::Split { bbox, .. } => bbox,
    };

    let mut ordered_lefts: Vec<_> = left_candidates
        .iter()
        .map(|candidate| (optimistic_lower_bound(&candidate.signature, whole_bbox), candidate))
        .collect();
    ordered_lefts.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut best = seed;
    for (bound, left) in ordered_lefts {
        if bound >= best.logloss {
            continue;
        }
        query_tree(left, right_tree, &mut best);
    }
    best
}

fn query_tree(left: &Candidate, node: &KdNode, best: &mut SearchResult) {
    let bbox = match node {
        KdNode::Leaf { bbox, .. } | KdNode::Split { bbox, .. } => bbox,
    };
    if optimistic_lower_bound(&left.signature, bbox) >= best.logloss {
        return;
    }

    match node {
        KdNode::Leaf { points, .. } => {
            for right in points {
                if let Some(signature) = left.signature.eml(&right.signature) {
                    let candidate = SearchResult {
                        logloss: logloss(&signature),
                        auc: auc(&signature),
                        expr: format!("({} {})", left.expr, right.expr),
                    };
                    if compare_results(&candidate, best) == Ordering::Less {
                        *best = candidate;
                    }
                }
            }
        }
        KdNode::Split { left: left_node, right: right_node, .. } => {
            let left_bound = optimistic_lower_bound(
                &left.signature,
                match left_node.as_ref() {
                    KdNode::Leaf { bbox, .. } | KdNode::Split { bbox, .. } => bbox,
                },
            );
            let right_bound = optimistic_lower_bound(
                &left.signature,
                match right_node.as_ref() {
                    KdNode::Leaf { bbox, .. } | KdNode::Split { bbox, .. } => bbox,
                },
            );

            if left_bound <= right_bound {
                if left_bound < best.logloss {
                    query_tree(left, left_node, best);
                }
                if right_bound < best.logloss {
                    query_tree(left, right_node, best);
                }
            } else {
                if right_bound < best.logloss {
                    query_tree(left, right_node, best);
                }
                if left_bound < best.logloss {
                    query_tree(left, left_node, best);
                }
            }
        }
    }
}
