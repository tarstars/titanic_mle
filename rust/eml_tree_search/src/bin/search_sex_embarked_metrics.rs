use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use eml_tree_search::{ValueSignature, sex_embarked_domain};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const NEGATIVE_COUNTS: [u32; 6] = [66, 38, 364, 9, 9, 63];
const POSITIVE_COUNTS: [u32; 6] = [29, 3, 77, 64, 27, 142];

#[derive(Clone, Debug)]
struct RankedCandidate {
    expr: String,
    auc: f64,
    logloss: f64,
}

fn main() {
    let samples = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1_000_000);

    let exact = build_exact_with_exprs(4);
    let union_le_4 = build_union(&exact, 4);
    let union_le_3 = build_union(&exact, 3);

    let distinct_behaviors_le_4 = union_le_4.len();
    let exact_frontier = rank_union(&union_le_4);
    let sampled_height_5 = sample_height_five(&exact, &union_le_3, &union_le_4, samples, 20_260_420);

    println!("Distinct behaviors on Titanic rows, height <= 4: {distinct_behaviors_le_4}");
    println!("Exact top 10 by ROC AUC for height <= 4:");
    print_ranked(&exact_frontier.0);
    println!();
    println!("Exact top 10 by logloss for height <= 4:");
    print_ranked(&exact_frontier.1);
    println!();
    println!("Height-5 sampled search:");
    println!("  samples: {samples}");
    println!("  unique sampled signatures: {}", sampled_height_5.unique_sampled);
    println!("  best sampled by ROC AUC:");
    print_ranked(&sampled_height_5.top_auc);
    println!();
    println!("  best sampled by logloss:");
    print_ranked(&sampled_height_5.top_logloss);
}

fn print_ranked(rows: &[RankedCandidate]) {
    for (index, row) in rows.iter().enumerate() {
        println!(
            "  {}. auc={:.16}, logloss={:.16}, expr={}",
            index + 1,
            row.auc,
            row.logloss,
            row.expr
        );
    }
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

fn rank_union(
    signatures: &HashMap<ValueSignature, String>,
) -> (Vec<RankedCandidate>, Vec<RankedCandidate>) {
    let mut rows = Vec::with_capacity(signatures.len());
    for (signature, expr) in signatures {
        rows.push(RankedCandidate {
            expr: expr.clone(),
            auc: auc(signature),
            logloss: logloss(signature),
        });
    }

    let mut by_auc = rows.clone();
    by_auc.sort_by(compare_by_auc);
    by_auc.truncate(10);

    let mut by_logloss = rows;
    by_logloss.sort_by(compare_by_logloss);
    by_logloss.truncate(10);

    (by_auc, by_logloss)
}

fn compare_by_auc(left: &RankedCandidate, right: &RankedCandidate) -> Ordering {
    right
        .auc
        .total_cmp(&left.auc)
        .then_with(|| left.logloss.total_cmp(&right.logloss))
        .then_with(|| left.expr.cmp(&right.expr))
}

fn compare_by_logloss(left: &RankedCandidate, right: &RankedCandidate) -> Ordering {
    left.logloss
        .total_cmp(&right.logloss)
        .then_with(|| right.auc.total_cmp(&left.auc))
        .then_with(|| left.expr.cmp(&right.expr))
}

struct SampleSummary {
    unique_sampled: usize,
    top_auc: Vec<RankedCandidate>,
    top_logloss: Vec<RankedCandidate>,
}

fn sample_height_five(
    exact: &[HashMap<ValueSignature, String>],
    union_le_3: &HashMap<ValueSignature, String>,
    union_le_4: &HashMap<ValueSignature, String>,
    samples: usize,
    seed: u64,
) -> SampleSummary {
    let left_exact_4: Vec<(ValueSignature, String)> = exact[4]
        .iter()
        .filter(|(signature, _)| signature.is_left_safe())
        .map(|(signature, expr)| (signature.clone(), expr.clone()))
        .collect();
    let right_exact_4: Vec<(ValueSignature, String)> = exact[4]
        .iter()
        .filter(|(signature, _)| signature.is_right_safe())
        .map(|(signature, expr)| (signature.clone(), expr.clone()))
        .collect();
    let left_union_le_3: Vec<(ValueSignature, String)> = union_le_3
        .iter()
        .filter(|(signature, _)| signature.is_left_safe())
        .map(|(signature, expr)| (signature.clone(), expr.clone()))
        .collect();
    let right_union_le_4: Vec<(ValueSignature, String)> = union_le_4
        .iter()
        .filter(|(signature, _)| signature.is_right_safe())
        .map(|(signature, expr)| (signature.clone(), expr.clone()))
        .collect();

    let count_a = (left_exact_4.len() as u128) * (right_union_le_4.len() as u128);
    let count_b = (left_union_le_3.len() as u128) * (right_exact_4.len() as u128);
    let threshold = (count_a as f64) / ((count_a + count_b) as f64);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut seen = HashSet::new();
    let mut rows = Vec::new();

    for _ in 0..samples {
        let (left_signature, left_expr, right_signature, right_expr) = if rng.gen::<f64>() < threshold {
            let (left_signature, left_expr) = &left_exact_4[rng.gen_range(0..left_exact_4.len())];
            let (right_signature, right_expr) =
                &right_union_le_4[rng.gen_range(0..right_union_le_4.len())];
            (left_signature, left_expr, right_signature, right_expr)
        } else {
            let (left_signature, left_expr) =
                &left_union_le_3[rng.gen_range(0..left_union_le_3.len())];
            let (right_signature, right_expr) =
                &right_exact_4[rng.gen_range(0..right_exact_4.len())];
            (left_signature, left_expr, right_signature, right_expr)
        };

        if let Some(signature) = left_signature.eml(right_signature) {
            if !seen.insert(signature.clone()) {
                continue;
            }
            rows.push(RankedCandidate {
                expr: format!("({left_expr} {right_expr})"),
                auc: auc(&signature),
                logloss: logloss(&signature),
            });
        }
    }

    let mut top_auc = rows.clone();
    top_auc.sort_by(compare_by_auc);
    top_auc.truncate(10);

    let mut top_logloss = rows;
    top_logloss.sort_by(compare_by_logloss);
    top_logloss.truncate(10);

    SampleSummary {
        unique_sampled: seen.len(),
        top_auc,
        top_logloss,
    }
}
