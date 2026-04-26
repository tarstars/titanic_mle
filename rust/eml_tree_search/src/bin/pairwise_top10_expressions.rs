use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use csv::StringRecord;
use serde::Serialize;

const HEIGHT_MAX: usize = 3;
const TOP_K: usize = 10;

#[derive(Clone)]
struct Candidate {
    values: Box<[f64]>,
    expr: String,
    height: usize,
}

#[derive(Clone)]
struct GroupedData {
    feature_a: String,
    feature_b: String,
    x0: Box<[f64]>,
    x1: Box<[f64]>,
    positives: Box<[u32]>,
    negatives: Box<[u32]>,
}

#[derive(Clone)]
struct RankedCandidate {
    expr: String,
    height: usize,
    logloss: f64,
    auc: f64,
}

struct PairSearchResult {
    distinct_behavior_count: usize,
    top10: Vec<RankedCandidate>,
}

#[derive(Serialize)]
struct PairwiseTop10Report {
    search_height_max: usize,
    ranking_metric: &'static str,
    feature_pairs: Vec<PairTop10>,
}

#[derive(Serialize)]
struct PairTop10 {
    feature_a: String,
    feature_b: String,
    grouped_point_count: usize,
    distinct_behaviors_le_height_max: usize,
    top10: Vec<PairTop10Row>,
}

#[derive(Serialize)]
struct PairTop10Row {
    rank: usize,
    expr: String,
    height: usize,
    logloss: f64,
    auc: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let repo_root = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/home/tarstars/prj/titanic_mle".to_string()),
    );
    let train_path = repo_root
        .join("data")
        .join("interim")
        .join("titanic_unit_interval_train.csv");
    let output_json = repo_root
        .join("data")
        .join("processed")
        .join("pairwise_top10_expressions_height_le_3_logloss.json");
    let output_csv = repo_root
        .join("data")
        .join("processed")
        .join("pairwise_top10_expressions_height_le_3_logloss.csv");

    let rows = load_rows(&train_path)?;
    let feature_names = infer_feature_names(&rows[0]);

    let mut pair_reports = Vec::new();
    let mut csv_rows = Vec::new();

    for left_index in 0..feature_names.len() {
        for right_index in (left_index + 1)..feature_names.len() {
            let grouped = group_pair(
                &rows,
                &feature_names[left_index],
                &feature_names[right_index],
            )?;
            let ranked = search_pair_top10(&grouped);

            for (rank, row) in ranked.top10.iter().enumerate() {
                csv_rows.push(format!(
                    "{},{},{},{},{:.16},{:.16},\"{}\"",
                    grouped.feature_a,
                    grouped.feature_b,
                    rank + 1,
                    row.height,
                    row.logloss,
                    row.auc,
                    row.expr.replace('\"', "\"\""),
                ));
            }

            pair_reports.push(PairTop10 {
                feature_a: grouped.feature_a.clone(),
                feature_b: grouped.feature_b.clone(),
                grouped_point_count: grouped.x0.len(),
                distinct_behaviors_le_height_max: ranked.distinct_behavior_count,
                top10: ranked
                    .top10
                    .iter()
                    .enumerate()
                    .map(|(rank, row)| PairTop10Row {
                        rank: rank + 1,
                        expr: row.expr.clone(),
                        height: row.height,
                        logloss: row.logloss,
                        auc: row.auc,
                    })
                    .collect(),
            });
            eprintln!(
                "processed pair {:>16} x {:<16} grouped_points={} top1_logloss={:.6}",
                grouped.feature_a,
                grouped.feature_b,
                grouped.x0.len(),
                ranked.top10[0].logloss,
            );
        }
    }

    pair_reports.sort_by(|left, right| {
        left.feature_a
            .cmp(&right.feature_a)
            .then_with(|| left.feature_b.cmp(&right.feature_b))
    });

    let report = PairwiseTop10Report {
        search_height_max: HEIGHT_MAX,
        ranking_metric: "logloss",
        feature_pairs: pair_reports,
    };

    fs::create_dir_all(output_json.parent().expect("processed dir should exist"))?;
    fs::write(&output_json, serde_json::to_string_pretty(&report)?)?;

    let mut csv_output = String::from("feature_a,feature_b,rank,height,logloss,auc,expr\n");
    for row in csv_rows {
        csv_output.push_str(&row);
        csv_output.push('\n');
    }
    fs::write(&output_csv, csv_output)?;

    println!("{}", output_json.display());
    println!("{}", output_csv.display());
    Ok(())
}

fn load_rows(path: &PathBuf) -> Result<Vec<StringRecord>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut rows = Vec::new();
    for row in reader.records() {
        rows.push(row?);
    }
    Ok(rows)
}

fn infer_feature_names(_first_row: &StringRecord) -> Vec<String> {
    vec![
        "pclass_unit".to_string(),
        "sex_unit".to_string(),
        "age_unit".to_string(),
        "age_missing".to_string(),
        "sibsp_unit".to_string(),
        "parch_unit".to_string(),
        "fare_unit".to_string(),
        "fare_missing".to_string(),
        "embarked_unit".to_string(),
        "embarked_missing".to_string(),
        "cabin_known".to_string(),
        "family_size_unit".to_string(),
        "is_alone".to_string(),
    ]
}

fn feature_index(name: &str) -> usize {
    match name {
        "PassengerId" => 0,
        "pclass_unit" => 1,
        "sex_unit" => 2,
        "age_unit" => 3,
        "age_missing" => 4,
        "sibsp_unit" => 5,
        "parch_unit" => 6,
        "fare_unit" => 7,
        "fare_missing" => 8,
        "embarked_unit" => 9,
        "embarked_missing" => 10,
        "cabin_known" => 11,
        "family_size_unit" => 12,
        "is_alone" => 13,
        "Survived" => 14,
        _ => panic!("unknown feature name: {name}"),
    }
}

fn group_pair(
    rows: &[StringRecord],
    feature_a: &str,
    feature_b: &str,
) -> Result<GroupedData, Box<dyn Error>> {
    let index_a = feature_index(feature_a);
    let index_b = feature_index(feature_b);
    let survived_index = feature_index("Survived");

    let mut grouped: HashMap<(u64, u64), (f64, f64, u32, u32)> = HashMap::new();
    for row in rows {
        let x0 = row[index_a].parse::<f64>()?;
        let x1 = row[index_b].parse::<f64>()?;
        let survived = row[survived_index].parse::<u32>()?;
        let key = (x0.to_bits(), x1.to_bits());
        let entry = grouped.entry(key).or_insert((x0, x1, 0, 0));
        if survived == 1 {
            entry.2 += 1;
        } else {
            entry.3 += 1;
        }
    }

    let mut grouped_entries: Vec<_> = grouped.into_values().collect();
    grouped_entries.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });

    Ok(GroupedData {
        feature_a: feature_a.to_string(),
        feature_b: feature_b.to_string(),
        x0: grouped_entries.iter().map(|entry| entry.0).collect::<Vec<_>>().into_boxed_slice(),
        x1: grouped_entries.iter().map(|entry| entry.1).collect::<Vec<_>>().into_boxed_slice(),
        positives: grouped_entries.iter().map(|entry| entry.2).collect::<Vec<_>>().into_boxed_slice(),
        negatives: grouped_entries.iter().map(|entry| entry.3).collect::<Vec<_>>().into_boxed_slice(),
    })
}

fn bits(values: &[f64]) -> Box<[u64]> {
    values
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn one_signature(len: usize) -> Box<[f64]> {
    vec![1.0; len].into_boxed_slice()
}

fn exact_height_maps(grouped: &GroupedData) -> Vec<HashMap<Box<[u64]>, Candidate>> {
    let mut exact = Vec::new();

    let mut base = HashMap::new();
    let one = one_signature(grouped.x0.len());
    let x0 = grouped.x0.clone();
    let x1 = grouped.x1.clone();
    base.insert(
        bits(&one),
        Candidate {
            values: one,
            expr: "1".to_string(),
            height: 0,
        },
    );
    base.insert(
        bits(&x0),
        Candidate {
            values: x0,
            expr: "x0".to_string(),
            height: 0,
        },
    );
    base.insert(
        bits(&x1),
        Candidate {
            values: x1,
            expr: "x1".to_string(),
            height: 0,
        },
    );
    exact.push(base);

    for height in 1..=HEIGHT_MAX {
        let mut lower = HashMap::new();
        for counter in exact.iter().take(height) {
            for (signature, candidate) in counter {
                lower
                    .entry(signature.clone())
                    .or_insert_with(|| candidate.clone());
            }
        }

        let previous_exact = &exact[height - 1];
        let lower_entries: Vec<_> = lower.into_iter().collect();
        let mut current = HashMap::new();

        for (left_signature, left_candidate) in &lower_entries {
            let left_is_previous = previous_exact.contains_key(left_signature);
            for (right_signature, right_candidate) in &lower_entries {
                if !(left_is_previous || previous_exact.contains_key(right_signature)) {
                    continue;
                }
                if let Some(values) = eml_signature(&left_candidate.values, &right_candidate.values) {
                    let key = bits(&values);
                    current.entry(key).or_insert_with(|| Candidate {
                        values,
                        expr: format!("({} {})", left_candidate.expr, right_candidate.expr),
                        height,
                    });
                }
            }
        }

        exact.push(current);
    }

    exact
}

fn eml_signature(left: &[f64], right: &[f64]) -> Option<Box<[f64]>> {
    let mut out = Vec::with_capacity(left.len());
    for (&left_value, &right_value) in left.iter().zip(right.iter()) {
        if right_value <= 0.0 {
            return None;
        }
        let exp_left = left_value.exp();
        let ln_right = right_value.ln();
        let value = exp_left - ln_right;
        if !exp_left.is_finite() || !ln_right.is_finite() || !value.is_finite() {
            return None;
        }
        out.push(value);
    }
    Some(out.into_boxed_slice())
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

fn logloss(values: &[f64], grouped: &GroupedData) -> f64 {
    let mut total_loss = 0.0;
    let mut total_count = 0u32;
    for (index, value) in values.iter().enumerate() {
        let probability = sigmoid(*value).clamp(1e-15, 1.0 - 1e-15);
        total_loss += -(grouped.positives[index] as f64) * probability.ln();
        total_loss += -(grouped.negatives[index] as f64) * (1.0 - probability).ln();
        total_count += grouped.positives[index] + grouped.negatives[index];
    }
    total_loss / (total_count as f64)
}

fn auc(values: &[f64], grouped: &GroupedData) -> f64 {
    let mut wins = 0.0;
    let positives: u32 = grouped.positives.iter().sum();
    let negatives: u32 = grouped.negatives.iter().sum();
    for i in 0..values.len() {
        for j in 0..values.len() {
            let pair_count = (grouped.positives[i] * grouped.negatives[j]) as f64;
            if values[i] > values[j] {
                wins += pair_count;
            } else if values[i].to_bits() == values[j].to_bits() {
                wins += 0.5 * pair_count;
            }
        }
    }
    wins / ((positives * negatives) as f64)
}

fn compare_ranked(left: &RankedCandidate, right: &RankedCandidate) -> Ordering {
    left.logloss
        .total_cmp(&right.logloss)
        .then_with(|| right.auc.total_cmp(&left.auc))
        .then_with(|| left.height.cmp(&right.height))
        .then_with(|| left.expr.cmp(&right.expr))
}

fn search_pair_top10(grouped: &GroupedData) -> PairSearchResult {
    let exact = exact_height_maps(grouped);
    let mut union = HashMap::new();
    for counter in &exact {
        for (signature, candidate) in counter {
            union.entry(signature.clone()).or_insert_with(|| candidate.clone());
        }
    }

    let distinct_behavior_count = union.len();
    let mut ranked = Vec::with_capacity(distinct_behavior_count);
    for candidate in union.into_values() {
        ranked.push(RankedCandidate {
            expr: candidate.expr,
            height: candidate.height,
            logloss: logloss(&candidate.values, grouped),
            auc: 0.0,
        });
    }

    ranked.sort_by(compare_ranked);
    ranked.truncate(TOP_K);
    for row in &mut ranked {
        let signature = expression_values(grouped, &row.expr).expect("top expression should evaluate");
        row.auc = auc(&signature, grouped);
    }
    ranked.sort_by(compare_ranked);
    PairSearchResult {
        distinct_behavior_count,
        top10: ranked,
    }
}

fn expression_values(grouped: &GroupedData, expr: &str) -> Option<Vec<f64>> {
    fn parse_expr<'a>(
        chars: &mut std::iter::Peekable<std::str::Chars<'a>>,
        grouped: &GroupedData,
    ) -> Option<Vec<f64>> {
        while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
            chars.next();
        }
        match chars.peek().copied()? {
            '1' => {
                chars.next();
                Some(vec![1.0; grouped.x0.len()])
            }
            'x' => {
                chars.next();
                match chars.next()? {
                    '0' => Some(grouped.x0.to_vec()),
                    '1' => Some(grouped.x1.to_vec()),
                    _ => None,
                }
            }
            '(' => {
                chars.next();
                let left = parse_expr(chars, grouped)?;
                while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
                    chars.next();
                }
                let right = parse_expr(chars, grouped)?;
                while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
                    chars.next();
                }
                if chars.next()? != ')' {
                    return None;
                }
                let mut out = Vec::with_capacity(left.len());
                for (left_value, right_value) in left.iter().zip(right.iter()) {
                    if *right_value <= 0.0 {
                        return None;
                    }
                    let value = left_value.exp() - right_value.ln();
                    if !value.is_finite() {
                        return None;
                    }
                    out.push(value);
                }
                Some(out)
            }
            _ => None,
        }
    }

    let mut chars = expr.chars().peekable();
    parse_expr(&mut chars, grouped)
}
