use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use csv::StringRecord;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
struct GroupedData {
    x0: Box<[f64]>,
    x1: Box<[f64]>,
    positives: Box<[u32]>,
    negatives: Box<[u32]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Expr {
    One,
    X0,
    X1,
    Node(Box<Expr>, Box<Expr>),
}

#[derive(Clone)]
struct EvaluatedExpr {
    expr: Expr,
    expr_string: String,
    height: usize,
    size: usize,
    logloss: f64,
    auc: f64,
    calibrated_logloss: f64,
    calibration_scale: f64,
    calibration_bias: f64,
}

#[derive(Deserialize)]
struct PairwiseTop10Report {
    feature_pairs: Vec<PairTop10>,
}

#[derive(Deserialize)]
struct PairTop10 {
    feature_a: String,
    feature_b: String,
    top10: Vec<PairTop10Row>,
}

#[derive(Deserialize)]
struct PairTop10Row {
    expr: String,
    logloss: f64,
    auc: f64,
}

#[derive(Serialize)]
struct GaRunReport {
    feature_a: String,
    feature_b: String,
    objective: Objective,
    grouped_point_count: usize,
    population_size: usize,
    generations: usize,
    max_height: usize,
    seed: u64,
    run_tag: String,
    seeded_expressions: Vec<String>,
    best_expression: String,
    best_height: usize,
    best_size: usize,
    best_logloss: f64,
    best_auc: f64,
    best_calibrated_logloss: f64,
    best_calibration_scale: f64,
    best_calibration_bias: f64,
    history: Vec<GenerationSummary>,
}

#[derive(Serialize)]
struct GenerationSummary {
    generation: usize,
    best_expression: String,
    best_logloss: f64,
    best_auc: f64,
    best_calibrated_logloss: f64,
    best_height: usize,
    best_size: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::from_env();
    let train_path = args
        .repo_root
        .join("data")
        .join("interim")
        .join("titanic_unit_interval_train.csv");
    let pairwise_report_path = args
        .repo_root
        .join("data")
        .join("processed")
        .join("pairwise_top10_expressions_height_le_3_logloss.json");

    let rows = load_rows(&train_path)?;
    let pairwise_report: PairwiseTop10Report =
        serde_json::from_str(&fs::read_to_string(&pairwise_report_path)?)?;
    let chosen_pair = choose_pair(
        &pairwise_report,
        args.feature_a.as_deref(),
        args.feature_b.as_deref(),
        args.objective,
    );
    let grouped = group_pair(&rows, &chosen_pair.feature_a, &chosen_pair.feature_b)?;
    let mut seeded_expressions: Vec<String> = chosen_pair
        .top10
        .iter()
        .map(|row| row.expr.clone())
        .collect();
    seeded_expressions.extend(args.seed_expressions.iter().cloned());

    let run_tag = args
        .run_tag
        .clone()
        .unwrap_or_else(|| format!("{:?}_seed{}", args.objective, args.seed).to_lowercase());
    let (best, history) = run_ga(&grouped, &seeded_expressions, &args);

    let output_path = args.repo_root.join("data").join("processed").join(format!(
        "ga_best_expression_{}__{}__{}__{}.json",
        chosen_pair.feature_a,
        chosen_pair.feature_b,
        run_tag,
        match args.objective {
            Objective::Logloss => "logloss",
            Objective::Auc => "auc",
            Objective::CalibratedLogloss => "calibrated_logloss",
            Objective::AucCalibratedLogloss => "auc_calibrated_logloss",
        }
    ));

    let report = GaRunReport {
        feature_a: chosen_pair.feature_a.clone(),
        feature_b: chosen_pair.feature_b.clone(),
        objective: args.objective,
        grouped_point_count: grouped.x0.len(),
        population_size: args.population_size,
        generations: args.generations,
        max_height: args.max_height,
        seed: args.seed,
        run_tag: run_tag.clone(),
        seeded_expressions,
        best_expression: best.expr_string.clone(),
        best_height: best.height,
        best_size: best.size,
        best_logloss: best.logloss,
        best_auc: best.auc,
        best_calibrated_logloss: best.calibrated_logloss,
        best_calibration_scale: best.calibration_scale,
        best_calibration_bias: best.calibration_bias,
        history,
    };

    fs::create_dir_all(output_path.parent().expect("processed dir should exist"))?;
    fs::write(&output_path, serde_json::to_string_pretty(&report)?)?;

    println!("objective: {:?}", args.objective);
    println!("feature pair: {} x {}", chosen_pair.feature_a, chosen_pair.feature_b);
    println!("best logloss: {:.16}", best.logloss);
    println!("best auc: {:.16}", best.auc);
    println!(
        "best calibrated logloss: {:.16} (a={:.16}, b={:.16})",
        best.calibrated_logloss, best.calibration_scale, best.calibration_bias
    );
    println!("best expr: {}", best.expr_string);
    println!("{}", output_path.display());
    Ok(())
}

struct Args {
    repo_root: PathBuf,
    feature_a: Option<String>,
    feature_b: Option<String>,
    seed_expressions: Vec<String>,
    population_size: usize,
    generations: usize,
    max_height: usize,
    seed: u64,
    objective: Objective,
    run_tag: Option<String>,
}

impl Args {
    fn from_env() -> Self {
        let mut args = std::env::args().skip(1);
        let mut repo_root = PathBuf::from("/home/tarstars/prj/titanic_mle");
        let mut feature_a = None;
        let mut feature_b = None;
        let mut seed_expressions = Vec::new();
        let mut population_size = 256;
        let mut generations = 150;
        let mut max_height = 5;
        let mut seed = 20_260_420;
        let mut objective = Objective::Logloss;
        let mut run_tag = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--repo-root" => repo_root = PathBuf::from(args.next().expect("missing repo root value")),
                "--feature-a" => feature_a = Some(args.next().expect("missing feature-a value")),
                "--feature-b" => feature_b = Some(args.next().expect("missing feature-b value")),
                "--seed-expression" => {
                    seed_expressions.push(args.next().expect("missing seed-expression value"));
                }
                "--population" => {
                    population_size = args.next().expect("missing population value").parse().expect("invalid population")
                }
                "--generations" => {
                    generations = args.next().expect("missing generations value").parse().expect("invalid generations")
                }
                "--max-height" => {
                    max_height = args.next().expect("missing max-height value").parse().expect("invalid max-height")
                }
                "--seed" => seed = args.next().expect("missing seed value").parse().expect("invalid seed"),
                "--objective" => {
                    objective = Objective::parse(&args.next().expect("missing objective value"));
                }
                "--run-tag" => run_tag = Some(args.next().expect("missing run-tag value")),
                _ => {}
            }
        }

        if feature_a.is_some() ^ feature_b.is_some() {
            panic!("feature-a and feature-b must be provided together");
        }

        Self {
            repo_root,
            feature_a,
            feature_b,
            seed_expressions,
            population_size,
            generations,
            max_height,
            seed,
            objective,
            run_tag,
        }
    }
}

#[derive(Clone, Copy, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
enum Objective {
    Logloss,
    Auc,
    CalibratedLogloss,
    AucCalibratedLogloss,
}

impl Objective {
    fn parse(value: &str) -> Self {
        match value {
            "logloss" => Self::Logloss,
            "auc" | "roc_auc" => Self::Auc,
            "calibrated-logloss" | "calibrated_logloss" => Self::CalibratedLogloss,
            "auc-calibrated-logloss" | "auc_calibrated_logloss" => Self::AucCalibratedLogloss,
            _ => panic!("unknown objective: {value}"),
        }
    }
}

fn choose_pair<'a>(
    report: &'a PairwiseTop10Report,
    feature_a: Option<&str>,
    feature_b: Option<&str>,
    objective: Objective,
) -> &'a PairTop10 {
    if let (Some(feature_a), Some(feature_b)) = (feature_a, feature_b) {
        return report
            .feature_pairs
            .iter()
            .find(|pair| pair.feature_a == feature_a && pair.feature_b == feature_b)
            .or_else(|| {
                report
                    .feature_pairs
                    .iter()
                    .find(|pair| pair.feature_a == feature_b && pair.feature_b == feature_a)
            })
            .expect("requested feature pair not found in pairwise report");
    }

    report
        .feature_pairs
        .iter()
        .min_by(|left, right| match objective {
            Objective::Logloss | Objective::CalibratedLogloss => left.top10[0]
                .logloss
                .total_cmp(&right.top10[0].logloss)
                .then_with(|| left.feature_a.cmp(&right.feature_a))
                .then_with(|| left.feature_b.cmp(&right.feature_b)),
            Objective::Auc | Objective::AucCalibratedLogloss => right
                .top10
                .iter()
                .map(|row| row.auc)
                .fold(f64::NEG_INFINITY, f64::max)
                .total_cmp(
                    &left
                        .top10
                        .iter()
                        .map(|row| row.auc)
                        .fold(f64::NEG_INFINITY, f64::max),
                )
                .then_with(|| left.top10[0].logloss.total_cmp(&right.top10[0].logloss))
                .then_with(|| left.feature_a.cmp(&right.feature_a))
                .then_with(|| left.feature_b.cmp(&right.feature_b)),
        })
        .expect("pairwise report should contain at least one pair")
}

fn load_rows(path: &PathBuf) -> Result<Vec<StringRecord>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut rows = Vec::new();
    for row in reader.records() {
        rows.push(row?);
    }
    Ok(rows)
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
        x0: grouped_entries.iter().map(|entry| entry.0).collect::<Vec<_>>().into_boxed_slice(),
        x1: grouped_entries.iter().map(|entry| entry.1).collect::<Vec<_>>().into_boxed_slice(),
        positives: grouped_entries.iter().map(|entry| entry.2).collect::<Vec<_>>().into_boxed_slice(),
        negatives: grouped_entries.iter().map(|entry| entry.3).collect::<Vec<_>>().into_boxed_slice(),
    })
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
    logloss_with_affine(values, grouped, 1.0, 0.0)
}

fn logloss_with_affine(values: &[f64], grouped: &GroupedData, scale: f64, bias: f64) -> f64 {
    let mut total_loss = 0.0;
    let mut total_count = 0u32;
    for (index, value) in values.iter().enumerate() {
        let probability = sigmoid(scale * *value + bias).clamp(1e-15, 1.0 - 1e-15);
        total_loss += -(grouped.positives[index] as f64) * probability.ln();
        total_loss += -(grouped.negatives[index] as f64) * (1.0 - probability).ln();
        total_count += grouped.positives[index] + grouped.negatives[index];
    }
    total_loss / (total_count as f64)
}

fn calibrated_logloss(values: &[f64], grouped: &GroupedData) -> (f64, f64, f64) {
    let mut scale = 1.0;
    let mut bias = 0.0;
    let mut best_loss = logloss_with_affine(values, grouped, scale, bias);

    for _ in 0..25 {
        let mut g_scale = 0.0;
        let mut g_bias = 0.0;
        let mut h_ss = 0.0;
        let mut h_sb = 0.0;
        let mut h_bb = 0.0;

        for (index, value) in values.iter().enumerate() {
            let positives = grouped.positives[index] as f64;
            let negatives = grouped.negatives[index] as f64;
            let weight = positives + negatives;
            let probability = sigmoid(scale * *value + bias);
            let error = weight * probability - positives;
            let variance = weight * probability * (1.0 - probability);
            g_scale += error * *value;
            g_bias += error;
            h_ss += variance * *value * *value;
            h_sb += variance * *value;
            h_bb += variance;
        }

        let determinant = h_ss * h_bb - h_sb * h_sb;
        if determinant <= 1e-18 {
            break;
        }

        let step_scale = (h_bb * g_scale - h_sb * g_bias) / determinant;
        let step_bias = (-h_sb * g_scale + h_ss * g_bias) / determinant;

        let mut step_factor = 1.0;
        let mut improved = None;
        while step_factor > 1e-8 {
            let candidate_scale = scale - step_factor * step_scale;
            let candidate_bias = bias - step_factor * step_bias;
            if candidate_scale <= 1e-10 {
                step_factor *= 0.5;
                continue;
            }
            let candidate_loss =
                logloss_with_affine(values, grouped, candidate_scale, candidate_bias);
            if candidate_loss < best_loss - 1e-12 {
                improved = Some((candidate_scale, candidate_bias, candidate_loss, step_factor));
                break;
            }
            step_factor *= 0.5;
        }

        let Some((candidate_scale, candidate_bias, candidate_loss, accepted_step_factor)) = improved
        else {
            break;
        };

        scale = candidate_scale;
        bias = candidate_bias;
        best_loss = candidate_loss;

        if (accepted_step_factor * step_scale).abs() < 1e-10
            && (accepted_step_factor * step_bias).abs() < 1e-10
        {
            break;
        }
    }

    (best_loss, scale, bias)
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

fn evaluate_expr(expr: &Expr, grouped: &GroupedData) -> Option<Vec<f64>> {
    match expr {
        Expr::One => Some(vec![1.0; grouped.x0.len()]),
        Expr::X0 => Some(grouped.x0.to_vec()),
        Expr::X1 => Some(grouped.x1.to_vec()),
        Expr::Node(left, right) => {
            let left_values = evaluate_expr(left, grouped)?;
            let right_values = evaluate_expr(right, grouped)?;
            let mut out = Vec::with_capacity(left_values.len());
            for (left_value, right_value) in left_values.iter().zip(right_values.iter()) {
                if *right_value <= 0.0 {
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
            Some(out)
        }
    }
}

fn expr_height(expr: &Expr) -> usize {
    match expr {
        Expr::One | Expr::X0 | Expr::X1 => 0,
        Expr::Node(left, right) => 1 + expr_height(left).max(expr_height(right)),
    }
}

fn expr_size(expr: &Expr) -> usize {
    match expr {
        Expr::One | Expr::X0 | Expr::X1 => 1,
        Expr::Node(left, right) => 1 + expr_size(left) + expr_size(right),
    }
}

fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::One => "1".to_string(),
        Expr::X0 => "x0".to_string(),
        Expr::X1 => "x1".to_string(),
        Expr::Node(left, right) => format!("({} {})", expr_to_string(left), expr_to_string(right)),
    }
}

fn parse_expr(input: &str) -> Result<Expr, String> {
    fn parse(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Result<Expr, String> {
        while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
            chars.next();
        }

        match chars.peek().copied() {
            Some('1') => {
                chars.next();
                Ok(Expr::One)
            }
            Some('x') => {
                chars.next();
                match chars.next() {
                    Some('0') => Ok(Expr::X0),
                    Some('1') => Ok(Expr::X1),
                    _ => Err("unknown variable".to_string()),
                }
            }
            Some('(') => {
                chars.next();
                let left = parse(chars)?;
                let right = parse(chars)?;
                while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
                    chars.next();
                }
                match chars.next() {
                    Some(')') => Ok(Expr::Node(Box::new(left), Box::new(right))),
                    _ => Err("missing closing ')'".to_string()),
                }
            }
            _ => Err("unexpected token".to_string()),
        }
    }

    let mut chars = input.chars().peekable();
    let expr = parse(&mut chars)?;
    while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
        chars.next();
    }
    if chars.peek().is_some() {
        return Err("trailing input".to_string());
    }
    Ok(expr)
}

fn random_leaf(rng: &mut StdRng) -> Expr {
    match rng.gen_range(0..3) {
        0 => Expr::One,
        1 => Expr::X0,
        _ => Expr::X1,
    }
}

fn random_tree(rng: &mut StdRng, max_height: usize) -> Expr {
    if max_height == 0 || rng.gen_bool(0.35) {
        return random_leaf(rng);
    }

    let left_height = rng.gen_range(0..max_height);
    let right_height = rng.gen_range(0..max_height);
    Expr::Node(
        Box::new(random_tree(rng, left_height)),
        Box::new(random_tree(rng, right_height)),
    )
}

fn all_paths(expr: &Expr) -> Vec<Vec<bool>> {
    fn walk(expr: &Expr, path: &mut Vec<bool>, out: &mut Vec<Vec<bool>>) {
        out.push(path.clone());
        if let Expr::Node(left, right) = expr {
            path.push(false);
            walk(left, path, out);
            path.pop();

            path.push(true);
            walk(right, path, out);
            path.pop();
        }
    }

    let mut out = Vec::new();
    let mut path = Vec::new();
    walk(expr, &mut path, &mut out);
    out
}

fn subtree_at(expr: &Expr, path: &[bool]) -> Expr {
    if path.is_empty() {
        return expr.clone();
    }

    match expr {
        Expr::Node(left, right) => {
            if !path[0] {
                subtree_at(left, &path[1..])
            } else {
                subtree_at(right, &path[1..])
            }
        }
        _ => expr.clone(),
    }
}

fn replace_subtree(expr: &Expr, path: &[bool], replacement: &Expr) -> Expr {
    if path.is_empty() {
        return replacement.clone();
    }

    match expr {
        Expr::Node(left, right) => {
            if !path[0] {
                Expr::Node(
                    Box::new(replace_subtree(left, &path[1..], replacement)),
                    Box::new((**right).clone()),
                )
            } else {
                Expr::Node(
                    Box::new((**left).clone()),
                    Box::new(replace_subtree(right, &path[1..], replacement)),
                )
            }
        }
        _ => replacement.clone(),
    }
}

fn swap_random_children(expr: &Expr, rng: &mut StdRng) -> Expr {
    let internal_paths: Vec<_> = all_paths(expr)
        .into_iter()
        .filter(|path| matches!(subtree_at(expr, path), Expr::Node(_, _)))
        .collect();
    if internal_paths.is_empty() {
        return expr.clone();
    }
    let path = &internal_paths[rng.gen_range(0..internal_paths.len())];
    let subtree = subtree_at(expr, path);
    if let Expr::Node(left, right) = subtree {
        let swapped = Expr::Node(right, left);
        replace_subtree(expr, path, &swapped)
    } else {
        expr.clone()
    }
}

fn mutate(expr: &Expr, max_height: usize, rng: &mut StdRng) -> Expr {
    for _ in 0..32 {
        let paths = all_paths(expr);
        let path = &paths[rng.gen_range(0..paths.len())];
        let candidate = match rng.gen_range(0..5) {
            0 => replace_subtree(expr, path, &random_leaf(rng)),
            1 => {
                let subtree = subtree_at(expr, path);
                let wrapped = if rng.gen_bool(0.5) {
                    Expr::Node(Box::new(subtree), Box::new(random_tree(rng, 1)))
                } else {
                    Expr::Node(Box::new(random_tree(rng, 1)), Box::new(subtree))
                };
                replace_subtree(expr, path, &wrapped)
            }
            2 => replace_subtree(expr, path, &random_tree(rng, 2)),
            3 => swap_random_children(expr, rng),
            _ => {
                let subtree = subtree_at(expr, path);
                match subtree {
                    Expr::Node(left, _) => replace_subtree(expr, path, &left),
                    _ => replace_subtree(expr, path, &random_leaf(rng)),
                }
            }
        };
        if expr_height(&candidate) <= max_height {
            return candidate;
        }
    }
    expr.clone()
}

fn crossover(left: &Expr, right: &Expr, max_height: usize, rng: &mut StdRng) -> Expr {
    for _ in 0..32 {
        let left_paths = all_paths(left);
        let right_paths = all_paths(right);
        let left_path = &left_paths[rng.gen_range(0..left_paths.len())];
        let right_path = &right_paths[rng.gen_range(0..right_paths.len())];
        let right_subtree = subtree_at(right, right_path);
        let candidate = replace_subtree(left, left_path, &right_subtree);
        if expr_height(&candidate) <= max_height {
            return candidate;
        }
    }
    left.clone()
}

fn compare_eval(left: &EvaluatedExpr, right: &EvaluatedExpr, objective: Objective) -> Ordering {
    match objective {
        Objective::Logloss => left
            .logloss
            .total_cmp(&right.logloss)
            .then_with(|| right.auc.total_cmp(&left.auc))
            .then_with(|| left.calibrated_logloss.total_cmp(&right.calibrated_logloss))
            .then_with(|| left.height.cmp(&right.height))
            .then_with(|| left.size.cmp(&right.size))
            .then_with(|| left.expr_string.cmp(&right.expr_string)),
        Objective::Auc => right
            .auc
            .total_cmp(&left.auc)
            .then_with(|| left.logloss.total_cmp(&right.logloss))
            .then_with(|| left.calibrated_logloss.total_cmp(&right.calibrated_logloss))
            .then_with(|| left.height.cmp(&right.height))
            .then_with(|| left.size.cmp(&right.size))
            .then_with(|| left.expr_string.cmp(&right.expr_string)),
        Objective::AucCalibratedLogloss => right
            .auc
            .total_cmp(&left.auc)
            .then_with(|| left.calibrated_logloss.total_cmp(&right.calibrated_logloss))
            .then_with(|| left.logloss.total_cmp(&right.logloss))
            .then_with(|| left.height.cmp(&right.height))
            .then_with(|| left.size.cmp(&right.size))
            .then_with(|| left.expr_string.cmp(&right.expr_string)),
        Objective::CalibratedLogloss => left
            .calibrated_logloss
            .total_cmp(&right.calibrated_logloss)
            .then_with(|| right.auc.total_cmp(&left.auc))
            .then_with(|| left.logloss.total_cmp(&right.logloss))
            .then_with(|| left.height.cmp(&right.height))
            .then_with(|| left.size.cmp(&right.size))
            .then_with(|| left.expr_string.cmp(&right.expr_string)),
    }
}

fn evaluate_population_member(
    expr: Expr,
    grouped: &GroupedData,
    cache: &mut HashMap<String, EvaluatedExpr>,
) -> Option<EvaluatedExpr> {
    let expr_string = expr_to_string(&expr);
    if let Some(cached) = cache.get(&expr_string) {
        return Some(cached.clone());
    }

    let values = evaluate_expr(&expr, grouped)?;
    let (calibrated_logloss, calibration_scale, calibration_bias) =
        calibrated_logloss(&values, grouped);
    let evaluated = EvaluatedExpr {
        expr,
        expr_string: expr_string.clone(),
        height: 0,
        size: 0,
        logloss: logloss(&values, grouped),
        auc: auc(&values, grouped),
        calibrated_logloss,
        calibration_scale,
        calibration_bias,
    };
    let evaluated = EvaluatedExpr {
        height: expr_height(&evaluated.expr),
        size: expr_size(&evaluated.expr),
        ..evaluated
    };
    cache.insert(expr_string, evaluated.clone());
    Some(evaluated)
}

fn tournament_select<'a>(
    population: &'a [EvaluatedExpr],
    rng: &mut StdRng,
    objective: Objective,
) -> &'a EvaluatedExpr {
    let mut best = &population[rng.gen_range(0..population.len())];
    for _ in 0..3 {
        let candidate = &population[rng.gen_range(0..population.len())];
        if compare_eval(candidate, best, objective) == Ordering::Less {
            best = candidate;
        }
    }
    best
}

fn run_ga(
    grouped: &GroupedData,
    seeded_expressions: &[String],
    args: &Args,
) -> (EvaluatedExpr, Vec<GenerationSummary>) {
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut cache = HashMap::new();
    let mut seen_population = HashSet::new();
    let mut population = Vec::new();

    for expr_text in seeded_expressions {
        if let Ok(expr) = parse_expr(expr_text) {
            if expr_height(&expr) <= args.max_height {
                if let Some(evaluated) = evaluate_population_member(expr, grouped, &mut cache) {
                    if seen_population.insert(evaluated.expr_string.clone()) {
                        population.push(evaluated);
                    }
                }
            }
        }
    }

    while population.len() < args.population_size {
        let expr = if !population.is_empty() && rng.gen_bool(0.6) {
            let seed = &population[rng.gen_range(0..population.len())].expr;
            mutate(seed, args.max_height, &mut rng)
        } else {
            random_tree(&mut rng, args.max_height)
        };
        if let Some(evaluated) = evaluate_population_member(expr, grouped, &mut cache) {
            if seen_population.insert(evaluated.expr_string.clone()) {
                population.push(evaluated);
            }
        }
    }

    let mut history = Vec::new();
    for generation in 0..args.generations {
        population.sort_by(|left, right| compare_eval(left, right, args.objective));
        let best = population[0].clone();
        history.push(GenerationSummary {
            generation,
            best_expression: best.expr_string.clone(),
            best_logloss: best.logloss,
            best_auc: best.auc,
            best_calibrated_logloss: best.calibrated_logloss,
            best_height: best.height,
            best_size: best.size,
        });
        eprintln!(
            "generation {:>3}: logloss={:.6} calibrated_logloss={:.6} auc={:.6} height={} size={} expr={}",
            generation,
            best.logloss,
            best.calibrated_logloss,
            best.auc,
            best.height,
            best.size,
            best.expr_string
        );

        let elite_count = ((args.population_size as f64) * 0.10).ceil() as usize;
        let elite_count = elite_count.max(4).min(population.len());
        let mut next_population = population[..elite_count].to_vec();
        let mut seen_next: HashSet<_> = next_population
            .iter()
            .map(|row| row.expr_string.clone())
            .collect();

        while next_population.len() < args.population_size {
            let expr = match rng.gen_range(0..100) {
                0..=44 => {
                    let parent = tournament_select(&population, &mut rng, args.objective);
                    mutate(&parent.expr, args.max_height, &mut rng)
                }
                45..=89 => {
                    let left = tournament_select(&population, &mut rng, args.objective);
                    let right = tournament_select(&population, &mut rng, args.objective);
                    crossover(&left.expr, &right.expr, args.max_height, &mut rng)
                }
                _ => random_tree(&mut rng, args.max_height),
            };

            if let Some(evaluated) = evaluate_population_member(expr, grouped, &mut cache) {
                if seen_next.insert(evaluated.expr_string.clone()) {
                    next_population.push(evaluated);
                }
            }
        }

        population = next_population;
    }

    population.sort_by(|left, right| compare_eval(left, right, args.objective));
    let best = population[0].clone();
    history.push(GenerationSummary {
        generation: args.generations,
        best_expression: best.expr_string.clone(),
        best_logloss: best.logloss,
        best_auc: best.auc,
        best_calibrated_logloss: best.calibrated_logloss,
        best_height: best.height,
        best_size: best.size,
    });
    (best, history)
}
