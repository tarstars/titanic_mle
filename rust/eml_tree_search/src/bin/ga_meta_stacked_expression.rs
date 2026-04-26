use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use csv::StringRecord;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
struct GaReport {
    feature_a: String,
    feature_b: String,
    best_expression: String,
}

#[derive(Clone)]
struct TerminalSource {
    id: &'static str,
    report_file: &'static str,
}

#[derive(Deserialize)]
struct ExternalTerminalLibrary {
    terminal_library: Vec<ExternalTerminalEntry>,
}

#[derive(Clone, Deserialize)]
struct ExternalTerminalEntry {
    id: String,
    feature_a: String,
    feature_b: String,
    source_expression: String,
    source_report: Option<String>,
}

#[derive(Clone, Serialize)]
struct TerminalSummary {
    id: String,
    source_report: String,
    feature_a: String,
    feature_b: String,
    source_expression: String,
    source_logloss: f64,
    source_auc: f64,
    fitted_calibration_scale: f64,
    fitted_calibration_bias: f64,
    calibrated_logloss: f64,
    calibrated_auc: f64,
}

#[derive(Clone)]
struct TerminalData {
    summary: TerminalSummary,
    probabilities: Vec<f64>,
}

#[derive(Clone)]
struct Dataset {
    terminal_values: Vec<Vec<f64>>,
    labels: Vec<u8>,
}

#[derive(Clone, Debug)]
enum BaseExpr {
    X0,
    X1,
    One,
    Node(Box<BaseExpr>, Box<BaseExpr>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MetaExpr {
    Leaf(usize),
    Node(Box<MetaExpr>, Box<MetaExpr>),
}

#[derive(Clone)]
struct EvaluatedExpr {
    expr: MetaExpr,
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
struct MetaSearchSeedReport {
    top10_by_logloss: Vec<RankedSeed>,
    top10_by_auc: Vec<RankedSeed>,
}

#[derive(Deserialize)]
struct RankedSeed {
    expr: String,
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

#[derive(Serialize)]
struct GaRunReport {
    experiment: String,
    objective: Objective,
    terminal_count: usize,
    population_size: usize,
    generations: usize,
    max_height: usize,
    seed: u64,
    run_tag: String,
    seed_report: String,
    seeded_expressions: Vec<String>,
    terminals: Vec<TerminalSummary>,
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

struct Args {
    repo_root: PathBuf,
    population_size: usize,
    generations: usize,
    max_height: usize,
    seed: u64,
    objective: Objective,
    run_tag: Option<String>,
    seed_expressions: Vec<String>,
    terminal_library: Option<PathBuf>,
}

impl Args {
    fn from_env() -> Self {
        let mut args = std::env::args().skip(1);
        let mut repo_root = PathBuf::from("/home/tarstars/prj/titanic_mle");
        let mut population_size = 256;
        let mut generations = 180;
        let mut max_height = 5;
        let mut seed = 20_260_420;
        let mut objective = Objective::CalibratedLogloss;
        let mut run_tag = None;
        let mut seed_expressions = Vec::new();
        let mut terminal_library = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--repo-root" => repo_root = PathBuf::from(args.next().expect("missing repo-root value")),
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
                "--seed-expression" => {
                    seed_expressions.push(args.next().expect("missing seed-expression value"));
                }
                "--terminal-library" => {
                    terminal_library = Some(PathBuf::from(
                        args.next().expect("missing terminal-library value"),
                    ));
                }
                _ => {}
            }
        }

        Self {
            repo_root,
            population_size,
            generations,
            max_height,
            seed,
            objective,
            run_tag,
            seed_expressions,
            terminal_library,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::from_env();
    let terminal_sources = terminal_sources();
    let train_path = args
        .repo_root
        .join("data")
        .join("interim")
        .join("titanic_unit_interval_train.csv");
    let seed_report_path = args
        .repo_root
        .join("data")
        .join("processed")
        .join("meta_stacked_exact_search_top5_height_le_3.json");

    let rows = load_rows(&train_path)?;
    let labels: Vec<u8> = rows
        .iter()
        .map(|row| row[feature_index("Survived")].parse::<u8>())
        .collect::<Result<_, _>>()?;
    let terminals = if let Some(path) = &args.terminal_library {
        load_external_terminal_library(path, &rows, &labels)?
    } else {
        load_terminal_library(&args.repo_root, &rows, &labels, &terminal_sources)?
    };
    let dataset = build_dataset(&terminals, &labels);
    let seed_report: MetaSearchSeedReport =
        serde_json::from_str(&fs::read_to_string(&seed_report_path)?)?;

    let mut seeded_expressions = Vec::new();
    seeded_expressions.extend(
        seed_report
            .top10_by_logloss
            .into_iter()
            .map(|row| row.expr),
    );
    seeded_expressions.extend(seed_report.top10_by_auc.into_iter().map(|row| row.expr));
    seeded_expressions.extend(terminals.iter().map(|terminal| terminal.summary.id.clone()));
    seeded_expressions.extend(args.seed_expressions.iter().cloned());

    let run_tag = args
        .run_tag
        .clone()
        .unwrap_or_else(|| format!("{:?}_seed{}", args.objective, args.seed).to_lowercase());
    let (best, history) = run_ga(&dataset, &terminals, &seeded_expressions, &args);

    let output_path = args.repo_root.join("data").join("processed").join(format!(
        "ga_meta_stacked_top{}__{}__{}.json",
        terminals.len(),
        run_tag,
        match args.objective {
            Objective::Logloss => "logloss",
            Objective::Auc => "auc",
            Objective::CalibratedLogloss => "calibrated_logloss",
            Objective::AucCalibratedLogloss => "auc_calibrated_logloss",
        }
    ));

    let report = GaRunReport {
        experiment: "ga_meta_stacked_expression".to_string(),
        objective: args.objective,
        terminal_count: terminals.len(),
        population_size: args.population_size,
        generations: args.generations,
        max_height: args.max_height,
        seed: args.seed,
        run_tag: run_tag.clone(),
        seed_report: seed_report_path
            .strip_prefix(&args.repo_root)
            .unwrap_or(&seed_report_path)
            .display()
            .to_string(),
        seeded_expressions,
        terminals: terminals.iter().map(|terminal| terminal.summary.clone()).collect(),
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

    fs::write(&output_path, serde_json::to_string_pretty(&report)?)?;

    println!("objective: {:?}", args.objective);
    println!("terminals: {}", terminals.len());
    println!("threads: {}", configured_thread_count());
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

fn terminal_sources() -> [TerminalSource; 5] {
    [
        TerminalSource {
            id: "ps_prob",
            report_file: "ga_best_expression_pclass_unit__sex_unit__iter4_seeded__calibrated_logloss.json",
        },
        TerminalSource {
            id: "ps_log",
            report_file: "ga_best_expression_pclass_unit__sex_unit__iter2__logloss.json",
        },
        TerminalSource {
            id: "sf_auc",
            report_file: "ga_best_expression_sex_unit__fare_unit__iter2__auc.json",
        },
        TerminalSource {
            id: "sa_log",
            report_file: "ga_best_expression_sex_unit__age_unit__iter1__logloss.json",
        },
        TerminalSource {
            id: "sf_prob",
            report_file: "ga_best_expression_sex_unit__fare_unit__iter3__calibrated_logloss.json",
        },
    ]
}

fn load_rows(path: &Path) -> Result<Vec<StringRecord>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut rows = Vec::new();
    for row in reader.records() {
        rows.push(row?);
    }
    Ok(rows)
}

fn build_dataset(terminals: &[TerminalData], labels: &[u8]) -> Dataset {
    Dataset {
        terminal_values: terminals
            .iter()
            .map(|terminal| terminal.probabilities.clone())
            .collect(),
        labels: labels.to_vec(),
    }
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

fn load_terminal_library(
    repo_root: &Path,
    rows: &[StringRecord],
    labels: &[u8],
    sources: &[TerminalSource],
) -> Result<Vec<TerminalData>, Box<dyn Error>> {
    let mut terminals = Vec::new();
    for source in sources {
        let report_path = repo_root.join("data").join("processed").join(source.report_file);
        let report: GaReport = serde_json::from_str(&fs::read_to_string(&report_path)?)?;
        terminals.push(build_terminal_data(
            source.id.to_string(),
            source.report_file.to_string(),
            report.feature_a,
            report.feature_b,
            report.best_expression,
            rows,
            labels,
        )?);
    }
    Ok(terminals)
}

fn load_external_terminal_library(
    library_path: &Path,
    rows: &[StringRecord],
    labels: &[u8],
) -> Result<Vec<TerminalData>, Box<dyn Error>> {
    let library: ExternalTerminalLibrary = serde_json::from_str(&fs::read_to_string(library_path)?)?;
    let mut terminals = Vec::new();
    for entry in library.terminal_library {
        terminals.push(build_terminal_data(
            entry.id,
            entry
                .source_report
                .unwrap_or_else(|| library_path.display().to_string()),
            entry.feature_a,
            entry.feature_b,
            entry.source_expression,
            rows,
            labels,
        )?);
    }
    Ok(terminals)
}

fn build_terminal_data(
    id: String,
    source_report: String,
    feature_a: String,
    feature_b: String,
    source_expression: String,
    rows: &[StringRecord],
    labels: &[u8],
) -> Result<TerminalData, Box<dyn Error>> {
    let expr = parse_base_expr(&source_expression)?;
    let mut scores = Vec::with_capacity(rows.len());
    for row in rows {
        let x0 = row[feature_index(&feature_a)].parse::<f64>()?;
        let x1 = row[feature_index(&feature_b)].parse::<f64>()?;
        let value = eval_base_expr(&expr, x0, x1)
            .ok_or_else(|| format!("terminal expression is invalid on train rows: {source_expression}"))?;
        scores.push(value);
    }
    let raw_logloss = logloss(&scores, labels);
    let raw_auc = auc(&scores, labels);
    let (calibrated_logloss_value, scale, bias) = calibrated_logloss(&scores, labels);
    let probabilities: Vec<f64> = scores
        .iter()
        .map(|score| sigmoid(scale * *score + bias).clamp(1e-15, 1.0 - 1e-15))
        .collect();
    Ok(TerminalData {
        summary: TerminalSummary {
            id,
            source_report,
            feature_a,
            feature_b,
            source_expression,
            source_logloss: raw_logloss,
            source_auc: raw_auc,
            fitted_calibration_scale: scale,
            fitted_calibration_bias: bias,
            calibrated_logloss: calibrated_logloss_value,
            calibrated_auc: auc(&probabilities, labels),
        },
        probabilities,
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

fn logloss(values: &[f64], labels: &[u8]) -> f64 {
    logloss_with_affine(values, labels, 1.0, 0.0)
}

fn logloss_with_affine(values: &[f64], labels: &[u8], scale: f64, bias: f64) -> f64 {
    let mut total_loss = 0.0;
    for (value, label) in values.iter().zip(labels.iter()) {
        let probability = sigmoid(scale * *value + bias).clamp(1e-15, 1.0 - 1e-15);
        total_loss += if *label == 1 {
            -probability.ln()
        } else {
            -(1.0 - probability).ln()
        };
    }
    total_loss / (values.len() as f64)
}

fn calibrated_logloss(values: &[f64], labels: &[u8]) -> (f64, f64, f64) {
    let mut scale = 1.0;
    let mut bias = 0.0;
    let mut best_loss = logloss_with_affine(values, labels, scale, bias);

    for _ in 0..25 {
        let mut g_scale = 0.0;
        let mut g_bias = 0.0;
        let mut h_ss = 0.0;
        let mut h_sb = 0.0;
        let mut h_bb = 0.0;

        for (value, label) in values.iter().zip(labels.iter()) {
            let probability = sigmoid(scale * *value + bias);
            let target = f64::from(*label);
            let error = probability - target;
            let variance = probability * (1.0 - probability);
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
            let candidate_loss = logloss_with_affine(values, labels, candidate_scale, candidate_bias);
            if candidate_loss < best_loss - 1e-12 {
                improved = Some((candidate_scale, candidate_bias, candidate_loss, step_factor));
                break;
            }
            step_factor *= 0.5;
        }

        let Some((candidate_scale, candidate_bias, candidate_loss, accepted_step_factor)) = improved else {
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

fn auc(values: &[f64], labels: &[u8]) -> f64 {
    let mut pairs: Vec<(f64, u8)> = values.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));

    let positives = labels.iter().filter(|label| **label == 1).count() as f64;
    let negatives = (labels.len() as f64) - positives;
    if positives == 0.0 || negatives == 0.0 {
        return 0.5;
    }

    let mut i = 0usize;
    let mut negatives_before = 0.0;
    let mut wins = 0.0;

    while i < pairs.len() {
        let score_bits = pairs[i].0.to_bits();
        let mut group_positives = 0.0;
        let mut group_negatives = 0.0;
        while i < pairs.len() && pairs[i].0.to_bits() == score_bits {
            if pairs[i].1 == 1 {
                group_positives += 1.0;
            } else {
                group_negatives += 1.0;
            }
            i += 1;
        }
        wins += group_positives * negatives_before;
        wins += 0.5 * group_positives * group_negatives;
        negatives_before += group_negatives;
    }

    wins / (positives * negatives)
}

fn eval_base_expr(expr: &BaseExpr, x0: f64, x1: f64) -> Option<f64> {
    match expr {
        BaseExpr::X0 => Some(x0),
        BaseExpr::X1 => Some(x1),
        BaseExpr::One => Some(1.0),
        BaseExpr::Node(left, right) => {
            let left_value = eval_base_expr(left, x0, x1)?;
            let right_value = eval_base_expr(right, x0, x1)?;
            if right_value <= 0.0 {
                return None;
            }
            let exp_left = left_value.exp();
            let ln_right = right_value.ln();
            let value = exp_left - ln_right;
            if !exp_left.is_finite() || !ln_right.is_finite() || !value.is_finite() {
                return None;
            }
            Some(value)
        }
    }
}

fn evaluate_expr(expr: &MetaExpr, dataset: &Dataset) -> Option<Vec<f64>> {
    match expr {
        MetaExpr::Leaf(index) => Some(dataset.terminal_values[*index].clone()),
        MetaExpr::Node(left, right) => {
            let left_values = evaluate_expr(left, dataset)?;
            let right_values = evaluate_expr(right, dataset)?;
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

fn parse_base_expr(input: &str) -> Result<BaseExpr, String> {
    fn parse(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Result<BaseExpr, String> {
        while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
            chars.next();
        }
        match chars.peek().copied() {
            Some('1') => {
                chars.next();
                Ok(BaseExpr::One)
            }
            Some('x') => {
                chars.next();
                match chars.next() {
                    Some('0') => Ok(BaseExpr::X0),
                    Some('1') => Ok(BaseExpr::X1),
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
                    Some(')') => Ok(BaseExpr::Node(Box::new(left), Box::new(right))),
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

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in input.chars() {
        match ch {
            '(' | ')' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(ch.to_string());
            }
            ch if ch.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn parse_meta_expr(input: &str, terminal_lookup: &HashMap<String, usize>) -> Result<MetaExpr, String> {
    fn parse(tokens: &[String], index: &mut usize, terminal_lookup: &HashMap<String, usize>) -> Result<MetaExpr, String> {
        if *index >= tokens.len() {
            return Err("unexpected end of input".to_string());
        }
        let token = &tokens[*index];
        if token == "(" {
            *index += 1;
            let left = parse(tokens, index, terminal_lookup)?;
            let right = parse(tokens, index, terminal_lookup)?;
            if *index >= tokens.len() || tokens[*index] != ")" {
                return Err("missing closing ')'".to_string());
            }
            *index += 1;
            Ok(MetaExpr::Node(Box::new(left), Box::new(right)))
        } else {
            *index += 1;
            terminal_lookup
                .get(token)
                .copied()
                .map(MetaExpr::Leaf)
                .ok_or_else(|| format!("unknown terminal: {token}"))
        }
    }

    let tokens = tokenize(input);
    let mut index = 0;
    let expr = parse(&tokens, &mut index, terminal_lookup)?;
    if index != tokens.len() {
        return Err("trailing input".to_string());
    }
    Ok(expr)
}

fn expr_to_string(expr: &MetaExpr, terminals: &[TerminalData]) -> String {
    match expr {
        MetaExpr::Leaf(index) => terminals[*index].summary.id.clone(),
        MetaExpr::Node(left, right) => format!(
            "({} {})",
            expr_to_string(left, terminals),
            expr_to_string(right, terminals)
        ),
    }
}

fn expr_height(expr: &MetaExpr) -> usize {
    match expr {
        MetaExpr::Leaf(_) => 0,
        MetaExpr::Node(left, right) => 1 + expr_height(left).max(expr_height(right)),
    }
}

fn expr_size(expr: &MetaExpr) -> usize {
    match expr {
        MetaExpr::Leaf(_) => 1,
        MetaExpr::Node(left, right) => 1 + expr_size(left) + expr_size(right),
    }
}

fn random_leaf(rng: &mut StdRng, leaf_count: usize) -> MetaExpr {
    MetaExpr::Leaf(rng.gen_range(0..leaf_count))
}

fn random_tree(rng: &mut StdRng, max_height: usize, leaf_count: usize) -> MetaExpr {
    if max_height == 0 || rng.gen_bool(0.35) {
        return random_leaf(rng, leaf_count);
    }
    let left_height = rng.gen_range(0..max_height);
    let right_height = rng.gen_range(0..max_height);
    MetaExpr::Node(
        Box::new(random_tree(rng, left_height, leaf_count)),
        Box::new(random_tree(rng, right_height, leaf_count)),
    )
}

fn all_paths(expr: &MetaExpr) -> Vec<Vec<bool>> {
    fn walk(expr: &MetaExpr, path: &mut Vec<bool>, out: &mut Vec<Vec<bool>>) {
        out.push(path.clone());
        if let MetaExpr::Node(left, right) = expr {
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

fn subtree_at(expr: &MetaExpr, path: &[bool]) -> MetaExpr {
    if path.is_empty() {
        return expr.clone();
    }
    match expr {
        MetaExpr::Node(left, right) => {
            if !path[0] {
                subtree_at(left, &path[1..])
            } else {
                subtree_at(right, &path[1..])
            }
        }
        _ => expr.clone(),
    }
}

fn replace_subtree(expr: &MetaExpr, path: &[bool], replacement: &MetaExpr) -> MetaExpr {
    if path.is_empty() {
        return replacement.clone();
    }
    match expr {
        MetaExpr::Node(left, right) => {
            if !path[0] {
                MetaExpr::Node(
                    Box::new(replace_subtree(left, &path[1..], replacement)),
                    Box::new((**right).clone()),
                )
            } else {
                MetaExpr::Node(
                    Box::new((**left).clone()),
                    Box::new(replace_subtree(right, &path[1..], replacement)),
                )
            }
        }
        _ => replacement.clone(),
    }
}

fn swap_random_children(expr: &MetaExpr, rng: &mut StdRng) -> MetaExpr {
    let internal_paths: Vec<_> = all_paths(expr)
        .into_iter()
        .filter(|path| matches!(subtree_at(expr, path), MetaExpr::Node(_, _)))
        .collect();
    if internal_paths.is_empty() {
        return expr.clone();
    }
    let path = &internal_paths[rng.gen_range(0..internal_paths.len())];
    let subtree = subtree_at(expr, path);
    if let MetaExpr::Node(left, right) = subtree {
        let swapped = MetaExpr::Node(right, left);
        replace_subtree(expr, path, &swapped)
    } else {
        expr.clone()
    }
}

fn mutate(expr: &MetaExpr, max_height: usize, leaf_count: usize, rng: &mut StdRng) -> MetaExpr {
    for _ in 0..32 {
        let paths = all_paths(expr);
        let path = &paths[rng.gen_range(0..paths.len())];
        let candidate = match rng.gen_range(0..5) {
            0 => replace_subtree(expr, path, &random_leaf(rng, leaf_count)),
            1 => {
                let subtree = subtree_at(expr, path);
                let wrapped = if rng.gen_bool(0.5) {
                    MetaExpr::Node(
                        Box::new(subtree),
                        Box::new(random_tree(rng, 1, leaf_count)),
                    )
                } else {
                    MetaExpr::Node(
                        Box::new(random_tree(rng, 1, leaf_count)),
                        Box::new(subtree),
                    )
                };
                replace_subtree(expr, path, &wrapped)
            }
            2 => replace_subtree(expr, path, &random_tree(rng, 2, leaf_count)),
            3 => swap_random_children(expr, rng),
            _ => {
                let subtree = subtree_at(expr, path);
                match subtree {
                    MetaExpr::Node(left, _) => replace_subtree(expr, path, &left),
                    _ => replace_subtree(expr, path, &random_leaf(rng, leaf_count)),
                }
            }
        };
        if expr_height(&candidate) <= max_height {
            return candidate;
        }
    }
    expr.clone()
}

fn crossover(left: &MetaExpr, right: &MetaExpr, max_height: usize, rng: &mut StdRng) -> MetaExpr {
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

fn configured_thread_count() -> usize {
    std::env::var("EML_SEARCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        })
}

fn evaluate_population_member_uncached(
    expr: MetaExpr,
    dataset: &Dataset,
    _terminals: &[TerminalData],
    expr_string: String,
) -> Option<EvaluatedExpr> {
    let values = evaluate_expr(&expr, dataset)?;
    let (calibrated_logloss_value, calibration_scale, calibration_bias) =
        calibrated_logloss(&values, &dataset.labels);
    let evaluated = EvaluatedExpr {
        expr,
        expr_string,
        height: 0,
        size: 0,
        logloss: logloss(&values, &dataset.labels),
        auc: auc(&values, &dataset.labels),
        calibrated_logloss: calibrated_logloss_value,
        calibration_scale,
        calibration_bias,
    };
    Some(EvaluatedExpr {
        height: expr_height(&evaluated.expr),
        size: expr_size(&evaluated.expr),
        ..evaluated
    })
}

fn evaluate_population_member(
    expr: MetaExpr,
    dataset: &Dataset,
    terminals: &[TerminalData],
    cache: &mut HashMap<String, EvaluatedExpr>,
) -> Option<EvaluatedExpr> {
    let expr_string = expr_to_string(&expr, terminals);
    if let Some(cached) = cache.get(&expr_string) {
        return Some(cached.clone());
    }

    let evaluated = evaluate_population_member_uncached(expr, dataset, terminals, expr_string.clone())?;
    cache.insert(expr_string, evaluated.clone());
    Some(evaluated)
}

fn evaluate_population_batch(
    exprs: Vec<MetaExpr>,
    dataset: &Dataset,
    terminals: &[TerminalData],
    cache: &mut HashMap<String, EvaluatedExpr>,
) -> Vec<EvaluatedExpr> {
    let mut ready = Vec::new();
    let mut missing = Vec::new();
    let mut seen_missing = HashSet::new();

    for expr in exprs {
        let expr_string = expr_to_string(&expr, terminals);
        if let Some(cached) = cache.get(&expr_string) {
            ready.push(cached.clone());
        } else if seen_missing.insert(expr_string.clone()) {
            missing.push((expr, expr_string));
        }
    }

    if missing.is_empty() {
        return ready;
    }

    let thread_count = configured_thread_count().min(missing.len().max(1));
    let mut evaluated_missing = if thread_count <= 1 || missing.len() < 64 {
        missing
            .into_iter()
            .filter_map(|(expr, expr_string)| {
                evaluate_population_member_uncached(expr, dataset, terminals, expr_string)
            })
            .collect::<Vec<_>>()
    } else {
        let chunk_count = thread_count * 4;
        let chunk_size = ((missing.len() + chunk_count - 1) / chunk_count).max(1);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in missing.chunks(chunk_size) {
                handles.push(scope.spawn(move || {
                    chunk
                        .iter()
                        .filter_map(|(expr, expr_string)| {
                            evaluate_population_member_uncached(
                                expr.clone(),
                                dataset,
                                terminals,
                                expr_string.clone(),
                            )
                        })
                        .collect::<Vec<_>>()
                }));
            }

            let mut out = Vec::new();
            for handle in handles {
                out.extend(handle.join().expect("worker thread failed"));
            }
            out
        })
    };

    for evaluated in &evaluated_missing {
        cache.insert(evaluated.expr_string.clone(), evaluated.clone());
    }
    ready.append(&mut evaluated_missing);
    ready
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

fn candidate_batch_size(remaining: usize) -> usize {
    (remaining.saturating_mul(4)).max(64)
}

fn run_ga(
    dataset: &Dataset,
    terminals: &[TerminalData],
    seeded_expressions: &[String],
    args: &Args,
) -> (EvaluatedExpr, Vec<GenerationSummary>) {
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut cache = HashMap::new();
    let mut seen_population = HashSet::new();
    let mut population = Vec::new();
    let terminal_lookup: HashMap<String, usize> = terminals
        .iter()
        .enumerate()
        .map(|(index, terminal)| (terminal.summary.id.clone(), index))
        .collect();

    for expr_text in seeded_expressions {
        if let Ok(expr) = parse_meta_expr(expr_text, &terminal_lookup) {
            if expr_height(&expr) <= args.max_height {
                if let Some(evaluated) = evaluate_population_member(expr, dataset, terminals, &mut cache) {
                    if seen_population.insert(evaluated.expr_string.clone()) {
                        population.push(evaluated);
                    }
                }
            }
        }
    }

    while population.len() < args.population_size {
        let batch_len = candidate_batch_size(args.population_size - population.len());
        let mut expr_batch = Vec::with_capacity(batch_len);
        for _ in 0..batch_len {
            expr_batch.push(if !population.is_empty() && rng.gen_bool(0.6) {
                let seed = &population[rng.gen_range(0..population.len())].expr;
                mutate(seed, args.max_height, terminals.len(), &mut rng)
            } else {
                random_tree(&mut rng, args.max_height, terminals.len())
            });
        }
        for evaluated in evaluate_population_batch(expr_batch, dataset, terminals, &mut cache) {
            if seen_population.insert(evaluated.expr_string.clone()) {
                population.push(evaluated);
                if population.len() >= args.population_size {
                    break;
                }
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
            let batch_len = candidate_batch_size(args.population_size - next_population.len());
            let mut expr_batch = Vec::with_capacity(batch_len);
            for _ in 0..batch_len {
                expr_batch.push(match rng.gen_range(0..100) {
                    0..=44 => {
                        let parent = tournament_select(&population, &mut rng, args.objective);
                        mutate(&parent.expr, args.max_height, terminals.len(), &mut rng)
                    }
                    45..=89 => {
                        let left = tournament_select(&population, &mut rng, args.objective);
                        let right = tournament_select(&population, &mut rng, args.objective);
                        crossover(&left.expr, &right.expr, args.max_height, &mut rng)
                    }
                    _ => random_tree(&mut rng, args.max_height, terminals.len()),
                });
            }

            for evaluated in evaluate_population_batch(expr_batch, dataset, terminals, &mut cache) {
                if seen_next.insert(evaluated.expr_string.clone()) {
                    next_population.push(evaluated);
                    if next_population.len() >= args.population_size {
                        break;
                    }
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
