use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use csv::StringRecord;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
struct GaReport {
    feature_a: String,
    feature_b: String,
    best_expression: String,
    best_logloss: f64,
    best_auc: f64,
}

#[derive(Clone)]
struct TerminalSource {
    id: &'static str,
    report_file: &'static str,
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

#[derive(Clone, Debug)]
enum MetaExpr {
    Leaf(usize),
    Node(Arc<MetaExpr>, Arc<MetaExpr>),
}

#[derive(Clone)]
struct CandidateResult {
    expr_string: String,
    height: usize,
    size: usize,
    logloss: f64,
    auc: f64,
    calibrated_logloss: f64,
    calibration_scale: f64,
    calibration_bias: f64,
}

#[derive(Serialize)]
struct SearchReport {
    experiment: String,
    max_height: usize,
    total_expressions: usize,
    valid_expressions: usize,
    terminal_count: usize,
    terminals: Vec<TerminalSummary>,
    best_by_logloss: RankedExpr,
    best_by_auc: RankedExpr,
    top10_by_logloss: Vec<RankedExpr>,
    top10_by_auc: Vec<RankedExpr>,
}

#[derive(Serialize)]
struct RankedExpr {
    expr: String,
    height: usize,
    size: usize,
    logloss: f64,
    auc: f64,
    calibrated_logloss: f64,
    calibration_scale: f64,
    calibration_bias: f64,
}

struct Args {
    repo_root: PathBuf,
    max_height: usize,
}

impl Args {
    fn from_env() -> Self {
        let mut args = std::env::args().skip(1);
        let mut repo_root = PathBuf::from("/home/tarstars/prj/titanic_mle");
        let mut max_height = 3;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--repo-root" => repo_root = PathBuf::from(args.next().expect("missing repo-root value")),
                "--max-height" => {
                    max_height = args.next().expect("missing max-height value").parse().expect("invalid max-height")
                }
                _ => {}
            }
        }

        Self { repo_root, max_height }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::from_env();
    let terminal_sources = [
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
    ];

    let train_path = args
        .repo_root
        .join("data")
        .join("interim")
        .join("titanic_unit_interval_train.csv");
    let rows = load_rows(&train_path)?;
    let labels: Vec<u8> = rows
        .iter()
        .map(|row| row[feature_index("Survived")].parse::<u8>())
        .collect::<Result<_, _>>()?;

    let terminals = load_terminal_library(&args.repo_root, &rows, &labels, &terminal_sources)?;
    let dataset = build_dataset(&terminals, &labels);

    let (total_expressions, valid_expressions, top_logloss, top_auc) =
        exact_search(&dataset, &terminals, args.max_height);

    let report = SearchReport {
        experiment: "stacked_meta_terminals_exact_height_le_3".to_string(),
        max_height: args.max_height,
        total_expressions,
        valid_expressions,
        terminal_count: terminals.len(),
        terminals: terminals.iter().map(|terminal| terminal.summary.clone()).collect(),
        best_by_logloss: ranked(&top_logloss[0]),
        best_by_auc: ranked(&top_auc[0]),
        top10_by_logloss: top_logloss.iter().take(10).map(ranked).collect(),
        top10_by_auc: top_auc.iter().take(10).map(ranked).collect(),
    };

    let output_path = args
        .repo_root
        .join("data")
        .join("processed")
        .join(format!(
            "meta_stacked_exact_search_top{}_height_le_{}.json",
            terminals.len(),
            args.max_height
        ));
    fs::write(&output_path, serde_json::to_string_pretty(&report)?)?;

    println!("total expressions: {}", total_expressions);
    println!("valid expressions: {}", valid_expressions);
    println!("threads: {}", configured_thread_count());
    println!(
        "best logloss: {:.16} auc={:.16} calibrated_logloss={:.16}",
        top_logloss[0].logloss, top_logloss[0].auc, top_logloss[0].calibrated_logloss
    );
    println!("best logloss expr: {}", top_logloss[0].expr_string);
    println!(
        "best auc: {:.16} logloss={:.16} calibrated_logloss={:.16}",
        top_auc[0].auc, top_auc[0].logloss, top_auc[0].calibrated_logloss
    );
    println!("best auc expr: {}", top_auc[0].expr_string);
    println!("{}", output_path.display());
    Ok(())
}

fn ranked(result: &CandidateResult) -> RankedExpr {
    RankedExpr {
        expr: result.expr_string.clone(),
        height: result.height,
        size: result.size,
        logloss: result.logloss,
        auc: result.auc,
        calibrated_logloss: result.calibrated_logloss,
        calibration_scale: result.calibration_scale,
        calibration_bias: result.calibration_bias,
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
        let expr = parse_base_expr(&report.best_expression)?;
        let mut scores = Vec::with_capacity(rows.len());
        for row in rows {
            let x0 = row[feature_index(&report.feature_a)].parse::<f64>()?;
            let x1 = row[feature_index(&report.feature_b)].parse::<f64>()?;
            let value = eval_base_expr(&expr, x0, x1)
                .ok_or_else(|| format!("saved report expression is invalid on train rows: {}", source.report_file))?;
            scores.push(value);
        }
        let (calibrated_logloss_value, scale, bias) = calibrated_logloss(&scores, labels);
        let probabilities: Vec<f64> = scores
            .iter()
            .map(|score| sigmoid(scale * *score + bias).clamp(1e-15, 1.0 - 1e-15))
            .collect();

        terminals.push(TerminalData {
            summary: TerminalSummary {
                id: source.id.to_string(),
                source_report: source.report_file.to_string(),
                feature_a: report.feature_a.clone(),
                feature_b: report.feature_b.clone(),
                source_expression: report.best_expression.clone(),
                source_logloss: report.best_logloss,
                source_auc: report.best_auc,
                fitted_calibration_scale: scale,
                fitted_calibration_bias: bias,
                calibrated_logloss: calibrated_logloss_value,
                calibrated_auc: auc(&probabilities, labels),
            },
            probabilities,
        });
    }
    Ok(terminals)
}

fn build_dataset(terminals: &[TerminalData], labels: &[u8]) -> Dataset {
    let terminal_values = terminals
        .iter()
        .map(|terminal| terminal.probabilities.clone())
        .collect();
    Dataset {
        terminal_values,
        labels: labels.to_vec(),
    }
}

fn exact_search(
    dataset: &Dataset,
    terminals: &[TerminalData],
    max_height: usize,
) -> (usize, usize, Vec<CandidateResult>, Vec<CandidateResult>) {
    let mut exact_levels: Vec<Vec<Arc<MetaExpr>>> = Vec::new();
    let leaves: Vec<Arc<MetaExpr>> = (0..terminals.len())
        .map(|index| Arc::new(MetaExpr::Leaf(index)))
        .collect();
    exact_levels.push(leaves.clone());

    let mut all_exprs = leaves.len();
    let mut valid_exprs = 0usize;
    let mut top_logloss = Vec::new();
    let mut top_auc = Vec::new();

    let leaf_results = evaluate_expr_batch(&leaves, dataset, terminals);
    valid_exprs += leaf_results.len();
    for result in leaf_results {
        push_top_logloss(&mut top_logloss, result.clone());
        push_top_auc(&mut top_auc, result);
    }

    for height in 1..=max_height {
        let (exact_level, level_results, level_total) =
            build_and_evaluate_exact_level(height, &exact_levels, dataset, terminals);
        all_exprs += level_total;
        valid_exprs += level_results.len();
        for result in level_results {
            push_top_logloss(&mut top_logloss, result.clone());
            push_top_auc(&mut top_auc, result);
        }
        exact_levels.push(exact_level);
    }

    top_logloss.sort_by(compare_logloss);
    top_auc.sort_by(compare_auc);
    (all_exprs, valid_exprs, top_logloss, top_auc)
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

fn evaluate_expr_batch(
    exprs: &[Arc<MetaExpr>],
    dataset: &Dataset,
    terminals: &[TerminalData],
) -> Vec<CandidateResult> {
    let thread_count = configured_thread_count().min(exprs.len().max(1));
    if thread_count <= 1 || exprs.len() < 2048 {
        return exprs
            .iter()
            .filter_map(|expr| evaluate_meta_expr(expr.clone(), dataset, terminals))
            .collect();
    }

    let chunk_count = thread_count * 4;
    let chunk_size = ((exprs.len() + chunk_count - 1) / chunk_count).max(1);

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in exprs.chunks(chunk_size) {
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .filter_map(|expr| evaluate_meta_expr(expr.clone(), dataset, terminals))
                    .collect::<Vec<_>>()
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            results.extend(handle.join().expect("worker thread failed"));
        }
        results
    })
}

fn build_and_evaluate_exact_level(
    height: usize,
    exact_levels: &[Vec<Arc<MetaExpr>>],
    dataset: &Dataset,
    terminals: &[TerminalData],
) -> (Vec<Arc<MetaExpr>>, Vec<CandidateResult>, usize) {
    let thread_count = configured_thread_count();
    let mut total_exprs = 0usize;

    #[derive(Clone, Copy)]
    struct Job {
        left_height: usize,
        right_height: usize,
        start: usize,
        end: usize,
    }

    let mut jobs = Vec::new();
    for left_height in 0..height {
        for right_height in 0..height {
            if left_height.max(right_height) != height - 1 {
                continue;
            }
            let lefts = &exact_levels[left_height];
            let rights = &exact_levels[right_height];
            let pair_total = lefts.len() * rights.len();
            total_exprs += pair_total;

            let chunk_count = (thread_count * 2).max(1);
            let chunk_size = ((lefts.len() + chunk_count - 1) / chunk_count).max(1);
            for start in (0..lefts.len()).step_by(chunk_size) {
                let end = (start + chunk_size).min(lefts.len());
                jobs.push(Job {
                    left_height,
                    right_height,
                    start,
                    end,
                });
            }
        }
    }

    if thread_count <= 1 || jobs.len() <= 1 {
        let mut exact_level = Vec::with_capacity(total_exprs);
        let mut level_results = Vec::new();
        for job in jobs {
            let lefts = &exact_levels[job.left_height][job.start..job.end];
            let rights = &exact_levels[job.right_height];
            for left in lefts {
                for right in rights {
                    let expr = Arc::new(MetaExpr::Node(left.clone(), right.clone()));
                    exact_level.push(expr.clone());
                    if let Some(result) = evaluate_meta_expr(expr, dataset, terminals) {
                        level_results.push(result);
                    }
                }
            }
        }
        return (exact_level, level_results, total_exprs);
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for job in jobs {
            handles.push(scope.spawn(move || {
                let lefts = &exact_levels[job.left_height][job.start..job.end];
                let rights = &exact_levels[job.right_height];
                let mut local_exprs = Vec::with_capacity(lefts.len() * rights.len());
                let mut local_results = Vec::new();
                for left in lefts {
                    for right in rights {
                        let expr = Arc::new(MetaExpr::Node(left.clone(), right.clone()));
                        local_exprs.push(expr.clone());
                        if let Some(result) = evaluate_meta_expr(expr, dataset, terminals) {
                            local_results.push(result);
                        }
                    }
                }
                (local_exprs, local_results)
            }));
        }

        let mut exact_level = Vec::with_capacity(total_exprs);
        let mut level_results = Vec::new();
        for handle in handles {
            let (mut local_exprs, mut local_results) =
                handle.join().expect("worker thread failed");
            exact_level.append(&mut local_exprs);
            level_results.append(&mut local_results);
        }
        (exact_level, level_results, total_exprs)
    })
}

fn evaluate_meta_expr(
    expr: Arc<MetaExpr>,
    dataset: &Dataset,
    terminals: &[TerminalData],
) -> Option<CandidateResult> {
    let values = eval_meta_expr(&expr, dataset)?;
    let logloss_value = logloss(&values, &dataset.labels);
    let auc_value = auc(&values, &dataset.labels);
    let (calibrated_logloss_value, scale, bias) = calibrated_logloss(&values, &dataset.labels);
    Some(CandidateResult {
        expr_string: meta_expr_to_string(&expr, terminals),
        height: meta_expr_height(&expr),
        size: meta_expr_size(&expr),
        logloss: logloss_value,
        auc: auc_value,
        calibrated_logloss: calibrated_logloss_value,
        calibration_scale: scale,
        calibration_bias: bias,
    })
}

fn eval_meta_expr(expr: &MetaExpr, dataset: &Dataset) -> Option<Vec<f64>> {
    match expr {
        MetaExpr::Leaf(index) => Some(dataset.terminal_values[*index].clone()),
        MetaExpr::Node(left, right) => {
            let left_values = eval_meta_expr(left, dataset)?;
            let right_values = eval_meta_expr(right, dataset)?;
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

fn meta_expr_to_string(expr: &MetaExpr, terminals: &[TerminalData]) -> String {
    match expr {
        MetaExpr::Leaf(index) => terminals[*index].summary.id.clone(),
        MetaExpr::Node(left, right) => format!(
            "({} {})",
            meta_expr_to_string(left, terminals),
            meta_expr_to_string(right, terminals)
        ),
    }
}

fn meta_expr_height(expr: &MetaExpr) -> usize {
    match expr {
        MetaExpr::Leaf(_) => 0,
        MetaExpr::Node(left, right) => 1 + meta_expr_height(left).max(meta_expr_height(right)),
    }
}

fn meta_expr_size(expr: &MetaExpr) -> usize {
    match expr {
        MetaExpr::Leaf(_) => 1,
        MetaExpr::Node(left, right) => 1 + meta_expr_size(left) + meta_expr_size(right),
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

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let ez = (-z).exp();
        1.0 / (1.0 + ez)
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn compare_logloss(left: &CandidateResult, right: &CandidateResult) -> std::cmp::Ordering {
    left.logloss
        .total_cmp(&right.logloss)
        .then_with(|| right.auc.total_cmp(&left.auc))
        .then_with(|| left.calibrated_logloss.total_cmp(&right.calibrated_logloss))
        .then_with(|| left.height.cmp(&right.height))
        .then_with(|| left.size.cmp(&right.size))
        .then_with(|| left.expr_string.cmp(&right.expr_string))
}

fn compare_auc(left: &CandidateResult, right: &CandidateResult) -> std::cmp::Ordering {
    right
        .auc
        .total_cmp(&left.auc)
        .then_with(|| left.logloss.total_cmp(&right.logloss))
        .then_with(|| left.calibrated_logloss.total_cmp(&right.calibrated_logloss))
        .then_with(|| left.height.cmp(&right.height))
        .then_with(|| left.size.cmp(&right.size))
        .then_with(|| left.expr_string.cmp(&right.expr_string))
}

fn push_top_logloss(top: &mut Vec<CandidateResult>, candidate: CandidateResult) {
    top.push(candidate);
    top.sort_by(compare_logloss);
    if top.len() > 10 {
        top.truncate(10);
    }
}

fn push_top_auc(top: &mut Vec<CandidateResult>, candidate: CandidateResult) {
    top.push(candidate);
    top.sort_by(compare_auc);
    if top.len() > 10 {
        top.truncate(10);
    }
}

fn load_rows(path: &Path) -> Result<Vec<StringRecord>, Box<dyn Error>> {
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
