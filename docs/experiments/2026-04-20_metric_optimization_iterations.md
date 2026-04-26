# Metric Optimization Iterations

## Goal

Push the current search frontier in two different directions:

- lower `logloss`
- higher `ROC AUC`

without waiting for a full multi-feature symbolic search.

## Situation Before This Iteration

From the existing exact pairwise `height <= 3` report:

- `sex_unit × age_unit` was the best top-1 pair by `logloss`
  - seed expression: `((x0 (1 x1)) ((1 1) (1 x1)))`
  - `logloss = 0.531675095952892`
  - `ROC AUC = 0.7618370455586446`
- `pclass_unit × sex_unit` was the best saved pair by `ROC AUC`
  - seed expression: `(x1 ((1 1) (x0 1)))`
  - `logloss = 0.5548208146648084`
  - `ROC AUC = 0.8328353518891339`
- `sex_unit × fare_unit` was the second-best saved pair by `ROC AUC`
  - seed expression: `(x0 ((1 1) (x1 1)))`
  - `logloss = 0.5602550495787212`
  - `ROC AUC = 0.8284387349673515`

The key question was whether deeper objective-specific GA runs would keep the same frontier, or move it.

## Idea

Use separate search modes for separate metrics instead of a single shared GA:

- objective-specific GA for `logloss`
- objective-specific GA for `ROC AUC`
- then monotone post-hoc calibration `sigmoid(a * score + b)` for the best found expressions

The calibration step is important because it preserves ranking when `a > 0`, so it can repair `logloss` without changing `ROC AUC`.

## Commands

`logloss`-oriented run on the best calibration seed:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-best-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --feature-a sex_unit \
  --feature-b age_unit \
  --objective logloss \
  --population 512 \
  --generations 300 \
  --max-height 6 \
  --seed 20260420 \
  --run-tag iter1
```

`ROC AUC`-oriented run on the strongest AUC seed:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-best-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --feature-a pclass_unit \
  --feature-b sex_unit \
  --objective auc \
  --population 512 \
  --generations 300 \
  --max-height 6 \
  --seed 20260420 \
  --run-tag iter1
```

Second iteration on the now-dominant pair for `logloss`:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-best-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --feature-a pclass_unit \
  --feature-b sex_unit \
  --objective logloss \
  --population 512 \
  --generations 300 \
  --max-height 6 \
  --seed 20260421 \
  --run-tag iter2
```

Second iteration to challenge the AUC leader with a more continuous pair:

```bash
cargo run --manifest-path rust/eml_tree_search/Cargo.toml --bin ga-best-expression -- \
  --repo-root /home/tarstars/prj/titanic_mle \
  --feature-a sex_unit \
  --feature-b fare_unit \
  --objective auc \
  --population 512 \
  --generations 300 \
  --max-height 6 \
  --seed 20260421 \
  --run-tag iter2
```

Monotone calibration sweep over the two best reports:

```bash
python3 - <<'PY'
import csv, json, math, pathlib
repo = pathlib.Path('/home/tarstars/prj/titanic_mle')
train_path = repo / 'data' / 'interim' / 'titanic_unit_interval_train.csv'
out_dir = repo / 'data' / 'processed'
feature_index = {
    'PassengerId': 0, 'pclass_unit': 1, 'sex_unit': 2, 'age_unit': 3,
    'age_missing': 4, 'sibsp_unit': 5, 'parch_unit': 6, 'fare_unit': 7,
    'fare_missing': 8, 'embarked_unit': 9, 'embarked_missing': 10,
    'cabin_known': 11, 'family_size_unit': 12, 'is_alone': 13, 'Survived': 14,
}
class Expr:
    def __init__(self, kind, left=None, right=None):
        self.kind = kind
        self.left = left
        self.right = right
def parse_expr(text):
    i = 0
    n = len(text)
    def skip_ws():
        nonlocal i
        while i < n and text[i].isspace():
            i += 1
    def parse():
        nonlocal i
        skip_ws()
        if text[i] == '1':
            i += 1
            return Expr('1')
        if text.startswith('x0', i):
            i += 2
            return Expr('x0')
        if text.startswith('x1', i):
            i += 2
            return Expr('x1')
        i += 1
        left = parse()
        right = parse()
        skip_ws()
        i += 1
        return Expr('node', left, right)
    return parse()
def eval_expr(expr, x0, x1):
    if expr.kind == '1':
        return 1.0
    if expr.kind == 'x0':
        return x0
    if expr.kind == 'x1':
        return x1
    left = eval_expr(expr.left, x0, x1)
    right = eval_expr(expr.right, x0, x1)
    return math.exp(left) - math.log(right)
def grouped_data(feature_a, feature_b):
    ia = feature_index[feature_a]
    ib = feature_index[feature_b]
    isurv = feature_index['Survived']
    grouped = {}
    with train_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            key = (float(row[ia]), float(row[ib]))
            pos, neg = grouped.get(key, (0, 0))
            if int(row[isurv]) == 1:
                pos += 1
            else:
                neg += 1
            grouped[key] = (pos, neg)
    xs0, xs1, poss, negs = [], [], [], []
    for (x0, x1), (pos, neg) in sorted(grouped.items()):
        xs0.append(x0); xs1.append(x1); poss.append(pos); negs.append(neg)
    return xs0, xs1, poss, negs
def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)
def avg_logloss(scores, poss, negs, a=1.0, b=0.0):
    total = 0.0
    count = 0
    for s, pos, neg in zip(scores, poss, negs):
        p = min(max(sigmoid(a * s + b), 1e-15), 1 - 1e-15)
        total += -pos * math.log(p) - neg * math.log(1 - p)
        count += pos + neg
    return total / count
def fit_monotone(scores, poss, negs):
    a = 1.0
    b = 0.0
    prev = avg_logloss(scores, poss, negs, a, b)
    for _ in range(100):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for s, pos, neg in zip(scores, poss, negs):
            w = pos + neg
            p = sigmoid(a * s + b)
            err = w * p - pos
            var = w * p * (1 - p)
            g_a += err * s
            g_b += err
            h_aa += var * s * s
            h_ab += var * s
            h_bb += var
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 1e-18:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (-h_ab * g_a + h_aa * g_b) / det
        scale = 1.0
        improved = False
        while scale > 1e-8:
            cand_a = a - scale * step_a
            cand_b = b - scale * step_b
            if cand_a <= 1e-10:
                scale *= 0.5
                continue
            cand = avg_logloss(scores, poss, negs, cand_a, cand_b)
            if cand < prev:
                a, b, prev = cand_a, cand_b, cand
                improved = True
                break
            scale *= 0.5
        if not improved:
            break
    return a, b, prev
for report_name in [
    'ga_best_expression_pclass_unit__sex_unit__iter2__logloss.json',
    'ga_best_expression_sex_unit__fare_unit__iter2__auc.json',
]:
    report_path = out_dir / report_name
    report = json.loads(report_path.read_text())
    expr = parse_expr(report['best_expression'])
    xs0, xs1, poss, negs = grouped_data(report['feature_a'], report['feature_b'])
    scores = [eval_expr(expr, x0, x1) for x0, x1 in zip(xs0, xs1)]
    a, b, ll = fit_monotone(scores, poss, negs)
    out_path = out_dir / f'{report_path.stem}__monotone_calibration.json'
    out_path.write_text(json.dumps({
        'source_report': str(report_path),
        'feature_a': report['feature_a'],
        'feature_b': report['feature_b'],
        'expression': report['best_expression'],
        'original_logloss': avg_logloss(scores, poss, negs),
        'original_auc': report['best_auc'],
        'calibrated_auc': report['best_auc'],
        'calibration': {'scale_a': a, 'bias_b': b},
        'calibrated_logloss': ll,
    }, indent=2))
PY
```

## Results

First iteration:

- `sex_unit × age_unit`, objective `logloss`
  - result: `logloss = 0.5075161174667232`
  - result: `ROC AUC = 0.8002268878023839`
  - best expression: `((1 ((1 (x0 1)) ((x0 (1 x1)) (x0 1)))) ((1 (x1 (1 (1 x1)))) ((x0 x1) ((1 x1) ((x1 1) (1 x1))))))`
- `pclass_unit × sex_unit`, objective `ROC AUC`
  - result: `logloss = 0.45867954598326716`
  - result: `ROC AUC = 0.8328353518891339`
  - best expression: `(x1 ((1 (x0 1)) (((x1 (1 1)) (x1 (x1 1))) ((1 (x0 1)) ((x0 1) (x0 1))))))`

Second iteration:

- `pclass_unit × sex_unit`, objective `logloss`
  - result: `logloss = 0.4499610394590956`
  - result: `ROC AUC = 0.8328353518891339`
  - best expression: `((x1 ((1 (1 1)) ((x0 1) (x1 1)))) ((1 (x0 (x1 (x1 1)))) (((x1 (x0 1)) ((x0 1) (1 1))) ((x1 1) ((x0 1) (1 1))))))`
- `sex_unit × fare_unit`, objective `ROC AUC`
  - result: `logloss = 1.0999105072716195`
  - result: `ROC AUC = 0.8345796184450196`
  - best expression: `((x0 (1 ((x1 (x0 1)) ((x1 1) (x1 1))))) (((x1 (1 1)) 1) (((x1 1) (1 (x1 1))) ((x1 (1 1)) (x0 1)))))`

Monotone calibration:

- calibrated `pclass_unit × sex_unit` best-`logloss` expression
  - original: `logloss = 0.4499610394590957`
  - calibrated: `logloss = 0.44992720749344484`
  - unchanged `ROC AUC = 0.8328353518891339`
  - calibration: `a = 0.9987498095950472`, `b = 0.020760631683378074`
- calibrated `sex_unit × fare_unit` best-`ROC AUC` expression
  - original: `logloss = 1.0999105072716202`
  - calibrated: `logloss = 0.5004204338671839`
  - unchanged `ROC AUC = 0.8345796184450196`
  - calibration: `a = 1.7212180266120736`, `b = -3.717273426529641`

Third iteration, calibration-aware objective:

- `pclass_unit × sex_unit`, objective `calibrated_logloss`
  - result: raw `logloss = 0.7955365168224754`
  - result: `ROC AUC = 0.8328353518891339`
  - result: calibrated `logloss = 0.4505200533353892`
  - best expression: `((x1 ((1 (x0 1)) (1 ((x1 1) (x1 1))))) ((1 1) ((x1 ((x1 1) (1 1))) (1 (x0 1)))))`
- `sex_unit × fare_unit`, objective `calibrated_logloss`
  - result: raw `logloss = 0.871233723908837`
  - result: `ROC AUC = 0.829812844193057`
  - result: calibrated `logloss = 0.4836832627668904`
  - best expression: `((x0 ((1 ((x1 1) (x1 1))) (x1 1))) (((x1 1) ((x1 1) ((1 1) (1 1)))) (((1 (x1 1)) ((x1 1) (1 1))) (((1 1) (x0 1)) (x0 1)))))`
- `sex_unit × fare_unit`, objective `auc_calibrated_logloss`
  - result: raw `logloss = 0.6239138605351247`
  - result: `ROC AUC = 0.8313733635850403`
  - result: calibrated `logloss = 0.4981720908349277`
  - best expression: `((x0 ((1 (1 (x0 1))) ((1 (x1 1)) (x1 1)))) ((1 (x1 (x0 (x0 1)))) (((x1 (x1 1)) (x1 (x0 1))) (x0 (x0 1)))))`

Fourth iteration, seeded calibration-aware refinement:

- `pclass_unit × sex_unit`, objective `calibrated_logloss`, seeded by the best `iter2` expression
  - result: raw `logloss = 0.4741512132351344`
  - result: `ROC AUC = 0.8328353518891339`
  - result: calibrated `logloss = 0.4494937812644100`
  - best expression: `((x1 ((1 (1 1)) (x0 (x1 (x1 1))))) ((1 (x0 (x1 (x1 1)))) (((x1 (x0 1)) ((x0 1) (1 1))) (1 ((x0 1) (1 1))))))`

## Interpretation

Five conclusions matter immediately:

- The original pairwise `height <= 3` report understated how strong `pclass_unit × sex_unit` really is. Once the tree is allowed to grow to `height <= 6`, that pair becomes the strongest probability model we have seen so far.
- A calibration-aware objective is not automatically better than post-hoc calibration. The unseeded `iter3` `pclass_unit × sex_unit` run was still slightly worse than the earlier post-hoc calibrated winner.
- Seed quality matters. Once the calibration-aware run was seeded with the best `iter2` probability expression, it moved the frontier again.
- The best pairwise probability frontier at the end of this stage became `pclass_unit × sex_unit` with calibrated `logloss = 0.4494937812644100`.
- The best current `ROC AUC` frontier is `sex_unit × fare_unit` with `ROC AUC = 0.8345796184450196`, but its raw score is badly calibrated. The monotone calibration repair is large and real, yet it still does not beat the calibrated `pclass_unit × sex_unit` model on `logloss`.

This gives us a clean split:

- if the target is probability quality, `pclass_unit × sex_unit` is the current winner
- if the target is ranking only, `sex_unit × fare_unit` is the current winner

## Output Artifacts

- `data/processed/ga_best_expression_sex_unit__age_unit__iter1__logloss.json`
- `data/processed/ga_best_expression_pclass_unit__sex_unit__iter1__auc.json`
- `data/processed/ga_best_expression_pclass_unit__sex_unit__iter2__logloss.json`
- `data/processed/ga_best_expression_sex_unit__fare_unit__iter2__auc.json`
- `data/processed/ga_best_expression_pclass_unit__sex_unit__iter2__logloss__monotone_calibration.json`
- `data/processed/ga_best_expression_sex_unit__fare_unit__iter2__auc__monotone_calibration.json`
- `data/processed/ga_best_expression_pclass_unit__sex_unit__iter3__calibrated_logloss.json`
- `data/processed/ga_best_expression_sex_unit__fare_unit__iter3__calibrated_logloss.json`
- `data/processed/ga_best_expression_sex_unit__fare_unit__iter3__auc_calibrated_logloss.json`
- `data/processed/ga_best_expression_pclass_unit__sex_unit__iter4_seeded__calibrated_logloss.json`

## Status

These results are approximate because the search stage is genetic. The calibration stage is deterministic for a fixed saved expression.

## Next Step

The next practical step is to stop treating pairwise expressions as terminal results and instead treat them as a library of building blocks:

- keep the separate pairwise frontiers
- convert the strongest saved expressions into calibrated probability terminals
- run an exact small-depth search above those terminals before attempting a full multi-feature GA
