# Applying the EML operator to the Titanic dataset
## A sourced research note as of 2026-04-20

## Scope and evidence standard

**Naming note:** the public literature cited here uses **EML** for the Exp-Minus-Log operator. In this repository, the current code and project shorthand use `mle(x, y) = exp(x) - ln(y)` for the same operator. This note keeps the external `EML` spelling so the terminology stays aligned with the cited sources.

This note answers a narrow question: **does it make sense to search over EML trees on the Titanic dataset, using log loss as the objective?**

I separate:

- **Sourced facts** — statements tied to cited papers, documentation, or dataset metadata.
- **Interpretation / recommendation** — my own synthesis from those sources.

I also stay conservative: for the recent EML work, I only rely on details that were recoverable from current public abstracts, project documentation, or official dataset/docs pages during this review.

---

## Executive conclusion

A **full unrestricted genetic-programming search over EML trees** for Titanic is **possible in principle**, but it is **not the best first experiment**.

The reason is simple:

1. The original EML paper supports **shallow** trainable EML trees and reports exact recovery of closed-form functions only at **small depths (up to 4)**. It does **not** establish that large free-form EML trees are easy to optimize on noisy classification data. [1]
2. The two follow-up April 2026 EML papers both lean toward the same practical conclusion: use EML as a **structured symbolic layer / parametrization device**, not as a large unconstrained runtime tree. One says a hybrid DNN-EML model should use a **depth-bounded** EML head and is unlikely to speed up training or inference on ordinary CPU/GPU hardware. [2] The other says direct EML simulation is slower and recommends using EML only on the **parametrization side**, while keeping the classical recursion at runtime. [3]
3. Titanic is a **small**, **partly missing**, **imbalanced** binary-classification dataset; the OpenML version has **1309 rows**, **14 features**, **2 classes**, a **61.8% majority class**, and substantial missingness in variables such as `age`, `cabin`, and `home.dest`. [4]

**My recommendation:** use Titanic as a **pilot benchmark** for a **shallow, fixed-topology or weakly-mutated EML head**, trained with continuous optimization for log loss, and compare it to ordinary symbolic classification and classical baselines. Do **not** begin with a large unrestricted GA over deep EML trees.

---

## 1. What is currently established about EML

### 1.1 The basic claim

EML is the binary operator

\[
\operatorname{eml}(x,y)=\exp(x)-\ln(y).
\]

The original 2026 paper claims that **EML together with the constant `1` generates the standard repertoire of a scientific calculator**, including arithmetic operations, exponentiation, and the usual transcendental / algebraic functions. It also gives the simple uniform grammar

\[
S \to 1 \mid \operatorname{eml}(S,S).
\]

The same paper states that this uniform representation enables **gradient-based symbolic regression**, and that exact recovery of closed-form elementary functions was demonstrated at **shallow depths up to 4**. [1]

### 1.2 Software and implementation state

A public Rust crate, **OxiEML**, appeared in April 2026. Its documentation states that it implements EML trees, symbolic regression, lowering of EML trees to ordinary operations, code generation, and a CLI; it also explicitly describes the same grammar `S -> 1 | eml(S, S)` and symbolic-regression workflow. The repository page shows a public release **0.1.0** dated **2026-04-14**. [5]

### 1.3 What the April 2026 follow-up papers imply

The **hybrid DNN-EML** paper proposes using a normal neural-network trunk with a **depth-bounded, sparse EML head** whose weights can be snapped into a symbolic expression. Its abstract explicitly says that EML is **unlikely to accelerate training**, and on commodity **CPU/GPU is also unlikely to accelerate inference**; the claimed benefit appears only for **custom EML hardware** such as FPGA / analog cells. [2]

The **battery-characterization** paper reaches a similarly pragmatic conclusion. Its abstract says direct EML simulation is slower than the classical exponential-Euler scheme, with a reported **~25x instruction overhead per RC branch**, and recommends: **use EML only on the parametrization side of the workflow, keeping the classical recursion at runtime**. [3]

### 1.4 Immediate mathematical constraint relevant to tabular data

EML contains a natural logarithm, so the second input must be **strictly positive**. That is not a research claim but a direct mathematical fact from the definition:

\[
\ln(y) \text{ is undefined for } y \le 0.
\]

So if your engineered features lie in `[0,1]`, then **zeros are invalid on the log branch** and values near zero are numerically risky. This matters immediately for Titanic.

---

## 2. What is established about symbolic classification with log loss

There is already prior work on **symbolic classification** rather than symbolic regression.

### 2.1 gplearn

The official `gplearn` documentation describes `SymbolicClassifier` as a **genetic-programming symbolic classifier**. The API reference lists `transformer='sigmoid'`, `metric='log loss'`, and a `parsimony_coefficient` parameter for size control. [6]

The introductory documentation explains the same thing in words: `SymbolicClassifier` transforms a program output through a **sigmoid** to obtain class probabilities, and it uses **log loss** as its default optimization metric. [7]

This means that the generic idea

\[
p(x)=\sigma(s(x)),
\]

with `s(x)` a symbolic expression and log loss as the training criterion, is fully standard. It is not exotic.

### 2.2 Prior research papers

The paper **Explainable Fraud Detection with Deep Symbolic Classification** presents symbolic classification as a search problem over analytic expressions and explicitly states that its framework extends symbolic regression with **a sigmoid layer** for binary classification and metric-driven optimization. [8][9]

So there is no conceptual problem with “symbolic expression + sigmoid + classification objective”. The open question is not *whether that setup is meaningful*; the open question is *whether EML is the right symbolic vocabulary and whether genetic search is the right optimizer for it on Titanic*.

---

## 3. Why Titanic is a special case

The OpenML Titanic dataset used in many ML benchmarks has the following properties:

- **1309 instances**
- **14 features**
- **2 classes**
- **61.8% majority class**
- missing values in multiple important fields, including **263 missing ages**, **1014 missing cabins**, **564 missing `home.dest`**, and other gaps. [4]

This matters for EML in three ways.

### 3.1 Small-sample risk

Titanic is too small to forgive a large symbolic search space. A big tree grammar plus GP-style structure search can easily overfit or chase CV noise.

### 3.2 Missing-data risk

If your pipeline turns missingness into engineered features in `[0,1]`, that is fine — but EML still requires **strictly positive** arguments on log branches, so you need either:

- clipping to `[ε, 1]`,
- a positive reparameterization such as `softplus(z) + ε`,
- or a topology rule saying which leaves may feed the log input.

### 3.3 Interpretability risk

Titanic is famous because very simple signals already work well: sex, class, age / child status, fare, embarked, family-size features, and interactions among them. A very deep symbolic tree might produce something “symbolic” but not actually more interpretable than a shallow logistic model with a few handcrafted interactions.

---

## 4. Does a GA over EML trees make sense?

## Short answer

**Yes in principle, no as the default first move.**

## Why “yes”

- GP classifiers with **sigmoid outputs** and **log-loss objectives** already exist in mainstream tooling (`gplearn`). [6][7]
- Symbolic classification research already supports the general pattern of searching over analytic expressions for classification. [8][9]
- EML provides a **uniform symbolic basis**, so it is a natural candidate vocabulary if your real interest is symbolic unification, snapping, or hardware regularity. [1][2][5]

## Why “not first”

- Current EML evidence favors **shallow** trees and **structured training**, not large free-form evolutionary search. [1][2][3]
- The recent EML follow-up papers are both cautionary about speed and depth. [2][3]
- Titanic is small enough that unrestricted GP bloat is a real risk; even standard GP libraries need an explicit **parsimony coefficient** to counter tree growth. [6]

So the right framing is:

> **Use EML as a constrained symbolic head, and allow only limited structural search around that head.**

That is much more faithful to the current evidence.

---

## 5. A better experiment design for Titanic

## 5.1 Data representation

Assume you already have engineered features \(z_1,\dots,z_d \in [0,1]\).

I would build two leaf types:

1. **exp-eligible leaves**: affine combinations of features, e.g.
   \[
   a_0 + \sum_j a_j z_j
   \]
2. **log-eligible leaves**: positive transforms only, e.g.
   \[
   \operatorname{softplus}\!\left(b_0 + \sum_j b_j z_j\right) + \varepsilon
   \]
   or simple clipping to `[ε, 1]`.

This keeps the log branch legal.

## 5.2 Model family

### Model A — shallow fixed EML head
A fixed tree of depth 2 or 3 that returns a scalar score \(s(x)\), followed by a sigmoid:

\[
p(x)=\sigma(s(x)).
\]

Train all continuous coefficients with gradient descent on **log loss**.

### Model B — weakly evolutionary EML head
Use a GA **only for the outer structure**:
- choose among a small set of shallow EML templates,
- mutate which features appear in which leaves,
- optionally mutate one or two subtree patterns,

but keep inner coefficients optimized by gradient descent.

This is a **hybrid continuous-discrete search**, not “pure GP over everything”.

### Model C — standard symbolic classifier baseline
Run `gplearn.SymbolicClassifier` on the same engineered features with its default sigmoid / log-loss setup, plus a carefully tuned parsimony coefficient. [6][7]

### Model D — classical tabular baselines
At minimum:
- logistic regression,
- gradient-boosted trees,
- perhaps a small GAM or explainable boosting model.

The point is not just performance; it is **performance per unit of symbolic complexity**.

## 5.3 Evaluation

Primary metric:
- **log loss**. The scikit-learn documentation defines it as the negative log-likelihood / binary cross-entropy over predicted probabilities. [10]

Recommended secondary metrics:
- ROC-AUC,
- Brier score,
- calibration curve,
- symbolic complexity (tree depth, node count, number of unique input features used).

Protocol:
- repeated stratified cross-validation,
- fixed random seeds,
- report mean and standard deviation,
- keep preprocessing inside each CV fold.

Because Titanic is small, single train/validation splits are too noisy for symbolic-structure conclusions.

---

## 6. The concrete research result I would stand behind

If your question is:

> “What is the strongest justified statement I can make today?”

Then it is this:

### Result

A **Titanic pilot using EML is justified**, but the evidence does **not** currently support starting from a **large unrestricted genetic search over EML trees**. The most evidence-aligned setup is a **shallow EML symbolic head**, trained with continuous optimization for log loss, optionally wrapped in a **small outer evolutionary search over topology**. This design matches:
- the original EML paper’s shallow symbolic-regression evidence, [1]
- the hybrid DNN-EML paper’s depth-bounded head design, [2]
- the battery paper’s recommendation to use EML as a **parametrization language** rather than as the whole runtime model, [3]
- and existing symbolic-classification practice using **sigmoid outputs** and **log-loss objectives**. [6][7][8][9]

### What would count as a success on Titanic

Not “beating CatBoost”.

A real success would be:

1. Competitive CV log loss versus logistic regression / ordinary symbolic classification.
2. A much **smaller symbolic expression** than unconstrained GP produces.
3. Stable formulas across repeated seeds / folds.
4. Evidence that EML’s uniformity buys you something:
   - cleaner snapping,
   - easier symbolic simplification,
   - or better complexity/performance tradeoff.

That would be a legitimate pilot result.

---

## 7. What is publishable and what is not

### Not publishable by itself
“EML on Titanic gives X log loss.”

That is a benchmark note, not a paper.

### Potentially publishable
A short methods paper or workshop paper on:

**Hybrid optimization of universal-operator symbolic classifiers**  
using:
- EML as the symbolic basis,
- gradient descent for coefficients,
- constrained topology search for structure,
- and complexity-aware model selection.

Titanic could be the pilot benchmark, but the real paper would need at least one stronger tabular or scientific benchmark afterward.

A better next step after Titanic would be:
- a small scientific time-series classification task,
- a physically interpretable binary decision task,
- or a low-dimensional system where a symbolic decision boundary is scientifically meaningful.

---

## 8. One-day implementation plan

### Day-1 objective
Answer one practical question:

> “Can a shallow EML head produce competitive log loss on Titanic without exploding in complexity?”

### Minimal build

1. Prepare Titanic features.
   - impute missing values,
   - encode categoricals,
   - scale engineered features to `[0,1]`,
   - create a positive-safe version for log leaves: `x_pos = clip(x, ε, 1.0)`.

2. Implement a depth-2 EML score function.
   Example pattern:
   \[
   s(x)=\operatorname{eml}(u_1(x), v_1(x)) + \alpha \,\operatorname{eml}(u_2(x), v_2(x)) + \beta,
   \]
   where all \(v_i(x)\) are constrained positive.

3. Wrap with sigmoid:
   \[
   p(x)=\sigma(s(x)).
   \]

4. Train by minimizing log loss with an \(L_1\) or explicit node-count / coefficient sparsity penalty.

5. Compare against:
   - logistic regression,
   - `gplearn.SymbolicClassifier`,
   - one boosted-tree baseline.

6. Report:
   - CV log loss,
   - AUC,
   - formula size,
   - number of used features,
   - seed stability.

### If you want one small evolutionary ingredient
Use evolution only to choose among:
- a handful of depth-2 / depth-3 templates,
- leaf-feature subsets,
- and whether a branch is exp-eligible or log-eligible.

Do **not** evolve arbitrary deep trees on day 1.

---

## 9. Final recommendation

If your real goal is a **research path**, do this:

1. **Titanic first** — as a debugging benchmark for the method.
2. Use **shallow EML + sigmoid + log loss**.
3. Add **limited topology search**, not full GP.
4. Measure **interpretability and stability**, not just predictive score.
5. Only if that works, move to a stronger scientific or time-series dataset.

That is the most defensible route given the current state of EML research.

---

## References

[1] Andrzej Odrzywołek, **All elementary functions from a single binary operator**, arXiv:2603.21852, first posted 2026-03-23; public summaries/reporting indicate a revised version in early April 2026.  
- Public abstract summary: <https://papers.cool/arxiv/2603.21852>  
- arXiv landing page: <https://arxiv.org/abs/2603.21852>

[2] Eymen Ipek, **Hardware-Efficient Neuro-Symbolic Networks with the Exp-Minus-Log Operator**, arXiv:2604.13871, posted 2026-04-15.  
- Public abstract summary: <https://papers.cool/arxiv/2604.13871>  
- arXiv landing page: <https://arxiv.org/abs/2604.13871>

[3] Eymen Ipek, **Evaluating the Exp-Minus-Log Sheffer Operator for Battery Characterization**, arXiv:2604.13873, posted 2026-04-15.  
- Public abstract summary: <https://papers.cool/arxiv/2604.13873>  
- arXiv landing page: <https://arxiv.org/abs/2604.13873>

[4] **OpenML Titanic dataset (ID 40945)** — dataset metadata and feature-level missingness.  
<https://api.openml.org/d/40945>

[5] **OxiEML** — public Rust crate / GitHub project implementing EML trees and symbolic regression.  
- GitHub: <https://github.com/cool-japan/oxieml>  
- docs.rs: <https://docs.rs/oxieml/latest/oxieml/>

[6] **gplearn API reference** — `SymbolicClassifier` parameters include `transformer='sigmoid'`, `metric='log loss'`, and `parsimony_coefficient`.  
<https://gplearn.readthedocs.io/en/stable/reference.html>

[7] **gplearn Introduction to GP** — explains that `SymbolicClassifier` applies a sigmoid transform and uses log loss by default.  
<https://gplearn.readthedocs.io/en/latest/intro.html>

[8] Samantha Visbeek, Erman Acar, Floris den Hengst, **Explainable Fraud Detection with Deep Symbolic Classification**, arXiv:2312.00586 / xAI 2024.  
- Public abstract / publication page: <https://research.vu.nl/en/publications/explainable-fraud-detection-withdeep-symbolic-classification/>  
- arXiv PDF: <https://arxiv.org/pdf/2312.00586>

[9] Samantha Visbeek, Erman Acar, Floris den Hengst, **Explainable Fraud Detection with Deep Symbolic Classification**, xAI 2024, Springer, DOI listed on publication page.  
<https://doi.org/10.1007/978-3-031-63800-8_18>

[10] scikit-learn documentation, **`sklearn.metrics.log_loss`**.  
<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html>

---
## Provenance note

This note was prepared from currently indexed public sources on 2026-04-20. For the very recent EML papers, the most reliable currently accessible material during preparation was the recoverable abstract-level metadata plus official project documentation.
