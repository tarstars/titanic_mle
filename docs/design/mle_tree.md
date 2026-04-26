# MLE Tree Design Sketch

The `mle tree` is the project-specific structure that will organize how `mle(x, y)` is applied to Titanic data.

## Initial Root Structure

Current root branches:

- `function_knowledge`
- `dataset_knowledge`
- `modeling_context`

## Why This Split

- `function_knowledge` stores facts about the custom `mle` function
- `dataset_knowledge` stores facts about Titanic columns, targets, and constraints
- `modeling_context` stores the current plan for turning those facts into a working solver

## Planned Expansion

Possible later branches:

- feature mapping candidates
- positivity transforms for `y`
- missing-value handling
- score aggregation
- submission generation

## Immediate Goal

The first useful executable version of the tree should answer:

- which Titanic variables can feed `x`
- which transformed variables can safely feed `y`
- how node outputs combine into a final survival decision

## Immediate Next Steps

Based on the 2026-04-20 feasibility note, the next implementation round should stay conservative:

- start with a shallow tree only, depth 2 or 3
- treat the tree output as a scalar score and wrap it with a sigmoid for binary classification
- keep separate rules for:
  - exp-eligible branches
  - log-eligible branches with guaranteed positive inputs
- optimize continuous coefficients first
- delay unrestricted topology search until a fixed-topology baseline is working

## Recommended Phase Order

1. Build preprocessing that produces:
   - normalized features for general use
   - positive-safe features for the log branch
2. Implement one fixed EML/MLE head that returns a scalar score.
3. Train and evaluate it with log loss.
4. Compare it against:
   - logistic regression
   - one boosted-tree baseline
   - one symbolic-classification baseline
5. Only then add weak structural mutation over a small template family.
