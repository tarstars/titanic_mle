# MLE Function Notes

This project defines:

```text
mle(x, y) = exp(x) - ln(y)
```

## Immediate Facts

- `exp(x)` is defined for every real `x`
- `ln(y)` is defined only for `y > 0`
- therefore the natural domain is:

```text
x ∈ R
y > 0
```

## Basic Behavior

- increasing `x` increases `mle(x, y)`
- increasing `y` decreases `mle(x, y)` because `ln(y)` grows with `y`
- if `y = exp(exp(x))`, then `mle(x, y) = 0`

## Derivatives

For `y > 0`:

```text
∂/∂x mle(x, y) = exp(x)
∂/∂y mle(x, y) = -1 / y
```

So:

- the function is strictly increasing in `x`
- the function is strictly decreasing in `y`

Second derivatives:

```text
∂²/∂x² mle(x, y) = exp(x)
∂²/∂y² mle(x, y) = 1 / y²
```

This means the function is convex in `x` and also convex in `y` over `y > 0`.

## Implications For Titanic

The constraint `y > 0` matters immediately. Any Titanic feature routed into the `y` position must be:

- already positive
- or transformed to stay strictly positive

Examples of raw features that may need transformation before use as `y`:

- `Age` when missing
- `Fare` if zeros must be avoided
- encoded binary features if they contain `0`

## Naming Note

In this repository, `mle` does not mean maximum likelihood estimation.

The external literature you are currently collecting uses **EML** ("Exp-Minus-Log") for the same operator:

```text
eml(x, y) = exp(x) - ln(y)
```

So in project terms:

```text
mle(x, y) == eml(x, y)
```

That ambiguity should be avoided in:

- notebook titles
- code comments
- research notes
- future model descriptions

## Open Questions

- What Titanic features should map to `x` and `y`?
- Should `mle(x, y)` be used directly as a score, or as a node transform inside a larger tree?
- How should missing values and zero values be handled before the `ln(y)` term?
