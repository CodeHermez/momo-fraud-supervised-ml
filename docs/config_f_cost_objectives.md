# Configuration F: which cost objective applies where

**Status:** documentation of an existing design choice. No metric, threshold or
result was changed. Raised as audit finding M4.

## The issue

Configuration F trains with per-instance severity weights but is thresholded and
scored under the class-constant cost model. Three different objectives are in
play and only one of them is the one F optimises.

| Stage | Objective | Implementation |
|---|---|---|
| Training | `C_FP = 1`, `C_FN = R · sᵢ` | `models.sample_weights` — `w⁺ᵢ = R · sᵢ`, `w⁻ = 1` |
| Threshold selection | `C_FP = 1`, `C_FN = R` | `experiment.evaluate_configs` → `evaluate.select_threshold(..., severity=None)` |
| Evaluation (`ner`) | `C_FP = 1`, `C_FN = R` | `evaluate.risk_metrics` |
| Evaluation (`ner_severity_weighted`) | `C_FP = 1`, `C_FN = R · sᵢ` | `evaluate.risk_metrics`, reported alongside |

`sᵢ` is the percentile rank of `amount` in the training distribution
(`risk.SeverityScaler`), so `sᵢ ∈ [0, 1]` and `E[sᵢ] ≈ 0.5`.

## Why it matters

Two distinct things separate F from D, not one:

1. **Per-instance versus constant** weighting — the intended manipulation.
2. **Effective magnitude.** D weights every positive by `R`; F weights them by
   `R · sᵢ`, averaging about `R/2`. F is a weaker weighting as well as a
   differently-shaped one.

F is then scored on D's objective. So F optimises `E[R · sᵢ]` while being judged
on `E[R]`, which is a mismatch that systematically disadvantages it.

## Why threshold selection was *not* made severity-aware

This is deliberate and remains correct. Letting severity into
`select_threshold` would make *every* configuration severity-aware, including
config C, whose entire definition is "the NER-optimal threshold under the
study's declared class-constant cost model". Fixing F's mismatch that way would
break the control it is measured against. The comment at
`experiment.py:evaluate_configs` records this reasoning.

## What configuration F is, and is not

**It is:** a sensitivity-analysis training variant, asking whether shaping the
training loss by transaction size helps a model that will be deployed and judged
under class-constant costs.

**It is not:** a self-consistent example-dependent cost-sensitive experiment.
That requires the same objective at all three stages — train, threshold and
score under `C_FN = R · sᵢ` — and this study has not run it.

## Consequence for the reported result

Configuration F's PR-AUC deltas at the `no_origin_balance` rung are −0.0144
(decision tree) to −0.2647 (logistic regression). Those numbers stand as
measurements of the variant described above.

They do **not** support the claim that per-instance severity weighting does not
work. Nothing in the current design can support that claim in either direction.

## Deferred

A dedicated example-dependent experiment — training, thresholding and evaluating
under a single `C_FN = R · sᵢ` objective, with config D re-expressed at matched
effective weight so magnitude and shape are separated — is flagged for the next
research phase. **It is not run here.** Deciding whether the study wants that
arm is a research decision, listed in section E of the repair report.
