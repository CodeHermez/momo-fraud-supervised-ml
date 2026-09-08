# Literature note — Hájek, Abedin & Sivarajah (2023)

**Why this note exists.** `report/RESULTS_SYNTHESIS.md` §5.1 flagged this paper as
the closest published work to configuration G (RUS + XGBoost on PaySim) and
predicted "a reviewer will ask". It is the study's single largest literature gap
and it is closed here to the extent that open sources allow.

**Status: citation verified, full text not obtained.** See §E for exactly what
remains unverified, so nothing in this note is mistaken for a reading of the
paper itself.

---

## A. Citation

> Hajek, P., Abedin, M. Z., & Sivarajah, U. (2023). Fraud detection in mobile
> payment systems using an XGBoost-based framework. *Information Systems
> Frontiers*, **25**(5), 1985–2003.
> https://doi.org/10.1007/s10796-022-10346-6

Online first 14 October 2022; issue publication October 2023. ISSN 1387-3326.
Funded by Czech Science Foundation grant 19-15498S.

Volume, issue, page range, year, DOI and ISSN cross-checked against two
independent institutional repository records (Kingston University and Teesside
University). The paper is paywalled and **no green open-access accepted
manuscript is offered** by any of the Kingston, Teesside or Swansea repositories.
Obtaining it needs University of Pretoria institutional access.

## B. What is established from open sources

From the publisher abstract as reproduced by the Teesside repository, and
corroborating secondary sources:

| | |
|---|---|
| Dataset | **PaySim** — "more than 6 million samples", nine attributes, **0.13% positive** |
| Second dataset | **BankSim** (a second synthetic simulator) |
| Framework | XGBoost-based; a **semi-supervised ensemble** integrating several unsupervised outlier-detection algorithms |
| Best classification result | the semi-supervised ensemble |
| **Best cost saving** | **random under-sampling combined with XGBoost** |
| Framing | explicitly considers "the financial consequences of fraud detection systems" |

The dataset is the same one this study uses, at the same prevalence. The
cost-saving winner is **configuration G of this study** — RUS at `N'_neg`,
natural test set — and the economic framing is the currency-denominated
equivalent of this study's unitless NER.

So this is not merely adjacent work. It is the same dataset, the same headline
technique, and a cost objective of the same shape, reaching an apparently
opposite conclusion. That is why it must be addressed rather than cited.

## C. The apparent disagreement

| | Hájek et al. (2023) | This study |
|---|---|---|
| RUS + XGBoost | **highest cost saving** | config G; PR-AUC delta negative in 12/12 comparisons (`04_h1_paired_bootstrap.csv`) |
| CSL training generally | a route to better economic outcomes | never improved ranking quality; 116/116 negative paired PR-AUC comparisons in E3 |
| Feature treatment | not established from open sources | origin-balance columns **ablated** (`no_origin_balance`) |

A reviewer will read this as a contradiction. It is narrower than it looks, and
§D sets out why — but the study must make that argument explicitly rather than
hope the question is not asked.

## D. Four candidate reconciliations, ranked

These are the study's argument, to be confirmed or corrected once the full text
is read. **None of them is a claim about what Hájek et al. actually did.**

**1. Different feature sets mean different ceilings — almost certainly the
dominant factor.** This study's `notebooks/02` establishes that three boolean
clauses over the origin-balance columns recover PaySim's label at precision
0.9999 and recall 0.9770. Any model retaining those columns is scoring against a
deterministic generator rule, and this study's own full-feature benchmark reaches
PR-AUC 0.9963 (random forest) and 0.9977 (XGBoost) there. At that ceiling the
gap between resampling strategies is a gap in how each lands on a simulator
artefact, not in fraud detection. This study ablated those columns and its
numbers are consequently, and correctly, lower. **If Hájek et al. retained them —
which §E lists as the first thing to check — then the two sets of numbers are not
comparable at all, and this is the reconciliation.**

**2. "Cost saving" is a decision-layer quantity, and may not isolate
resampling.** A cost saving is evaluated at an operating point. If the reported
saving is computed at a tuned threshold, then a RUS + XGBoost win is consistent
with — not contrary to — this study's central finding, which is that the
*decision* layer does the work while the *data* and *algorithm* layers do not.
The resampling would be receiving credit that belongs to the thresholding.

This study is in a strong position to raise that point and a weak position to
raise it smugly: **it made the same conflation itself.** Audit finding C2 caught
`06_r_sweep.csv` being read as a training-cost sweep when every model in all 36
cells was trained at a single `R = 773.70` and only the decision threshold moved.
The file was relabelled `06_decision_cost_sweep.csv` with explicit `R_train` and
`R_decision` columns (CHANGE-05), and E2 was later run to sweep the training
ratio properly. The right tone in the report is that separating the training and
decision layers is difficult and easy to get wrong, evidenced by this study
having to repair exactly that error in its own work.

**3. The metrics answer different questions.** Twelve negative PR-AUC deltas are a
statement about **ranking quality**. A cost saving is a statement about
**end-to-end decision risk**. This study's own results separate cleanly on that
line, and its decision-layer result is *not* uniformly negative: E3 found the NER
effect learner-dependent, with logistic regression and the decision tree
benefiting in 64/64 pairs. Agreement may be substantially greater than the
headline comparison suggests.

**4. RUS is stochastic, and single-seed RUS results are weakly determined.** E3
measured decision-tree RUS at σ = 0.042 PR-AUC — the largest training-seed
variance anywhere in this study, an order of magnitude above the 0.001–0.003
typical of the other arms. Whether Hájek et al. report seed variance is listed in
§E. This is a question to raise, not an accusation to make.

**The honest qualifier the study must state.** This study does **not** claim RUS
harms XGBoost. E3's own conclusion records XGBoost/RUS at a mean NER effect of
**+0.0028**, small relative to its split spread, and describes it as
"**indistinguishable from no effect**" rather than as a harm. The distance
between this study and Hájek et al. on their headline configuration is therefore
smaller than a careless reading of either would suggest.

## E. To verify from the full text

In priority order. Until each is checked, the corresponding argument in §D is a
hypothesis.

1. **The feature set.** Were `oldbalanceOrg`, `newbalanceOrig` and any derived
   error-balance columns retained? This decides reconciliation 1 and is the
   single most important question.
2. **How the cost saving is computed.** At a fixed threshold, a tuned threshold,
   or integrated over thresholds? This decides reconciliation 2.
3. **The cost matrix.** Fixed `C_FP` / `C_FN`, or example-dependent on `amount`?
   Relevant to configuration F and to `docs/config_f_cost_objectives.md`.
4. **Reported discrimination metrics.** AUC / PR-AUC values, to compare against
   this study's full-rung benchmark (RF 0.9963, XGB 0.9977) and so infer whether
   they sat on the drain signature.
5. **Train/test protocol.** Split proportions, whether the test set retained
   natural prevalence, and whether resampling was confined to training rows.
6. **Seeds and repetitions.** Single run or repeated? Any variance reported?
7. **The nine attributes.** Which nine, and whether `isFlaggedFraud` was among
   them.
8. **The BankSim results**, and whether the RUS + XGBoost conclusion holds on both
   datasets or only PaySim.

## F. Where this belongs in the report

- **Related work** — as the closest published treatment of the configuration this
  study calls G, on the same dataset and at the same prevalence.
- **Discussion** — as the primary point of contact and disagreement, argued
  through §D. This is the paragraph that most improves the report's robustness,
  because it converts the most obvious examiner question into an answered one.
- **Not in Results.** Nothing here is a result of this study.

## G. Remaining literature actions

`report/RESULTS_SYNTHESIS.md` §5.1 named three papers to obtain. This note closes
one at citation level. Still outstanding:

- **Lopez-Rojas, Elmir & Axelsson (2016)** — the PaySim paper itself. The
  simulator's generative rules are the direct mechanism behind this study's
  three-clause finding, which would turn an empirical curiosity into an explained
  one. Highest remaining value.
- **Johnson & Khoshgoftaar (2022)** — already central; cite the specific claim
  replicated (thresholding outperforms weighting and sampling), not the paper in
  general.
- **Elkan (2001)** — the canonical derivation of `λ = 1/(1+R)`, against which
  this study's equivalence theorem and E6 should be positioned.
