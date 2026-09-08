# Report outline — written from the evidence, in the order the research happened

**Principle.** The report narrates the actual research evolution rather than
presenting the final methodology as though it were known at the start. This is
both more honest and considerably stronger: the study's best material *is* the
correction sequence, and a report that hides it throws away its own contribution.

The arc:

> **Problem → initial CSL hypothesis → audit and reliability intervention →
> frozen baseline → E2/E3/E4/E6 → findings that challenge the original
> hypothesis → design contribution**

Every section below names the artifacts it is written from. Nothing is written
from memory. Claim strength is governed by `docs/claim_taxonomy.md`.

**One thing the report must not do:** describe the rung selection as
pre-registered. CHANGE-01 records that the chronology is `UNPROVEN`. The
supportable phrase is *criterion-driven and mechanically enforced*.

---

## 1. Introduction — the problem

The motivating threat is mobile money fraud in telco networks: vishing, SIM swap,
agent-side fraud, mule laundering. Extreme class imbalance (0.129%) makes
accuracy meaningless and makes the cost of the two error types asymmetric.

**Set up the gap honestly and early:** the introduction motivates authentication
compromise, and the dataset measures account draining. Those are different
things. Naming it here — rather than in Limitations — converts the study's
biggest vulnerability into evidence of care. (`RESULTS_SYNTHESIS` §6.1)

## 2. Background and related work

- Cost-sensitive learning; the proposal's §3.3 three-level taxonomy (data /
  algorithm / decision), all three derived from one `R`.
- **Johnson & Khoshgoftaar (2022)** — the replication target. Cite the *specific*
  claim replicated: thresholding outperforms weighting and sampling. They asked
  for replication on other severely imbalanced datasets; PaySim at 0.129% is one.
- **Elkan (2001)** — `λ = 1/(1+R)`. Position the equivalence theorem relative to
  him. *Not yet obtained.*
- **Hájek, Abedin & Sivarajah (2023)** — same dataset, same headline technique
  (RUS + XGBoost), apparently opposite conclusion. Handle in Discussion §8, not
  here. (`docs/literature_note_hajek_2023.md`)
- **Thar & Wai (2025)** — the published PaySim benchmark used as a pipeline
  correctness check.
- **Authentication threat models** — Ali, Dida & Sam (2020) plus the USSD and SIM-
  swap literature. This is the *only* evidence for RQ 2.1.1 and must carry that
  section alone. (`RESULTS_SYNTHESIS` §5.3; several still unread.)

## 3. Data and the leakage audit — the pivot

**This chapter is where the study earns its credibility.** Present it as a
finding, not as preprocessing.

1. PaySim, 6,362,620 rows, 8,213 frauds (0.129%), 31 days.
2. Three boolean clauses recover the label at **P 0.9999 / R 0.9770**
   (`02_rule_recoverability.csv`). Flag it as an *in-sample descriptive statistic
   of the generator* — audit finding m5 — not a held-out detector result.
3. `errorBalanceOrig` reaches solo AUC 0.9020; the drain signature holds for
   97.82% of fraud against 0.0013% of legitimate rows, median gap **exactly 0.0**.
4. **The proposal's own headline engineered feature is the leak.** State this
   plainly. It is a finding, and it explains why this study's numbers sit below
   the published work it compares against.
5. The ablation ladder and the `no_origin_balance` rung; the destination-side
   diagnostic that measured the symmetric argument and **declined** to extend it
   (`docs/destination_artefact_diagnostic.md`) — `dest_not_credited` runs 0.60
   legitimate vs 0.52 fraud, a lift of 0.86 pointing the *wrong way*.
6. CHANGE-01: the declared ablation was two features, the executed one six, and
   the chronology is unproven.

## 4. Method

- The risk framework: one interpretable `R`, unitless throughout; NER as the
  proposal's `TotalCost` divided by the do-nothing baseline.
- The equivalence theorem (taxonomy category 1) — derive it here.
- Splits, seeds, the threshold protocol: **TRAIN → FIT → VAL → SELECT → LOCK →
  TEST → REPORT**.
- The configuration grid A–G across all three CSL levels.
- **Category 5 boundaries declared here, not deferred to Limitations.**

## 5. The reliability intervention — audit, repair, freeze

**Do not omit this chapter to save space.** It is the strongest methodological
material in the study and most honours reports have no equivalent.

- The audit found 10 major and 10 minor issues; 16 fixed, 4 deferred with stated
  reasons (`docs/repair_report.md`).
- The three that most changed the study: **C1** the temporal split realised
  95.6/3.0/1.4 with test prevalence 1.391%; **C3** rung, best-config and learner
  were selected on *test* NER; **M1** CI-overlap used as a significance rule,
  replaced by a paired stratified bootstrap.
- Superseded artifacts retained, versioned, and indexed by status — never
  deleted.
- The frozen baseline and its guards; `FROZEN_RUNG_CLAIM` asserting the wording
  is not overstated.
- **C2 as an object lesson:** an R-sweep read as a training-cost sweep when only
  the decision threshold moved. This is the same conflation Hájek et al. may have
  made, which is why §8 can raise it without condescension.

## 6. Experiments

Present as a ladder, each answering what the previous left open.

| | Question | Outcome |
|---|---|---|
| **Baseline grid** | Does CSL help at all? | 12/12 negative PR-AUC deltas |
| **E3** | Does that survive seeds and splits? | 220 fits; **116/116 negative**; NER effect learner-dependent and stably so |
| **E2** | Does it hold at other *training* cost ratios? | 364 cells; ranking XGB>RF>DT>LR with 0 violations in 112 |
| **E4** | How much of the precision collapse is prevalence alone? | 104 cells; model and threshold frozen; measured, not suggestive |
| **E6** | Where is the B-vs-C comparison actually informative? | equivalence exact in 104/104; divergence asymmetric |

Each ran against the frozen baseline, each carries QC and a smoke test, and each
report states what it does *not* establish.

## 7. Results

Ordered by how much they should change the reader's mind:

1. Fraud is recoverable by three clauses — published ML results on this dataset
   score below a rule with no parameters.
2. Cost-sensitive **training** does not help; cost-sensitive **thresholding**
   does (49–62% risk reduction).
3. The threshold equivalence — derived and confirmed exactly (E6).
4. `amount` is not the top feature — **with M9 named in the same paragraph.**
5. The base-rate collapse, now measured (E4).

## 8. Discussion

- **The Hájek et al. reconciliation.** The four candidate explanations, feature
  sets first. The paragraph that most improves robustness.
- **Dissolving the threshold disagreement.** Aleksandrova & Armianova reported
  "statistical vs business" thresholds diverging; E6 shows it is a distance
  between two *cost ratios*, predictable in direction and magnitude.
- **What replication means here.** Johnson & Khoshgoftaar's finding reproduces in
  a new domain at 0.129% — and E3 shows it is not a single-split artefact.
- **The synthetic-data problem**, stated as a limit on external validity rather
  than apologised for.

## 9. The design contribution — RQ 2.1.1

Per CHANGE-06 and `docs/authentication_design_contribution.md`. One `R` sets both
the alerting threshold and the authentication escalation ladder. Include the
operational load (97.04% unescalated; 2.96% carrying 90.2% of fraud) as *friction
cost*, and §D's boundaries in full — including the accessibility limit, that the
`High` band's biometric factor is unavailable to the USSD feature-phone users
whose access USSD exists to provide.

## 10. Limitations

Ranked by how much each constrained the finding, per `RESULTS_SYNTHESIS` §6 —
these were *discovered by running the experiments*, which is what makes the
section strong. Carry the still-open items: M8 partly (notebook-04 grid),
**M9 permutation importance**, M10 validation reuse, config F's objective
mismatch.

## 11. Conclusion and future work

The ranked next experiments: the example-dependent CSL arm, grouped permutation
importance (closing M9), threshold re-selection under prevalence shift. The one
extra data column that would be worth most: account age / first-seen timestamp.

---

## Open decisions before drafting prose

1. **Document format and template.** Nothing in the repository records whether
   COS700 expects LaTeX or Word, a department template, or a length limit. This
   blocks drafting, not planning.
2. **Citation style.** `RESULTS_SYNTHESIS` §7.3 flags it as unrecorded.
3. **The freeze staleness** in `docs/claim_taxonomy.md` — one guard is factually
   false post-E3 and two are stale. Resolve before writing §6 and §10, since both
   quote those guards.
4. **Hájek et al. full text** — needed for §8's strongest form.
