# Critical review of the report draft

Reviewed: `report/main.tex` + 12 chapters, 37 pages, compiling clean.
Emphasis on Chapters 4–9, where the audit intervention, the CSL experiments, the
E-series results and the design contribution have to form one argument.

Findings are ranked within each section. **Fixed** means I corrected it during
this review; everything else is open.

---

## A. Do the research questions actually get answered?

**The single highest-value structural gap: there is no RQ → answer mapping.** An
examiner checks RQ coverage first, and at present they must reconstruct it. Add a
table — in Chapter 1 or Chapter 11 — with four columns: the RQ, the chapter that
answers it, the one-sentence answer, and the claim category from
`docs/claim_taxonomy.md`. This is half a page and it materially changes how the
report is assessed.

| RQ | Coverage | Verdict |
|---|---|---|
| **1** Authentication vulnerabilities | Ch 9 | **Good.** Answered as design, boundary enforced in prose and in code. |
| **2** How effectively can a CSL framework detect fraud | Ch 7 | **Weakest.** Numbers are present but never framed as answering RQ2, and "how effectively" invites an absolute performance claim that the leakage finding deliberately undercuts. Needs an explicit answer of the form: *at the ablated rung, XGBoost reaches NER 0.1314; absolute figures on the full feature set are not interpretable, and that is itself the finding.* |
| **3** CSL versus threshold adjustment | Ch 6–8 | **Strong.** This is the spine and it holds. |
| **4** Which features contribute most | Ch 7 §4 | **Answered but provisional.** The report says so. The RQ mapping must repeat that, or a reader will take the type-over-amount result as settled. |

**A structural point the report never explains: the E-numbering.** Chapters 6–8
refer to E2, E3, E4 and E6 throughout, with no E1 and no E5. A reader has no idea
whether two experiments were lost, failed, or never existed. Either explain the
numbering once (it is the audit's original enumeration, of which some were not
run) or renumber the executed experiments 1–4 in the report. As it stands this
reads as internal jargon leaking into a dissertation.

## B. Do the conclusions follow from E2/E3/E4/E6?

Broadly yes, with one substantive omission.

**E2's actual result is missing.** Chapter 6 reports E2 and then states only the
learner ranking (0 violations in 112). But E2 was commissioned as the
*training-cost-ratio sweep* — the experiment that answers whether the CSL result
depends on the choice of `R_train`. Its finding on that question does not appear
in the report at all. The reader is told the experiment ran and given a ranking
check that was incidental to its purpose. This is the clearest case in the draft
of an experiment being reported rather than used. Fix: state what E2 established
about `R_train`, including the boundary-minimum caveat and the corrected
monotonicity claim.

**Correct but worth tightening:**

- Chapter 7 §2's heading, "Cost-sensitive training does not help; thresholding
  does", is stronger than the section beneath it, which correctly separates
  ranking quality (uniform) from decision risk (learner-dependent). Qualify the
  heading to *…on ranking quality*.
- **Fixed:** Chapter 1 claimed "692 model cells". The true figures are **480
  distinct model fits** reported over **792 evaluation cells** once cached fits
  are reused. Neither is 692; the number was wrong in both readings.

**A structural argument the report should make and does not.** PR-AUC is
threshold-free. Threshold adjustment therefore *cannot* change PR-AUC at all,
while weighting and resampling change the fitted probability surface and so can
only degrade or improve the ranking. That single observation explains the whole
asymmetry the study reports — why the ranking result is uniformly negative while
the decision result is learner-dependent — and it is why the comparison is
constructed across two metrics rather than one. Adding it to Chapter 4 or the
start of Chapter 7 converts a surprising empirical pattern into an explained one.

## C. Overclaiming and unsupported causal language

The draft is disciplined here; the claim taxonomy did its job. Remaining items
are small:

1. **"never improved"** (abstract, Ch 7) is correctly scoped in the body but
   reads as universal in the abstract. Add "across the grid tested" once.
2. **Chapter 5's closing line** — "a freeze is a record with a date on it, not a
   permanent authority" — is editorialising. It is a good sentence and it is not
   evidence; keep it only if the surrounding paragraph stays factual.
3. **No causal language problems found** in Chapters 6–8. Associations are
   labelled as such, and the one weak claim (feature importance) carries its
   confound in the same section rather than in a distant limitation.

## D. Methodology and reproducibility

Strong overall — the threshold protocol, the provenance records and the frozen
baseline are better documented than most reports at this level.

**Gaps:**

1. **No reproducibility statement.** The report never says what software, what
   versions, what seeds, or how to regenerate the results, despite the repository
   carrying all of it. Add a short appendix or a Method subsection: dataset
   checksum, library versions, seeds, and the one-command test invocation.
2. **The account-overlap gap.** **Fixed** — now in Chapter 10 as the most
   consequential unaddressed methodological difference, with the finding that
   Johnson & Khoshgoftaar partitioned folds by provider while this study splits
   at transaction level with 63.28% account overlap. This was found by reading
   the replication target properly, and it is exactly the question a strong
   examiner asks.

## E. Chapter ordering and narrative flow

The arc works. Two adjustments:

1. **Chapter 6 is ordered by experiment ID, which is experiment-log ordering.**
   Retitle each section by the question it answers, with the identifier in
   parentheses — e.g. "Does the result survive seeds and splits? (E3)". Two of
   the four already read this way; make all four consistent.
2. **Chapter 2 is thin for a dissertation.** Six short sections, and the
   mobile-money fraud ML literature is surveyed only through the two papers this
   study engages with directly. For COS700 this is the chapter most likely to be
   marked light. The `Literature/` folder holds 25 papers; perhaps eight are
   currently used.

## F. Does Chapter 5 spend too much space on self-criticism?

**My assessment: the length is defensible; the framing is the risk.**

At roughly two and a half pages it is not disproportionate for what is genuinely
a contribution. Every section earns its place: the three findings changed
conclusions, the "what the repair did not do" section pre-empts the obvious
suspicion that results were quietly dropped, the C2 object lesson is what
licenses Chapter 8 to raise the same conflation in published work, and the freeze
is the study's reproducibility backbone.

The problem is that it currently reads as a **sequence of confessions**. Three
fixes, none of them large:

1. Open with an explicit frame: this chapter describes controls and what they
   changed, not an apology.
2. Close with a "what survives" paragraph — the audit's independent recomputation
   confirmed the main grid, the NER framework and the threshold protocol
   unchanged. Without this the chapter ends on damage.
3. Cut perhaps 20% from the C1/C3/M1 narration, which is more detailed than the
   argument needs.

**Do not move it to an appendix.** It is chronologically load-bearing — the
frozen baseline that Chapter 6 depends on does not exist without it — and it is
the strongest differentiator the report has. But discuss the framing with your
supervisor before submission, because reception varies and that is a judgement
about your examiners, not about the evidence.

## G. Does the discussion explain the surprising findings?

**Partly.** The threshold disagreement is now genuinely explained, with
Aleksandrova & Armianova's own cost ratio (≈2.28) shown to sit away from their
80:20 class distribution — precisely the divergence condition the theorem
predicts. That is a strong section.

**The central surprise is not explained.** Why does cost-sensitive training
degrade ranking quality? The report establishes it 116 times over and never
offers a mechanism. See §B above for the structural answer, which is available
and short. Leaving it unexplained is the most likely "so why does this happen?"
question in a viva.

**Hájek et al. remains hypothetical** and is correctly flagged as such. See
`docs/hajek_comparability.md` for the extraction template.

## H. Are the limitations appropriately scoped?

Yes — and this is among the strongest chapters. Limits are ranked by
consequence, separated into data / method / robustness, and the framing that they
were *discovered by running the experiments* is both true and to the report's
credit. The account-partitioning gap is now included.

One addition: the limitations do not mention that **only one of thirty-one
generated figures is used** (see §J), which is a presentation limitation rather
than a methodological one — but it belongs in a revision plan, not the chapter.

## I. Examiner and reviewer weaknesses, ranked

1. **The unverifiable benchmark.** `constants.THAR_WAI_PR_AUC` supplies published
   comparison values attributed to a 2025 source that could not be located.
   **Fixed to the extent possible** — Chapter 3 now carries a visible draft
   footnote, and `refs.bib` marks the entry `UNRESOLVED`. This must be resolved
   or removed. Submitting numbers attributed to an untraceable source is the
   single most damaging finding available to an examiner, and it is avoidable.
2. **No RQ → answer map** (§A).
3. **E2 reported but not used** (§B).
4. **The unexplained mechanism** behind the ranking result (§G).
5. **Account overlap** — disclosed, unfixed. Defensible as disclosed.
6. **E6's grid was analyst-chosen** — disclosed. Fine.
7. **Chronology unproven** — disclosed, and the wording is careful.
8. **Two threat-model papers cited but unread** (§J).
9. **The E-numbering** (§A).

## J. Citation and reference problems

**Fixed during this review** — five references verified against primary sources
rather than memory:

| Reference | Verified from |
|---|---|
| Johnson & Khoshgoftaar (2022) | the PDF in `Literature/`; ICMLA, pp. 1427–1434 |
| Aleksandrova & Armianova (2022) | the PDF in `Literature/`; ICAI'22, Varna, p. 89 |
| Lopez-Rojas et al. (2016) | the open proceedings PDF; EMSS, pp. 249–255, ISBN read from the paper |
| Elkan (2001) | dblp and ACM agree: IJCAI'01 vol. 2, pp. 973–978 |
| Saito & Rehmsmeier (2015) | the reference list of Johnson & Khoshgoftaar |

Two findings came out of reading rather than citing:

- **The PaySim paper does not document fraud generation.** It states that "the
  injection of malicious fraud behaviour … [is] outside the scope of this paper".
  The advice in `RESULTS_SYNTHESIS` §5.1 to "read section 3 and quote the
  fraud-agent behaviour" rests on a false premise. Chapters 2 and 3 now say so
  explicitly, and the three-clause result is framed as a property of the released
  dataset rather than a restatement of a documented rule.
- **Johnson & Khoshgoftaar's cost parameterisation is identical to this
  study's** — they fix `C(1,0)=1`, set `C(0,1)=N₋/N₊`, and derive both
  `λ = 1/(1+C(0,1))` and `N'₋ = N₋·C(1,0)/C(0,1)`. The replication is
  like-for-like at the level of the cost model, not only the headline claim. This
  is a materially stronger statement than the draft made and Chapter 2 now makes
  it.

**Open:**

- **Thar & Wai (2025)** — unresolved (§I.1).
- **Two Literature papers cited in Chapter 9 but not read closely** — the USSD
  vulnerability and SIM-swap papers. Chapter 9's specific mechanisms must not be
  asserted until these are read. `refs.bib` marks both.
- **Citation style unlocked** — now a one-line switch in `main.tex` with the
  three candidate styles pre-written, so confirming it later costs nothing and
  no prose needs rewriting either way.

## K. Grammar and academic tone

Sound. It reads as academic prose rather than as notebook commentary. Two habits
to watch on a revision pass: several sentences run past forty words, and the
em-dash is doing a lot of work — both are stylistic, neither is an error.

Voice is consistent. Hedging is calibrated rather than reflexive, which is the
harder thing to get right.

## L. Consistency between tables, figures and prose

**Every quality-control figure quoted in Chapter 6 matches its source report**
(E3 22/22 and 16/16; E2; E4 14/14 and 7/7; E6 15/15 and 8/8). Every number I
spot-checked against `results/` reconciles. Tables are `\input` from generated
files, so no figure is transcribed by hand — this is the report's strongest
traceability property and it should be stated in the reproducibility appendix
recommended in §D.

**Two problems:**

1. **Fixed — a whole experiment was missing.** `report/tables/replication.tex`
   was generated but never included, and the three-arm published-comparison
   replication appeared nowhere in the report, despite Chapter 6 referring to E4
   as *superseding* it. Notebooks 09 and 10 (temporal generalisation, velocity
   features, learner selection, tuning, stacking, the failed fit) appeared only in
   the limitations. Chapter 6 now has sections covering all of it, and every
   generated table is used.
2. **Open — one figure in 37 pages.** The project has produced 31 figure pairs
   and the report uses exactly one. Candidates that would each carry real
   argumentative weight: the class balance and drain-signature plots in Chapter 3,
   the ablation ladder, the PR curves and calibration plots in Chapter 7, and the
   risk-band plot in Chapter 9. For a dissertation this is the most visible
   presentation gap remaining.

## M. Does it read like a COS700 dissertation or an experiment log?

**Mostly a dissertation**, and the narrative arc is the reason: the report argues
a position and shows how it was arrived at, rather than enumerating runs.

Three things still read as log:

1. Section titles in Chapter 6 keyed to experiment identifiers (§E.1).
2. The unexplained E-numbering (§A).
3. One figure, thirty-one available (§L.2) — a dissertation is a visual document
   and this one currently is not.

Fixing those three, adding the RQ map, and stating E2's actual finding would put
the draft in good shape. None of them requires new experiments, which is the
right constraint.

---

## Recommended order of work

1. Resolve or remove the Thar & Wai benchmark. **Blocking.**
2. Add the RQ → answer table.
3. State E2's finding on `R_train`.
4. Add the PR-AUC-is-threshold-free mechanism paragraph.
5. Add figures to Chapters 3, 7 and 9.
6. Reframe Chapter 5's opening and closing; trim ~20%.
7. Explain or remove the E-numbering.
8. Read the two threat-model papers; make Chapter 9's citations real.
9. Add the reproducibility appendix.
10. Confirm the citation style and flip the switch.
