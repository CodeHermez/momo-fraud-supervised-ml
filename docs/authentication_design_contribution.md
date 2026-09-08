# RQ 2.1.1 — resolved as a design contribution, not an empirical result

**Status: research decision taken.** This resolves outstanding decision 5 in
`docs/repair_report.md` §E. Recorded as CHANGE-06 in
`docs/methodology_change_log.md`.

**The decision: reframe RQ 2.1.1 as a design contribution — a mapping from
transaction-risk bands to step-up authentication requirements — and state
explicitly that this is not an empirical security evaluation.**

No new experiment was run. No model was fitted. Nothing in the frozen baseline
changed. The only numbers here are band populations already produced by
`notebooks/06_risk_analysis.ipynb`.

---

## A. Why the empirical route is closed

RQ 2.1.1 asks which authentication vulnerabilities matter in telco mobile money —
USSD session hijacking, SIM swap, static-PIN weakness. PaySim's schema is eleven
columns:

```
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
```

There is no authentication event, session identifier, device fingerprint, SIM
change flag, PIN retry count, channel (USSD / app / agent), or location. The
proposal's own architecture diagram names a USSD Gateway, an SMSC issuing OTPs
and an Auth Server; **none of those three components emits a single field into
the dataset**. The study models the Transaction Switch and the E-Wallet Ledger
only.

This is already a supported conclusion in the frozen baseline
(`baseline.SUPPORTED_CONCLUSIONS`), with `constants.RAW_COLUMNS` as its
evidence. Manufacturing an experiment around variables that do not exist would
be stretching PaySim past what it contains, and the study declines to do it.

## B. What the design contribution is

The study already produces a per-transaction risk score and already partitions it
into four operational bands, with an action attached to each
(`constants.BAND_ACTIONS`, `risk.band_cutoffs`). Band `High` is literally named
`step_up_auth`. The contribution is to make that mapping explicit, principled and
derived rather than incidental.

### B.1 The cutoffs are cost-derived, not chosen

`risk.band_cutoffs` anchors every boundary on the Bayes threshold
`λ = 1/(1+R)`, not on round numbers:

| Band | Score range (0–100) | At `R = 773.70` |
|---|---|---|
| Low | `< λ/10` | `< 0.0129` |
| Medium | `λ/10 … λ` | `0.0129 … 0.1291` |
| High | `λ … 10λ` | `0.1291 … 1.2908` |
| Critical | `≥ 10λ` | `≥ 1.2908` |

`High` begins exactly at `λ` — the point at which flagging becomes
risk-reducing. So the band structure is a consequence of the study's single
declared parameter.

> **The design claim.** One parameter — *R*, "missing one fraud is as bad as R
> false alarms" — sets both the alerting threshold **and** the authentication
> escalation ladder. An institution states its risk appetite once and gets both,
> instead of tuning a detection threshold and a friction policy independently
> against each other.

This is the part that is genuinely a contribution: not that step-up
authentication is a good idea, which is well established, but that the
*trigger points* can be derived from the same cost ratio that governs the
classifier, rather than set by committee.

### B.2 The escalation rule is factor-category-based

Ali, Dida & Sam (2020) — a review of 97 papers on mobile money 2FA threat models
— records that the deployed scheme authenticates a subscriber with **a registered
SIM card (ownership factor) plus a mobile money PIN (knowledge factor)**, and
that this "only uses a personal identification number (PIN) and a subscriber
identity module (SIM) to authenticate users, which are susceptible to attacks."
They classify factors as knowledge, ownership and biometric (inherence).

Mapping the study's motivating threats onto those categories:

| Threat | Factor it defeats | Baseline 2FA after the attack |
|---|---|---|
| Vishing / social engineering | knowledge (PIN disclosed) | ownership only |
| SIM swap / remote SIM provisioning abuse | ownership (SIM re-issued) | knowledge only |
| Both together | knowledge + ownership | nothing remains |

**The rule this implies:** a step-up must add a factor from a category the
suspected attack does not already defeat, and it must travel over a channel
independent of the compromised one. Escalating from "PIN" to "PIN again" adds
nothing. Escalating to an SMS OTP adds nothing against SIM swap, because the SMS
arrives on the swapped SIM.

### B.3 The ladder

| Band | Action | Authentication requirement | Category added | Threat addressed |
|---|---|---|---|---|
| **Low** | `allow` | Baseline 2FA: SIM + PIN | — (status quo) | — |
| **Medium** | `monitor` | Baseline 2FA, plus out-of-band notification opening a repudiation window | none — detection, not prevention | gives the account holder a chance to repudiate |
| **High** | `step_up_auth` | A third factor from the **biometric** category, bound to the handset | inherence | vishing (PIN known) and SIM swap (SIM held) both fail this |
| **Critical** | `block` | Hold, plus human verification over a channel **not** tied to the SIM | out-of-band identity | SIM swap, agent-side collusion |

The escalation is deliberately *non-monotone in factor count* and monotone in
*factor independence*: what increases with risk is not how many credentials are
demanded but how independent the added credential is from the ones the attack is
presumed to have taken.

## C. What is measured — the operational load

This is the one part of the design that PaySim **can** quantify: how much friction
the ladder imposes, and where. XGBoost at the frozen rung, test split
(`results/06_risk_bands.csv`):

| Band | Action | Transactions | Share of traffic | Fraud | Share of fraud | Fraud rate |
|---|---|---|---|---|---|---|
| Low | `allow` | 710,254 | 74.42% | 15 | 1.22% | 0.002% |
| Medium | `monitor` | 215,866 | 22.62% | 106 | 8.60% | 0.049% |
| High | `step_up_auth` | 25,175 | 2.64% | 91 | 7.39% | 0.361% |
| Critical | `block` | 3,099 | 0.32% | 1,020 | **82.79%** | 32.91% |

> **97.04% of transactions never see additional friction, while the two escalated
> bands — 2.96% of traffic — carry 90.2% of the fraud.**

That is a statement about *cost of friction*, and it is measurable because it
depends only on the score distribution. It is the honest empirical component of a
design contribution.

## D. What is explicitly NOT claimed

This section exists so the distinction the decision rests on cannot be blurred.

**Not claimed — nothing here is an empirical security evaluation.**

1. **No control was tested.** Nothing in this study shows that a biometric
   step-up prevents a vished transaction, or that out-of-band verification
   defeats a SIM swap. Those are design arguments from the threat-model
   literature, not results.
2. **The bands are calibrated on the wrong threat.** PaySim's fraud is an
   account-draining signature — `amount == oldbalanceOrg` in 97.82% of frauds —
   not an authentication compromise. The load figures in §C describe how the
   ladder would behave on *this* fraud process. They do not transfer to vishing
   or SIM swap, whose score distribution is unknown and unmeasured here.
3. **No false-reject or usability measurement exists.** A biometric step-up has a
   false-reject rate, and rejecting a legitimate customer at `High` is a real
   cost that appears nowhere in NER.
4. **Channel independence is assumed, not verified.** "Out-of-band" is only
   meaningful if the second channel is genuinely independent of the first; on a
   single handset with a swapped SIM, several plausible channels are not.
5. **No attacker adaptation is modelled.** A published, deterministic band
   boundary is a target: an adversary who knows `λ` can size transactions to sit
   in `Medium`. The static-threshold assumption is exactly what an adaptive
   adversary breaks.

**The accessibility limitation, stated plainly.** The `High` band requires a
biometric factor, which assumes a handset that can capture one. A large share of
mobile money users transact over USSD on feature phones precisely because that is
what they have — the Ali et al. review notes USSD is "handset-independent," which
is the source of its reach. A ladder whose escalation step is unavailable to
feature-phone users either fails open for them or excludes them from higher-value
transactions. **This is a limitation of the design, not an oversight in it**, and
the report should carry it rather than let a reader assume universal
applicability. A defensible fallback for the USSD case — agent-mediated
verification, or a callback to a registered alternate number — is a design
question this study does not resolve.

## E. What would be needed to evaluate this empirically

For the record, and to make the gap constructive rather than merely
acknowledged. The minimum fields:

| Field | Makes testable |
|---|---|
| SIM-change flag with timestamp | whether recent SIM change predicts fraud — the SIM-swap hypothesis |
| Device / handset fingerprint | device-change anomalies; whether biometric step-up is even available |
| Channel (USSD / app / agent / STK) | whether risk differs by channel, which the ladder assumes |
| Failed-PIN count, session duration | brute force and session-hijacking signals |
| Per-typology fraud label | whether the ladder routes *the right threat* to the right control |
| Step-up outcome (passed / failed / abandoned) | whether the control actually stops anything |

The last row is the one that converts this from design to evaluation: without
the outcome of an escalation, no dataset can say whether escalating helped. This
matches `report/RESULTS_SYNTHESIS.md` §6.9, which ranks account age, typology
label and SIM/device flags as the highest-value missing columns.

## F. Literature this section rests on

Properly read and cited above:

- **Ali, G., Dida, M. A. & Sam, A. E. (2020).** *Two-Factor Authentication Scheme
  for Mobile Money: A Review of Threat Models and Countermeasures.* Future
  Internet 12(9), 160. doi:10.3390/fi12100160. — the factor taxonomy, the
  deployed SIM+PIN scheme, and the five threat-model categories.

Present in `Literature/` and required for the section, **not yet read closely**:

- *Critical Vulnerabilities in USSD Banking Authentication Protocols* — the USSD
  session-hijacking claim rests on this and should not be asserted without it.
- *Securing SIM Toolkit-Based Mobile Money Applications Against SIM Swap Attacks*
- *Security Analysis of the Consumer Remote SIM Provisioning Protocol*
- *Role of authentication factors in Fin-tech mobile transaction security*
- *Mobile Money Security*

`report/RESULTS_SYNTHESIS.md` §5.3 is right that these must carry RQ 2.1.1 alone,
and that citing them in passing is not enough. This document does not discharge
that obligation; it fixes the *frame* the section is written in.

## G. How the report should present it

1. **Methodology** states that RQ 2.1.1 is not answerable from PaySim, with
   `constants.RAW_COLUMNS` as the evidence — before any result is presented, not
   as a limitation discovered late.
2. **The contribution is placed as design**, in its own subsection, with §B.1's
   derivation (one `R`, both the threshold and the ladder) as the claim.
3. **§C is presented as operational load**, never as security effectiveness.
4. **§D is carried in full**, including the accessibility limitation.
5. **The threat-model claims are cited to the literature**, not to this study's
   experiments.

The resulting claim is narrower than the proposal implied and considerably
stronger than a stretched empirical one: the study derives an authentication
escalation policy from its cost model, quantifies the friction that policy would
impose, and is explicit that whether the escalations work is not something PaySim
can be asked.
