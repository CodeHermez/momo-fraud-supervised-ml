# Fraud detection prototype

An interactive demonstration of the fraud classifier built in this repository:
score a transaction you type in, build a table of them, upload a CSV, and browse
the evidence behind the model.

The prototype is a UI layer only. All inference goes through
`src/momo_fraud/predict.py`, which the project wrote for exactly this purpose —
one code path scores a single transaction and six million, so the demo cannot
drift from the experiment.

## Running it

From the repository root, with the project's virtual environment active:

```bash
pip install -r requirements.txt
streamlit run prototype/app.py
```

It opens on <http://localhost:8501>.

### One extra step for the test-set explorer

The *Test set explorer* page needs a small labelled slice of the held-out split.
It is derived from the raw PaySim CSV, so it is not committed:

```bash
python prototype/scripts/build_demo_sample.py
```

This needs `data/raw/PS_20174392719_1491204439457_log.csv` and takes a couple of
minutes. It writes `prototype/data/demo_sample.parquet` (about 2 MB: every fraud
in the test split plus 40,000 legitimate transactions). Every other page works
without it.

## What it needs on disk

| Path | Used for | Committed? |
|---|---|---|
| `artifacts/` | the model bundle every page loads | yes |
| `figures/`, `results/` | the experiment results gallery | yes |
| `predictions/xgboost__unweighted__test__seed42__no_origin_balance.parquet` | the threshold page's live metrics | no — regenerate with notebook 03 |
| `prototype/data/demo_sample.parquet` | the test set explorer | no — build it with the script above |

Pages degrade with an explanation when something is missing; none of them
traceback.

## The pages

1. **Score a transaction** — a form, plus five real transactions from the
   held-out split covering a clear pass, a near miss, a caught fraud, a false
   alarm and a missed fraud. Returns a risk score, band, recommended action and
   a SHAP-backed explanation in plain English.
2. **Build a table** — an editable grid for composing several transactions and
   scoring them together.
3. **Batch upload** — a CSV in the PaySim schema. Extra columns are ignored; if
   the file carries `isFraud`, a confusion matrix is produced as well.
   `assets/sample_transactions.csv` is a valid 200-row file to start with.
4. **Threshold and risk** — move the severity ratio and the alert line, and
   watch precision, recall, NER and alert volume move across all 954,394
   held-out transactions.
5. **Test set explorer** — the confusion matrix, with every quadrant browsable
   and any row openable.
6. **Experiment results** — the leakage finding, then all 31 figures and 41
   tables in study order.
7. **Model card** — what the model was trained on, how it scored, and its stated
   limitations.

## Input contract

Seven raw fields, matching `FeatureBuilder.transform_one`:

| Field | Notes |
|---|---|
| `step` | hour of the simulation, 1–743 |
| `type` | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | value moved |
| `oldbalanceOrg`, `newbalanceOrig` | required, but **not used by this model** |
| `oldbalanceDest`, `newbalanceDest` | recipient balances |

`nameOrig`, `nameDest`, `isFlaggedFraud` and `isFraud` are accepted in an upload
and ignored — the first two are dropped identifiers, `isFlaggedFraud` is
leakage, and `isFraud` is the label.

The origin-balance columns are collected because the shared feature pipeline
validates their presence, then dropped before the model sees them. This is the
`no_origin_balance` feature set: on PaySim those two columns encode the fraud
label almost perfectly, so a model given them learns the simulator's bookkeeping
rather than fraud. `test_origin_balances_do_not_change_the_score` pins this.

## Two models, one repository

`artifacts/` is a refit on train **and** validation (85% of the data), while the
scores recorded in `predictions/` — and every metric in the model card and the
figures — come from the **train-only** model, as held-out evaluation requires.
They correlate above 0.98 and agree on roughly 99% of flag decisions, but they
are not the same model. Pages that mix the two say so.

## Tests

```bash
pytest prototype/tests
```

- `test_scoring.py` pins scorer parity, single-versus-batch agreement, and the
  input contract.
- `test_pages.py` executes every page through Streamlit's `AppTest` harness, so
  a traceback surfaces here rather than in front of an audience.
- `test_walkthrough.py` drives the demo itself — clicking a sample profile,
  submitting the form, scoring the grid, uploading both a good and a broken
  CSV, moving the alert line — and checks what comes back. Rendering a page and
  demonstrating with it are not the same test.

All three run as part of the repository suite (`pytest` from the root).

## Docker

From the repository root, after building the demo sample:

```bash
docker build -t momo-fraud-demo .          # about 12 minutes cold
docker run --rm -p 8501:8501 momo-fraud-demo
```

Then open <http://localhost:8501>. Build from the repository root, not from this
directory: the image needs `artifacts/`, `figures/`, `results/` and `src/`,
which live above it.

The raw CSV is not copied into the image; the derived demo sample and the single
prediction file the threshold page needs are.

The image installs `requirements.txt` from *this* directory rather than the one
at the repository root. The root file is the research environment — JupyterLab,
LightGBM, imbalanced-learn, the Kaggle client — none of which inference imports;
leaving them out takes the image from 2.5 GB to 1.5 GB. Versions are otherwise
pinned identically, so the container scores exactly as the notebooks do.

To check an image rather than trust it, run the suite inside it:

```bash
docker run -d --name momo-demo -p 8501:8501 momo-fraud-demo
docker exec momo-demo python -m pytest prototype/tests -q   # 54 passed
docker stop momo-demo && docker rm momo-demo
```

Unlike `streamlit run`, which binds to every interface by default, the container
is reachable only on the ports you publish.

## Caveat

PaySim is a simulator, and this model deliberately gives up the shortcut its
fraud offers. Nothing here should be read as a measurement of real mobile money
fraud — see the *Model card* page for the full list of limitations.
