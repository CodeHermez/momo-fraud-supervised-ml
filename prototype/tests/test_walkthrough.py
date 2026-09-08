"""The demo path itself, driven end to end.

``test_pages`` proves each page *renders*. That is not the same as proving the
demo *works*: every interesting thing here happens after a click. These tests
walk the path someone would actually walk in front of an audience -- load a
profile, submit the form, score a grid, upload a file, move the alert line --
and assert on what comes back, not merely on the absence of a traceback.

Two steps of the walkthrough cannot be driven from here, because ``AppTest``
has no handle for them: editing cells in ``st.data_editor`` (the seeded grid is
scored instead) and clicking a row in ``st.dataframe`` on the explorer page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROTOTYPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTOTYPE))

from core import schema  # noqa: E402

TIMEOUT = 180

SAMPLE_CSV = PROTOTYPE / "assets" / "sample_transactions.csv"

#: One valid row followed by the three ways a row goes wrong.
BROKEN_CSV = (
    "step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest\n"
    "1,PAYMENT,1000.00,0,0,0,0\n"
    "1,WIRE,1000.00,0,0,0,0\n"
    "1,PAYMENT,-5.00,0,0,0,0\n"
    "99999,PAYMENT,1000.00,0,0,0,0\n"
).encode("utf-8")


def _open(page: str) -> AppTest:
    app = AppTest.from_file(str(PROTOTYPE / "pages" / page), default_timeout=TIMEOUT)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _click(app: AppTest, label: str) -> AppTest:
    for button in app.button:
        if button.label == label:
            button.click()
            app.run()
            assert not app.exception, [e.value for e in app.exception]
            return app
    raise AssertionError(f"no button labelled {label!r}; found {[b.label for b in app.button]}")


def _metrics(app: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in app.metric}


# --------------------------------------------------------------------------
# Page 1 -- score a transaction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", schema.SAMPLE_PROFILES, ids=lambda p: p.name)
def test_profile_button_loads_that_transaction(profile: schema.Profile) -> None:
    app = _click(_open("1_Score_a_Transaction.py"), profile.name)
    assert app.session_state["transaction"] == profile.transaction


@pytest.mark.parametrize("profile", schema.SAMPLE_PROFILES, ids=lambda p: p.name)
def test_every_profile_states_its_ground_truth(profile: schema.Profile) -> None:
    """The demo must own its mistakes as loudly as its successes.

    Two of the five profiles are a false alarm and a missed fraud, so the page
    is expected to render ``st.error`` for them -- that is the honest outcome,
    not a failure.
    """
    app = _click(_open("1_Score_a_Transaction.py"), profile.name)
    verdicts = [s.value for s in app.success] + [e.value for e in app.error]
    stated = [v for v in verdicts if "Ground truth" in v]
    assert len(stated) == 1, verdicts
    assert ("was fraud" if profile.actual_fraud else "was legitimate") in stated[0]


def test_the_drain_transfer_is_caught() -> None:
    fraud = next(
        p for p in schema.SAMPLE_PROFILES if p.actual_fraud and p.recorded_score > 0.5
    )
    app = _click(_open("1_Score_a_Transaction.py"), fraud.name)

    assert _metrics(app)["Alert"] == "FLAGGED"
    assert any("got this right" in s.value for s in app.success)


def test_submitting_the_form_scores_what_was_typed() -> None:
    app = _open("1_Score_a_Transaction.py")

    app.selectbox[0].set_value("TRANSFER")
    for widget in app.number_input:
        if widget.label == "Amount":
            widget.set_value(250_000.0)
    app = _click(app, "Score it")

    typed = app.session_state["transaction"]
    assert typed["type"] == "TRANSFER"
    assert typed["amount"] == pytest.approx(250_000.0)
    assert "Risk score" in _metrics(app)


# --------------------------------------------------------------------------
# Page 2 -- build a table
# --------------------------------------------------------------------------


def test_scoring_the_seeded_grid() -> None:
    app = _click(_open("2_Build_a_Table.py"), "Score all rows")
    metrics = _metrics(app)

    assert metrics["Rows scored"] == str(len(schema.SAMPLE_PROFILES[:3]))
    assert int(metrics["Flagged"]) >= 1
    assert not app.error, [e.value for e in app.error]


def test_adding_a_blank_row_then_scoring() -> None:
    app = _click(_open("2_Build_a_Table.py"), "Add a blank row")
    app = _click(app, "Score all rows")

    assert _metrics(app)["Rows scored"] == "4"


# --------------------------------------------------------------------------
# Page 3 -- batch upload
# --------------------------------------------------------------------------


def _upload(app: AppTest, name: str, content: bytes) -> AppTest:
    app.file_uploader[0].upload(name, content, "text/csv")
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_uploading_the_bundled_sample() -> None:
    app = _upload(_open("3_Batch_Upload.py"), SAMPLE_CSV.name, SAMPLE_CSV.read_bytes())
    metrics = _metrics(app)

    assert metrics["Rows scored"] == "200"
    assert not app.warning, [w.value for w in app.warning]

    # The file carries ``isFraud``, so the scorecard must appear too.
    assert "Precision" in metrics and "Recall" in metrics


def test_the_sample_names_the_columns_it_ignores() -> None:
    app = _upload(_open("3_Batch_Upload.py"), SAMPLE_CSV.name, SAMPLE_CSV.read_bytes())
    ignored = app.dataframe[0].value

    assert list(ignored.columns) == ["column", "why it is ignored"]
    assert set(ignored["column"]) <= set(schema.IGNORED_FIELDS)


def test_a_broken_upload_reports_each_problem_and_scores_the_rest() -> None:
    app = _upload(_open("3_Batch_Upload.py"), "broken.csv", BROKEN_CSV)

    assert app.warning, "three bad rows should raise a warning"
    assert "3 problem(s)" in app.warning[0].value
    assert _metrics(app)["Rows scored"] == "1"


# --------------------------------------------------------------------------
# Page 4 -- threshold and risk
# --------------------------------------------------------------------------


def test_default_controls_reproduce_the_model_card() -> None:
    from core import scorer

    card = scorer.get_model_card()["held_out_performance"]
    metrics = _metrics(_open("4_Threshold_and_Risk.py"))

    assert metrics["Recall"] == f"{card['recall']:.2%}"
    assert metrics["Precision"] == f"{card['precision']:.2%}"
    assert metrics["NER"] == f"{card['ner']:.4f}"


def test_raising_the_threshold_trades_recall_for_precision() -> None:
    app = _open("4_Threshold_and_Risk.py")
    before = _metrics(app)

    # Well above the shipped 0.0019, but inside the slider's 0.05 ceiling.
    app.slider[0].set_value(0.02)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    after = _metrics(app)

    def percent(value: str) -> float:
        return float(value.rstrip("%"))

    assert percent(after["Recall"]) < percent(before["Recall"])
    assert percent(after["Precision"]) > percent(before["Precision"])
    assert percent(after["Alert rate"]) < percent(before["Alert rate"])


def test_the_risk_optimal_cut_beats_the_shipped_one_on_this_split() -> None:
    """Lower NER is better, and the optimum is measured on the test split.

    The shipped threshold was frozen on validation, so it cannot win here. This
    pins the direction of that gap rather than pretending it does not exist.
    """
    app = _open("4_Threshold_and_Risk.py")
    shipped_ner = float(_metrics(app)["NER"])

    app.checkbox[1].check()
    app.run()
    assert not app.exception, [e.value for e in app.exception]

    assert float(_metrics(app)["NER"]) < shipped_ner
