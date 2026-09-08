"""Write the frozen baseline manifest and stamp the freeze into the index.

Three outputs:

* ``results/frozen_experimental_baseline.json`` -- the machine-readable manifest
  every future experiment validates itself against.
* an updated ``results/experiment_index.json`` carrying the freeze decision.
* a provenance record for the freeze itself.

Nothing is computed here. The values come from ``momo_fraud.baseline``, which
records decisions already evidenced in ``docs/repair_report.md`` and the
diagnostics it cites.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from momo_fraud import baseline as B          # noqa: E402
from momo_fraud import evaluate as E          # noqa: E402
from momo_fraud import provenance as P        # noqa: E402

RESULTS = PROJECT_ROOT / "results"


def main() -> None:
    checks_path = RESULTS / "frozen_baseline_checks.json"
    if not checks_path.exists():
        raise SystemExit(
            "Run scripts/freeze/consistency_checks.py --json first: the manifest "
            "records the check verdict and must not claim FROZEN without it."
        )

    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    if checks["verdict"] != "PASS":
        raise SystemExit(
            f"Consistency checks report {checks['verdict']} "
            f"({checks['n_passed']}/{checks['n_checks']}). The baseline is not "
            f"frozen until every control is in force."
        )

    payload = B.manifest()
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["consistency_checks"] = {
        "verdict": checks["verdict"],
        "n_passed": checks["n_passed"],
        "n_checks": checks["n_checks"],
        "artifact": "results/frozen_baseline_checks.json",
    }
    payload["guards"] = {
        "superseded_artifacts": "baseline.load_results refuses status SUPERSEDED",
        "diagnostic_rungs": "experiment.choose_rung excludes them by default",
        "frozen_rung": "baseline.assert_frozen_rung raises on any other rung",
        "experiment_schema": "baseline.validate_experiment_frame",
    }
    payload["unfreezing"] = (
        "Changing anything in this manifest requires a new entry in "
        "docs/methodology_change_log.md stating what changed, why, whether it "
        "was decided before or after seeing downstream results, and which "
        "experiments are affected. A better score is not a reason."
    )

    E.save_json(payload, B.MANIFEST_NAME)
    print(f"results/{B.MANIFEST_NAME}.json written")
    print(f"  status       {payload['status']}")
    print(f"  rung         {payload['feature_set']['frozen_rung']}  "
          f"({payload['feature_set']['claim']})")
    print(f"  split        stratified {payload['split']['train']}/"
          f"{payload['split']['val']}/{payload['split']['test']} "
          f"seed {payload['split']['split_seed']}")
    print(f"  R_default    {payload['cost']['R_default']:.4f}")
    print(f"  learners     {payload['models']['learners']}")
    print(f"  checks       {checks['n_passed']}/{checks['n_checks']} PASS")

    # Stamp the freeze into the artifact index so a reader of either file finds it.
    index_path = RESULTS / f"{B.INDEX_NAME}.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["frozen_baseline"] = {
        "status": "FROZEN",
        "frozen_at_utc": payload["frozen_at_utc"],
        "manifest": f"results/{B.MANIFEST_NAME}.json",
        "protocol": "docs/frozen_baseline_protocol.md",
        "frozen_rung": B.FROZEN_RUNG,
        "claim": B.FROZEN_RUNG_CLAIM,
        "diagnostic_rungs": sorted(B.DIAGNOSTIC_RUNGS),
        "rationale": payload["feature_set"]["rationale"],
    }
    for stem in ("frozen_experimental_baseline", "frozen_baseline_checks"):
        if not any(r["artifact"] == stem for r in index["artifacts"]):
            index["artifacts"].append({
                "artifact": stem,
                "produced_by": "scripts/freeze/",
                "status": "VALID",
                "note": "frozen baseline definition and its verification",
                "present": True,
            })
    E.save_json(index, B.INDEX_NAME)
    print(f"results/{B.INDEX_NAME}.json stamped with the freeze decision")

    P.write(P.experiment_record(
        name="frozen_experimental_baseline",
        seed=payload["seeds"]["model_seed"],
        split=payload["split"],
        model=payload["models"],
        csl={"taxonomy": list(B.CSL_TAXONOMY)},
        cost_ratio=payload["cost"],
        thresholds=payload["threshold_protocol"],
        evaluation=payload["metrics"],
        notes=("Freeze phase. No experiment was run, no model fitted, no "
               "hyperparameter tuned, no feature added or removed."),
    ), "frozen_experimental_baseline_provenance")
    print("results/frozen_experimental_baseline_provenance.json written")


if __name__ == "__main__":
    main()
