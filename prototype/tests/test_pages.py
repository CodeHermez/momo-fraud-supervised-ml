"""Every page must render without raising.

A demo fails in exactly one way that matters: a traceback on screen while
someone is watching. ``AppTest`` executes each page's script the way Streamlit
would and surfaces anything it threw, so that failure happens here instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROTOTYPE = Path(__file__).resolve().parents[1]
PAGES = sorted((PROTOTYPE / "pages").glob("*.py"))

#: Pages do real work -- loading a booster, scoring 41k rows, reading 31 figures.
TIMEOUT = 120


def _run(path: Path) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    app.run()
    return app


def test_landing_page_renders() -> None:
    app = _run(PROTOTYPE / "app.py")
    assert not app.exception, [e.value for e in app.exception]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders(page: Path) -> None:
    app = _run(page)
    assert not app.exception, [e.value for e in app.exception]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_reports_no_error(page: Path) -> None:
    """``st.error`` on load means a missing artifact, not a working page."""
    app = _run(page)
    assert not app.error, [e.value for e in app.error]
