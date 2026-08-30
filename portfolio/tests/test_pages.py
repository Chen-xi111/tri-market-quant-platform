from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "page",
    [
        "streamlit_app.py",
        "app_pages/methodology.py",
        "app_pages/comparison.py",
        "app_pages/cn_market.py",
        "app_pages/hk_market.py",
        "app_pages/us_market.py",
    ],
)
def test_public_page_renders_without_exceptions(monkeypatch, page):
    monkeypatch.setenv("QUANT_PUBLIC_MODE", "true")
    app = AppTest.from_file(str(ROOT / page), default_timeout=30).run()

    assert not app.exception


@pytest.mark.parametrize(
    ("page", "market_key"),
    [
        ("app_pages/cn_market.py", "cn"),
        ("app_pages/hk_market.py", "hk"),
        ("app_pages/us_market.py", "us"),
    ],
)
def test_public_demo_controls_and_lazy_evidence_tabs(monkeypatch, page, market_key):
    monkeypatch.setenv("QUANT_PUBLIC_MODE", "true")
    app = AppTest.from_file(str(ROOT / page), default_timeout=30).run()

    app.selectbox[1].set_value("breakout")
    app.selectbox[2].set_value(25)
    app.button[0].click()
    app.run()
    assert not app.exception
    assert len(app.metric) == 4
    assert app.session_state[f"demo_cost_{market_key}"] == 25

    app.session_state[f"demo_tabs_{market_key}"] = "Robustness"
    app.run()
    assert not app.exception
    assert app.tabs[1].children

    app.session_state[f"demo_tabs_{market_key}"] = "Data and assumptions"
    app.run()
    assert not app.exception
    assert app.tabs[2].children


def test_cross_market_controls_and_lazy_scorecards(monkeypatch):
    monkeypatch.setenv("QUANT_PUBLIC_MODE", "true")
    app = AppTest.from_file(str(ROOT / "app_pages/comparison.py"), default_timeout=30).run()

    assert not app.exception
    assert len(app.metric) == 4
    app.selectbox[0].set_value("breakout")
    app.segmented_control[0].set_value("Common cost")
    app.run()
    app.selectbox[1].set_value(25)
    app.run()
    assert not app.exception

    app.session_state["comparison_tabs"] = "Scorecard"
    app.run()
    assert not app.exception
    assert app.tabs[1].children

    app.session_state["comparison_tabs"] = "Comparison contract"
    app.run()
    assert not app.exception
    assert app.tabs[2].children
