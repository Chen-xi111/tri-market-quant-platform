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
        "app_pages/cn_market.py",
        "app_pages/hk_market.py",
        "app_pages/us_market.py",
    ],
)
def test_public_page_renders_without_exceptions(monkeypatch, page):
    monkeypatch.setenv("QUANT_PUBLIC_MODE", "true")
    app = AppTest.from_file(str(ROOT / page), default_timeout=30).run()

    assert not app.exception
