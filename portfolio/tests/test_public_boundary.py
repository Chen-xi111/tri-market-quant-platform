from __future__ import annotations

import pytest

from platform_hub import public_mode_enabled


@pytest.mark.parametrize("value", [None, "true", "1", "yes", "on"])
def test_public_mode_is_the_safe_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("QUANT_PUBLIC_MODE", raising=False)
    else:
        monkeypatch.setenv("QUANT_PUBLIC_MODE", value)

    assert public_mode_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_local_mode_requires_an_explicit_opt_out(monkeypatch, value):
    monkeypatch.setenv("QUANT_PUBLIC_MODE", value)

    assert public_mode_enabled() is False
