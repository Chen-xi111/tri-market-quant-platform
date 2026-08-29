from __future__ import annotations

from scripts.run_all import active_services


def test_public_launcher_starts_only_the_portal():
    services = active_services({"QUANT_PUBLIC_MODE": "true"})
    assert [(service.name, service.port) for service in services] == [("portal", 8500)]


def test_local_launcher_keeps_process_isolation():
    services = active_services({"QUANT_PUBLIC_MODE": "false"})
    assert [(service.name, service.port) for service in services] == [
        ("us", 8501),
        ("hk", 8502),
        ("cn", 8503),
        ("portal", 8500),
    ]
