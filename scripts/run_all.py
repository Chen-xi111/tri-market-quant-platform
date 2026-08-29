#!/usr/bin/env python3
"""Run the public portal or the complete four-process local workspace."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".runtime"
LOG_DIR = RUNTIME_DIR / "logs"


@dataclass(frozen=True)
class Service:
    name: str
    cwd: Path
    script: Path
    port: int


SERVICES = (
    Service("us", ROOT / "systems/us", ROOT / "systems/us/dashboard.py", 8501),
    Service("hk", ROOT / "systems/hk", ROOT / "systems/hk/streamlit_app.py", 8502),
    Service("cn", ROOT / "systems/cn", ROOT / "systems/cn/streamlit_app.py", 8503),
    Service("portal", ROOT, ROOT / "streamlit_app.py", 8500),
)


def public_mode_requested(environment: dict[str, str]) -> bool:
    return environment.get("QUANT_PUBLIC_MODE", "false").strip().lower() not in {
        "false",
        "0",
        "no",
        "off",
    }


def active_services(environment: dict[str, str]) -> tuple[Service, ...]:
    if public_mode_requested(environment):
        return tuple(service for service in SERVICES if service.name == "portal")
    return SERVICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for all Streamlit services (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--public-host",
        default="localhost",
        help="Hostname used by browser links and embedded market pages.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Python interpreter to use. Defaults to .venv/bin/python when present.",
    )
    return parser.parse_args()


def python_executable(explicit: Path | None) -> Path:
    if explicit:
        return explicit.expanduser().resolve()
    project_python = ROOT / ".venv/bin/python"
    return project_python if project_python.exists() else Path(sys.executable)


def port_is_open(host: str, port: int, timeout: float = 0.25) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((probe_host, port), timeout=timeout):
            return True
    except OSError:
        return False


def health_is_ready(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with urlopen(f"http://{probe_host}:{port}/_stcore/health", timeout=0.5) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def tail(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "No log output was captured."


def stop_processes(processes: list[subprocess.Popen], timeout: float = 8.0) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + timeout
    for process in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        if process.poll() is None:
            process.wait()


def main() -> int:
    args = parse_args()
    python = python_executable(args.python)
    if not python.exists():
        print(f"Python interpreter not found: {python}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment.update(
        {
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            "QUANT_INTERNAL_HOST": "127.0.0.1",
            "QUANT_PUBLIC_HOST": args.public_host,
            "QUANT_PUBLIC_SCHEME": "http",
        }
    )
    environment.setdefault("QUANT_PUBLIC_MODE", "false")
    services = active_services(environment)

    reused = {
        service.name
        for service in services
        if service.name != "portal" and health_is_ready(args.host, service.port)
    }
    blocked = [
        service.port
        for service in services
        if service.name not in reused and port_is_open(args.host, service.port)
    ]
    if blocked:
        print(
            "Cannot start the unified platform because these ports are in use: "
            + ", ".join(map(str, blocked)),
            file=sys.stderr,
        )
        return 2

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    managed: list[tuple[Service, subprocess.Popen]] = []
    log_handles = []
    stopping = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        for service in services:
            if service.name in reused:
                continue
            market_cache = service.cwd / ".market_cache"
            mpl_cache = service.cwd / ".mpl_cache"
            market_cache.mkdir(exist_ok=True)
            mpl_cache.mkdir(exist_ok=True)

            service_env = environment.copy()
            service_env["MPLCONFIGDIR"] = str(mpl_cache)

            log_path = LOG_DIR / f"{service.name}.log"
            log_handle = log_path.open("w")
            log_handles.append(log_handle)
            command = [
                str(python),
                "-m",
                "streamlit",
                "run",
                str(service.script),
                f"--server.address={args.host}",
                f"--server.port={service.port}",
                "--server.headless=true",
                "--server.runOnSave=true",
                "--browser.gatherUsageStats=false",
            ]
            process = subprocess.Popen(
                command,
                cwd=service.cwd,
                env=service_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            managed.append((service, process))

        deadline = time.monotonic() + 35
        pending = {service.name: service for service in services if service.name not in reused}
        while pending and time.monotonic() < deadline and not stopping:
            for service, process in managed:
                if service.name not in pending:
                    continue
                if process.poll() is not None:
                    print(f"Service {service.name} exited during startup.", file=sys.stderr)
                    print(tail(LOG_DIR / f"{service.name}.log"), file=sys.stderr)
                    return 1
                if health_is_ready(args.host, service.port):
                    pending.pop(service.name)
            time.sleep(0.25)

        if pending:
            print("Startup timed out for: " + ", ".join(sorted(pending)), file=sys.stderr)
            for name in sorted(pending):
                print(f"\n--- {name}.log ---", file=sys.stderr)
                print(tail(LOG_DIR / f"{name}.log"), file=sys.stderr)
            return 1

        print("Tri-market quant platform is ready.")
        print(f"Portal:       http://{args.public_host}:8500")
        if public_mode_requested(environment):
            print("Public mode: fixed snapshots, read-only UI, no child market services.")
        else:
            print(f"US equities:  http://{args.public_host}:8501")
            print(f"Hong Kong:    http://{args.public_host}:8502")
            print(f"China A:      http://{args.public_host}:8503")
        if reused:
            print("Reused existing healthy services: " + ", ".join(sorted(reused)))
            print("Those reused services are not stopped by this launcher.")
        print("Press Control-C to stop all services.")

        while not stopping:
            for service, process in managed:
                return_code = process.poll()
                if return_code is not None:
                    print(
                        f"Service {service.name} stopped unexpectedly with exit code {return_code}.",
                        file=sys.stderr,
                    )
                    print(tail(LOG_DIR / f"{service.name}.log"), file=sys.stderr)
                    return 1
            time.sleep(0.5)
        return 0
    finally:
        stop_processes([process for _, process in managed])
        for handle in log_handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
