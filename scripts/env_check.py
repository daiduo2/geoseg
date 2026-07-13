#!/usr/bin/env python3
"""Environment health checker for geoseg.

Run with uv:

    uv run scripts/env_check.py
    uv run scripts/env_check.py --fix
    uv run scripts/env_check.py --json

The script validates that the project is using uv/pnpm as the single source of
package management and that the virtual environment is clean (no system
site-packages leakage).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PATH = PROJECT_ROOT / ".venv"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

REQUIRED_PYTHON = (3, 10)

PYTHON_DEPS: list[tuple[str, str | None]] = [
    ("numpy", "ndarray"),
    ("PIL", "Image"),
    ("pydantic", "BaseModel"),
    ("skimage", "measure"),
    ("scipy", "ndimage"),
    ("sklearn", "cluster"),
    ("cv2", None),
    ("napari", "Viewer"),
    ("fitz", None),
    ("requests", None),
]

PROJECT_MODULES = [
    "geoseg",
    "geoseg.controller",
    "geoseg.session_state",
    "geoseg.modules.cv_detect",
    "geoseg.modules.segment_engines",
    "geoseg.modules.post_process",
    "geoseg.modules.exporter",
    "geoseg.modules.editor",
    "geoseg.modules.pdf_extractor",
    "geoseg.modules.mineru_client",
]


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        **kwargs,
    )


def get_uv_version() -> str | None:
    proc = run(["uv", "--version"])
    return proc.stdout.strip() if proc.returncode == 0 else None


def get_pnpm_version() -> str | None:
    proc = run(["pnpm", "--version"])
    return proc.stdout.strip() if proc.returncode == 0 else None


def get_python_version() -> tuple[int, int, int] | None:
    proc = run([sys.executable, "--version"])
    if proc.returncode != 0:
        return None
    parts = proc.stdout.strip().replace("Python ", "").split(".")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def venv_config() -> dict[str, str]:
    cfg_path = VENV_PATH / "pyvenv.cfg"
    result: dict[str, str] = {}
    if not cfg_path.exists():
        return result
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def is_uv_managed_venv() -> bool:
    cfg = venv_config()
    return "uv" in cfg


def venv_allows_system_site_packages() -> bool:
    cfg = venv_config()
    value = cfg.get("include-system-site-packages", "false").lower()
    return value in ("true", "1", "yes")


def check_import(module_name: str, attr: str | None) -> tuple[bool, float, str]:
    start = time.perf_counter()
    try:
        mod = __import__(module_name, fromlist=[attr] if attr else [])
        if attr:
            getattr(mod, attr)
        elapsed = time.perf_counter() - start
        return True, elapsed, ""
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return False, elapsed, str(exc)


def check_command(name: str, args: list[str]) -> tuple[bool, str, str]:
    proc = run([name, *args])
    if proc.returncode != 0:
        return False, "", proc.stderr.strip() or f"{name} failed"
    return True, proc.stdout.strip(), ""


def format_duration(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}µs"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def print_section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(title)
    print("─" * 50)


def perform_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "venv_path": str(VENV_PATH),
        "checks": {},
    }
    all_ok = True

    # Python version
    py_version = get_python_version()
    py_ok = py_version is not None and py_version[:2] >= REQUIRED_PYTHON
    checks["checks"]["python_version"] = {
        "ok": py_ok,
        "value": ".".join(map(str, py_version)) if py_version else None,
        "required": ".".join(map(str, REQUIRED_PYTHON)),
    }
    all_ok &= py_ok

    # uv
    uv_version = get_uv_version()
    uv_ok = uv_version is not None
    checks["checks"]["uv"] = {
        "ok": uv_ok,
        "value": uv_version,
        "command": "uv --version",
    }
    all_ok &= uv_ok

    # pnpm
    pnpm_version = get_pnpm_version()
    pnpm_ok = pnpm_version is not None
    checks["checks"]["pnpm"] = {
        "ok": pnpm_ok,
        "value": pnpm_version,
        "command": "pnpm --version",
    }
    all_ok &= pnpm_ok

    # venv state
    venv_exists = VENV_PATH.exists()
    cfg = venv_config()
    uv_managed = is_uv_managed_venv()
    system_site = venv_allows_system_site_packages()
    venv_ok = venv_exists and uv_managed and not system_site
    checks["checks"]["venv"] = {
        "ok": venv_ok,
        "exists": venv_exists,
        "uv_managed": uv_managed,
        "include_system_site_packages": system_site,
        "config": cfg,
    }
    all_ok &= venv_ok

    # Python deps
    dep_results: dict[str, Any] = {}
    for module, attr in PYTHON_DEPS:
        ok, elapsed, error = check_import(module, attr)
        dep_results[module] = {
            "ok": ok,
            "duration_seconds": elapsed,
            "duration": format_duration(elapsed),
            "error": error,
        }
        all_ok &= ok
    checks["checks"]["python_dependencies"] = dep_results

    # Project modules
    module_results: dict[str, Any] = {}
    for module in PROJECT_MODULES:
        ok, elapsed, error = check_import(module, None)
        module_results[module] = {
            "ok": ok,
            "duration_seconds": elapsed,
            "duration": format_duration(elapsed),
            "error": error,
        }
        all_ok &= ok
    checks["checks"]["project_modules"] = module_results

    checks["all_ok"] = all_ok
    return checks


def print_text_report(checks: dict[str, Any]) -> None:
    print(f"geoseg environment check")
    print(f"Project root: {checks['project_root']}")
    print(f"Venv path:    {checks['venv_path']}")

    print_section("Package managers")
    for key in ("uv", "pnpm"):
        item = checks["checks"][key]
        status = "✅" if item["ok"] else "❌"
        value = item["value"] or "not found"
        print(f"{status} {key:12s} {value}")

    print_section("Python")
    py = checks["checks"]["python_version"]
    status = "✅" if py["ok"] else "❌"
    print(f"{status} version {py['value']} (required ≥ {py['required']})")

    print_section("Virtual environment")
    venv = checks["checks"]["venv"]
    if not venv["exists"]:
        print("❌ .venv does not exist")
    else:
        print(f"{'✅' if venv['uv_managed'] else '❌'} managed by uv")
        print(
            f"{'❌' if venv['include_system_site_packages'] else '✅'} "
            f"system site-packages disabled"
        )

    print_section("Python dependencies")
    for name, item in checks["checks"]["python_dependencies"].items():
        status = "✅" if item["ok"] else "❌"
        print(f"{status} {name:20s} {item['duration']:>10s} {item['error']}")

    print_section("Project modules")
    for name, item in checks["checks"]["project_modules"].items():
        status = "✅" if item["ok"] else "❌"
        print(f"{status} {name:40s} {item['duration']:>10s} {item['error']}")

    print()
    if checks["all_ok"]:
        print("✅ All checks passed.")
    else:
        print("❌ Some checks failed. Run with --fix to recreate the venv.")


def fix_environment() -> int:
    print("Recreating virtual environment with uv...")
    proc = run(["uv", "venv", "--clear"])
    if proc.returncode != 0:
        print(f"Failed to recreate venv:\n{proc.stderr}", file=sys.stderr)
        return 1
    print("Syncing dependencies...")
    proc = run(["uv", "sync"])
    if proc.returncode != 0:
        print(f"Failed to sync dependencies:\n{proc.stderr}", file=sys.stderr)
        return 1
    print("✅ Environment recreated and synced.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check geoseg environment health")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Recreate .venv and run uv sync if checks fail",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    if args.fix:
        return fix_environment()

    checks = perform_checks()

    if args.json:
        print(json.dumps(checks, indent=2, default=str))
        return 0 if checks["all_ok"] else 1

    print_text_report(checks)
    return 0 if checks["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
