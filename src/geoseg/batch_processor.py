"""Backward-compatible batch processor entry point."""

from __future__ import annotations

import sys

from geoseg.batch.cli import main
from geoseg.batch.service import export_reviewed, process_directory

__all__ = ["export_reviewed", "main", "process_directory"]


if __name__ == "__main__":
    sys.exit(main())
