"""Backward-compatible entry point for engine comparison diagnostics."""

from geoseg.modules.segment_engines.diagnostics.compare_results import *  # noqa: F401,F403


if __name__ == "__main__":
    import sys

    from geoseg.modules.segment_engines.diagnostics.compare_results import main

    sys.exit(main())
