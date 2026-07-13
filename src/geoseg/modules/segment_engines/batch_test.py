"""Backward-compatible entry point for batch segmentation diagnostics."""

from geoseg.modules.segment_engines.diagnostics.batch_test import *  # noqa: F401,F403


if __name__ == "__main__":
    import sys

    from geoseg.modules.segment_engines.diagnostics.batch_test import main

    sys.exit(main())
