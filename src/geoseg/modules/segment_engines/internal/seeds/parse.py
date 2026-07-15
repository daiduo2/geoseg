"""Seed metadata parsing helpers."""

from __future__ import annotations


def _parse_count_from_tag(tag: str) -> int:
    """Extract count from tags like 'online(count=123)' -> 123."""
    if "count=" in tag:
        try:
            return int(tag.split("count=")[1].rstrip(")"))
        except ValueError:
            return 0
    return 0

__all__ = ["_parse_count_from_tag"]
