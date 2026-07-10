#!/usr/bin/env python

from __future__ import annotations


def clamp_optional(value: float, minimum: float | None, maximum: float | None) -> float:
    if minimum is not None:
        value = max(value, float(minimum))
    if maximum is not None:
        value = min(value, float(maximum))
    return value


__all__ = ["clamp_optional"]
