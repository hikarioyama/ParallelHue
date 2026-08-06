"""Prometheus metrics helpers used by the tmux summary."""
from __future__ import annotations

import sys
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def _metrics_url(endpoint: str) -> str:
    """Convert an OpenAI endpoint, normally /v1/chat/completions, to /metrics."""
    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc or parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    parts = [part for part in parsed.path.split("/") if part]
    prefix = parts[:parts.index("v1")] if "v1" in parts else parts[:-1]
    return urlunsplit((parsed.scheme, parsed.netloc, "/" + "/".join([*prefix, "metrics"]), "", ""))


def _fetch_metrics(endpoint: str, timeout: float) -> str | None:
    try:
        request = Request(_metrics_url(endpoint), headers={"Accept": "text/plain"})
        with urlopen(request, timeout=max(1.0, min(timeout, 15.0))) as response:  # noqa: S310
            return response.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        # Don't print a user-supplied URL: it can contain credentials.
        print("parallelhue: metrics unavailable", file=sys.stderr)
        return None


def _parse_prometheus_counters(text: str | None) -> dict[str, float]:
    counters: dict[str, float] = {}
    for line in (text or "").splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0].startswith("#"):
            continue
        name = fields[0].split("{", 1)[0]
        if not (name.endswith("_total") or name.endswith("_sum")):
            continue
        try:
            counters[name] = counters.get(name, 0.0) + float(fields[1])
        except ValueError:
            pass
    return counters


def _counter_delta(before: dict[str, float], after: dict[str, float], *candidates: str) -> float | None:
    """Return a non-reset delta for the first exact counter name available."""
    for name in candidates:
        if name in before and name in after and after[name] >= before[name]:
            return after[name] - before[name]
    return None
