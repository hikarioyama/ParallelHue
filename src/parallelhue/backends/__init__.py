"""Backend-profile selection."""
from __future__ import annotations

import os

from .base import BackendProfile
from .dspark import PROFILE as DSPARK_PROFILE
from .generic import PROFILE as GENERIC_PROFILE
from .mtp import PROFILE as MTP_PROFILE

_PROFILES = {
    GENERIC_PROFILE.name: GENERIC_PROFILE,
    MTP_PROFILE.name: MTP_PROFILE,
    DSPARK_PROFILE.name: DSPARK_PROFILE,
}


def _infer_backend(model: str | None) -> str:
    """Infer the profile from the model and server-launch environment."""
    environment = " ".join(
        os.environ.get(name, "")
        for name in (
            "BACKEND",
            "VLLM_SPECULATIVE_CONFIG",
            "VLLM_KV_TRANSFER_CONFIG",
            "PARALLELHUE_SERVER_ARGS",
        )
    ).lower()
    if "dspark" in environment:
        return "dspark"
    if "mtp" in environment:
        return "mtp"

    model_name = (model or "").lower()
    if any(marker in model_name for marker in ("dspark", "deepseek-v4", "deepseek")):
        return "dspark"
    if any(marker in model_name for marker in ("qwen", "mtp", "a3b", "a6b")):
        return "mtp"
    return "generic"


def get_backend(name: str | None = None, model: str | None = None) -> BackendProfile:
    """Return a named backend profile, resolving ``auto`` when requested."""
    selected = (name or "auto").strip().lower()
    if selected == "auto":
        selected = _infer_backend(model)
    try:
        return _PROFILES[selected]
    except KeyError as exc:
        choices = ", ".join(("auto", *_PROFILES))
        raise ValueError(f"unknown backend {name!r}; expected one of: {choices}") from exc


__all__ = ["BackendProfile", "get_backend"]
