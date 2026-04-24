"""
Configuration module for voicebox backend.

Handles data directory configuration for production bundling.
"""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Default data directory (used in development)
_data_dir = Path("data").resolve()


def _sync_hf_constants(cache_dir: Path) -> None:
    """Keep huggingface_hub constants aligned with runtime env overrides."""
    try:
        from huggingface_hub import constants as hf_constants

        cache_str = str(cache_dir)
        hf_home = cache_dir.parent
        xet_cache = hf_home / "xet"

        os.environ["HF_HUB_CACHE"] = cache_str
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_str
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_XET_CACHE"] = str(xet_cache)
        hf_constants.HF_HUB_CACHE = cache_str
        hf_constants.HUGGINGFACE_HUB_CACHE = cache_str
        if hasattr(hf_constants, "HF_HOME"):
            hf_constants.HF_HOME = str(hf_home)
    except Exception:
        # huggingface_hub may not be imported yet; env vars are still set above.
        pass


def set_models_cache_dir(path: str | Path) -> Path:
    """Set and create the HuggingFace cache directory used by all engines."""
    cache_dir = Path(path).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_home = cache_dir.parent
    xet_cache = hf_home / "xet"
    hf_home.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)

    cache_str = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = cache_str
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_str
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    _sync_hf_constants(cache_dir)
    logger.info("Model cache directory set to: %s", cache_dir)
    return cache_dir


def _is_dir_writable(path: Path) -> bool:
    """Return True if path exists (or can be created) and is writable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".voicebox_hf_probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _get_current_hf_cache_dir() -> Path:
    """Resolve the active HF cache directory from env/constants/defaults."""
    env_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_cache:
        return Path(env_cache).expanduser().resolve()

    try:
        from huggingface_hub import constants as hf_constants

        return Path(hf_constants.HF_HUB_CACHE).expanduser().resolve()
    except Exception:
        return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def ensure_hf_cache_writable() -> Path:
    """Ensure HF cache path is writable, with safe fallback for constrained runtimes."""
    current_cache = _get_current_hf_cache_dir()
    if _is_dir_writable(current_cache):
        _sync_hf_constants(current_cache)
        return current_cache

    candidates = [
        get_cache_dir() / "huggingface" / "hub",
        Path(tempfile.gettempdir()) / "voicebox" / "huggingface" / "hub",
    ]

    for candidate in candidates:
        if _is_dir_writable(candidate):
            logger.warning(
                "HF cache directory is not writable (%s). Falling back to %s",
                current_cache,
                candidate,
            )
            return set_models_cache_dir(candidate)

    raise PermissionError(
        "Unable to find a writable HuggingFace cache directory. "
        f"Checked: {current_cache}, {', '.join(str(c) for c in candidates)}"
    )


# Allow users to override the HuggingFace model download directory.
# Set VOICEBOX_MODELS_DIR to an absolute path before starting the server.
# This sets HF_HUB_CACHE so all huggingface_hub downloads go to that path.
_custom_models_dir = os.environ.get("VOICEBOX_MODELS_DIR")
if _custom_models_dir:
    set_models_cache_dir(_custom_models_dir)


def _path_relative_to_any_data_dir(path: Path) -> Path | None:
    """Extract the path within a data dir from an absolute or relative path."""
    parts = path.parts
    for idx, part in enumerate(parts):
        if part != "data":
            continue

        tail = parts[idx + 1 :]
        if tail:
            return Path(*tail)
        return Path()

    return None


def set_data_dir(path: str | Path):
    """
    Set the data directory path.

    Args:
        path: Path to the data directory
    """
    global _data_dir
    _data_dir = Path(path).resolve()
    _data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Data directory set to: %s", _data_dir)


def get_data_dir() -> Path:
    """
    Get the data directory path.

    Returns:
        Path to the data directory
    """
    return _data_dir


def to_storage_path(path: str | Path) -> str:
    """Convert a filesystem path to a DB-safe path relative to the data dir."""
    resolved_path = Path(path).resolve()

    relative_to_any_data_dir = _path_relative_to_any_data_dir(resolved_path)
    if relative_to_any_data_dir is not None:
        return str(relative_to_any_data_dir)

    try:
        return str(resolved_path.relative_to(_data_dir))
    except ValueError:
        return str(resolved_path)


def resolve_storage_path(path: str | Path | None) -> Path | None:
    """Resolve a DB-stored path against the configured data dir."""
    if path is None:
        return None

    stored_path = Path(path)
    if stored_path.is_absolute():
        rebased_path = _path_relative_to_any_data_dir(stored_path)
        if rebased_path is not None:
            candidate = (_data_dir / rebased_path).resolve()
            if candidate.exists() or not stored_path.exists():
                return candidate

        return stored_path

    # 0.3.0 records sometimes stored relative paths with the data-dir name
    # baked in (e.g. "data/profiles/..."). Joining those directly with
    # _data_dir produces a spurious "<data_dir>/data/profiles/..." nest.
    if stored_path.parts and stored_path.parts[0] == "data":
        stored_path = (
            Path(*stored_path.parts[1:]) if len(stored_path.parts) > 1 else Path()
        )

    return (_data_dir / stored_path).resolve()


def get_db_path() -> Path:
    """Get database file path."""
    return _data_dir / "voicebox.db"


def get_profiles_dir() -> Path:
    """Get profiles directory path."""
    path = _data_dir / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_generations_dir() -> Path:
    """Get generations directory path."""
    path = _data_dir / "generations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir() -> Path:
    """Get cache directory path."""
    path = _data_dir / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    """Get models directory path."""
    path = _data_dir / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
