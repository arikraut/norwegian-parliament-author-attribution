"""Shared utilities used across data_pipeline modules."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def find_project_root(*starts: Path | None) -> Path:
    """Search for the project root from one or more candidate starting paths.

    Tries each non-None candidate in order, walking up the directory tree from
    each one. Falls back to the current working directory when no candidates are
    given. Raises FileNotFoundError if no candidate leads to a valid root.
    """
    candidates = [s for s in starts if s is not None] or [Path.cwd()]
    for start in candidates:
        for ancestor in [start.resolve(), *start.resolve().parents]:
            if (ancestor / "pyproject.toml").exists() and (ancestor / "data").exists():
                return ancestor
    tried = ", ".join(str(s) for s in candidates)
    raise FileNotFoundError(f"Could not locate the project root from: {tried}")


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    """Resolve a config path against the project root without touching disk."""
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def relative_to_project(project_root: Path, path_value: Path) -> str:
    """Render artifact paths relative to the project root for manifests."""
    try:
        return path_value.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path_value)


def write_json(path: Path, payload: dict) -> None:
    """Write a UTF-8 JSON artifact with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _toml_value(value: Any) -> str:
    """Serialize the simple scalar/list/table values used in generated TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items()) + " }"
    if value is None:
        raise ValueError("TOML does not support null values.")
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def write_toml(path: Path, payload: dict[str, Any]) -> None:
    """Write the simple TOML shapes used by resolved pipeline configs."""

    lines: list[str] = []

    def emit_table(prefix: str, table: dict[str, Any]) -> None:
        """Emit one TOML table and recursively emit nested tables."""
        scalars = {
            key: value
            for key, value in table.items()
            if not isinstance(value, dict | list)
            or (isinstance(value, list) and not value)
            or (isinstance(value, list) and not isinstance(value[0], dict))
        }
        nested_tables = {key: value for key, value in table.items() if isinstance(value, dict)}
        arrays = {
            key: value
            for key, value in table.items()
            if isinstance(value, list) and value and isinstance(value[0], dict)
        }

        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_toml_value(value)}")
        if prefix and (scalars or nested_tables or arrays):
            lines.append("")

        for key, value in nested_tables.items():
            emit_table(f"{prefix}.{key}" if prefix else key, value)

        for key, values in arrays.items():
            array_prefix = f"{prefix}.{key}" if prefix else key
            for item in values:
                lines.append(f"[[{array_prefix}]]")
                for item_key, item_value in item.items():
                    lines.append(f"{item_key} = {_toml_value(item_value)}")
                lines.append("")

    emit_table("", payload)
    text = "\n".join(lines).rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_config_outputs(config_path: Path, *output_paths: Path) -> None:
    """Copy a source config beside result and artifact outputs."""
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, output_path)


def create_progress_bar(*, total: int, desc: str, unit: str = "it", show_progress: bool = True):
    """Create an optional tqdm progress bar for long-running pipeline loops."""
    if not show_progress:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def read_required_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV artifact and fail at the file boundary if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    """Read an optional CSV artifact, returning an empty frame when absent."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
