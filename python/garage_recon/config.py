"""Configuration: packaged defaults deep-merged with user YAML / overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).with_name("default_config.yaml")


class Config(dict):
    """Dict with attribute access for nested keys (``cfg.loops.top_k``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict) and not isinstance(obj, Config):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _leaves(table: dict, prefix: tuple = ()):
    """(key path, value) of the non-dict values of a nested dict."""
    for key, value in table.items():
        if isinstance(value, dict):
            yield from _leaves(value, prefix + (key,))
        else:
            yield prefix + (key,), value


def apply_imu_grade(cfg: dict, user: dict) -> dict:
    """The IMU grade's settings (`imu_grades[imu_grade]`) where the user's
    files do not set them.

    Applied once: the result is marked `imu_grade_applied`, so a saved
    config.used.yaml loaded again keeps its values (gr/io/config.cc does the
    same).
    """
    grade = cfg.get("imu_grade")
    if not grade or cfg.get("imu_grade_applied"):
        return cfg
    table = (cfg.get("imu_grades") or {}).get(grade)
    if table is None:
        raise ValueError(f"unknown imu_grade {grade!r}")
    out = copy.deepcopy(cfg)
    for path, value in _leaves(table):
        given = user
        for key in path:
            given = given.get(key) if isinstance(given, dict) else None
        if given is not None:
            continue  # Set explicitly.
        section = out
        for key in path[:-1]:
            section = section.setdefault(key, {})
        section[path[-1]] = copy.deepcopy(value)
    out["imu_grade_applied"] = True
    return out


def load_config(path: str | Path | None = None,
                overrides: dict | None = None) -> Config:
    cfg = yaml.safe_load(DEFAULT_PATH.read_text())
    user = {}
    if path is not None:
        user = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        user = deep_merge(user, overrides)
    cfg = apply_imu_grade(deep_merge(cfg, user), user)
    return Config.wrap(cfg)


def save_config(cfg: dict, path: str | Path) -> None:
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
