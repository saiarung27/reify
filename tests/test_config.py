"""Tests for :mod:`reify.config`."""

from __future__ import annotations

from pathlib import Path

from reify.config import ReifySettings, get_settings


def test_default_paths_under_home() -> None:
    s = ReifySettings()
    assert s.db_path == s.home / "store.db"
    assert s.cache_dir == s.home / "cache"


def test_get_settings_is_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_ensure_dirs_is_idempotent(tmp_path: Path) -> None:
    s = ReifySettings(home=tmp_path / "reify_home")
    s.ensure_dirs()
    s.ensure_dirs()  # second call must not raise
    assert s.home.is_dir()
    assert s.cache_dir.is_dir()


def test_default_llm_specs_are_well_formed() -> None:
    s = ReifySettings()
    for spec in (s.llm_extract, s.llm_reconsolidate, s.llm_intent):
        provider, _, model = spec.partition(":")
        assert provider and model


def test_default_retention_settings_are_sane() -> None:
    s = ReifySettings()
    assert s.default_ttl_days > 0
    assert s.reinforcement_factor > 1.0
    assert s.decay_sweep_interval_seconds > 0
