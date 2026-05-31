"""Installed CLI integration tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _cli_path() -> str:
    return str(Path(sys.executable).with_name("reify"))


def _run_cli(tmp_home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["REIFY_HOME"] = str(tmp_home)
    return subprocess.run(
        [_cli_path(), *args],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_cli_remember_recall_and_export(tmp_path: Path) -> None:
    remember = _run_cli(
        tmp_path,
        "remember",
        "I got promoted from Senior to Staff Engineer",
        "--identity",
        "user123",
        "--at",
        "2025-01-01T00:00:00+00:00",
    )
    assert "Committed" in remember.stdout

    reflect = _run_cli(tmp_path, "reflect")
    assert "Processed" in reflect.stdout

    recall = _run_cli(
        tmp_path,
        "recall",
        "what is my title?",
        "--identity",
        "user123",
        "--at",
        "2025-01-30T00:00:00+00:00",
    )
    assert "Staff Engineer" in recall.stdout

    knowledge = _run_cli(
        tmp_path,
        "what-do-you-know",
        "promotion",
        "--identity",
        "user123",
        "--at",
        "2025-01-30T00:00:00+00:00",
    )
    assert "Staff Engineer" in knowledge.stdout

    export_path = tmp_path / "export.md"
    export = _run_cli(tmp_path, "export", str(export_path), "--identity", "user123")
    assert "Exported" in export.stdout
    assert export_path.exists()
    assert "Staff Engineer" in export_path.read_text()


def test_cli_init_and_no_llm_reflect(tmp_path: Path) -> None:
    init = _run_cli(tmp_path, "init")
    assert "Reify initialized" in init.stdout
    assert (tmp_path / "store.db").exists() is False

    reflect = _run_cli(tmp_path, "reflect")
    assert "Processed 0 queued memory(ies)" in reflect.stdout


def test_cli_forget_by_tag(tmp_path: Path) -> None:
    _run_cli(
        tmp_path,
        "remember",
        "I work at Stripe",
        "--identity",
        "user123",
        "--at",
        "2025-01-01T00:00:00+00:00",
    )
    _run_cli(
        tmp_path,
        "remember",
        "I live in Seattle",
        "--identity",
        "user123",
        "--at",
        "2025-01-01T00:00:00+00:00",
    )

    forgot = _run_cli(tmp_path, "forget", "--tag", "employer", "--identity", "user123")
    assert "Forgot" in forgot.stdout

    recall = _run_cli(
        tmp_path,
        "recall",
        "employer",
        "--identity",
        "user123",
    )
    assert "Stripe" not in recall.stdout


def test_cli_eval_user_value(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "eval", "user-value")
    assert "reify eval user-value" in result.stdout
    assert "30-day recall accuracy" in result.stdout
    assert "100.0%" in result.stdout


def test_cli_eval_longmemeval_fixture(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "eval", "longmemeval-fixture")
    assert "reify eval longmemeval-fixture" in result.stdout
    assert "dataset: bundled fixture" in result.stdout
    assert "overall accuracy: 100.0%" in result.stdout


def test_cli_eval_longmemeval_requires_dataset(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["REIFY_HOME"] = str(tmp_path)
    result = subprocess.run(
        [_cli_path(), "eval", "longmemeval"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires --dataset" in result.stdout


def test_cli_eval_longmemeval_dataset_runs_proxy_and_labels_official_gap(tmp_path: Path) -> None:
    dataset = tmp_path / "official.json"
    dataset.write_text(
        (
            '[{"question_id":"q1","question_type":"single-session-user",'
            '"question":"Where do I live?","answer":"Seattle","question_date":"2026/01/31",'
            '"haystack_session_ids":["s1"],"haystack_dates":["2026/01/01 (Thu) 12:00"],'
            '"haystack_sessions":[[{"role":"user","content":"I live in Seattle.","has_answer":true}]],'
            '"answer_session_ids":["s1"]}]'
        ),
        encoding="utf-8",
    )
    result = _run_cli(tmp_path, "eval", "longmemeval", "--dataset", str(dataset))
    assert "Reify Retrieval Proxy" in result.stdout
    assert "Official LongMemEval QA score not run." in result.stdout


def test_cli_eval_multisession_synthesis_dataset_runs_reify_native_tracks(tmp_path: Path) -> None:
    dataset = tmp_path / "official.json"
    dataset.write_text(
        (
            '[{"question_id":"q1","question_type":"single-session-user",'
            '"question":"Where do I live?","answer":"Seattle","question_date":"2026/01/31",'
            '"haystack_session_ids":["s1"],"haystack_dates":["2026/01/01 (Thu) 12:00"],'
            '"haystack_sessions":[[{"role":"user","content":"I live in Seattle.","has_answer":true}]],'
            '"answer_session_ids":["s1"]}]'
        ),
        encoding="utf-8",
    )
    result = _run_cli(tmp_path, "eval", "multisession-synthesis", "--dataset", str(dataset))
    assert "reify eval multisession-synthesis" in result.stdout
    assert "Capability Track Breakdown" in result.stdout
    assert "scoring: current Reify retrieval proxy" in result.stdout
    assert "track direct-recall: 100.0%" in result.stdout


def test_cli_eval_reify_native_alias_mentions_alias_name(tmp_path: Path) -> None:
    dataset = tmp_path / "official.json"
    dataset.write_text(
        (
            '[{"question_id":"q1","question_type":"single-session-user",'
            '"question":"Where do I live?","answer":"Seattle","question_date":"2026/01/31",'
            '"haystack_session_ids":["s1"],"haystack_dates":["2026/01/01 (Thu) 12:00"],'
            '"haystack_sessions":[[{"role":"user","content":"I live in Seattle.","has_answer":true}]],'
            '"answer_session_ids":["s1"]}]'
        ),
        encoding="utf-8",
    )
    result = _run_cli(tmp_path, "eval", "reify-native", "--dataset", str(dataset))
    assert "reify eval reify-native" in result.stdout
