"""The ledger-branch scripts against a local bare remote: no network, no GitHub."""

# Fixed commands (git, bash, this interpreter) on paths the test creates.
# ruff: noqa: S603, S607

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PUSH_STANDINGS = ROOT / "scripts" / "push_standings.sh"
ROUNDS = ROOT / "scripts" / "standings_rounds.py"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash"
)


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    done = subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    )
    return done.stdout


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--quiet", "--bare", "--initial-branch=main", str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "clone", "--quiet", str(origin), str(work))
    (work / "README.md").write_text("x\n", encoding="utf-8")
    _git(work, "add", "README.md")
    _git(work, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")
    _git(work, "push", "--quiet", "origin", "HEAD:main")
    return work


def _push(clone: Path, snapshots: dict[str, str]) -> subprocess.CompletedProcess[str]:
    out = clone / "standings"
    out.mkdir(exist_ok=True)
    for old in out.iterdir():
        old.unlink()
    for name, text in snapshots.items():
        (out / name).write_text(text, encoding="utf-8")
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "LEDGER_RETRY_SECONDS": "0",
    }
    return subprocess.run(
        ["bash", str(PUSH_STANDINGS), "standings"],
        cwd=clone,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _ledger_files(clone: Path) -> list[str]:
    _git(clone, "fetch", "--quiet", "origin", "ledger")
    return sorted(_git(clone, "ls-tree", "-r", "--name-only", "FETCH_HEAD").split())


def test_push_standings_starts_the_branch_then_adds_a_file_per_snapshot(clone: Path) -> None:
    first = _push(clone, {"minibench.json": "{}\n", "standings.txt": "rank 1\n"})
    assert "no ledger branch on origin yet" in first.stdout
    assert "pushed 1 snapshot(s)" in first.stdout
    files = _ledger_files(clone)
    assert files[0] == "README.md"
    assert [f.rsplit("-", 1)[-1] for f in files[1:]] == ["minibench.json", "standings.txt"]
    assert all(f.startswith("standings/") for f in files[1:])

    _push(clone, {"33121.json": "{}\n"})
    files = _ledger_files(clone)
    assert sum(f.endswith("33121.json") for f in files) == 1
    assert sum(f.endswith("minibench.json") for f in files) == 1  # the first reading is kept


def test_push_standings_with_nothing_to_push_leaves_origin_alone(clone: Path) -> None:
    done = _push(clone, {})
    assert "nothing to push" in done.stdout
    assert "ledger" not in _git(clone, "ls-remote", "--heads", "origin")


def test_standings_rounds_names_the_two_rounds_before_the_current_one(tmp_path: Path) -> None:
    snapshot = tmp_path / "minibench.json"
    snapshot.write_text(
        json.dumps({"round": {"slug": "minibench", "start_date": "2026-11-16T00:00:00Z"}}),
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, str(ROUNDS), str(snapshot)], check=True, capture_output=True, text=True
    )
    assert done.stdout.split() == ["minibench-2026-11-02", "minibench-2026-10-19"]


def test_standings_rounds_refuses_a_snapshot_without_a_start(tmp_path: Path) -> None:
    snapshot = tmp_path / "bad.json"
    snapshot.write_text(json.dumps({"round": {"slug": "minibench"}}), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(ROUNDS), str(snapshot)], capture_output=True, text=True
    )
    assert done.returncode == 1
    assert "start_date" in done.stderr
