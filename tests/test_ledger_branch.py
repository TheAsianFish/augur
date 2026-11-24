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

from augur.clock import FixedClock
from augur.leakage import RunMode
from augur.ledger import Ledger

from .conftest import T0

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


# -- M11: the shadow tree --------------------------------------------------------------

PUSH_SHADOW = ROOT / "scripts" / "push_shadow.sh"
SHADOW_FETCH = ROOT / "scripts" / "shadow_fetch.sh"
NAME = "20261130T031500Z-123-1"


def _env() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "LEDGER_RETRY_SECONDS": "0",
        "AUGUR_PYTHON": sys.executable,
    }


def _script(clone: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", *args], cwd=clone, env=_env(), check=check, capture_output=True, text=True
    )


def _shadow_ledger(path: Path, template: bool) -> None:
    with Ledger.open(path, FixedClock(T0)) as ledger:
        if not ledger.connection.execute("SELECT 1 FROM runs").fetchone():
            ledger.start_run(
                mode=RunMode.DRY_RUN,
                tournament="minibench",
                provider="openrouter",
                model="m",
                settings={},
                code_version="t",
            )
        if template:
            ledger.record_reference(
                question_id=501,
                post_id=500,
                question_type="binary",
                forecaster="template",
                detail="d",
                values={"probability": 0.3},
                as_of=T0,
                source="s",
            )


def test_push_shadow_twice_under_one_name_keeps_one_superset_file(
    clone: Path, tmp_path: Path
) -> None:
    db = clone / "run.sqlite"
    _shadow_ledger(db, template=False)
    (clone / "run.txt").write_text("tick output\n", encoding="utf-8")
    first = _script(clone, str(PUSH_SHADOW), "run.sqlite", "run.txt", NAME)
    assert "pushed shadow/2026-11-30/" in first.stdout
    again = _script(clone, str(PUSH_SHADOW), "run.sqlite", "run.txt", NAME)
    assert "nothing to push" in again.stdout
    _shadow_ledger(db, template=True)
    _script(clone, str(PUSH_SHADOW), "run.sqlite", "run.txt", NAME)
    files = _ledger_files(clone)
    assert files == [
        "README.md",
        f"shadow/2026-11-30/{NAME}.jsonl",
        f"shadow/2026-11-30/{NAME}.txt",
    ]
    text = _git(clone, "show", f"FETCH_HEAD:shadow/2026-11-30/{NAME}.jsonl")
    tables = [json.loads(line)["table"] for line in text.splitlines()]
    assert "reference_forecasts" in tables, "the second push replaced the first"

    out = tmp_path / "fetched"
    fetched = _script(clone, str(SHADOW_FETCH), str(out))
    assert "1 export(s)" in fetched.stdout
    assert (out / "shadow" / "2026-11-30" / f"{NAME}.jsonl").read_text("utf-8") == text


def test_push_shadow_refuses_a_name_without_its_timestamp(clone: Path) -> None:
    done = _script(clone, str(PUSH_SHADOW), "run.sqlite", "run.txt", "../x", check=False)
    assert done.returncode == 1
    assert "UTC timestamp" in done.stderr


def test_shadow_fetch_without_a_branch_or_a_tree_is_empty(clone: Path, tmp_path: Path) -> None:
    out = tmp_path / "fetched"
    done = _script(clone, str(SHADOW_FETCH), str(out))
    assert "no ledger branch yet" in done.stdout
    assert list(out.iterdir()) == []
    _push(clone, {"minibench.json": "{}\n"})
    done = _script(clone, str(SHADOW_FETCH), str(out))
    assert "no shadow tree yet" in done.stdout


def test_shadow_fetch_fails_loudly_when_the_remote_is_unreachable(
    clone: Path, tmp_path: Path
) -> None:
    env = _env() | {"AUGUR_LEDGER_REMOTE": str(tmp_path / "nowhere.git")}
    done = subprocess.run(
        ["bash", str(SHADOW_FETCH), str(tmp_path / "out")],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1
    assert "cannot reach" in done.stderr
