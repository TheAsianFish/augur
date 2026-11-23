"""The release workflow publishes only from a tag, by trusted publishing.

Read as text (no YAML parser is a dependency): jobs are the two-space-indented
keys under ``jobs:``, and each job's block runs to the next one.
"""

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "release.yml"
TAG_GUARD = "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"


def _code(text: str) -> str:
    """The workflow without comments, so prose cannot satisfy or break a check."""
    return "\n".join(
        line.split(" #", 1)[0] for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _jobs() -> dict[str, str]:
    text = _code(WORKFLOW.read_text(encoding="utf-8"))
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def test_the_workflow_runs_on_version_tags_and_a_dispatch_that_defaults_to_dry_run() -> None:
    text = _code(WORKFLOW.read_text(encoding="utf-8"))
    head = text.split("\njobs:\n", 1)[0]
    assert re.search(r"push:\n\s+tags: \['v\*'\]", head)
    assert re.search(r"dry_run:\n(?:\s+.*\n)*?\s+type: boolean\n\s+default: true", head)
    assert re.search(r"^permissions:\n  contents: read$", head, re.M)
    assert re.search(r"^concurrency:", head, re.M)


def test_only_a_pushed_tag_reaches_the_publish_job() -> None:
    jobs = _jobs()
    assert set(jobs) == {"build", "publish"}
    publish = jobs["publish"]
    assert f"    {TAG_GUARD}\n" in publish, "the job-level guard is missing"
    assert "needs: build" in publish
    assert "environment:\n      name: pypi" in publish
    assert "pypa/gh-action-pypi-publish@release/v1" in publish


def test_id_token_write_is_granted_to_the_publish_job_alone() -> None:
    jobs = _jobs()
    text = _code(WORKFLOW.read_text(encoding="utf-8"))
    assert text.count("id-token: write") == 1
    assert "permissions:\n      id-token: write" in jobs["publish"]
    assert "id-token" not in jobs["build"]


def test_no_token_or_password_is_handed_to_the_publisher() -> None:
    text = _code(WORKFLOW.read_text(encoding="utf-8"))
    assert "password:" not in text
    assert not re.search(r"secrets\.\w*(PYPI|TOKEN)", text, re.I)


def test_every_job_has_a_timeout() -> None:
    for name, block in _jobs().items():
        assert re.search(r"^    timeout-minutes: \d+$", block, re.M), name


def test_the_build_job_runs_the_gates_and_checks_the_tag() -> None:
    build = _jobs()["build"]
    for step in ("make check", "make install-check", "twine check", "upload-artifact"):
        assert step in build, step
    assert TAG_GUARD in build, "the tag-equals-version step must run on tags"
    assert "v$version" in build
