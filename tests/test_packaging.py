"""The package metadata agrees with the code, and the wheel's data files exist."""

import tomllib
from pathlib import Path

import augur

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_package_version_is_the_one_the_code_reports() -> None:
    assert PYPROJECT["project"]["version"] == augur.__version__


def test_every_file_forced_into_the_wheel_exists_and_is_in_the_sdist() -> None:
    hatch = PYPROJECT["tool"]["hatch"]["build"]["targets"]
    forced: dict[str, str] = hatch["wheel"]["force-include"]
    sdist = hatch["sdist"]["include"]
    for source, target in forced.items():
        assert (ROOT / source).is_dir(), source
        assert target.startswith("augur/"), target
        # The wheel is built from the sdist, so the source must travel with it.
        assert any(source.startswith(entry.strip("/")) for entry in sdist), source


def test_the_sdist_leaves_out_the_ledgers_and_runs() -> None:
    include = PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "/data" not in include
    assert not any(
        entry.startswith(("/data/ledgers", "/data/runs", "/.github")) for entry in include
    )
