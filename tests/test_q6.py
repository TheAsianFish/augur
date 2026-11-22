"""Q6: AUROC, the measures, leave-one-out fits, skip gains, the live half, the snapshot."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist, fmean

import pytest

from augur import q6
from augur.calibrate import platt_targets
from augur.clock import FixedClock
from augur.leakage import RunMode
from augur.ledger import Ledger
from augur.replay import clip, logit, sigmoid
from augur.report import REPO, render_q6, render_q6_preregistration
from augur.score import ScoringError
from augur.sourcecal import fit_logistic

from .helpers import binary_question

T0 = datetime(2026, 11, 17, 18, 0, tzinfo=UTC)

# -- AUROC ------------------------------------------------------------------------


def test_auroc_extremes_and_ties() -> None:
    assert q6.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert q6.auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert q6.auroc([5, 5, 5, 5], [0, 1, 0, 1]) == 0.5


def test_auroc_hand_case_with_a_tie() -> None:
    # positives {2, 3}, negatives {1, 2}: 2>1 (1), 2=2 (.5), 3>1 (1), 3>2 (1) = 3.5 / 4
    assert q6.auroc([1, 2, 2, 3], [0, 1, 0, 1]) == 0.875


@pytest.mark.parametrize("labels", [[1, 1, 1], [0, 0, 0]])
def test_auroc_needs_both_classes(labels: list[int]) -> None:
    with pytest.raises(ScoringError):
        q6.auroc([0.1, 0.2, 0.3], labels)


def test_auroc_interval_is_deterministic_and_contains_the_estimate() -> None:
    scores = [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.5, 0.6, 0.3]
    labels = [0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
    a = q6.auroc_interval(scores, labels, resamples=500, seed=3)
    b = q6.auroc_interval(scores, labels, resamples=500, seed=3)
    assert a == b
    assert a.value == q6.auroc(scores, labels)
    assert a.low <= a.value <= a.high
    assert (a.n, a.positives) == (10, 5)


def test_auroc_interval_redraws_one_class_resamples() -> None:
    # One positive in four: a resample of four has no positive about 32% of the time.
    a = q6.auroc_interval([0.1, 0.2, 0.3, 0.9], [0, 0, 0, 1], resamples=200, seed=1)
    assert 0.0 <= a.low <= a.value <= a.high <= 1.0
    assert a.value == 1.0


def test_auroc_se_at_one_half_is_the_null_variance() -> None:
    for n1, n0 in [(20, 20), (69, 69), (5, 11)]:
        assert q6.auroc_se(0.5, n1, n0) == pytest.approx(math.sqrt((n1 + n0 + 1) / (12 * n1 * n0)))


def test_auroc_needed_is_the_smallest_balanced_n() -> None:
    n = q6.auroc_needed()
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(q6.POWER)
    t = q6.AUROC_TARGET

    def passes(m: int) -> bool:
        return (t - 0.5) >= z * q6.auroc_se(t, m // 2, m - m // 2)

    assert n % 2 == 0
    assert passes(n)
    assert not passes(n - 2)


# -- the measures -----------------------------------------------------------------


def test_draw_spread() -> None:
    assert q6.draw_spread([0.5, 0.5]) == 0.0
    assert q6.draw_spread([0.4]) == 0.0
    assert q6.draw_spread([]) == 0.0
    # log-odds -1 and +1: population sd 1
    assert q6.draw_spread([sigmoid(-1.0), sigmoid(1.0)]) == pytest.approx(1.0)
    assert q6.pass_spread([sigmoid(-1.0), sigmoid(1.0)]) == pytest.approx(1.0)


def test_model_gap_uses_the_medians() -> None:
    # medians 0.5 (log-odds 0) and sigmoid(2): gap 2, in either order
    first = [0.1, 0.5, 0.9]
    second = [sigmoid(2.0)] * 3
    assert q6.model_gap(first, second) == pytest.approx(2.0)
    assert q6.model_gap(second, first) == pytest.approx(2.0)
    assert q6.model_gap(first, first) == 0.0


def test_confidence_is_distance_from_one_half() -> None:
    assert q6.confidence(0.5) == pytest.approx(0.0)
    assert q6.confidence(sigmoid(1.5)) == pytest.approx(1.5)
    assert q6.confidence(sigmoid(-1.5)) == pytest.approx(1.5)


def test_spread_shrink() -> None:
    assert q6.spread_shrink(0.8, 0.0) == pytest.approx(clip(0.8))
    assert q6.spread_shrink(0.5, 3.0) == pytest.approx(0.5)
    # d = 1 halves the log-odds
    assert logit(q6.spread_shrink(sigmoid(2.0), 1.0)) == pytest.approx(1.0)
    assert q6.spread_shrink(0.0, 0.0) == pytest.approx(clip(0.0))
    assert q6.spread_shrink(1.0, 0.0) == pytest.approx(clip(1.0))
    assert 0.0 < q6.spread_shrink(0.0, 0.0) < q6.spread_shrink(1.0, 0.0) < 1.0


def test_constant_shrink() -> None:
    assert q6.constant_shrink(0.8, 1.0) == pytest.approx(clip(0.8))
    assert q6.constant_shrink(0.8, 0.0) == pytest.approx(0.5)
    assert logit(q6.constant_shrink(sigmoid(2.0), 0.25)) == pytest.approx(0.5)
    assert q6.constant_shrink(1.0, 1.0) == pytest.approx(clip(1.0))


def test_large_errors_are_strictly_above_the_median_brier() -> None:
    # Briers: .01, .16, .16, .81 (p, y): median .16, so only the last is large
    assert q6.large_errors([0.9, 0.6, 0.4, 0.1], [1, 0, 1, 1]) == [0, 0, 0, 1]
    # all equal: nothing is strictly above the median
    assert q6.large_errors([0.5, 0.5, 0.5], [1, 0, 1]) == [0, 0, 0]


def test_wrong_side_excludes_one_half() -> None:
    assert q6.wrong_side([0.5, 0.5], [1, 0]) == [0, 0]
    assert q6.wrong_side([0.6, 0.4, 0.4, 0.6], [1, 0, 1, 0]) == [0, 0, 1, 1]


# -- leave-one-out fits -----------------------------------------------------------


def test_loo_never_trains_on_the_held_out_index() -> None:
    seen: list[tuple[int, list[int]]] = []

    def fit(train: Sequence[int]) -> Callable[[int], float]:
        held = next(i for i in range(5) if i not in train)
        seen.append((held, list(train)))
        return lambda i: 0.25 + 0.1 * i

    out = q6.loo(5, fit)
    assert [h for h, _ in seen] == [0, 1, 2, 3, 4]
    for held, train in seen:
        assert held not in train
        assert sorted(train) == [j for j in range(5) if j != held]
    assert out == pytest.approx([0.25, 0.35, 0.45, 0.55, 0.65])


def _item(qid: int, final: float, outcome: int, source: str = "s") -> q6.Item:
    return q6.Item(qid, source, outcome, 0.5, final, (final,))


ITEMS = [
    _item(1, 0.8, 1),
    _item(2, 0.7, 0),
    _item(3, 0.3, 0),
    _item(4, 0.2, 1),
    _item(5, 0.9, 1),
    _item(6, 0.4, 0),
    _item(7, 0.6, 1),
]


def test_control_a_is_platt_fitted_on_the_other_questions() -> None:
    held = 3
    train = [j for j in range(len(ITEMS)) if j != held]
    xs = [logit(clip(i.final)) for i in ITEMS]
    (a, b), _ = fit_logistic(
        [[xs[j], 1.0] for j in train],
        platt_targets([ITEMS[j].outcome for j in train]),
        [0.0, 0.0],
        [1.0, 0.0],
    )
    got = q6.control_a(ITEMS)
    assert len(got) == len(ITEMS)
    assert got[held] == pytest.approx(clip(sigmoid(a * xs[held] + b)))
    # a fit that never saw question 3 differs from one that did
    assert len(set(round(p, 6) for p in got)) > 1


def test_fitted_slope_with_a_constant_measure_is_control_b_like() -> None:
    items = [_item(k + 1, i.final, i.outcome, "x" if k % 2 else "y") for k, i in enumerate(ITEMS)]
    flat = q6.fitted_slope(items, [1.0] * len(items))
    again = q6.fitted_slope(items, [7.0] * len(items))
    # a constant measure standardises to z = 0 whatever its value
    assert flat == pytest.approx(again, abs=1e-9)
    b = q6.control_b(items)
    assert flat == pytest.approx(b, abs=1e-6)
    assert all(0.0 < p < 1.0 for p in flat)


# -- comparisons and skip gains ---------------------------------------------------


def test_needed_matches_the_formula() -> None:
    for sd, delta in [(0.1, 0.01), (0.25, 0.01), (0.3, 0.02)]:
        want = math.ceil(((1.959964 + 0.841621) * sd / delta) ** 2)
        assert q6.needed(sd, delta) == want
    assert q6.needed(0.1) == q6.needed(0.1, q6.BRIER_TARGET)


def test_skip_values() -> None:
    qids = [20, 21, 22, 23, 24, 25, 26, 27]
    measure = [3.0, 1.0, 3.0, 3.0, 0.0, 2.0, 1.0, 0.0]
    control = [0.8, 0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5]
    prices = [0.5] * 8
    ys = [1, 1, 0, 1, 0, 1, 0, 1]
    against_all, against_random, k = q6.skip_values(control, prices, ys, measure, qids)
    assert k == 2
    # three tie at 3.0; the two with the smaller qids (20, 22: indices 0, 2) are skipped
    peer0 = 100 * math.log(0.8 / 0.5)
    peer2 = 100 * math.log(0.4 / 0.5)
    assert against_all[0] == pytest.approx(-peer0)
    assert against_all[2] == pytest.approx(-peer2)
    assert [v for i, v in enumerate(against_all) if i not in (0, 2)] == [0.0] * 6
    peers = [100 * math.log((c if y else 1 - c) / 0.5) for c, y in zip(control, ys, strict=True)]
    assert fmean(against_random) == pytest.approx(fmean(against_all) + (2 / 8) * fmean(peers))


# -- the live half ----------------------------------------------------------------


def _unit(qid: int, outcome: int | None, draws: tuple[float, ...], opened: date) -> q6.LiveUnit:
    return q6.LiveUnit("x.sqlite", qid, opened, "m", draws, outcome)


def test_live_unit_half_and_week() -> None:
    assert _unit(7, None, (0.5,), date(2026, 11, 17)).half == "policy"
    assert _unit(8, None, (0.5,), date(2026, 11, 17)).half == "control"
    assert _unit(1, None, (0.5,), date(2026, 11, 17)).week == "2026-W47"
    # ISO year differs from the calendar year at the turn
    assert _unit(1, None, (0.5,), date(2027, 1, 3)).week == "2026-W53"
    assert _unit(1, None, (0.5,), date(2027, 1, 4)).week == "2027-W01"


def _live_units() -> list[q6.LiveUnit]:
    d = date(2026, 12, 1)
    out = []
    for k in range(1, 9):
        draws = (0.7, 0.8, 0.9) if k % 3 else (0.2, 0.5, 0.8)
        out.append(_unit(k, k % 2, draws, d + timedelta(days=7 * (k % 4))))
    return out


def test_live_analysis_not_due_reports_counts_only() -> None:
    out = q6.live_analysis(_live_units(), 0.6, date(2027, 1, 15))
    assert out["due"] is False
    assert out["units"] == 8
    assert out["by_half"]["policy"] == {"units": 4, "resolved": 4}
    assert out["by_half"]["control"] == {"units": 4, "resolved": 4}
    assert all("comparisons" not in h for h in out["by_half"].values())
    assert out["target"] == q6.LIVE_TARGET


def test_live_analysis_due_at_the_deadline_computes_comparisons() -> None:
    out = q6.live_analysis(_live_units(), 0.6, q6.LIVE_DEADLINE)
    assert out["due"] is True
    for half in ("policy", "control"):
        cs = out["by_half"][half]["comparisons"]
        assert [c["label"] for c in cs] == [
            "against the default",
            "against the matched constant",
        ]
        assert all(c["brier"]["n"] == 4 for c in cs)


def test_live_analysis_due_with_one_resolved_in_a_half_skips_it() -> None:
    d = date(2026, 12, 1)
    units = [
        _unit(1, 1, (0.7, 0.8, 0.9), d),
        _unit(3, 0, (0.4, 0.5, 0.9), d),
        _unit(2, 1, (0.7, 0.8, 0.9), d),
        _unit(4, None, (0.7, 0.8, 0.9), d),
    ]
    out = q6.live_analysis(units, 0.5, q6.LIVE_DEADLINE + timedelta(days=1))
    assert "comparisons" in out["by_half"]["policy"]
    assert "comparisons" not in out["by_half"]["control"]
    assert out["by_half"]["control"] == {"units": 2, "resolved": 1}


def test_live_analysis_due_at_the_target_before_the_deadline() -> None:
    d = date(2026, 12, 1)
    units = [_unit(2 * k + 1, k % 2, (0.6, 0.7, 0.8), d) for k in range(q6.LIVE_TARGET)]
    out = q6.live_analysis(units, 0.5, date(2027, 1, 1))
    assert out["due"] is True
    assert out["by_half"]["policy"]["resolved"] == q6.LIVE_TARGET


# -- live_units over a ledger -----------------------------------------------------


def _add_run(
    led: Ledger,
    *,
    mode: RunMode,
    tournament: str,
    questions: dict[int, list[tuple[float, str]]],
) -> None:
    run = led.start_run(
        mode=mode,
        tournament=tournament,
        provider="openrouter",
        model="m",
        settings={},
        code_version="test",
    )
    for qid, draws in questions.items():
        q, post = binary_question(T0, post_id=qid + 1000, question_id=qid)
        led.record_question(run, q, T0, post)
        for member, (p, model) in enumerate(draws):
            call = led.record_call(
                run_id=run,
                question_id=qid,
                role="forecaster",
                member=member,
                provider="openrouter",
                model=model,
                model_release_date=None,
                request_hash=f"h{qid}-{member}-{run}",
                request={},
                response_text="x",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                error=None,
                started_at=T0,
                source_ids=[],
            )
            led.record_forecast(
                run_id=run,
                question_id=qid,
                kind="member",
                member=member,
                method="draw",
                probability=p,
                reasoning="r",
                call_id=call,
            )


def test_live_units_selects_the_pre_registered_units(tmp_path: Path) -> None:
    clock = FixedClock(T0)
    (tmp_path / "data/ledgers").mkdir(parents=True)
    three = [(0.6, "a"), (0.7, "a"), (0.8, "a")]
    with Ledger.open(tmp_path / "data/ledgers/x.sqlite", clock) as led:
        _add_run(
            led,
            mode=RunMode.DRY_RUN,
            tournament="minibench",
            questions={
                1: three,
                2: three[:2],  # two draws: too few
                3: [(0.6, "a"), (0.7, "b"), (0.8, "a")],  # two models
            },
        )
        clock.advance(60)
        _add_run(  # the first forecast per question wins
            led,
            mode=RunMode.DRY_RUN,
            tournament="minibench",
            questions={1: [(0.1, "a")] * 3, 4: three},
        )
        clock.advance(60)
        _add_run(led, mode=RunMode.DRY_RUN, tournament="bot-testing-area", questions={5: three})
        _add_run(led, mode=RunMode.BACKTEST, tournament="minibench", questions={6: three})
        led.record_resolution(
            question_id=1,
            post_id=1001,
            question_type="binary",
            status="resolved",
            value="yes",
            resolved_at=T0,
            source="manual",
            evidence="e",
        )
        led.record_resolution(
            question_id=4,
            post_id=1004,
            question_type="binary",
            status="annulled",
            value=None,
            resolved_at=T0,
            source="manual",
            evidence="e",
        )
    units = q6.live_units(tmp_path)
    assert [u.question_id for u in units] == [1, 4]
    first = units[0]
    assert first.draws == (0.6, 0.7, 0.8)
    assert first.outcome == 1
    assert first.model == "a"
    assert first.ledger == "x.sqlite"
    assert first.opened == T0.date()
    assert units[1].outcome is None


def test_live_units_without_ledgers_is_empty(tmp_path: Path) -> None:
    assert q6.live_units(tmp_path) == []


# -- the snapshot and the report --------------------------------------------------


def test_committed_snapshot_is_reproduced() -> None:
    # Takes about 15 s: it rebuilds both samples and every leave-one-out fit.
    snap = q6.snapshot(REPO, date(2026, 11, 22))
    committed = (REPO / q6.SNAPSHOT).read_text(encoding="utf-8")
    assert committed == json.dumps(snap, indent=1, sort_keys=True) + "\n"


def test_preregistration_is_planned_text_verbatim() -> None:
    plan = (REPO / "PLAN.md").read_text(encoding="utf-8")
    start, end = "<!-- q6-preregistration:begin -->\n", "<!-- q6-preregistration:end -->"
    want = plan.split(start, 1)[1].split(end, 1)[0]
    assert render_q6_preregistration(REPO) == want
    results = (REPO / "RESULTS.md").read_text(encoding="utf-8")
    m = re.search(
        r"<!-- augur:begin q6-preregistration -->\n(.*?)<!-- augur:end q6-preregistration -->",
        results,
        re.S,
    )
    assert m is not None
    assert m.group(1) == want


def test_render_q6_without_a_snapshot(tmp_path: Path) -> None:
    assert render_q6(tmp_path) == "M9's analysis has not been run (`augur q6`).\n"
