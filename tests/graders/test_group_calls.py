"""Several submissions to one input in one call per judge: ``CriterionGrader.grade_group``.

Each LLM judge gets one call listing the criteria once and every submission under its own id
(``s1``, ``s2``, ...), in an order shuffled per judge, and answers every criterion for every
submission (``GroupJudgment``); the answers map back to one report per submission, in the order
the submissions were given. A missing or unusable entry fails only its submission, or only its
criterion of its submission; a failed call fails every criterion of every submission; the
call's usage rides on the first submission's first criterion.

Nothing here reaches the network: ``GroupLLM`` is patched in for ``LLMClient`` and builds its
reply from the prompt's ids, then runs it through the real validation.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from test_error_routing import FAILURES
from test_per_item_calls import (
    BINARY,
    CLARITY,
    COST,
    GUIDELINES,
    GUIDELINES_PREFIX,
    LIGHT,
    LLM,
    MIXED,
    QUERY,
    USAGE,
    Call,
    dm,
    make_grader,
    presented_labels,
    rubric_blocks,
)

from autorubric import CriterionVerdict, Rubric
from autorubric.graders import JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.prompts import build_group_system_prompt
from autorubric.types import GroupJudgment

SUBMISSION_BLOCK = re.compile(
    r'<submission id="(?P<id>[^"]+)">\n(?P<text>.*?)\n</submission>', re.S
)
SUBMISSIONS = [
    "Plants use light to make sugar.",
    "Plants eat soil.",
    "Light, water and air become sugar in the leaves.",
]
LIT, DIM = {"Very clear", "Formal"}, {"Unclear", "Playful"}


def shown(user_prompt: str) -> dict[str, str]:
    """Each submission's text by the id the prompt shows it under, in the prompt's order."""
    return {m["id"]: m["text"] for m in SUBMISSION_BLOCK.finditer(user_prompt)}


class GroupLLM:
    """Stand-in for ``LLMClient`` that records every call and answers ``GroupJudgment``.

    For each submission of the prompt and each criterion: a binary criterion ``MET`` when the
    submission mentions light and ``UNMET`` otherwise; a multi-choice criterion its best option
    (``LIT``) when it mentions light and its worst (``DIM``) otherwise, wherever the call
    presents it. ``drop`` names texts the reply leaves out; ``bad`` maps a text to a criterion
    id whose entry is made unusable; ``raw`` replaces the whole reply; ``error`` is raised by
    every call.
    """

    def __init__(
        self,
        *,
        drop: tuple[str, ...] = (),
        bad: dict[str, str] | None = None,
        raw: Any = None,
        error: BaseException | None = None,
    ) -> None:
        self.drop, self.bad, self.raw, self.error = drop, bad or {}, raw, error
        self.calls: list[Call] = []

    def reply(self, user_prompt: str) -> dict[str, Any]:
        out = []
        for submission_id, text in shown(user_prompt).items():
            if text in self.drop:
                continue
            lit = "light" in text.lower()
            judgments = []
            for criterion_id, block in rubric_blocks(user_prompt).items():
                if "<options>" in block:
                    labels = presented_labels(block)
                    pick = next(i for i, lab in enumerate(labels) if lab in (LIT if lit else DIM))
                    entry = {
                        "criterion_id": criterion_id,
                        "criterion_status": None,
                        "selected_option": pick + 1,
                    }
                else:
                    entry = {
                        "criterion_id": criterion_id,
                        "criterion_status": "MET" if lit else "UNMET",
                        "selected_option": None,
                    }
                entry["explanation"] = f"{submission_id} {criterion_id}"
                if self.bad.get(text) == criterion_id:
                    entry = {"criterion_id": criterion_id, "explanation": None}
                judgments.append(entry)
            out.append({"submission_id": submission_id, "judgments": judgments})
        return {"submissions": out}

    async def generate(
        self, system_prompt: str, user_prompt: str, response_format: Any = None, **kwargs: Any
    ) -> GenerateResult:
        self.calls.append(Call(system_prompt, user_prompt, response_format))
        if self.error is not None:
            raise self.error
        assert response_format is GroupJudgment
        data = self.raw if self.raw is not None else self.reply(user_prompt)
        return GenerateResult(
            content="{}", usage=USAGE, cost=COST, parsed=GroupJudgment.model_validate(data)
        )


def panel(*ids: str, **kwargs: Any) -> tuple[Any, dict[str, GroupLLM]]:
    fakes = {i: GroupLLM(**kwargs) for i in ids}
    grader = make_grader(
        fakes, judges=[JudgeSpec(LLM, i) for i in ids], llm_calls="per_item", auto_na_option=False
    )
    return grader, fakes


def light_report(report: Any) -> Any:
    (light,) = [r for r in report.report if r.criterion.name == "light"]
    return light


class TestGroupCalls:
    @pytest.mark.asyncio
    async def test_one_call_per_judge_and_a_report_per_submission_in_order(self):
        grader, fakes = panel("a", "b")
        reports = await Rubric(MIXED).grade_group(SUBMISSIONS, grader, query=QUERY)

        assert [len(f.calls) for f in fakes.values()] == [1, 1]
        assert [light_report(r).final_verdict for r in reports] == [
            CriterionVerdict.MET,
            CriterionVerdict.UNMET,
            CriterionVerdict.MET,
        ]
        clarity = [next(c for c in r.report if c.criterion.name == "clarity") for r in reports]
        assert [c.final_multi_choice_verdict.selected_label for c in clarity] == [
            "Very clear",
            "Unclear",
            "Very clear",
        ]
        assert reports[0].score > reports[1].score and reports[0].score == reports[2].score
        assert all(r.error is None for r in reports)

    @pytest.mark.asyncio
    async def test_the_prompt_lists_the_criteria_once_and_every_submission_by_id(self):
        grader, fakes = panel("a")
        await Rubric(MIXED).grade_group(SUBMISSIONS, grader, query=QUERY)
        (call,) = fakes["a"].calls

        assert list(rubric_blocks(call.user_prompt)) == ["c0", "c1", "c2", "c3"]
        assert list(shown(call.user_prompt)) == ["s1", "s2", "s3"]
        assert sorted(shown(call.user_prompt).values()) == sorted(SUBMISSIONS)
        assert f"<input>{QUERY}</input>" in call.user_prompt
        assert call.system_prompt == build_group_system_prompt(
            grader._system_prompt, grader._multi_choice_system_prompt
        )

    @pytest.mark.asyncio
    async def test_each_judge_sees_its_own_order_unless_told_not_to_shuffle(self):
        grader, fakes = panel(*"abcdef")
        await Rubric(BINARY).grade_group(SUBMISSIONS, grader)
        orders = {tuple(shown(f.calls[0].user_prompt).values()) for f in fakes.values()}
        assert len(orders) > 1

        grader, fakes = panel("a", "b")
        await Rubric(BINARY).grade_group(SUBMISSIONS, grader, shuffle_submissions=False)
        assert {tuple(shown(f.calls[0].user_prompt).values()) for f in fakes.values()} == {
            tuple(SUBMISSIONS)
        }

    @pytest.mark.asyncio
    async def test_the_order_is_the_same_on_a_second_grading(self):
        grader, fakes = panel("a")
        for _ in range(2):
            await Rubric(BINARY).grade_group(SUBMISSIONS, grader)
        first, second = (c.user_prompt for c in fakes["a"].calls)
        assert first == second

    @pytest.mark.asyncio
    async def test_a_group_of_one(self):
        grader, fakes = panel("a")
        (report,) = await Rubric(MIXED).grade_group(SUBMISSIONS[:1], grader)
        assert light_report(report).final_verdict == CriterionVerdict.MET
        assert list(shown(fakes["a"].calls[0].user_prompt)) == ["s1"]

    @pytest.mark.asyncio
    async def test_no_submissions_makes_no_call(self):
        grader, fakes = panel("a")
        assert await Rubric(MIXED).grade_group([], grader) == []
        assert fakes["a"].calls == []

    @pytest.mark.asyncio
    async def test_usage_is_counted_once_per_call(self):
        grader, _ = panel("a")
        reports = await Rubric(MIXED).grade_group(SUBMISSIONS, grader)
        totals = [r.token_usage.total_tokens if r.token_usage else 0 for r in reports]
        assert sum(totals) == USAGE.total_tokens
        assert sum(r.completion_cost or 0 for r in reports) == pytest.approx(COST)

    @pytest.mark.asyncio
    async def test_guidelines_prefix_the_prompt(self):
        grader, fakes = panel("a")
        await Rubric(BINARY, guidelines=GUIDELINES).grade_group(SUBMISSIONS, grader)
        assert fakes["a"].calls[0].user_prompt.startswith(GUIDELINES_PREFIX)

    @pytest.mark.asyncio
    async def test_dict_submissions_are_shown_with_their_markers(self):
        grader, fakes = panel("a")
        await Rubric(BINARY).grade_group([{"thinking": "hmm", "output": "light"}], grader)
        assert "<thinking>hmm</thinking>\n<output>light</output>" in fakes["a"].calls[0].user_prompt


class TestGroupFailures:
    @pytest.mark.asyncio
    async def test_a_submission_left_out_fails_alone(self):
        grader, _ = panel("a", drop=(SUBMISSIONS[1],))
        reports = await Rubric(MIXED).grade_group(SUBMISSIONS, grader)
        assert reports[0].error is None and reports[2].error is None
        assert all(
            c.error and "no usable judgments for submission" in c.error for c in reports[1].report
        )
        assert reports[1].score is None

    @pytest.mark.asyncio
    async def test_an_unusable_judgment_fails_only_its_criterion_of_its_submission(self):
        grader, _ = panel("a", bad={SUBMISSIONS[0]: "c1"})
        reports = await Rubric(MIXED).grade_group(SUBMISSIONS, grader)
        errors = [[c.criterion.name for c in r.report if c.error] for r in reports]
        assert errors == [["clarity"], [], []]

    @pytest.mark.asyncio
    async def test_a_failed_call_fails_every_criterion_of_every_submission(self):
        grader, _ = panel("a", error=RuntimeError("boom"))
        reports = await Rubric(MIXED).grade_group(SUBMISSIONS, grader)
        assert all(c.error for r in reports for c in r.report)

    @pytest.mark.asyncio
    async def test_in_a_panel_the_other_judge_still_grades(self):
        fakes = {"a": GroupLLM(error=FAILURES["infrastructure"]()), "b": GroupLLM()}
        grader = make_grader(
            fakes, judges=[JudgeSpec(LLM, "a"), JudgeSpec(LLM, "b")], llm_calls="per_item"
        )
        reports = await Rubric([LIGHT, CLARITY]).grade_group(SUBMISSIONS, grader)
        assert [light_report(r).final_verdict for r in reports] == [
            CriterionVerdict.MET,
            CriterionVerdict.UNMET,
            CriterionVerdict.MET,
        ]

    @pytest.mark.asyncio
    async def test_a_decision_model_is_refused(self):
        grader = make_grader({}, judge_model_config=dm())
        with pytest.raises(ValueError, match="LLM judges only"):
            await grader.grade_group(SUBMISSIONS, BINARY)

    @pytest.mark.asyncio
    async def test_a_grader_without_grade_group_is_refused(self):
        class Plain:
            pass

        with pytest.raises(TypeError, match="cannot grade a group"):
            await Rubric(BINARY).grade_group(SUBMISSIONS, Plain())  # type: ignore[arg-type]
