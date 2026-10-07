import asyncio

import pytest

from agent.bargein import BargeInGate, OverTalkGuard, classify, ends_with_continuation, is_backchannel
from agent.judge import parse_jev, parse_verdict

STATEMENT = "Kuryeniz yola çıktı, yirmi beş dakika içinde kapınızda olur."
QUESTION = "Siparişinizi iptal etmemi ister misiniz?"


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    ("text", "question", "seconds", "action", "reason"),
    [
        ("hıhı", False, 0.3, "ignore", "onay"),
        ("evet evet", False, 0.6, "ignore", "onay"),
        ("tamam anladım", False, 1.5, "ignore", "onay"),
        ("evet", True, 0.3, "interrupt", "soruya cevap"),
        ("hayır istemiyorum", True, 0.8, "interrupt", "soruya cevap"),
        ("dur", False, 0.2, "interrupt", "durdurma"),
        ("bir saniye", False, 0.4, "interrupt", "durdurma"),
        ("ama ben adresimi değiştirdim", False, 1.0, "interrupt", "anlamlı söz"),
        ("insan", False, 0.3, "ignore", "kısa ses"),
        ("pardon adresim", False, 0.7, "judge", "belirsiz"),
        ("", False, 0.0, "ignore", "boş"),
    ],
)
def test_classify(text, question, seconds, action, reason):
    decision = classify(text, question=question, speech_seconds=seconds)
    assert (decision.action, decision.reason) == (action, reason)


def test_echo_of_own_sentence_is_ignored():
    decision = classify(
        "sizi",
        question=False,
        speech_seconds=0.9,
        agent_text="Size nasıl yardımcı olabilirim?",
    )
    assert decision.reason == "yankı"


def test_turkish_casing_and_punctuation():
    assert is_backchannel("Hı hı, TAMAM.")
    assert ends_with_continuation("Siparişim gelmedi ve")
    assert not ends_with_continuation("Siparişim gelmedi.")


def make_gate(clock, **kwargs):
    calls = {"interrupt": 0, "warn": [], "events": []}
    gate = BargeInGate(
        interrupt=lambda: calls.__setitem__("interrupt", calls["interrupt"] + 1),
        warn=lambda remaining: calls["warn"].append(remaining),
        emit=calls["events"].append,
        clock=clock,
        **kwargs,
    )
    return gate, calls


def test_backchannel_during_statement_does_not_interrupt_and_drops_turn():
    clock = Clock()
    gate, calls = make_gate(clock)
    gate.add_sentence(STATEMENT, 4.0)
    gate.set_agent_speaking(True)
    clock.now += 1.0
    gate.set_user_speaking(True)
    clock.now += 0.3
    gate.on_interim("hıhı")
    assert calls["interrupt"] == 0
    assert gate.should_drop_turn("hıhı")
    assert not gate.should_drop_turn("hıhı")


def test_yes_after_question_interrupts():
    clock = Clock()
    gate, calls = make_gate(clock)
    gate.add_sentence(QUESTION, 3.0)
    gate.add_sentence(STATEMENT, 4.0)
    gate.set_agent_speaking(True)
    clock.now += 3.5
    gate.set_user_speaking(True)
    clock.now += 0.3
    gate.on_interim("evet")
    assert calls["interrupt"] == 1


def test_yes_during_plain_statement_is_backchannel():
    clock = Clock()
    gate, calls = make_gate(clock)
    gate.add_sentence(STATEMENT, 4.0)
    gate.add_sentence(STATEMENT, 4.0)
    gate.set_agent_speaking(True)
    clock.now += 5.0
    gate.set_user_speaking(True)
    gate.on_interim("evet")
    assert calls["interrupt"] == 0


def test_repeated_interrupts_trigger_one_warning():
    clock = Clock()
    gate, calls = make_gate(clock, guard=OverTalkGuard(max_interrupts=3, window=30, cooldown=60))
    for _ in range(4):
        gate.add_sentence(STATEMENT, 4.0)
        gate.set_agent_speaking(True)
        clock.now += 0.5
        gate.set_user_speaking(True)
        gate.on_interim("dur")
        gate.set_user_speaking(False)
        gate.set_agent_speaking(False)
        clock.now += 2.0
    assert calls["interrupt"] == 3
    assert len(calls["warn"]) == 1
    assert calls["events"][-2]["action"] == "blocked"


def test_long_overlap_triggers_warning():
    clock = Clock()
    gate, calls = make_gate(clock)
    gate.add_sentence(STATEMENT, 10.0)
    gate.set_agent_speaking(True)
    gate.set_user_speaking(True)
    clock.now += 6.5
    gate.on_interim("hıhı")
    assert calls["interrupt"] == 0
    assert len(calls["warn"]) == 1


def test_judge_runs_in_background_and_can_interrupt():
    async def scenario():
        clock = Clock()
        seen = []

        async def judge(text, **context):
            seen.append((text, context["sentence"]))
            await asyncio.sleep(0.01)
            return {"action": "interrupt", "manipulation": False}

        gate, calls = make_gate(clock, judge=judge)
        gate.add_sentence(STATEMENT, 4.0)
        gate.set_agent_speaking(True)
        gate.set_user_speaking(True)
        clock.now += 0.7
        decision = gate.on_interim("pardon adresim")
        assert decision.action == "judge"
        assert calls["interrupt"] == 0
        await asyncio.gather(*gate.tasks)
        assert calls["interrupt"] == 1
        assert seen[0][1] == STATEMENT

    asyncio.run(scenario())


def test_slow_judge_is_ignored():
    async def scenario():
        clock = Clock()

        async def judge(text, **context):
            await asyncio.sleep(1)
            return {"action": "interrupt", "manipulation": False}

        gate, calls = make_gate(clock, judge=judge, judge_timeout=0.05)
        gate.add_sentence(STATEMENT, 4.0)
        gate.set_agent_speaking(True)
        gate.set_user_speaking(True)
        clock.now += 0.7
        gate.on_interim("pardon adresim")
        await asyncio.gather(*gate.tasks)
        assert calls["interrupt"] == 0
        assert calls["events"][-1]["reason"] == "hakem yetişmedi"

    asyncio.run(scenario())


def test_judge_decides_meaningful_speech():
    async def scenario():
        clock = Clock()

        async def judge(text, **context):
            return {"action": "ignore", "manipulation": False}

        gate, calls = make_gate(clock, judge=judge)
        gate.add_sentence(STATEMENT, 4.0)
        gate.set_agent_speaking(True)
        gate.set_user_speaking(True)
        clock.now += 1.0
        assert gate.on_interim("ama ben adresimi değiştirdim").action == "judge"
        await asyncio.gather(*gate.tasks)
        assert calls["interrupt"] == 0

    asyncio.run(scenario())


def test_slow_judge_falls_back_to_rule_for_meaningful_speech():
    async def scenario():
        clock = Clock()

        async def judge(text, **context):
            await asyncio.sleep(1)
            return {"action": "ignore", "manipulation": False}

        gate, calls = make_gate(clock, judge=judge, judge_timeout=0.05)
        gate.add_sentence(STATEMENT, 4.0)
        gate.set_agent_speaking(True)
        gate.set_user_speaking(True)
        clock.now += 1.0
        gate.on_interim("ama ben adresimi değiştirdim")
        await asyncio.gather(*gate.tasks)
        assert calls["interrupt"] == 1

    asyncio.run(scenario())


def test_stop_phrase_skips_judge():
    async def scenario():
        clock = Clock()
        seen = []

        async def judge(text, **context):
            seen.append(text)
            return {"action": "ignore", "manipulation": False}

        gate, calls = make_gate(clock, judge=judge)
        gate.add_sentence(STATEMENT, 4.0)
        gate.set_agent_speaking(True)
        gate.set_user_speaking(True)
        gate.on_interim("dur")
        assert calls["interrupt"] == 1
        assert seen == []

    asyncio.run(scenario())


def test_each_new_interim_is_judged_and_latest_waits():
    async def scenario():
        clock = Clock()
        seen = []
        release = asyncio.Event()

        async def judge(text, **context):
            seen.append(text)
            await release.wait()
            return {"action": "interrupt" if "iptal" in text else "ignore", "manipulation": False}

        gate, calls = make_gate(clock, judge=judge)
        gate.add_sentence(STATEMENT, 6.0)
        gate.set_agent_speaking(True)
        gate.set_user_speaking(True)
        for text in ("pardon", "pardon ben", "pardon ben siparişi", "pardon ben siparişi iptal"):
            gate.on_interim(text)
        await asyncio.sleep(0)
        assert seen == ["pardon", "pardon ben"]
        release.set()
        while gate.tasks:
            await asyncio.gather(*list(gate.tasks))
        assert seen[-1] == "pardon ben siparişi iptal"
        assert calls["interrupt"] == 1

    asyncio.run(scenario())


def test_judge_gets_question_and_reply_and_reason():
    async def scenario():
        clock = Clock()
        seen = {}

        async def judge(text, **context):
            seen.update(context)
            return {"action": "interrupt", "manipulation": False, "reason": "soruya cevap"}

        gate, calls = make_gate(clock, judge=judge)
        gate.add_sentence(QUESTION, 3.0)
        gate.add_sentence(STATEMENT, 4.0)
        gate.set_agent_speaking(True)
        clock.now += 3.5
        gate.set_user_speaking(True)
        clock.now += 1.0
        gate.on_interim("ben adresi değiştirmek istiyorum")
        await asyncio.gather(*gate.tasks)
        assert seen == {"sentence": STATEMENT, "reply": f"{QUESTION} {STATEMENT}", "question": QUESTION}
        event = calls["events"][-1]
        assert (event["action"], event["reason"], event["by"]) == ("interrupt", "soruya cevap", "hakem")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("raw", "action"),
    [
        ('{"interrupt": true, "manipulation": false, "reason": "itiraz"}', "interrupt"),
        ('```json\n{"interrupt": false, "manipulation": false, "reason": "onay"}\n```', "ignore"),
        ("", "ignore"),
    ],
)
def test_parse_verdict(raw, action):
    assert parse_verdict(raw)["action"] == action


def test_parse_jev_uses_yes_probability():
    body = {"answers": {"interrupt": {"type": "noul", "noul": 0.82}, "manipulation": {"type": "noul", "noul": 0.1}}}
    assert parse_jev(body) == {"action": "interrupt", "manipulation": False, "reason": "jev %82"}
    assert parse_jev({"answers": {}})["action"] == "ignore"


def test_rules_mode_interrupts_meaningful_speech_without_judge():
    async def scenario():
        clock = Clock()

        async def judge(text, **context):
            return {"action": "ignore", "manipulation": False}

        gate, calls = make_gate(clock, judge=judge, judge_first=False)
        gate.add_sentence(STATEMENT, 4.0)
        gate.set_agent_speaking(True)
        gate.set_user_speaking(True)
        clock.now += 1.0
        gate.on_interim("ama ben adresimi değiştirdim")
        assert calls["interrupt"] == 1

    asyncio.run(scenario())


def test_continuation_waits_and_merges():
    async def scenario():
        gate, _ = make_gate(Clock())
        waiter = asyncio.ensure_future(gate.wait_user_resume(0.5))
        await asyncio.sleep(0)
        gate.set_user_speaking(True)
        assert await waiter
        gate.hold("siparişim gelmedi ve")
        assert gate.merge_held("kurye de aramadı") == "siparişim gelmedi ve kurye de aramadı"
        assert gate.merge_held("tamam") == "tamam"

    asyncio.run(scenario())
