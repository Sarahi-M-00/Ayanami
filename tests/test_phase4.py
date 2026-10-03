"""Phase 4 acceptance tests: adapter interface, router, schema, scrubber.

All tests run against the fake (precomputed) teacher. No model download,
no network.
"""

from pathlib import Path

import pytest

from ayanami_distill.data.records import is_valid, validate_record
from ayanami_distill.teachers import (
    base as TB,
    cache as TC,
    fake as TF,
    router as TR,
    scrub as TS,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fake_teacher.jsonl"
STUDENT_FP = "563a701b87f87d8076a6faf47ea67cc1292b2321e25f18d09e2d11b6189140aa"


def make_meta(fp=STUDENT_FP, tos=True, lic="test-fixture"):
    return TB.TeacherMetadata(
        id="fake-qwen3-test", revision="fixture-v1", backend="precomputed",
        tokenizer_id="student_base", tokenizer_fingerprint=fp,
        license=lic, tos_allows_distillation=tos)


def make_teacher(logprobs=True):
    return TF.PrecomputedTeacher(
        make_meta(),
        TB.TeacherCapabilities(text=True, logprobs_topk=logprobs),
        FIXTURE)


# ---- adapter interface ----
def test_fake_is_adapter_with_metadata_and_capabilities():
    t = make_teacher()
    assert isinstance(t, TB.TeacherAdapter)
    assert t.metadata.id == "fake-qwen3-test"
    assert t.capabilities.text is True
    t.close()


def test_fake_generate_returns_valid_record():
    t = make_teacher()
    rec = t.generate([{"role": "user", "content": "hi"}], {"temperature": 0.0})
    assert is_valid(rec), validate_record(rec)
    assert rec["teacher_id"] == "fake-qwen3-test"
    t.close()


def test_fake_score_replays_stored_topk():
    t = make_teacher()
    t.generate([{"role": "user", "content": "a"}], {})
    t.generate([{"role": "user", "content": "b"}], {})  # serves fake-0002
    topk = t.score([{"role": "user", "content": "b"}], "whatever")
    assert topk["k"] == 4 and len(topk["ids"]) == 4
    t.close()


# ---- router ----
def test_router_same_tokenizer_with_logprobs_is_logit_kd():
    d = TR.select_mode(make_meta(), TB.TeacherCapabilities(logprobs_topk=True), STUDENT_FP)
    assert d.mode == TR.LOGIT_KD and "same tokenizer" in d.reason


def test_router_different_tokenizer_defaults_to_text_sft():
    d = TR.select_mode(make_meta(fp="other"), TB.TeacherCapabilities(), STUDENT_FP)
    assert d.mode == TR.TEXT_SFT and "default" in d.reason


def test_router_different_tokenizer_with_flag_is_cross_tokenizer():
    d = TR.select_mode(make_meta(fp="other"), TB.TeacherCapabilities(), STUDENT_FP,
                       cross_tokenizer_enabled=True)
    assert d.mode == TR.CROSS_TOKENIZER


def test_router_same_tokenizer_without_logprobs_is_text_sft():
    d = TR.select_mode(make_meta(), TB.TeacherCapabilities(), STUDENT_FP)
    assert d.mode == TR.TEXT_SFT and "no logprobs" in d.reason


def test_router_hidden_kd_flag_requires_states_and_same_tokenizer():
    caps = TB.TeacherCapabilities(logprobs_topk=True, hidden_states=True)
    d = TR.select_mode(make_meta(), caps, STUDENT_FP, hidden_kd_enabled=True)
    assert d.mode == TR.LOGIT_KD and d.hidden_kd is True
    d2 = TR.select_mode(make_meta(fp="other"), caps, STUDENT_FP, hidden_kd_enabled=True)
    assert d2.hidden_kd is False


def test_router_blocked_license_raises():
    with pytest.raises(TB.TeacherBlockedError):
        TR.select_mode(make_meta(tos=False), TB.TeacherCapabilities(), STUDENT_FP)
    with pytest.raises(TB.TeacherBlockedError):
        TR.select_mode(make_meta(lic=""), TB.TeacherCapabilities(), STUDENT_FP)


# ---- record schema ----
def test_fixture_records_are_valid():
    for rec in TC.read_records(FIXTURE):
        assert is_valid(rec), validate_record(rec)


def test_schema_rejects_missing_field_bad_domain_bad_role():
    good = TC.read_records(FIXTURE)[0]
    bad1 = {k: v for k, v in good.items() if k != "completion"}
    assert not is_valid(bad1)
    bad2 = dict(good, domain="cooking")
    assert any("domain" in e for e in validate_record(bad2))
    bad3 = dict(good, messages=[{"role": "hacker", "content": "x"}])
    assert any("messages[0]" in e for e in validate_record(bad3))
    bad4 = dict(good, logprobs_topk={"k": 2, "ids": [1], "logprobs": [0.0, -1.0],
                                     "tail_mass": 0.0})
    assert any("length k" in e for e in validate_record(bad4))


# ---- scrubber ----
def test_scrub_rewrites_self_id_and_vendor():
    cleaned, hits = TS.scrub_completion(
        "I am Qwen, an AI developed by Alibaba Cloud. I can help.")
    assert "I am Ayanami" in cleaned
    assert "developed by Ling" in cleaned
    assert "Qwen" not in cleaned and "Alibaba" not in cleaned
    assert len(hits) == 2


def test_scrub_handles_gpt_claude_variants():
    cleaned, hits = TS.scrub_completion(
        "I'm GPT-4o, created by OpenAI. My name is Claude.")
    assert "Ayanami" in cleaned and "created by Ling" in cleaned
    assert len(hits) == 3


def test_scrub_leaves_code_and_prose_untouched():
    text = "Run `llama-quantize` then `import transformers`. Llama drücken capacity is high."
    cleaned, hits = TS.scrub_completion(text)
    assert cleaned == text and hits == []


# ---- cache ----
def test_cache_estimate_math():
    est = TC.estimate_topk_cache(1000, k=32)
    assert est["bytes_per_token"] == 32 * 6 + 2
    assert est["total_bytes"] == 1000 * (32 * 6 + 2)
    assert est["total_mib"] == round(194000 / 2**20, 2)


def test_cache_append_read_roundtrip(tmp_path):
    p = tmp_path / "c.jsonl"
    TC.append_record(p, {"a": 1})
    TC.append_record(p, {"a": 2})
    assert TC.read_records(p) == [{"a": 1}, {"a": 2}]


def test_cache_dir_layout():
    got = TC.cache_dir(Path("/repo"), "org/model", "abc123")
    assert str(got) == "/repo/data/teacher_cache/org_model/abc123"
