"""End-to-end smoke tests (run with: pytest -q). They use whatever providers are available, offline included."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pipeline  # noqa: E402
import rag_core as rc  # noqa: E402
from eval_data import TEST_SET  # noqa: E402


@pytest.fixture(scope="module")
def bot():
    pipeline.build_index(log=lambda m: None)
    return rc.RAGChatbot.from_persisted()


def test_documents_load_in_three_formats():
    types = {d.metadata["file_type"] for d in rc.load_documents()}
    assert {"pdf", "csv", "txt"} <= types


def test_chunks_have_ids_and_headers():
    chunks = rc.split_documents(rc.load_documents())
    assert len(chunks) > 20
    assert all("chunk_id" in c.metadata for c in chunks)
    assert any(" | " in c.page_content.split("\n")[0] for c in chunks)


def test_retrieval_finds_attendance_rule(bot):
    docs, _ = bot.retriever.retrieve("What is the minimum attendance required?", k=3)
    assert any("75%" in d.page_content for d in docs)


def test_answer_has_sources(bot):
    r = bot.ask("What is the late fee for delayed fee payment?", use_history=False)
    assert not r["refused"] and r["sources"] and "200" in r["answer"]


def test_unrelated_question_is_refused(bot):
    r = bot.ask("Who won the FIFA World Cup in 2022?", use_history=False)
    assert r["refused"]


def test_follow_up_is_rewritten(bot):
    bot.reset()
    bot.ask("How many books can an undergraduate student borrow?")
    r = bot.ask("And for postgraduates?")
    assert "book" in r["standalone_question"].lower()
    bot.reset()


def test_retrieval_quality_on_test_set(bot):
    import evaluation
    m = evaluation.retrieval_metrics(bot.retriever, TEST_SET, "hybrid")
    assert m["hit@3"] >= 0.8
