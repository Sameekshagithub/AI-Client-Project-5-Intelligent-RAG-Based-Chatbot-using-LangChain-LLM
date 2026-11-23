"""Retrieval + generation metrics on the held-out TEST_SET (used by the Streamlit Evaluation tab and the CLI)."""
import re

import numpy as np
import pandas as pd

from eval_data import TEST_SET
from rag_core import STOPWORDS, tokenize


def is_relevant(doc, it):
    return (doc.metadata.get("file") == it["source"]
            and any(k.lower() in doc.page_content.lower() for k in it["keywords"]))


def retrieval_metrics(retriever, items=TEST_SET, mode="hybrid", k=5):
    rows = []
    for it in items:
        if it["source"] is None:
            continue
        docs, _ = retriever.retrieve(it["question"], k=k, mode=mode)
        rel = [is_relevant(d, it) for d in docs]
        first = next((i + 1 for i, r in enumerate(rel) if r), None)
        text = " ".join(d.page_content.lower() for d in docs)
        rows.append({"hit@1": int(any(rel[:1])), "hit@3": int(any(rel[:3])), "hit@5": int(any(rel[:5])),
                     "mrr": 1 / first if first else 0.0,
                     "keyword_recall": float(np.mean([kw.lower() in text for kw in it["keywords"]]))})
    return pd.DataFrame(rows).mean().round(3)


def support_score(answer, contexts):
    toks = [t for t in tokenize(re.sub(r"\[\d+\]", " ", answer)) if t not in STOPWORDS]
    ctx = set(tokenize(" ".join(contexts)))
    return float(np.mean([t in ctx for t in toks])) if toks else np.nan


def run_generation(bot, items=TEST_SET, progress=None):
    rows = []
    for n, it in enumerate(items):
        bot.reset()
        r = bot.ask(it["question"], use_history=False)
        in_scope = it["source"] is not None
        rows.append({"question": it["question"], "type": it["qtype"], "expected": it["ground_truth"],
                     "answer": r["answer"], "refused": r["refused"], "in_scope": in_scope, "latency_s": r["latency_s"],
                     "correctness": float(np.mean([k.lower() in r["answer"].lower() for k in it["keywords"]])) if in_scope else np.nan,
                     "faithfulness": support_score(r["answer"], r["contexts"]) if not r["refused"] else np.nan})
        if progress:
            progress((n + 1) / len(items))
    bot.reset()
    df = pd.DataFrame(rows)
    ins, oos = df[df.in_scope], df[~df.in_scope]
    summary = {"Answer correctness": ins.correctness.mean(), "Faithfulness": df.faithfulness.mean(),
               "Refusal accuracy": oos.refused.mean(), "False-refusal rate": ins.refused.mean(),
               "Avg latency (s)": df.latency_s.mean()}
    return df, {k: round(float(v), 3) for k, v in summary.items()}


def run_all(bot, progress=None):
    ret = {m: retrieval_metrics(bot.retriever, mode=m) for m in ("similarity", "mmr", "hybrid")}
    df, summary = run_generation(bot, progress=progress)
    return pd.DataFrame(ret).T, df, summary


if __name__ == "__main__":
    from rag_core import RAGChatbot
    ret, df, summary = run_all(RAGChatbot.from_persisted())
    print("\nRetrieval (test set)\n", ret.to_string(), "\n\nGeneration\n", pd.Series(summary).to_string())
    df.to_csv("logs/eval_results.csv", index=False)
    print("\nSaved logs/eval_results.csv")
