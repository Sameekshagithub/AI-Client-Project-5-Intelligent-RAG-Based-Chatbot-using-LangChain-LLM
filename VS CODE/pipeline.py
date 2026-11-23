"""Index-building pipeline: load -> chunk -> embed -> ChromaDB (+ calibrated out-of-scope threshold)."""
from pathlib import Path

import numpy as np

import rag_core as rc
from eval_data import DEV_SET
from make_sample_data import SAMPLE_FILES, write_sample_kb


def index_exists():
    return (Path(rc.CONFIG["persist_dir"]) / "rag_meta.json").exists()


def build_index(log=print):
    """(Re)build the persistent index from everything in data/. Returns a summary dict."""
    data_dir = Path(rc.CONFIG["data_dir"])
    data_dir.mkdir(exist_ok=True)
    if not any(p.suffix.lower() in (".pdf", ".csv", ".txt", ".md") for p in data_dir.glob("*")):
        log("data/ is empty -> creating the sample NovaTech knowledge base")
        write_sample_kb(str(data_dir))

    log("Step 1/4  loading documents")
    docs = rc.load_documents()
    log("Step 2/4  chunking")
    chunks = rc.split_documents(docs)
    log("Step 3/4  loading embedding model (first run downloads ~90 MB)")
    emb, label = rc.build_embeddings()
    log(f"Step 4/4  indexing {len(chunks)} chunks into ChromaDB ({label})")
    vdb = rc.build_chroma(chunks, emb, label)
    retriever = rc.HybridRetriever(vdb, chunks)

    files = {p.name for p in data_dir.glob("*")}
    if SAMPLE_FILES <= files:            # the dev questions only make sense for the sample knowledge base
        ins = [retriever.vector_search(i["question"], 1)[0][1] for i in DEV_SET if i["source"]]
        oos = [retriever.vector_search(i["question"], 1)[0][1] for i in DEV_SET if not i["source"]]
        lo, hi = float(np.percentile(ins, 10)), float(max(oos))
        min_score = hi + 0.35 * (lo - hi) if lo > hi else lo * 0.9
        how = "calibrated on dev questions"
    else:
        min_score = 0.08 if label == "tfidf-offline" else 0.25
        how = "default for custom documents - adjust in the sidebar"
    min_score = round(float(min_score), 3)
    rc._persist_meta(rc.CONFIG["persist_dir"], min_score=min_score)
    log(f"Out-of-scope threshold = {min_score} ({how})")
    return {"documents": len(docs), "chunks": len(chunks), "embeddings": label, "min_score": min_score}


if __name__ == "__main__":
    print(build_index())
