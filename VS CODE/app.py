"""NovaBot - Streamlit front-end for the RAG student-support chatbot.

Run:  streamlit run app.py
"""
import html
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import evaluation
import pipeline
import rag_core as rc
from eval_data import TEST_SET

st.set_page_config(page_title="NovaBot · NovaTech Student Assistant", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; max-width: 1100px;}
.hero {background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#EC4899 120%); color:#fff; border-radius:18px;
       padding:22px 28px; margin-bottom:14px; box-shadow:0 8px 24px rgba(79,70,229,.25);}
.hero h1 {margin:0; font-size:1.75rem; font-weight:750; color:#fff;}
.hero p {margin:4px 0 0; opacity:.92; font-size:.97rem; color:#fff;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem; font-weight:600; margin:2px 4px 2px 0;}
.pill.ok {background:#DCFCE7; color:#166534;} .pill.warn {background:#FEF3C7; color:#92400E;}
.pill.info {background:#E0E7FF; color:#3730A3;} .pill.bad {background:#FEE2E2; color:#991B1B;}
.src {border:1px solid rgba(120,120,160,.25); border-left:4px solid #4F46E5; border-radius:10px; padding:10px 14px;
      margin:8px 0; background:rgba(79,70,229,.04); font-size:.88rem;}
.src .meta {font-size:.76rem; opacity:.75; margin-bottom:4px;}
.src mark {background:#FEF08A; padding:0 2px; border-radius:3px;}
.metric-card {border:1px solid rgba(120,120,160,.25); border-radius:14px; padding:14px 16px; background:rgba(79,70,229,.04);}
.metric-card .v {font-size:1.7rem; font-weight:750; color:#4F46E5;} .metric-card .l {font-size:.8rem; opacity:.75;}
div[data-testid="stSidebar"] .stButton button {width:100%;}
</style>
""", unsafe_allow_html=True)

SAMPLES = [
    ("📅", "What is the minimum attendance required to write the end-semester exam?"),
    ("💰", "What is the late fee for delayed fee payment?"),
    ("🏠", "What time does the hostel gate close for first-year students?"),
    ("🎓", "Can I get a 100% tuition waiver on merit?"),
    ("💼", "If I accept a Rs. 5 LPA offer, can I still sit for a company offering Rs. 8 LPA?"),
    ("📚", "How many books can an undergraduate student borrow?"),
]
ICONS = {"pdf": "📕", "csv": "📊", "txt": "📄", "md": "📝"}
FEEDBACK_LOG = Path("logs/feedback.jsonl")


# ----------------------------------------------------------------------------- state & resources
@st.cache_resource(show_spinner="Starting NovaBot ... (first run builds the knowledge-base index)")
def load_bot():
    if not pipeline.index_exists():
        pipeline.build_index(log=lambda m: None)
    return rc.RAGChatbot.from_persisted()


def init_state(bot):
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("history", [])
    ss.setdefault("pending", None)
    ss.setdefault("mode", "hybrid")
    ss.setdefault("top_k", 4)
    ss.setdefault("min_score", float(bot.min_score))
    ss.setdefault("show_sources", True)
    ss.setdefault("show_debug", False)


def highlight(text, query):
    words = {w for w in rc.content_words(query) if len(w) > 3}
    out = html.escape(text)
    for w in sorted(words, key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(w)}\w*)", r"<mark>\1</mark>", out)
    return out


def save_feedback(idx):
    val = st.session_state.get(f"fb_{idx}")
    msg = st.session_state.messages[idx]
    FEEDBACK_LOG.parent.mkdir(exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "helpful": bool(val),
                            "question": msg["question"], "answer": msg["content"]}) + "\n")


def transcript():
    lines = ["# NovaBot conversation", f"_Exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
    for m in st.session_state.messages:
        lines.append(f"**{'You' if m['role'] == 'user' else 'NovaBot'}:** {m['content']}\n")
    return "\n".join(lines)


def stream_words(text):
    for token in re.findall(r"\S+\s*", text):
        yield token
        time.sleep(0.012)


# ----------------------------------------------------------------------------- UI pieces
def hero(bot):
    offline = bot.llm_label == "offline-extractive"
    st.markdown(f"""
<div class="hero"><h1>🎓 NovaBot — NovaTech Student Assistant</h1>
<p>Ask about attendance, fees, scholarships, hostel, placements or the library. Answers come only from official
documents and show their sources.</p></div>""", unsafe_allow_html=True)
    cls = "warn" if offline else "ok"
    st.markdown(
        f"<span class='pill {cls}'>LLM: {html.escape(bot.llm_label)}</span>"
        f"<span class='pill info'>Embeddings: {html.escape(bot.embedding_label)}</span>"
        f"<span class='pill info'>{len(bot.retriever.chunks)} chunks indexed</span>"
        f"<span class='pill info'>Retrieval: {st.session_state.mode}</span>", unsafe_allow_html=True)
    if offline:
        st.warning("**Offline demo mode** — no LLM API key found, so answers are extracted from the documents "
                   "instead of written by a language model. Add `GROQ_API_KEY` (free) to the `.env` file and restart "
                   "for fluent answers.", icon="⚠️")


def render_sources(msg):
    if msg.get("refused") or not msg.get("sources"):
        return
    with st.expander(f"📎 Sources ({len(msg['sources'])})"):
        for s, ctx in zip(msg["sources"], msg["contexts"]):
            ftype = (s["file"] or "").rsplit(".", 1)[-1]
            page = f" · page {s['page'] + 1}" if s["page"] is not None else ""
            st.markdown(
                f"<div class='src'><div class='meta'>[{s['id']}] {ICONS.get(ftype, '📄')} <b>{html.escape(s['file'] or '')}</b>"
                f"{page} · <code>{html.escape(s['chunk_id'] or '')}</code></div>"
                f"{highlight(ctx[:420], msg['question'])}{'…' if len(ctx) > 420 else ''}</div>", unsafe_allow_html=True)


def render_message(i, msg):
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return
        badge = "bad" if msg["refused"] else "ok"
        label = "no answer in documents" if msg["refused"] else "grounded answer"
        st.markdown(f"<span class='pill {badge}'>{label}</span><span class='pill info'>⏱ {msg['latency_s']}s</span>"
                    f"<span class='pill info'>🎯 similarity {msg['top_score']}</span>", unsafe_allow_html=True)
        if st.session_state.show_debug and msg["standalone_question"] != msg["question"]:
            st.caption(f"Rewritten for retrieval: *{msg['standalone_question']}*")
        if st.session_state.show_sources:
            render_sources(msg)
        st.feedback("thumbs", key=f"fb_{i}", on_change=save_feedback, args=(i,))


def answer_question(bot, question):
    ss = st.session_state
    ss.messages.append({"role": "user", "content": question})
    bot.history, bot.min_score = list(ss.history), ss.min_score
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching the documents ..."):
            r = bot.ask(question, mode=ss.mode, k=ss.top_k)
        st.write_stream(stream_words(r["answer"]))
    ss.history = list(bot.history)
    ss.messages.append({"role": "assistant", "content": r["answer"], "refused": r["refused"], "top_score": r["top_score"],
                        "latency_s": r["latency_s"], "sources": r["sources"], "contexts": r["contexts"],
                        "question": question, "standalone_question": r["standalone_question"]})
    st.rerun()


# ----------------------------------------------------------------------------- pages
def page_chat(bot):
    ss = st.session_state
    hero(bot)
    if not ss.messages:
        st.markdown("##### 👋 Try one of these")
        cols = st.columns(3)
        for i, (icon, q) in enumerate(SAMPLES):
            if cols[i % 3].button(f"{icon}  {q}", key=f"sample_{i}", width="stretch"):
                ss.pending = q
                st.rerun()
    for i, m in enumerate(ss.messages):
        render_message(i, m)
    question = ss.pending or st.chat_input("Ask about attendance, fees, hostel, placements, library ...")
    ss.pending = None
    if question:
        answer_question(bot, question)


def page_kb(bot):
    st.header("📚 Knowledge base")
    chunks = bot.retriever.chunks
    df = pd.DataFrame([{"file": c.metadata["file"], "type": c.metadata["file_type"], "chunk_id": c.metadata["chunk_id"],
                        "section": c.metadata.get("section", "—"), "tokens": rc.approx_tokens(c.page_content),
                        "text": c.page_content} for c in chunks])
    c1, c2, c3, c4 = st.columns(4)
    for col, (v, l) in zip((c1, c2, c3, c4), [(df.file.nunique(), "documents"), (len(df), "chunks"),
                                              (int(df.tokens.mean()), "avg tokens / chunk"), (df.type.nunique(), "file formats")]):
        col.markdown(f"<div class='metric-card'><div class='v'>{v}</div><div class='l'>{l}</div></div>", unsafe_allow_html=True)
    st.markdown("#### Files")
    files = df.groupby(["file", "type"]).agg(chunks=("chunk_id", "size"), avg_tokens=("tokens", "mean")).round(0).reset_index()
    st.dataframe(files, hide_index=True, width="stretch")

    st.markdown("#### 🔍 Retrieval playground")
    st.caption("See exactly what the retriever returns for any query — vector similarity vs keyword (BM25) vs the fused hybrid result.")
    q = st.text_input("Query", "How much is the late fee?")
    k = st.slider("Results", 2, 8, 4, key="pg_k")
    if q:
        vec = bot.retriever.vector_search(q, k)
        t1, t2, t3 = st.tabs(["🧭 Vector (semantic)", "🔤 BM25 (keyword)", "🧬 Hybrid (RRF)"])
        with t1:
            st.dataframe(pd.DataFrame([{"rank": i + 1, "similarity": round(s, 3), "chunk": d.metadata["chunk_id"],
                                        "text": d.page_content[:200]} for i, (d, s) in enumerate(vec)]), hide_index=True, width="stretch")
        with t2:
            bm = bot.retriever.bm25_search(q, k)
            st.dataframe(pd.DataFrame([{"rank": i + 1, "bm25": round(s, 2), "chunk": d.metadata["chunk_id"],
                                        "text": d.page_content[:200]} for i, (d, s) in enumerate(bm)]), hide_index=True, width="stretch")
        with t3:
            docs, top = bot.retriever.retrieve(q, k=k, mode="hybrid")
            st.dataframe(pd.DataFrame([{"rank": i + 1, "chunk": d.metadata["chunk_id"], "text": d.page_content[:200]}
                                       for i, d in enumerate(docs)]), hide_index=True, width="stretch")
            verdict = "✅ above threshold — the bot will answer" if top >= st.session_state.min_score else "⛔ below threshold — the bot will refuse"
            st.caption(f"Top similarity {top:.3f} vs threshold {st.session_state.min_score:.3f}: {verdict}")

    st.markdown("#### 🧩 Chunk explorer")
    pick = st.selectbox("Document", sorted(df.file.unique()))
    st.dataframe(df[df.file == pick][["chunk_id", "section", "tokens", "text"]], hide_index=True, width="stretch")


def page_eval(bot):
    st.header("📊 Evaluation")
    st.write(f"Runs the **{len(TEST_SET)} held-out test questions** through the live chatbot: retrieval quality "
             "(Hit@k, MRR) for all three retrieval modes, then answer correctness, faithfulness and refusal behaviour.")
    if bot.llm_label == "offline-extractive":
        st.info("You are in offline demo mode, so answer-side scores reflect the simple extractive fall-back. "
                "Add an API key for production-quality numbers.", icon="ℹ️")
    if st.button("▶ Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Evaluating ...")
        bot.min_score = st.session_state.min_score
        ret, df, summary = evaluation.run_all(bot, progress=lambda p: bar.progress(p, text="Evaluating answers ..."))
        bar.empty()
        st.session_state.eval = (ret, df, summary)
    if "eval" in st.session_state:
        ret, df, summary = st.session_state.eval
        cols = st.columns(len(summary))
        for col, (k, v) in zip(cols, summary.items()):
            col.markdown(f"<div class='metric-card'><div class='v'>{v}</div><div class='l'>{k}</div></div>", unsafe_allow_html=True)
        st.markdown("#### Retrieval quality by mode")
        st.bar_chart(ret[["hit@1", "hit@3", "mrr", "keyword_recall"]])
        st.dataframe(ret, width="stretch")
        st.markdown("#### Per-question results")
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("⬇ Download results (CSV)", df.to_csv(index=False), "eval_results.csv", "text/csv")


def page_about():
    st.header("🏗️ How NovaBot works")
    img = Path("assets/architecture.png")
    if img.exists():
        st.image(str(img), width="stretch")
    st.markdown("""
**Indexing (once):** PDF / CSV / TXT files → structure-aware chunks with context headers → embeddings → ChromaDB.

**Every question:**
1. *Query rewriting* turns follow-ups such as "and for postgraduates?" into a standalone question.
2. *Hybrid retrieval* merges semantic (vector) and keyword (BM25) rankings with Reciprocal Rank Fusion.
3. *Out-of-scope guard* refuses if even the best chunk is not similar enough.
4. *Grounded prompt* tells the LLM to use only the numbered context, cite `[n]`, and refuse otherwise.
5. The answer is shown with its **sources** so students can verify it.

**Use your own documents:** upload PDF / CSV / TXT files in the sidebar and press **Re-index**.
""")
    st.caption("All institute names, figures and contacts in the sample knowledge base are fictional.")


# ----------------------------------------------------------------------------- sidebar + router
def sidebar(bot):
    ss = st.session_state
    with st.sidebar:
        st.markdown("## 🎓 NovaBot")
        page = st.radio("Navigate", ["💬 Chat", "📚 Knowledge base", "📊 Evaluation", "🏗️ About"], label_visibility="collapsed")
        st.divider()
        st.markdown("**Settings**")
        ss.mode = st.selectbox("Retrieval mode", ["hybrid", "similarity", "mmr"], index=["hybrid", "similarity", "mmr"].index(ss.mode),
                               help="hybrid = semantic + keyword search fused with RRF")
        ss.top_k = st.slider("Chunks sent to the LLM (top-k)", 2, 8, ss.top_k)
        ss.min_score = st.slider("Out-of-scope threshold", 0.0, 0.8, float(ss.min_score), 0.01,
                                 help="Questions whose best chunk scores below this are refused. Higher = stricter.")
        ss.show_sources = st.toggle("Show sources", ss.show_sources)
        ss.show_debug = st.toggle("Show query rewriting", ss.show_debug)
        st.divider()
        if st.button("🗑 Clear conversation"):
            ss.messages, ss.history = [], []
            st.rerun()
        if ss.messages:
            st.download_button("⬇ Download transcript", transcript(), "novabot_conversation.md", "text/markdown")
        with st.expander("📤 Add your own documents"):
            up = st.file_uploader("PDF, CSV or TXT", type=["pdf", "csv", "txt", "md"], accept_multiple_files=True)
            if up and st.button("Save & re-index", type="primary"):
                for f in up:
                    Path("data").joinpath(f.name).write_bytes(f.getvalue())
                with st.status("Re-indexing ...", expanded=True) as status:
                    info = pipeline.build_index(log=st.write)
                    status.update(label=f"Done: {info['chunks']} chunks", state="complete")
                st.cache_resource.clear()
                ss.messages, ss.history = [], []
                ss.pop("min_score", None)
                st.rerun()
        st.caption("Answers are generated from indexed documents only. Always verify important decisions with the Student Help Desk.")
    return page


def main():
    bot = load_bot()
    init_state(bot)
    page = sidebar(bot)
    if page.endswith("Chat"):
        page_chat(bot)
    elif page.endswith("Knowledge base"):
        page_kb(bot)
    elif page.endswith("Evaluation"):
        page_eval(bot)
    else:
        page_about()


main()
