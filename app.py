"""Daraz Support Ops Assistant.

Loads the pre-built FAISS index from ./faiss_index (built once by ingest.py).
This app never reads or re-embeds the PDFs; it only embeds the user's question.
"""
import json
from pathlib import Path

import faiss-cpu
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
INDEX_DIR = Path(__file__).parent / "faiss_index"
LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_HISTORY_TURNS = 3  # previous Q&A pairs sent to the LLM for follow-ups

ALL_SECTIONS = "All sections"
SECTIONS = {
    "returns": ("↩️", "Returns"),
    "delivery": ("🚚", "Delivery"),
    "refunds": ("💸", "Refunds"),
    "sellers": ("🏪", "Sellers"),
    "payments": ("💳", "Payments"),
    "customer_support": ("🎧", "Customer Support"),
}

STARTERS = [
    "How many days does a customer have to return an item?",
    "When is a refund issued for a cancelled order?",
    "What should I tell a customer whose parcel is delayed?",
    "Which payment methods can customers use?",
]

SYSTEM_PROMPT = """You are the Daraz Customer Support Operations Assistant. You help Daraz \
support agents answer questions using ONLY the policy excerpts provided in the context.

Rules:
- Base every answer on the provided context. Do not invent policies, timeframes, fees or steps.
- If the context does not contain the answer, say so clearly and suggest the agent escalate \
to a team lead or check the relevant policy owner. Do not guess.
- Be concise and practical: lead with the direct answer, then give steps or conditions if needed.
- When the context gives conditions, exceptions or timeframes, state them exactly.
- Cite the excerpts you used with their numbers, like [1] or [2][3].
- Reply in the same language the agent writes in."""

# ----------------------------------------------------------------------------
# Page + branding
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Daraz Support Assistant",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root { --daraz: #F85606; --daraz-dark: #D94A04; --daraz-tint: #FFF1EA; --ink: #212121; }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { padding-top: 1.5rem; max-width: 820px; }

    /* Brand bar */
    .brand {
        display: flex; align-items: center; gap: 14px;
        background: var(--daraz); color: #fff;
        padding: 16px 20px; border-radius: 14px; margin-bottom: 18px;
    }
    .brand .logo { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.5px; line-height: 1; }
    .brand .tag { font-size: 0.95rem; opacity: 0.95; border-left: 1px solid rgba(255,255,255,.55); padding-left: 14px; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: var(--daraz-tint); border-right: 1px solid #FFD9C7; }
    section[data-testid="stSidebar"] .sb-title { font-weight: 700; color: var(--ink); margin: 0 0 2px 0; }
    section[data-testid="stSidebar"] .sb-note { font-size: 0.85rem; color: #6b6b6b; margin-bottom: 10px; }

    /* Buttons */
    .stButton > button {
        border: 1px solid var(--daraz); color: var(--daraz); background: #fff;
        border-radius: 999px; font-weight: 600; text-align: left;
    }
    .stButton > button:hover { background: var(--daraz); color: #fff; border-color: var(--daraz); }
    .stButton > button:focus-visible { outline: 3px solid #FFB894; }

    /* Chat */
    div[data-testid="stChatMessage"] { border-radius: 14px; padding: 0.75rem 1rem; }
    .chip {
        display: inline-block; background: var(--daraz-tint); color: var(--daraz-dark);
        border: 1px solid #FFD9C7; border-radius: 999px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600;
    }
    .src { font-size: 0.85rem; color: #555; margin-bottom: 0.6rem; }
    .src b { color: var(--ink); }
</style>
<div class="brand">
    <div class="logo">Daraz</div>
    <div class="tag">Support Operations Assistant</div>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Resources (loaded once per server process, never rebuilt from PDFs)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_knowledge_base():
    index_file = INDEX_DIR / "index.faiss"
    meta_file = INDEX_DIR / "metadata.json"
    if not index_file.exists() or not meta_file.exists():
        return None

    index = faiss.read_index(str(index_file))
    with open(meta_file, encoding="utf-8") as f:
        meta_by_id = {m["id"]: m for m in json.load(f)}

    model_name = DEFAULT_EMBED_MODEL
    cfg_file = INDEX_DIR / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            model_name = json.load(f).get("model", DEFAULT_EMBED_MODEL)

    embedder = SentenceTransformer(model_name)  # embeds queries only
    return index, meta_by_id, embedder


@st.cache_resource
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None
    return Groq(api_key=api_key)


kb = load_knowledge_base()
if kb is None:
    st.error(
        "The knowledge base index was not found. Put the `faiss_index` folder "
        "(with `index.faiss` and `metadata.json`) next to `app.py`."
    )
    st.stop()
index, meta_by_id, embedder = kb

client = get_groq_client()
if client is None:
    st.error(
        "Missing `GROQ_API_KEY`. Add it to `.streamlit/secrets.toml` locally, "
        "or to the app's Secrets settings on Streamlit Cloud."
    )
    st.stop()

# Sections shown in the sidebar: the six known ones, plus any extra found in the index.
counts = {}
for m in meta_by_id.values():
    counts[m["department"]] = counts.get(m["department"], 0) + 1
section_keys = list(SECTIONS) + sorted(d for d in counts if d not in SECTIONS)


def section_label(key: str) -> str:
    if key == ALL_SECTIONS:
        return "🔎  All sections"
    icon, name = SECTIONS.get(key, ("📁", key.replace("_", " ").title()))
    return f"{icon}  {name}"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="sb-title">Knowledge base</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sb-note">Choose a section to search only that part of the policies.</p>',
        unsafe_allow_html=True,
    )
    selected = st.radio(
        "Section",
        options=[ALL_SECTIONS] + section_keys,
        format_func=section_label,
        label_visibility="collapsed",
    )
    st.divider()
    top_k = st.slider("Excerpts to use", min_value=2, max_value=10, value=5)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{index.ntotal:,} excerpts indexed")

active_section = None if selected == ALL_SECTIONS else selected


# ----------------------------------------------------------------------------
# Retrieval + generation
# ----------------------------------------------------------------------------
def retrieve(query: str, section: str | None, k: int):
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    # FAISS can't filter by metadata, so fetch a wide candidate pool and filter here.
    fetch_k = index.ntotal if (section and index.ntotal <= 5000) else (k * 50 if section else k)
    fetch_k = min(fetch_k, index.ntotal)
    scores, ids = index.search(q, fetch_k)

    hits = []
    for score, cid in zip(scores[0], ids[0]):
        if cid == -1:
            continue
        m = meta_by_id.get(int(cid))
        if m is None or (section and m["department"] != section):
            continue
        hits.append({**m, "score": float(score)})
        if len(hits) == k:
            break
    return hits


def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        name = SECTIONS.get(h["department"], ("", h["department"]))[1]
        blocks.append(f"[{i}] ({name} | {h['source_file']}, p.{h.get('page', '?')})\n{h['text']}")
    return "\n\n".join(blocks)


def stream_answer(question: str, hits, history):
    context = build_context(hits) if hits else "(No relevant excerpts were found.)"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    )

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=2048,
        reasoning_effort="low",
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def render_sources(hits):
    if not hits:
        return
    with st.expander(f"Sources ({len(hits)})"):
        for i, h in enumerate(hits, start=1):
            name = SECTIONS.get(h["department"], ("", h["department"]))[1]
            st.markdown(
                f'<div class="src"><span class="chip">[{i}] {name}</span> '
                f'<b>{h["source_file"]}</b>, page {h.get("page", "?")} '
                f'(match {h["score"]:.2f})</div>',
                unsafe_allow_html=True,
            )
            st.caption(h["text"][:500] + ("..." if len(h["text"]) > 500 else ""))


# ----------------------------------------------------------------------------
# Chat UI
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

placeholder = (
    "Ask about any Daraz policy..."
    if active_section is None
    else f"Ask about {SECTIONS.get(active_section, ('', active_section))[1]}..."
)
typed = st.chat_input(placeholder)
prompt = typed or st.session_state.pending
st.session_state.pending = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🛍️"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"))

if not st.session_state.messages and not prompt:
    st.markdown("#### How can I help?")
    st.caption(
        "Ask a policy question and I'll answer from the Daraz knowledge base. "
        "Pick a section in the sidebar to narrow the search."
    )
    cols = st.columns(2)
    for i, text in enumerate(STARTERS):
        if cols[i % 2].button(text, key=f"starter_{i}", use_container_width=True):
            st.session_state.pending = text
            st.rerun()

if prompt:
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    history = list(st.session_state.messages)

    # Short follow-ups ("and for sellers?") retrieve better with the previous question attached.
    search_query = prompt
    prev_user = [m["content"] for m in history if m["role"] == "user"]
    if len(prompt.split()) < 6 and prev_user:
        search_query = f"{prev_user[-1]} {prompt}"

    with st.chat_message("assistant", avatar="🛍️"):
        try:
            with st.spinner("Searching policies..."):
                hits = retrieve(search_query, active_section, top_k)
            answer = st.write_stream(stream_answer(prompt, hits, history))
            render_sources(hits)
        except Exception as e:
            hits, answer = [], ""
            st.error(f"Could not get an answer from the model: {e}")

    if answer:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": hits}
        )
