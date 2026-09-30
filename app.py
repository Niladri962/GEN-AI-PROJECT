from html import escape

import streamlit as st

from subjectmate.rag import Turn, answer_question
from subjectmate.settings import Settings
from subjectmate.vectorstore import build_index, index_is_current, load_index, read_index_manifest, retrieve

settings = Settings()
st.set_page_config(page_title="SubjectMate", page_icon=":material/menu_book:", layout="centered")

SUGGESTIONS = [
    ("Machine learning", "What is principal component analysis?"),
    ("Algorithms", "Explain binary search tree insertion"),
    ("Databases", "What is a data warehouse?"),
]

# Open book on a rounded blue tile.
LOGO_SVG = """<svg width="{size}" height="{size}" viewBox="0 0 40 40" aria-hidden="true">
<rect width="40" height="40" rx="11" fill="#2563EB"/>
<path d="M9 13.2c3.6-1.4 7.3-1.1 11 1.2 3.7-2.3 7.4-2.6 11-1.2v14.6c-3.6-1.4-7.3-1.1-11 1.2-3.7-2.3-7.4-2.6-11-1.2z"
 fill="none" stroke="#fff" stroke-width="2.2" stroke-linejoin="round"/>
<path d="M20 14.4v14.6" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/>
<path d="M29.5 7.5l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z" fill="#BFDBFE"/>
</svg>"""

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
.stApp { background: #0B0E14; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], footer { display: none; }
section[data-testid="stSidebar"] { background: #0F131A; border-right: 1px solid #1B212C; }
code { font-family: 'JetBrains Mono', ui-monospace, monospace; }

.sm-brand { display: flex; align-items: center; gap: 12px; margin: 2px 0 22px; }
.sm-brand-name { font-size: 17px; font-weight: 700; color: #F1F4F9; letter-spacing: -0.01em; line-height: 1.1; }
.sm-brand-tag { font-size: 12px; color: #8C95A6; margin-top: 2px; }
.sm-label { font-size: 12px; font-weight: 600; color: #8C95A6; margin: 22px 0 8px; }
.sm-history { font-size: 13px; line-height: 1.45; color: #B4BCC9; padding: 8px 10px; border-radius: 8px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sm-history.current { background: #161C26; color: #F1F4F9; }
.sm-status { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #8C95A6; margin-top: 22px; }
.sm-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

.sm-hero { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 48px 0 28px; }
.sm-hero svg { margin-bottom: 20px; }
.sm-title { font-size: 34px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; color: #F1F4F9; margin: 0 0 10px; }
.sm-sub { font-size: 15px; line-height: 1.6; color: #A3ACBA; max-width: 520px; }

[data-testid="stChatInput"] { border: 1px solid #252D3A; border-radius: 14px; background: #121720; }
[data-testid="stChatInput"]:focus-within { border-color: #2563EB; box-shadow: 0 0 0 4px rgba(37,99,235,.16); }
[data-testid="stChatInput"] textarea { font-size: 16px; }
.stButton > button { border-radius: 10px; }
.st-key-suggestions .stButton > button { min-height: 76px; text-align: left; justify-content: flex-start;
  align-items: flex-start; background: #121720; border: 1px solid #1F2733; color: #D5DBE5; }
.st-key-suggestions .stButton > button:hover { border-color: #2563EB; color: #F1F4F9; }
.st-key-suggestions .stButton > button div { justify-content: flex-start; }
.st-key-suggestions .stButton > button p { white-space: normal; text-align: left; font-size: 14px; line-height: 1.45; margin: 0; }
.st-key-suggestions .stButton > button strong { display: block; font-size: 12px; font-weight: 600; color: #7EA6F5; margin-bottom: 4px; }
[data-testid="stChatMessage"] { background: transparent; }

.sm-meta { font-size: 12px; color: #8C95A6; margin: 4px 0 8px; }
.sm-src { border: 1px solid #1F2733; border-radius: 10px; background: #0F131A; padding: 12px 14px; margin-bottom: 8px; }
.sm-src-top { display: flex; align-items: center; gap: 10px; }
.sm-chip { font: 12px 'JetBrains Mono', monospace; padding: 1px 7px; border-radius: 5px;
  background: rgba(37,99,235,.18); color: #A9C7FF; }
.sm-file { font-size: 14px; font-weight: 600; color: #F1F4F9; word-break: break-word; }
.sm-score { margin-left: auto; font-size: 12px; color: #8C95A6; white-space: nowrap; }
.sm-loc { font-size: 12px; color: #8C95A6; margin-top: 4px; }
.sm-bar { height: 3px; border-radius: 2px; background: #1B212C; margin-top: 10px; }
.sm-bar i { display: block; height: 3px; border-radius: 2px; background: #3B82F6; }
</style>
"""

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/school:"


@st.cache_resource(show_spinner="Loading the course index...")
def load_index_cached():
    return load_index(settings)


@st.cache_data(ttl=60, show_spinner=False)
def index_status() -> tuple[bool, int]:
    """(index is current, passage count); checked at most once a minute, not on every click."""
    if not index_is_current(settings):
        return False, 0
    return True, (read_index_manifest(settings.manifest_path) or {}).get("chunk_count", 0)


def sources_html(sources: list[dict[str, object]]) -> str:
    cards = []
    for source in sources:
        score = max(0.0, min(float(source.get("score") or 0), 1.0))
        cards.append(
            '<div class="sm-src"><div class="sm-src-top">'
            f'<span class="sm-chip">{escape(str(source["label"]))}</span>'
            f'<span class="sm-file">{escape(str(source["filename"]))}</span>'
            f'<span class="sm-score">{score:.0%} match</span></div>'
            f'<div class="sm-loc">{escape(str(source["location"]))}</div>'
            f'<div class="sm-bar"><i style="width:{score * 100:.0f}%"></i></div></div>'
        )
    return "".join(cards)


def render_answer_details(message: dict) -> None:
    sources = message.get("sources", [])
    if not sources:
        return
    with st.expander(f"{len(sources)} source{'s' if len(sources) != 1 else ''}", icon=":material/menu_book:"):
        st.markdown(sources_html(sources), unsafe_allow_html=True)


# Each message: {"role", "content", and for answers "asked", "sources", "search_query", "model_used"}
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(STYLE, unsafe_allow_html=True)
current, passage_count = index_status()

with st.sidebar:
    st.markdown(
        f'<div class="sm-brand">{LOGO_SVG.format(size=38)}<div>'
        '<div class="sm-brand-name">SubjectMate</div><div class="sm-brand-tag">Study assistant</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("New question", icon=":material/add:", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    questions = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if questions:
        items = "".join(
            f'<div class="sm-history{" current" if index == len(questions) - 1 else ""}" '
            f'title="{escape(q)}">{escape(q)}</div>'
            for index, q in enumerate(questions)
        )
        st.markdown(f'<div class="sm-label">This chat</div>{items}', unsafe_allow_html=True)

    if current:
        status = f'<span class="sm-dot" style="background:#4ADE80"></span>Ready · {passage_count:,} passages'
    else:
        status = '<span class="sm-dot" style="background:#F59E0B"></span>Index needs updating'
    st.markdown(f'<div class="sm-status">{status}</div>', unsafe_allow_html=True)

    with st.expander("Maintenance"):
        model = settings.llm_model if settings.openai_api_key else (settings.fallback_model or "not set")
        st.caption(f"Model: `{model}`" + (f" · fallback `{settings.fallback_model}`" if settings.openai_api_key and settings.fallback_model else ""))
        st.caption("Embeds files added to `all data/`; rebuilds if files changed or were removed.")
        if st.button("Build / update index", use_container_width=True):
            try:
                with st.spinner("Extracting course files and embedding them into Qdrant..."):
                    built = build_index(settings)
                    load_index_cached.clear()
                    index_status.clear()
                st.success(f"Index holds {built['chunk_count']:,} passages.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

if not st.session_state.messages:
    st.markdown(
        f'<div class="sm-hero">{LOGO_SVG.format(size=56)}'
        '<div class="sm-title">What would you like to learn today?</div>'
        '<div class="sm-sub">Ask anything from your courses. Answers come from your lecture notes, '
        "slides and textbooks, with the source for every point.</div></div>",
        unsafe_allow_html=True,
    )

if not current:
    st.info(
        "The course index is being updated. Make sure Qdrant is running at "
        f"`{settings.qdrant_url}`; if nothing is updating, use **Maintenance → Build / update index**."
    )
elif not settings.llm_configured:
    st.warning("Set `OPENAI_API_KEY` (or a fallback model) in `.env` to enable answers.")

if not st.session_state.messages and current:
    with st.container(key="suggestions"):
        for column, (topic, suggestion) in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
            if column.button(f"**{topic}** {suggestion}", key=f"suggest-{topic}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("search_query") and message["search_query"] != message.get("asked"):
            st.markdown(
                f'<div class="sm-meta">Searched for: {escape(message["search_query"])}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_answer_details(message)

typed = st.chat_input(
    "Ask a question, or a follow-up…",
    disabled=not current or not settings.llm_configured,
)
question = typed or st.session_state.pop("pending_question", None)
if question and question.strip() and current and settings.llm_configured:
    history = [
        Turn(user["content"], assistant["content"])
        for user, assistant in zip(st.session_state.messages[0::2], st.session_state.messages[1::2])
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            with st.spinner("Looking through your course materials…"):
                index = load_index_cached()
                result = answer_question(question, lambda query: retrieve(query, index, settings), settings, history)
            message = {
                "role": "assistant",
                "content": result.answer,
                "asked": question,
                "sources": result.sources,
                "search_query": result.search_query,
                "model_used": result.model_used,
            }
        except Exception as exc:
            message = {"role": "assistant", "content": f"Unable to answer this question: {exc}", "sources": []}
        # Keep user/assistant turns paired so the history stays aligned.
        st.session_state.messages.append(message)
    st.rerun()
