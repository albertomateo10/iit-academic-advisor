import re
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.errors import GraphRecursionError
import markdown
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from agents.graph import advisor_agent

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IIT Academic Advisor",
    page_icon="🦅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --iit-red:      #C00000;
    --iit-red-dark: #8B0000;
    --radius:       16px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Remove default block padding */
.block-container {
    padding-top: 0.3rem !important;
    padding-bottom: 5rem !important;
    max-width: 780px !important;
}

/* ── Header ── */
.iit-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding-bottom: 0.9rem;
    border-bottom: 2px solid var(--iit-red);
    margin-bottom: 1.2rem;
}
.iit-header-icon { font-size: 2.4rem; line-height: 1; margin-bottom: 0.35rem; }
.iit-header-text h1 {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--iit-red);
    line-height: 1.2;
}
.iit-header-text p {
    margin: 0;
    font-size: 1.1rem;
    opacity: 0.7;
}

/* Dark mode header */
[data-theme="dark"] .iit-header-text h1 { color: #e85c5c; }
[data-theme="dark"] .iit-header         { border-bottom-color: var(--iit-red-dark); }

/* ── Message rows ── */
.msg-row {
    display: flex;
    width: 100%;
    margin-bottom: 0.6rem;
    animation: fadeUp 0.2s ease;
}
.msg-row.user      { justify-content: flex-start; }
.msg-row.assistant { justify-content: flex-end; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Avatar ── */
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    flex-shrink: 0;
    margin-top: 3px;
}
.avatar.user-av { background: #555; color: #fff; margin-right: 8px; }
.avatar.bot-av  { background: var(--iit-red); color: #fff; margin-left: 8px; }

/* ── Bubble ── */
.bubble {
    max-width: 70%;
    padding: 0.6rem 0.95rem;
    border-radius: var(--radius);
    font-size: 0.88rem;
    line-height: 1.55;
    word-wrap: break-word;
}

.bubble p { 
    margin-top: 0; 
    margin-bottom: 0.6rem; 
}
.bubble h1, .bubble h2, .bubble h3 { 
    margin-top: 0.8rem; 
    margin-bottom: 0.4rem; 
    line-height: 1.2;
}
.bubble h1 { font-size: 1.25rem; }
.bubble h2 { font-size: 1.15rem; }
.bubble h3 { font-size: 1.05rem; }
.bubble ul, .bubble ol { 
    margin-top: 0; 
    margin-bottom: 0.6rem; 
    padding-left: 1.2rem; 
}
.bubble li { 
    margin-bottom: 0.2rem; 
}
.bubble > :last-child { 
    margin-bottom: 0; 
}

/* Light mode */
.bubble.user-b {
    background: #dce8f5;
    color: #1a1a1a;
    border-bottom-left-radius: 4px;
}
.bubble.bot-b {
    background: var(--iit-red);
    color: #fff;
    border-bottom-right-radius: 4px;
}

/* Dark mode */
[data-theme="dark"] .bubble.user-b {
    background: #2e2e2e;
    color: #e8e8e8;
}
[data-theme="dark"] .bubble.bot-b {
    background: var(--iit-red-dark);
    color: #fff;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible,
[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] *:focus-visible {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] > div {
    border-radius: 24px !important;
    border: none !important;
    box-shadow: none !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within > div {
    box-shadow: 0 0 0 2px var(--iit-red) !important;
}
[data-testid="stChatInput"] button {
    background: var(--iit-red) !important;
    border-radius: 50% !important;
    color: #fff !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: var(--iit-red) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "completed_courses": [],
        "current_gpa": 4.0,
        "awaiting_prereq_confirmation": False,
        "summary": "",
    }

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="iit-header">
    <div class="iit-header-text">
        <h1>🦅 Academic Advisor &mdash; Illinois Institute of Technology</h1>
        <p>Ask me about courses, prerequisites, policies or your degree roadmap.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Collapse multiple blank lines into one to avoid excessive spacing in bubbles
def fmt(text: str) -> str:
    if not text:
        return ""
    html_text = markdown.markdown(text, extensions=['extra'])
    return html_text

# ─────────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.agent_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user-av"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white"><path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/></svg></div>
            <div class="bubble user-b">{msg.content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
        st.markdown(f"""
        <div class="msg-row assistant">
            <div class="bubble bot-b">{fmt(msg.content)}</div>
            <div class="avatar bot-av">🦅</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask me about IIT courses, prerequisites, or your degree plan…"):

    # Show user bubble immediately
    st.markdown(f"""
    <div class="msg-row user">
        <div class="avatar user-av">🧑‍🎓</div>
        <div class="bubble user-b">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)

    # Update LangGraph state
    st.session_state.agent_state["messages"].append(HumanMessage(content=prompt))

    # Invoke agent with recursion error handling
    with st.spinner("Searching the IIT catalog…"):
        try:
            new_state = advisor_agent.invoke(
                st.session_state.agent_state,
                config={"recursion_limit": 20},
            )
            st.session_state.agent_state = new_state
        except GraphRecursionError:
            fallback = "I'm sorry, I wasn't able to find the information you're looking for in my catalog. Please contact the ITM department at 312.567.5290 or visit the IIT website for help with this question."
            st.session_state.agent_state["messages"].append(
                AIMessage(content=fallback)
            )
            new_state = st.session_state.agent_state

    # Show AI response bubble
    final_message = new_state["messages"][-1]
    if isinstance(final_message, AIMessage) and final_message.content:
        st.markdown(f"""
        <div class="msg-row assistant">
            <div class="bubble bot-b">{fmt(final_message.content)}</div>
            <div class="avatar bot-av">🦅</div>
        </div>
        """, unsafe_allow_html=True)
