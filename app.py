import streamlit as st
import os
from rag_app.ingest import build_index
from rag_app.query import ask, is_index_stale
from rag_app.config import DATA_DIR, INDEX_DIR
 
st.set_page_config(
    page_title="DocuMind",
    page_icon=None,
    layout="wide"
)
 
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Georgia', serif;
            background-color: #ffffff;
            color: #1a1a1a;
        }
 
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
 
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
 
        .page-title {
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #0f0f0f;
            margin: 0;
        }
 
        .page-subtitle {
            font-size: 0.85rem;
            color: #888888;
            margin-top: 0.3rem;
            font-style: italic;
        }
 
        .section-label {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #888888;
            margin-bottom: 0.8rem;
        }
 
        .left-panel {
            border-right: 1px solid #e5e5e5;
            padding-right: 2rem;
            min-height: 80vh;
        }
 
        .stButton > button {
            background-color: #0f0f0f !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 4px !important;
            font-size: 0.8rem !important;
            font-family: 'Georgia', serif !important;
            letter-spacing: 0.04em !important;
            width: 100% !important;
            margin-top: 0.5rem !important;
        }
 
        .stButton > button:hover {
            background-color: #333333 !important;
        }
 
        /* Hide avatars but keep in DOM for CSS targeting */
        [data-testid="chatAvatarIcon-user"],
        [data-testid="chatAvatarIcon-assistant"] {
            display: none !important;
        }

        /* Base chat message — no border, no background */
        [data-testid="stChatMessage"] {
            background: none !important;
            border: none !important;
            padding: 0.4rem 0 !important;
            margin-bottom: 0 !important;
            gap: 0 !important;
        }

        /* User message — right-aligned gray bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            display: flex !important;
            justify-content: flex-end !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div {
            background: #f0f0f0 !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 10px 16px !important;
            max-width: 70% !important;
            font-size: 0.9rem !important;
            line-height: 1.6 !important;
        }

        /* Assistant message — full-width clean text */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-bottom: 1px solid #f4f4f4 !important;
            padding-bottom: 1rem !important;
            margin-bottom: 0.5rem !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div {
            background: none !important;
            padding: 0 !important;
            width: 100% !important;
            font-size: 0.9rem !important;
            line-height: 1.75 !important;
        }

        .stCaption {
            color: #aaaaaa;
            font-size: 0.75rem;
        }
 
        .step-text {
            font-size: 0.82rem;
            color: #555555;
            line-height: 1.8;
        }
 
        hr {
            border: none;
            border-top: 1px solid #e5e5e5;
            margin: 1.2rem 0;
        }
    </style>
""", unsafe_allow_html=True)
 
 
# ── Header ───────────────────────────────────────────────────────
st.markdown("""
    <p class="page-title">DocuMind</p>
    <p class="page-subtitle">Ask questions. Get answers from your documents.</p>
    <hr>
""", unsafe_allow_html=True)
 
 
# ── Stale index detection (once per session) ─────────────────────
if not st.session_state.get("_stale_check_done"):
    st.session_state["_stale_check_done"] = True
    _index_exists = os.path.exists(os.path.join(INDEX_DIR, "embeddings.npy"))
    _pdfs_exist = os.path.exists(DATA_DIR) and any(f.endswith(".pdf") for f in os.listdir(DATA_DIR))

    if _index_exists and is_index_stale():
        if _pdfs_exist:
            with st.spinner("Updating index..."):
                build_index(DATA_DIR)
        else:
            st.toast("Please re-upload your PDFs and click Build Index.", icon="📄")

# ── Two Column Layout ────────────────────────────────────────────
left, right = st.columns([1, 2.5], gap="large")

with left:
    st.markdown('<p class="section-label">Documents</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
 
    if uploaded_files:
        os.makedirs(DATA_DIR, exist_ok=True)
        for file in uploaded_files:
            save_path = os.path.join(DATA_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.read())
        st.success(f"{len(uploaded_files)} file(s) uploaded")
 
    if st.button("Build Index", use_container_width=True):
        if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
            with st.spinner("Processing..."):
                build_index(DATA_DIR)
            st.success("Index ready")
        else:
            st.error("Upload at least one PDF first")
 
    st.markdown("<hr>", unsafe_allow_html=True)
 
    st.markdown('<p class="section-label">How it works</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="step-text">
            1. Upload your PDF documents<br>
            2. Click Build Index<br>
            3. Ask questions on the right
        </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem; color:#bbbbbb;">Built on '
        '<a href="https://github.com/endee-io/endee" style="color:#888888;">Endee</a> '
        'vector database</p>',
        unsafe_allow_html=True
    )
 
with right:
    index_ready = os.path.exists(os.path.join(INDEX_DIR, "embeddings.npy"))
 
    if not index_ready:
        st.markdown(
            '<p style="color:#aaaaaa; font-size:0.88rem; font-style:italic; margin-top:1rem;">'
            'Upload a PDF and build the index to get started.</p>',
            unsafe_allow_html=True
        )
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
 
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.caption("Source: " + ", ".join(msg["sources"]))

        if query := st.chat_input("Ask a question about your documents"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = ask(query)
                st.markdown(result["answer"])
                st.caption("Source: " + ", ".join(result["sources"]))
 
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })
 