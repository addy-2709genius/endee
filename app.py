import streamlit as st
import os
from rag_app.ingest import build_index
from rag_app.query import ask
from rag_app.config import DATA_DIR, INDEX_DIR
 
st.set_page_config(
    page_title="DocuMind",
    page_icon=None,
    layout="wide"
)
 
st.markdown("""
    <style>
        /* ── Force light mode regardless of system/browser theme ── */
        :root {
            color-scheme: light !important;
        }

        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        [data-testid="stMain"],
        [data-testid="stSidebar"],
        .main, .block-container,
        [class*="css"] {
            font-family: 'Georgia', serif !important;
            background-color: #ffffff !important;
            color: #1a1a1a !important;
        }

        /* Override Streamlit CSS variables */
        :root,
        [data-theme="dark"],
        [data-theme="light"] {
            --background-color: #ffffff !important;
            --secondary-background-color: #f7f7f7 !important;
            --text-color: #1a1a1a !important;
            --font: 'Georgia', serif !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            padding-top: 2.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
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

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background-color: #f9f9f9 !important;
            border: 1px solid #efefef !important;
            border-radius: 6px !important;
            padding: 1rem !important;
            margin-bottom: 0.8rem !important;
            color: #1a1a1a !important;
        }

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span {
            color: #1a1a1a !important;
        }

        /* Chat avatar — replace default emoji with styled initials */
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            background-color: #0f0f0f !important;
            border-radius: 50% !important;
        }

        /* Chat input */
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInputTextArea"] {
            background-color: #f7f7f7 !important;
            color: #1a1a1a !important;
            border: 1px solid #e0e0e0 !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: #f7f7f7 !important;
            border: 1px dashed #cccccc !important;
            color: #1a1a1a !important;
        }

        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p {
            color: #555555 !important;
        }

        /* Success / error boxes */
        [data-testid="stAlert"] {
            background-color: #f0faf0 !important;
            color: #1a1a1a !important;
        }

        .stCaption, [data-testid="stCaptionContainer"] {
            color: #aaaaaa !important;
            font-size: 0.75rem !important;
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
 
        USER_AVATAR = "https://api.dicebear.com/9.x/initials/svg?seed=U&backgroundColor=0f0f0f&textColor=ffffff&fontSize=40"
        AI_AVATAR = "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=DocuMind&backgroundColor=e8e8e8"

        for msg in st.session_state.messages:
            avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.caption("Source: " + ", ".join(msg["sources"]))

        if query := st.chat_input("Ask a question about your documents"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(query)

            with st.chat_message("assistant", avatar=AI_AVATAR):
                with st.spinner("Thinking..."):
                    result = ask(query)
                st.markdown(result["answer"])
                st.caption("Source: " + ", ".join(result["sources"]))

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })
 