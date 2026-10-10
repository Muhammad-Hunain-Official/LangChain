
import streamlit as st
from Chatbot import extract_video_id, build_rag_chain, ask_question

st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="🎬",
    layout="centered"
)

# Initialize theme
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Theme toggle button
header_col, theme_col = st.columns([8, 1])

with header_col:
    st.title("YouTube AI Assistant")

with theme_col:
    icon = "☀️" if st.session_state.dark_mode else "🌙"

    if st.button(icon, key="theme_toggle", help="Change theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Theme colors
if st.session_state.dark_mode:
    bg = "#0E1117"
    card = "#1C2333"
    text = "#F0F2F6"
    input_bg = "#262D3D"
    border = "#384256"
else:
    bg = "#FFFFFF"
    card = "#F1F5F9"
    text = "#172033"
    input_bg = "#FFFFFF"
    border = "#D5DCE5"

# Apply theme using CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {text};
    }}

    [data-testid="stHeader"] {{
        background-color: {bg};
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {bg};
    }}

    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    p, label, h1, h2, h3 {{
        color: {text};
    }}

    [data-testid="stChatMessage"] {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 12px;
    }}

    [data-testid="stChatInput"] {{
        background-color: {input_bg};
        border: 1px solid {border};
        border-radius: 12px;
    }}

    input, textarea {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}

    div.stButton > button {{
        border-radius: 10px;
        border: 1px solid {border};
    }}

    [data-testid="stAlert"] {{
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Load a YouTube video and ask questions about its transcript.")
st.divider()


@st.cache_resource(show_spinner=False)
def get_cached_chain(video_id):
    return build_rag_chain(video_id)


# Session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# YouTube video input
video_url = st.text_input(
    "YouTube video URL or ID",
    placeholder="Paste your YouTube video link here"
)

if st.button("Load Video", type="primary", use_container_width=True):
    if not video_url.strip():
        st.warning("Please enter a YouTube video URL first.")
    else:
        try:
            video_id = extract_video_id(video_url)

            with st.spinner("Fetching transcript and building RAG system..."):
                chain = get_cached_chain(video_id)

            st.session_state.rag_chain = chain
            st.session_state.video_id = video_id
            st.session_state.messages = []

            st.success("Video loaded successfully!")

        except Exception as e:
            st.error(f"Could not load video: {e}")


# Chat interface
if st.session_state.rag_chain is not None:

    st.caption(f"Loaded video ID: {st.session_state.video_id}")
    st.subheader("Ask a question")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the video...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer = ask_question(
                        st.session_state.rag_chain,
                        question
                    )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Could not generate answer: {e}")

else:
    st.info("Paste a YouTube link above and click Load Video to begin.")
