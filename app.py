import streamlit as st
from summarizer_backend import TextSummarizer

# --- Page Configuration ---
st.set_page_config(page_title="AI Text Summarizer", page_icon="📄", layout="wide")

# --- Title and Header ---
st.title("📄 AI-Powered Hybrid Text Summarizer")
st.markdown("""
    A mini-project demonstrating advanced NLP techniques. 
    This application combines **Extractive** and **Abstractive** summarization 
    to provide comprehensive text analytics.
""")

# --- Load the Model (Cached so it only loads once) ---
@st.cache_resource
def load_summarizer():
    return TextSummarizer()

summarizer = load_summarizer()

# --- Input Section ---
st.header("1. Input Text")
user_text = st.text_area(
    "Paste your document, article, or report here:", 
    height=250, 
    placeholder="Artificial Intelligence is transforming various sectors..."
)

# --- Controls Section ---
st.header("2. Configuration")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Extractive Settings")
    num_sentences = st.slider("Number of sentences to extract:", 1, 10, 3)
    st.caption("*Syllabus Topic 1: Feature Extraction (TF-IDF)*")

with col2:
    st.subheader("Abstractive Settings")
    max_length = st.slider("Max summary length (words):", 30, 300, 130)
    st.caption("*Syllabus Topic 2: Neural Models (Transformers)*")

# --- Processing and Output Section ---
st.markdown("---")

if st.button("🚀 Generate Summaries", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text to summarize!")
    else:
        with st.spinner("Processing text through AI models..."):
            # Generate both summaries
            ext_summary = summarizer.extractive_summary(user_text, num_sentences=num_sentences)
            abs_summary = summarizer.abstractive_summary(user_text, max_length=max_length)

        # Display Results in Tabs
        st.header("3. Summarization Results")
        tab1, tab2 = st.tabs(["🤖 Abstractive Summary (Transformer/T5)", "📊 Extractive Summary (TF-IDF)"])

        with tab1:
            st.info(abs_summary)
            st.markdown("**How it works:** Uses a Transformer Neural Network (T5) to read the text and generate brand new sentences that capture the core meaning.")
            st.caption("Aligns with Syllabus Topic 2: Modelling Techniques for NLP (Transformers)")

        with tab2:
            st.success(ext_summary)
            st.markdown("**How it works:** Uses TF-IDF Feature Extraction to mathematically score and select the most important existing sentences from the original text.")
            st.caption("Aligns with Syllabus Topic 1: NLP Text Analytics (Feature Extraction)")

# --- Footer ---
st.markdown("---")
st.caption("Mini Project for 'Applications in Artificial Intelligence' | Built with Python, Scikit-Learn, and HuggingFace Transformers.")