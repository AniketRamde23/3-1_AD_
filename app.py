import streamlit as st
import PyPDF2
import os
from io import BytesIO



# --- NLP and Summarization Imports ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
nltk.data.path.append("nltk_data")


# DEEP LEARNING IMPORTS
from transformers import BartForConditionalGeneration, BartTokenizer

# --- Setup and Initialization ---

# ✅ Ensure required NLTK data
import nltk

# Ensure punkt and punkt_tab are available
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# Try to import python-docx
try:
    from docx import Document
except ImportError:
    Document = None


# --- Streamlit Page Setup ---
st.set_page_config(page_title="SumzUp: Intelligent Document Summarizer")

# --- (1) Theme Logic ---
ms = st.session_state
if "themes" not in ms:
    ms.themes = {
        "current_theme": "light",
        "refreshed": True,
        "light": {"theme.base": "light", "theme.backgroundColor": "#FFFFFF",
                  "theme.primaryColor": "#6200EE", "theme.secondaryBackgroundColor": "#F5F5F5",
                  "theme.textColor": "#000000", "button_face": "🌜"},
        "dark": {"theme.base": "dark", "theme.backgroundColor": "#121212",
                 "theme.primaryColor": "#BB86FC", "theme.secondaryBackgroundColor": "#1F1B24",
                 "theme.textColor": "#E0E0E0", "button_face": "🌞"}
    }

def ChangeTheme():
    prev = ms.themes["current_theme"]
    next_theme = "dark" if prev == "light" else "light"
    tdict = ms.themes[next_theme]
    for k, v in tdict.items():
        if k.startswith("theme"):
            st._config.set_option(k, v)
    ms.themes["current_theme"] = next_theme
    ms.themes["refreshed"] = False

btn_face = ms.themes[ms.themes["current_theme"]]["button_face"]
st.button(btn_face, on_click=ChangeTheme)
if not ms.themes["refreshed"]:
    ms.themes["refreshed"] = True
    st.rerun()


# --- (2) File Extraction Functions ---
def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def extract_text_from_docx(docx_file):
    if not Document:
        st.warning("Install 'python-docx' for DOCX support. Run: pip install python-docx")
        return ""
    docx_file.seek(0)
    document = Document(docx_file)
    return "\n".join([p.text for p in document.paragraphs])

def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

def extract_text_from_pptx(pptx_file):
    return "PPTX extraction not implemented yet."


# --- (3) Summarization Core Functions ---

def clean_text(text):
    """Basic cleaning to improve summarization quality."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text

# Lazy-load BART model only once to save memory and startup time
@st.cache_resource
def load_bart_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def generate_extractive_summary(text, sentence_count=5):
    """Generate a fluent extractive summary using Sumy TextRank."""
    text = clean_text(text)
    if len(text) < 200:
        return "Not enough content for a meaningful extractive summary. (Needs > 200 chars)"

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()

        # Dynamic sentence count: limit to 1/5 of total sentences or max 10
        total_sentences = len(list(parser.document.sentences))
        sentence_count = min(sentence_count, max(3, total_sentences // 5))

        summary_sentences = summarizer(parser.document, sentence_count)
        if not summary_sentences:
            return "Summary generation failed — text may be too short or repetitive."

        final_summary = "**Extractive Summary (TextRank):**\n\n"
        for i, s in enumerate(summary_sentences, 1):
            final_summary += f"{i}. {str(s)}\n"
        return final_summary.strip()

    except Exception as e:
        return f"Error during TextRank summarization: {e}"


def generate_abstractive_summary(text, max_len=150, min_len=40):
    """Generate a fluent abstractive summary using BART transformer."""
    text = clean_text(text)
    if len(text) < 200:
        return "Not enough content for a meaningful abstractive summary. (Needs > 200 chars)"

    try:
        tokenizer, model = load_bart_model()

        # Input is truncated to fit BART's max length of 1024 tokens
        inputs = tokenizer(
            [text],
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=min_len,
            max_length=max_len,
            length_penalty=2.0,
            no_repeat_ngram_size=3
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return f"**Abstractive Summary (BART):**\n\n{summary}"

    except Exception as e:
        # Catch errors if Hugging Face model loading fails (e.g., no internet, resource limits)
        return f"Error during abstractive summarization: {e}. Check internet connection or increase system resources."


# --- (4) Streamlit UI ---
st.title("SumzUp: Intelligent Document Summarizer")

with st.sidebar:
    st.header("Controls")
    sentence_count = st.slider("Summary Length (Sentences)", 1, 10, 5)
    mode = st.radio("Summarization Mode", ["Extractive (TextRank)", "Abstractive (BART)"])
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt", "pptx"])
    process = st.button("Generate Summary", type="primary")

    st.markdown("""
    ---
    Developed by Team **SumzUp** *Moottha Suraj, Mora Vamshi Reddy, Patlolla Shruthika, Aniket Ramde*
    """)


# --- (5) Main Processing Logic ---
if process and uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    raw_text = ""
    
    # 1. Extraction
    with st.spinner(f"Extracting text from {uploaded_file.name}..."):
        try:
            uploaded_file.seek(0)
            
            if ext == "pdf":
                raw_text = extract_text_from_pdf(uploaded_file)
            elif ext == "docx":
                raw_text = extract_text_from_docx(uploaded_file)
            elif ext == "txt":
                raw_text = extract_text_from_txt(uploaded_file)
            elif ext == "pptx":
                raw_text = extract_text_from_pptx(uploaded_file)
            else:
                st.error("Unsupported file format.")
                st.stop()
            
        except Exception as e:
            st.error(f"An error occurred during text extraction: {e}")
            st.stop()

    # 2. Summarization
    if not raw_text or len(raw_text) < 50:
        st.error("Not enough text extracted (minimum 50 chars). Please try another document.")
    else:
        st.subheader("Summary Output")
        with st.spinner("Generating summary..."):
            if "Extractive" in mode:
                result = generate_extractive_summary(raw_text, sentence_count)
            else: # Abstractive (BART)
                result = generate_abstractive_summary(raw_text)
            
            st.success("✅ Summary generated successfully!")
            st.markdown(result)

            with st.expander("📄 View Extracted Source Text (first 600 chars)"):
                st.text(raw_text[:600] + "...")

elif process and not uploaded_file:
    st.error("Please upload a file first.")

# Check to remind user about missing libraries (moved to the end of the script flow)
if not Document:
    st.warning("Install 'python-docx' for DOCX support. Run: pip install python-docx")

# You'd add a similar check for python-pptx if needed:
# try: from pptx import Presentation
# except ImportError: st.warning("Install 'python-pptx' for PPTX support. Run: pip install python-pptx")
