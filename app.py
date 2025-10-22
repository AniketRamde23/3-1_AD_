import streamlit as st
import PyPDF2
import os
from io import BytesIO
# Import necessary libraries for other formats
try:
    from docx import Document
    # from pptx import Presentation  # You'll need to install this: pip install python-pptx
except ImportError:
    st.warning("Install 'python-docx' for DOCX support. Run: pip install python-docx")
    # st.warning("Install 'python-pptx' for PPTX support. Run: pip install python-pptx")


# --- (1) Project Title and Configuration ---
st.set_page_config(page_title="SumzUp: Intelligent Document Summarizer")


# ----------------- Theme Toggle Logic (Kept as is) -----------------
ms = st.session_state
if "themes" not in ms: 
    ms.themes = {"current_theme": "light",
                 "refreshed": True,
                 # ... (Rest of theme definition) ...
                "light": {"theme.base": "dark",
                          "theme.backgroundColor": "#FFFFFF",
                          "theme.primaryColor": "#6200EE",
                          "theme.secondaryBackgroundColor": "#F5F5F5",
                          "theme.textColor": "000000",
                          "button_face": "🌜"},

                "dark":  {"theme.base": "light",
                          "theme.backgroundColor": "#121212",
                          "theme.primaryColor": "#BB86FC",
                          "theme.secondaryBackgroundColor": "#1F1B24",
                          "theme.textColor": "#E0E0E0",
                          "button_face": "🌞"},
                          }


def ChangeTheme():
    previous_theme = ms.themes["current_theme"]
    tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
    for vkey, vval in tdict.items(): 
        if vkey.startswith("theme"): st._config.set_option(vkey, vval)

    ms.themes["refreshed"] = False
    if previous_theme == "dark": ms.themes["current_theme"] = "light"
    elif previous_theme == "light": ms.themes["current_theme"] = "dark"

btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
st.button(btn_face, on_click=ChangeTheme)

if ms.themes["refreshed"] == False:
    ms.themes["refreshed"] = True
    st.rerun()
# ----------------- End Theme Toggle Logic -----------------


# --- (2) Text Extraction Functions (Consolidated) ---

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file using PyPDF2."""
    pdf_bytes = pdf_file.read()
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
             text += page_text + "\n"
    return text

def extract_text_from_docx(docx_file):
    """Extract text from an uploaded DOCX file using python-docx."""
    document = Document(docx_file)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(txt_file):
    """Extract text from an uploaded TXT file."""
    # Streamlit UploadedFile object is file-like, but we decode content
    stringio = BytesIO(txt_file.getvalue())
    return stringio.read().decode("utf-8")

def extract_text_from_pptx(pptx_file):
    """Placeholder for PPTX extraction logic."""
    # To implement this, you need to install python-pptx: pip install python-pptx
    # from pptx import Presentation 
    # prs = Presentation(pptx_file)
    # text = ""
    # for slide in prs.slides:
    #     for shape in slide.shapes:
    #         if hasattr(shape, "text"):
    #             text += shape.text + "\n"
    # return text
    return "PPTX Extraction Logic Not Yet Implemented."


# --- (3) Summarization Placeholder Functions ---
# IMPORTANT: REPLACE THESE with your actual ML (TextRank) and DL (BART/T5) implementation
def generate_extractive_summary(text):
    """Placeholder for TextRank/LexRank (ML) implementation."""
    if len(text) < 200:
        return "Not enough content for a meaningful extractive summary. (Needs > 200 chars)"
    # In the real code, this would call your TextRank logic (e.g., using sumy)
    return (f"**Extractive Summary (ML):** This summary was generated using TextRank, selecting the most important sentences.\n\n"
            f"1. The project focuses on creating a hybrid summarization tool called SumzUp.\n"
            f"2. It uses both Machine Learning (TextRank) and Deep Learning (BART/T5) methodologies.\n"
            f"3. The system supports multiple file formats like PDF, DOCX, and TXT for versatility. [PLACEHOLDER OUTPUT]")

def generate_abstractive_summary(text):
    """Placeholder for BART/T5 (DL) implementation."""
    if len(text) < 200:
        return "Not enough content for a meaningful abstractive summary. (Needs > 200 chars)"
    # In the real code, this would call your Hugging Face/BART/T5 logic
    return (f"**Abstractive Summary (DL):** SumzUp is an intelligent application developed to automatically create concise, human-like summaries from various document types, demonstrating the power of modern transformer models to combat information overload. [PLACEHOLDER OUTPUT]")


# --- (4) Sidebar and Main Interface ---
st.title('SumzUp: Intelligent Document Summarizer')

with st.sidebar:
    st.header("Project Controls")
    
    # User selects the summarization mode
    summarization_mode = st.radio(
        "Select Summarization Mode:",
        ('Extractive (ML/TextRank)', 'Abstractive (DL/BART/T5)')
    )
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Document (PDF, DOCX, TXT, PPTX)", 
        type=['pdf', 'docx', 'txt', 'pptx']
    )
    
    # Process Button
    process_button = st.button("Generate Summary", type="primary")

    # This is the section containing your team's details
    st.markdown('''
        ---
        Developed by Team **SumzUp** *Moottha Suraj, Mora Vamshi Reddy, Patlolla Shruthika, Aniket Ramde*
        ''', unsafe_allow_html=True)


# --- (5) Main Application Logic ---
if process_button and uploaded_file is not None:
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    raw_text = ""
    
    # 1. Text Extraction
    with st.spinner(f"Extracting text from {uploaded_file.name}..."):
        try:
            # Reset file pointer to beginning for reliable reading across formats
            uploaded_file.seek(0)
            
            if file_extension == 'pdf':
                raw_text = extract_text_from_pdf(uploaded_file)
            elif file_extension == 'docx':
                raw_text = extract_text_from_docx(uploaded_file)
            elif file_extension == 'txt':
                raw_text = extract_text_from_txt(uploaded_file)
            elif file_extension == 'pptx':
                raw_text = extract_text_from_pptx(uploaded_file)
            
            
        except Exception as e:
            st.error(f"An error occurred during text extraction: {e}")
            st.stop()
            
    # Check if extraction was successful
    if not raw_text or len(raw_text) < 50:
        st.error("Could not extract enough text (minimum 50 characters required). Please try a different document or format.")
    else:
        st.subheader("Summarization Result")
        # 2. Summarization
        with st.spinner(f"Generating summary using **{summarization_mode}**..."):
            
            if summarization_mode == 'Extractive (ML/TextRank)':
                final_summary = generate_extractive_summary(raw_text)
            else:
                final_summary = generate_abstractive_summary(raw_text)
                
            # 3. Display Output
            st.success("Summary Generated!")
            st.markdown(final_summary)
            
            # Optional: Show the first few hundred characters of the source text
            with st.expander("View Extracted Source Text (First 500 characters)"):
                 st.text(raw_text[:500] + "...")

elif process_button and uploaded_file is None:
    st.error("Please upload a file before clicking 'Generate Summary'.")