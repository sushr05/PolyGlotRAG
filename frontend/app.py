import streamlit as st
import os
import sys
import logging
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports from app.core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \'..\')))

# Load environment variables from .env file in the root directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), \'..\", \".env\"))

# Configure logging
logging.basicConfig(level=logging.INFO, format=\"%(asctime)s - %(levelname)s - %(message)s\")

# --- Attempt to import core modules ---
try:
    from app.core.rag_pipeline import query_pipeline, ingest_pdf, services_initialized
    from app.core.translation_service import NLLB_LANG_MAP
    core_loaded = True
except ImportError as e:
    st.error(f"Error importing core modules: {e}. Please ensure all dependencies are installed and paths are correct.")
    core_loaded = False
except Exception as e:
    st.error(f"Error initializing core services: {e}. Check API keys and model availability.")
    core_loaded = False

# --- Streamlit App UI ---
st.set_page_config(page_title="Polyglot PDF Q&A", layout="wide")
st.title("📚 Polyglot PDF Q&A System")
st.markdown("Upload a PDF, select languages, and ask questions about its content.")

# Check if core services are ready
if not core_loaded or not services_initialized:
    st.error("Core services failed to initialize. Please check the logs and ensure API keys (Pinecone, Groq) are set correctly in the .env file and required models are accessible. The application cannot proceed.")
    st.stop()

# --- PDF Upload Section ---
st.sidebar.header("PDF Upload")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_dir = "/tmp/pdf_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)

    try:
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded 	`{uploaded_file.name}`")

        # Ingestion Button
        if st.sidebar.button("Process and Ingest PDF"):
            with st.spinner(f"Processing and ingesting 	`{uploaded_file.name}`	... This may take a while depending on the PDF size and model loading times."):
                try:
                    # Use filename as document_id, sanitize if needed
                    doc_id = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")
                    success = ingest_pdf(temp_pdf_path, document_id=doc_id)
                    if success:
                        st.sidebar.success(f"Successfully ingested 	`{uploaded_file.name}`	.")
                        st.session_state[\'last_ingested_doc_id\'] = doc_id
                    else:
                        st.sidebar.error(f"Failed to ingest 	`{uploaded_file.name}`	. Check logs for details.")
                except Exception as e:
                    st.sidebar.error(f"An error occurred during ingestion: {e}")
                    logging.error(f"Ingestion error for {uploaded_file.name}: {e}", exc_info=True)

    except Exception as e:
        st.sidebar.error(f"Error handling uploaded file: {e}")
        logging.error(f"Error handling upload {uploaded_file.name}: {e}", exc_info=True)
    finally:
        # Clean up the temporary file if it exists
        # if os.path.exists(temp_pdf_path):
        #     os.remove(temp_pdf_path) # Keep it for now for potential debugging
        pass

# Display last ingested document ID if available
if \'last_ingested_doc_id\' in st.session_state:
    st.sidebar.info(f"Last ingested document ID: `{st.session_state[\'last_ingested_doc_id\']}`")

# --- Q&A Section ---
st.header("Ask a Question")

# Get available languages from the map
available_languages = list(NLLB_LANG_MAP.keys())

col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("Select Question Language", options=available_languages, index=available_languages.index("english"))
with col2:
    target_lang = st.selectbox("Select Answer Language", options=available_languages, index=available_languages.index("english"))

question = st.text_input("Enter your question here:")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner(f"Thinking... (Querying in {source_lang}, answering in {target_lang})"):
            try:
                logging.info(f"Received query: 	`{question}`	 (Source: {source_lang}, Target: {target_lang})")
                answer = query_pipeline(question, source_lang, target_lang)

                st.subheader("Answer:")
                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred during the query process: {e}")
                logging.error(f"Query pipeline error for question 	`{question}`	: {e}", exc_info=True)

# Add instructions or notes
st.sidebar.markdown("--- ")
st.sidebar.markdown("**Notes:**")
st.sidebar.markdown("- Ensure API keys for Pinecone and Groq are in the `.env` file.")
st.sidebar.markdown("- Model loading (Translation, Embeddings) can take time on first run.")
st.sidebar.markdown("- Ingestion processes the PDF into the vector store.")
st.sidebar.markdown("- Q&A uses the ingested documents.")

