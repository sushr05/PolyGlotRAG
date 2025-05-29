import logging
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Import custom modules
from app.core.translation_service import TranslationService, NLLB_LANG_MAP
from app.core.embedding_service import EmbeddingService
from app.core.vector_store_service import VectorStoreService

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

# --- Configuration ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "polyglot-rag-index")
# Assuming multilingual-e5-large dimension
EMBEDDING_DIMENSION = 1024
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192") # Or "mistral-7b-instruct"
# NLLB model can be resource intensive, consider smaller versions or API if needed
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# --- Initialize Services ---
try:
    translator = TranslationService(model_name=NLLB_MODEL_NAME)
    embedder = EmbeddingService()
    vector_store = VectorStoreService(index_name=PINECONE_INDEX_NAME, embedding_dimension=EMBEDDING_DIMENSION)
    llm = ChatGroq(temperature=0, model_name=GROQ_MODEL_NAME, groq_api_key=os.getenv("GROQ_API_KEY"))
    services_initialized = translator.translator and embedder.model and vector_store.index and llm
except Exception as e:
    logging.error(f"Failed to initialize one or more services: {e}")
    services_initialized = False

# --- RAG Pipeline Logic ---
def format_docs(docs: list[dict]) -> str:
    """Formats retrieved documents into a single string for the LLM context."""
    # Extracts the \'text\' field from the metadata of each retrieved document
    return "\n\n".join(doc.get(\'metadata\', {}).get(\'text\', \'\') for doc in docs if doc)

def get_rag_chain():
    if not services_initialized:
        raise RuntimeError("Services could not be initialized. Cannot create RAG chain.")

    # Define the RAG prompt template
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don\'t know the answer, just say that you don\'t know.
    Keep the answer concise and relevant to the question based *only* on the provided context.

    Context: {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Define the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {
            "context": lambda x: format_docs(x[\'retrieved_docs\']),
            "question": lambda x: x[\'translated_question\']
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def query_pipeline(original_question: str, source_lang: str, target_lang: str, top_k: int = 5):
    """Executes the full multilingual RAG query pipeline.

    Args:
        original_question: The question asked by the user in their language.
        source_lang: The language of the original question (e.g., \'hindi\').
        target_lang: The language the answer should be returned in (e.g., \'hindi\').
        top_k: Number of documents to retrieve.

    Returns:
        The final answer translated into the target language, or an error message.
    """
    if not services_initialized:
        return "Error: Core services are not initialized."

    # 1. Translate query to English (internal processing language)
    logging.info(f"Translating query from {source_lang} to English...")
    translated_question = translator.translate(original_question, source_lang, "english")
    if not translated_question:
        logging.error("Failed to translate the question to English.")
        return f"Error: Could not translate question from {source_lang} to English."
    logging.info(f"Translated question: {translated_question}")

    # 2. Generate embedding for the translated query
    logging.info("Generating embedding for the translated query...")
    # Add \'query: \' prefix as recommended for e5 models
    query_embedding = embedder.get_embeddings([f"query: {translated_question}"])
    if not query_embedding:
        logging.error("Failed to generate embedding for the translated question.")
        return "Error: Could not generate query embedding."
    query_embedding = query_embedding[0] # Get the single embedding vector

    # 3. Retrieve relevant documents from vector store
    logging.info(f"Retrieving top-{top_k} documents from vector store...")
    retrieved_docs = vector_store.query_vectors(query_embedding, top_k=top_k)
    if retrieved_docs is None: # Check for None explicitly, as empty list is valid
        logging.error("Failed to retrieve documents from the vector store.")
        return "Error: Failed to retrieve documents."
    if not retrieved_docs:
        logging.warning("No relevant documents found in the vector store.")
        # Decide how to handle: return specific message or try generation without context?
        # For now, let LLM handle it, context will be empty.
        pass
    logging.info(f"Retrieved {len(retrieved_docs)} documents.")

    # 4. Generate answer using RAG chain
    logging.info("Generating answer using RAG chain...")
    rag_chain = get_rag_chain()
    try:
        english_answer = rag_chain.invoke({
            "retrieved_docs": retrieved_docs,
            "translated_question": translated_question
        })
        logging.info(f"Generated English answer: {english_answer}")
    except Exception as e:
        logging.error(f"Error during RAG chain invocation: {e}")
        return "Error: Failed to generate answer using LLM."

    # 5. Translate answer back to the target language
    logging.info(f"Translating answer from English to {target_lang}...")
    final_answer = translator.translate(english_answer, "english", target_lang)
    if not final_answer:
        logging.error(f"Failed to translate the answer to {target_lang}. Returning English answer.")
        # Fallback: return the English answer if final translation fails
        return f"(Answer in English as translation failed): {english_answer}"

    logging.info(f"Final translated answer: {final_answer}")
    return final_answer

# --- Ingestion Pipeline Logic (Simplified Example) ---
from app.core.pdf_processor import extract_text_from_pdf
from app.core.text_chunker import TextChunker
import uuid

def ingest_pdf(pdf_path: str, document_id: str | None = None):
    """Processes a PDF, chunks it, embeds chunks, and upserts to vector store."""
    if not services_initialized:
        logging.error("Services not initialized. Cannot ingest PDF.")
        return False

    if not document_id:
        document_id = os.path.basename(pdf_path)

    logging.info(f"Starting ingestion for PDF: {pdf_path} with doc_id: {document_id}")

    # 1. Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logging.error(f"Failed to extract text from {pdf_path}")
        return False

    # 2. Chunk text
    chunker = TextChunker() # Use default chunk size/overlap
    chunks = chunker.chunk_text(text)
    if not chunks:
        logging.error(f"Failed to chunk text for {pdf_path}")
        return False
    logging.info(f"Created {len(chunks)} chunks.")

    # 3. Embed chunks
    # Add \'passage: \' prefix for e5 models
    logging.info("Generating embeddings for chunks...")
    chunk_embeddings = embedder.get_embeddings([f"passage: {chunk}" for chunk in chunks])
    if not chunk_embeddings or len(chunk_embeddings) != len(chunks):
        logging.error(f"Failed to generate embeddings for chunks of {pdf_path}")
        return False
    logging.info("Embeddings generated.")

    # 4. Prepare vectors for upsert
    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        vector_id = f"{document_id}_chunk_{i}" # Create a unique ID for each chunk
        metadata = {"text": chunk, "document_id": document_id, "chunk_index": i}
        vectors_to_upsert.append((vector_id, embedding, metadata))

    # 5. Upsert vectors
    logging.info("Upserting vectors to Pinecone...")
    success = vector_store.upsert_vectors(vectors_to_upsert)
    if success:
        logging.info(f"Successfully ingested PDF: {pdf_path}")
    else:
        logging.error(f"Failed to upsert vectors for PDF: {pdf_path}")

    return success

# Example Usage (requires API keys in .env and models downloaded/accessible)
# if __name__ == \'__main__\':
#     if not services_initialized:
#         print("Failed to initialize services. Exiting.")
#     else:
#         # --- Example Ingestion ---
#         # Create a dummy PDF first or use an existing one
#         # dummy_pdf_path = "/home/ubuntu/dummy.pdf"
#         # with open(dummy_pdf_path, "w") as f:
#         #     f.write("This is a test PDF document about multilingual AI systems.\nIt discusses RAG and translation.")
#         # print(f"Attempting ingestion of: {dummy_pdf_path}")
#         # ingest_success = ingest_pdf(dummy_pdf_path, document_id="test_doc_1")
#         # print(f"Ingestion successful: {ingest_success}")
#
#         # --- Example Query ---
#         print("\nAttempting query...")
#         question_hindi = "बहुभाषी एआई सिस्टम क्या हैं?" # What are multilingual AI systems?
#         answer = query_pipeline(question_hindi, source_lang="hindi", target_lang="hindi")
#         print(f"\nQuestion (Hindi): {question_hindi}")
#         print(f"Answer (Hindi): {answer}")
#
#         question_french = "Qu\'est-ce que RAG?" # What is RAG?
#         answer_french = query_pipeline(question_french, source_lang="french", target_lang="french")
#         print(f"\nQuestion (French): {question_french}")
#         print(f"Answer (French): {answer_french}")

