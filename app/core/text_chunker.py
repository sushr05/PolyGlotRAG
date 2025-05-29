import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        """Initializes the text chunker.

        Args:
            chunk_size: The target size for each text chunk (in characters).
            chunk_overlap: The overlap between consecutive chunks (in characters).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Using RecursiveCharacterTextSplitter as it tries to split based on semantic boundaries first
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True, # Useful for potential context window management
        )
        logging.info(f"Text chunker initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def chunk_text(self, text: str) -> list[str]:
        """Splits a large text into smaller chunks.

        Args:
            text: The input text string.

        Returns:
            A list of text chunks.
        """
        if not text:
            logging.warning("Input text is empty. Returning empty list of chunks.")
            return []
        try:
            logging.info(f"Chunking text of length {len(text)}...")
            # Langchain\\'s create_documents expects a list of texts, but we have one large text
            # We can use split_text directly for a single string
            chunks = self.splitter.split_text(text)
            logging.info(f"Successfully split text into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logging.error(f"Error during text chunking: {e}")
            return []

# Example usage:
# if __name__ == \'__main__\':
#     chunker = TextChunker()
#     long_text = "This is a very long text... " * 200 # Example long text
#     chunks = chunker.chunk_text(long_text)
#     if chunks:
#         print(f"Created {len(chunks)} chunks.")
#         print("First chunk:\n", chunks[0])
#         print("\nSecond chunk:\n", chunks[1])
#     else:
#         print("Chunking failed.")

