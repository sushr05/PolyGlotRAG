import logging
from sentence_transformers import SentenceTransformer
import torch
import os

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\")

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the Sentence Transformer model."""
        try:
            logging.info(f"Loading embedding model: {self.model_name}")
            # Check if CUDA is available and set device
            device = \'cuda\' if torch.cuda.is_available() else \'cpu\'
            if device == \'cuda\':
                logging.info("CUDA detected. Loading embedding model on GPU.")
            else:
                logging.info("CUDA not detected. Loading embedding model on CPU.")

            # Load the model
            self.model = SentenceTransformer(self.model_name, device=device)
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load embedding model {self.model_name}: {e}")
            self.model = None

    def get_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """Generates embeddings for a list of texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embeddings (each embedding is a list of floats), or None if embedding fails.
        """
        if not self.model:
            logging.error("Embedding model is not loaded. Cannot generate embeddings.")
            return None

        try:
            logging.info(f"Generating embeddings for {len(texts)} texts.")
            # The model's encode function handles batching internally
            # For multilingual-e5 models, prefixing with 'query: ' or 'passage: ' is recommended
            # However, for simplicity here, we'll omit it, but it should be considered for optimization
            # Example: embeddings = self.model.encode([f'passage: {text}' for text in texts])
            embeddings = self.model.encode(texts, convert_to_numpy=False, convert_to_tensor=False)
            # Convert numpy arrays or tensors to lists of floats if necessary
            # The encode method might return numpy arrays or tensors depending on version/config
            # Ensuring output is list[list[float]]
            if not isinstance(embeddings, list):
                embeddings = embeddings.tolist()
            if embeddings and not isinstance(embeddings[0], list):
                 embeddings = [emb.tolist() for emb in embeddings]

            logging.info("Embeddings generated successfully.")
            return embeddings
        except Exception as e:
            logging.error(f"Error during embedding generation: {e}")
            return None

# Example usage (for testing):
# if __name__ == '__main__':
#     # Ensure model is downloaded first
#     embedder = EmbeddingService()
#     if embedder.model:
#         sentences = ["This is an example sentence", "这是一个例句", "Ceci est une phrase d'exemple"]
#         embeddings = embedder.get_embeddings(sentences)
#         if embeddings:
#             print(f"Generated {len(embeddings)} embeddings.")
#             print(f"Dimension of first embedding: {len(embeddings[0])}")
#         else:
#             print("Failed to generate embeddings.")

