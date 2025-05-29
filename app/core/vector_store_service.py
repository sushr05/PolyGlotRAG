import logging
import os
from pinecone import Pinecone, ServerlessSpec, PodSpec
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

# Load environment variables from .env file
load_dotenv()

class VectorStoreService:
    def __init__(self, index_name: str, embedding_dimension: int):
        self.api_key = os.getenv("PINECONE_API_KEY")
        # Pinecone free tier uses cloud=\'aws\' and region=\'us-east-1\'
        # Adjust if using other tiers/configurations
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        self.pinecone_client = None
        self.index = None

        if not self.api_key:
            logging.error("PINECONE_API_KEY not found in environment variables.")
            raise ValueError("Pinecone API key is required.")

        self._initialize_pinecone()
        self._connect_to_index()

    def _initialize_pinecone(self):
        """Initializes the Pinecone client."""
        try:
            logging.info("Initializing Pinecone client...")
            self.pinecone_client = Pinecone(api_key=self.api_key)
            logging.info("Pinecone client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Pinecone client: {e}")
            self.pinecone_client = None

    def _connect_to_index(self):
        """Connects to the specified Pinecone index, creating it if it doesn\t exist."""
        if not self.pinecone_client:
            logging.error("Pinecone client not initialized. Cannot connect to index.")
            return

        try:
            if self.index_name not in self.pinecone_client.list_indexes().names:
                logging.info(f"Index 	'{self.index_name}	' not found. Creating new index...")
                # Using ServerlessSpec for potentially lower costs, adjust if needed
                # Requires specifying cloud and region
                # For hybrid search, metric should be \'dotproduct\' or \'cosine\'
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine", # Cosine similarity is common for sentence embeddings
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                    # If using PodSpec (older/standard tiers):
                    # spec=PodSpec(
                    #     environment=self.environment, # e.g., \'gcp-starter\', \'us-west1-gcp\'
                    #     pod_type=\'p1.x1\', # Example pod type
                    #     pods=1
                    # )
                )
                logging.info(f"Index 	'{self.index_name}	' created successfully.")
            else:
                logging.info(f"Connecting to existing index: {self.index_name}")

            self.index = self.pinecone_client.Index(self.index_name)
            logging.info(f"Successfully connected to Pinecone index: {self.index_name}")
            # Log index stats
            stats = self.index.describe_index_stats()
            logging.info(f"Index stats: {stats}")

        except Exception as e:
            logging.error(f"Failed to connect to or create Pinecone index 	'{self.index_name}	': {e}")
            self.index = None

    def upsert_vectors(self, vectors: list[tuple[str, list[float], dict]]) -> bool:
        """Upserts vectors into the Pinecone index.

        Args:
            vectors: A list of tuples, where each tuple is (id, vector, metadata).
                     Metadata should be a dictionary.

        Returns:
            True if upsert was successful, False otherwise.
        """
        if not self.index:
            logging.error("Pinecone index not available. Cannot upsert vectors.")
            return False

        try:
            logging.info(f"Upserting {len(vectors)} vectors into index 	'{self.index_name}	'...")
            # Pinecone recommends upserting in batches (e.g., 100 vectors per request)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logging.info(f"Upserted batch {i // batch_size + 1}")
            logging.info("Vector upsert completed successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to upsert vectors into Pinecone index: {e}")
            return False

    def query_vectors(self, query_embedding: list[float], top_k: int = 5, filter_dict: dict | None = None) -> list[dict] | None:
        """Queries the Pinecone index for similar vectors.

        Args:
            query_embedding: The embedding vector of the query.
            top_k: The number of similar vectors to retrieve.
            filter_dict: (Optional) A dictionary for metadata filtering.
                       Example: {"document_id": "doc123"}

        Returns:
            A list of query results (dictionaries with id, score, metadata), or None on failure.
        """
        if not self.index:
            logging.error("Pinecone index not available. Cannot query vectors.")
            return None

        try:
            logging.info(f"Querying index 	'{self.index_name}	' with top_k={top_k}...")
            query_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            logging.info(f"Query successful. Found {len(query_results.get(	'matches	', []))} matches.")
            return query_results.get("matches", []) # Return the list of matches
        except Exception as e:
            logging.error(f"Failed to query Pinecone index: {e}")
            return None

    def delete_index(self):
        """Deletes the Pinecone index."""
        if not self.pinecone_client:
            logging.error("Pinecone client not initialized. Cannot delete index.")
            return
        try:
            if self.index_name in self.pinecone_client.list_indexes().names:
                logging.warning(f"Deleting Pinecone index: {self.index_name}")
                self.pinecone_client.delete_index(self.index_name)
                logging.info(f"Index 	'{self.index_name}	' deleted successfully.")
                self.index = None
            else:
                logging.info(f"Index 	'{self.index_name}	' does not exist. No need to delete.")
        except Exception as e:
            logging.error(f"Failed to delete Pinecone index 	'{self.index_name}	': {e}")

# Example usage (requires PINECONE_API_KEY in .env):
# if __name__ == \'__main__\':
#     # Assuming multilingual-e5-large dimension is 1024
#     vector_store = VectorStoreService(index_name="polyglot-rag-test", embedding_dimension=1024)
#
#     if vector_store.index:
#         # Example Upsert
#         vectors_to_upsert = [
#             ("vec1", [0.1] * 1024, {"text": "This is chunk 1", "doc_id": "doc_A"}),
#             ("vec2", [0.2] * 1024, {"text": "This is chunk 2", "doc_id": "doc_A"}),
#             ("vec3", [0.3] * 1024, {"text": "Ceci est chunk 3", "doc_id": "doc_B"}),
#         ]
#         vector_store.upsert_vectors(vectors_to_upsert)
#
#         # Example Query
#         query_emb = [0.15] * 1024
#         results = vector_store.query_vectors(query_emb, top_k=2)
#         if results:
#             print("Query Results:")
#             for match in results:
#                 print(f"  ID: {match[\'id\']}, Score: {match[\'score\']:.4f}, Metadata: {match[\'metadata\']}")
#
#         # Example Query with Filter
#         results_filtered = vector_store.query_vectors(query_emb, top_k=2, filter_dict={"doc_id": "doc_A"})
#         if results_filtered:
#             print("\nFiltered Query Results (doc_id=\'doc_A\'):")
#             for match in results_filtered:
#                 print(f"  ID: {match[\'id\']}, Score: {match[\'score\']:.4f}, Metadata: {match[\'metadata\']}")
#
#         # Cleanup (optional)
#         # vector_store.delete_index()

