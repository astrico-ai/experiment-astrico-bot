"""
Value Normalizer - Handles typos and variations in user input.
Uses embeddings to match user input to actual database values.
"""
import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings

from ..db.connection import db
from ..config import settings

logger = logging.getLogger(__name__)


class ValueNormalizer:
    """
    Normalizes user input values to actual database values using embeddings.
    Handles typos, case variations, and synonyms.
    Uses ChromaDB for persistent storage.
    """

    def __init__(self):
        """Initialize value normalizer with ChromaDB persistence."""
        self.value_embeddings: Dict[str, Dict[str, List[float]]] = {}
        self.indexed_columns = {}
        self.embedding_provider = None

        # Initialize ChromaDB client
        chroma_path = Path(settings.chroma_persist_directory) / "value_embeddings"
        chroma_path.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create/get collection for value embeddings
        self.collection = self.chroma_client.get_or_create_collection(
            name="value_embeddings",
            metadata={"description": "Embeddings of categorical database values for normalization"}
        )

        logger.info("[VALUE-NORM] Value normalizer initialized with ChromaDB persistence")

    def set_embedding_provider(self, encode_func, encode_batch_func=None):
        """
        Set the embedding function to use.

        Args:
            encode_func: Function that takes text and returns embedding vector
            encode_batch_func: Optional function that takes list of texts and returns list of embeddings
        """
        self.embedding_provider = encode_func
        self.embedding_provider_batch = encode_batch_func
        logger.info("[VALUE-NORM] Embedding provider set")

    def load_from_storage(self):
        """Load all indexed columns from ChromaDB."""
        try:
            count = self.collection.count()
            if count == 0:
                logger.info("[VALUE-NORM] No stored embeddings found")
                return

            logger.info(f"[VALUE-NORM] Loading {count} stored embeddings from ChromaDB...")

            # Get all stored data
            results = self.collection.get(include=["embeddings", "metadatas", "documents"])

            # Organize by column
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                embedding = results['embeddings'][i]
                value = results['documents'][i]

                table = metadata['table']
                column = metadata['column']
                display_name = metadata.get('display_name', f"{table}.{column}")

                key = f"{table}.{column}"

                # Initialize structures if needed
                if key not in self.value_embeddings:
                    self.value_embeddings[key] = {}
                    self.indexed_columns[key] = {
                        'table': table,
                        'column': column,
                        'display_name': display_name,
                        'values': []
                    }

                # Store embedding and value
                self.value_embeddings[key][value] = embedding
                if value not in self.indexed_columns[key]['values']:
                    self.indexed_columns[key]['values'].append(value)

            logger.info(f"[VALUE-NORM] Loaded {len(self.indexed_columns)} columns from storage")

        except Exception as e:
            logger.error(f"[VALUE-NORM] Error loading from storage: {e}")

    def index_column_values(
        self,
        table: str,
        column: str,
        display_name: Optional[str] = None,
        force_reindex: bool = False
    ):
        """
        Index distinct values from a database column.
        Checks ChromaDB first - only indexes if not already stored.

        Args:
            table: Table name
            column: Column name
            display_name: Optional friendly name for logging
            force_reindex: If True, re-index even if already stored
        """
        if not self.embedding_provider:
            logger.error("[VALUE-NORM] Embedding provider not set. Call set_embedding_provider() first.")
            return

        display = display_name or f"{table}.{column}"
        key = f"{table}.{column}"

        # Check if already indexed in ChromaDB
        if not force_reindex:
            existing_results = self.collection.get(
                where={"$and": [{"table": table}, {"column": column}]},
                limit=1
            )
            if existing_results['ids']:
                logger.info(f"[VALUE-NORM] {display} already indexed, skipping...")
                return

        logger.info(f"[VALUE-NORM] Indexing values for {display}...")

        try:
            # Get distinct non-null values from database
            query = f"""
                SELECT DISTINCT {column}
                FROM {table}
                WHERE {column} IS NOT NULL
                ORDER BY {column}
            """

            with db.get_connection() as client:
                result = client.query(query)
                results = list(result.named_results())

            if not results:
                logger.warning(f"[VALUE-NORM] No values found for {display}")
                return

            # Create embeddings for each value
            values = [row[column] for row in results]
            embeddings = {}
            ids = []
            documents = []
            embedding_list = []
            metadatas = []

            # Use batch encoding if available (much faster)
            if self.embedding_provider_batch:
                try:
                    str_values = [str(value) for value in values]
                    batch_embeddings = self.embedding_provider_batch(str_values)

                    for value, embedding in zip(values, batch_embeddings):
                        embeddings[value] = embedding
                        doc_id = f"{table}_{column}_{value}".replace(" ", "_").replace("/", "_")
                        ids.append(doc_id)
                        documents.append(str(value))
                        embedding_list.append(embedding)
                        metadatas.append({
                            'table': table,
                            'column': column,
                            'display_name': display,
                            'value': str(value)
                        })

                    logger.info(f"[VALUE-NORM] Batch encoded {len(values)} values in one call")
                except Exception as e:
                    logger.error(f"[VALUE-NORM] Batch encoding failed: {e}, falling back to individual encoding")
                    # Fall back to individual encoding
                    self.embedding_provider_batch = None

            # Fall back to individual encoding if batch not available
            if not self.embedding_provider_batch:
                for value in values:
                    try:
                        embedding = self.embedding_provider(str(value))
                        embeddings[value] = embedding

                        # Prepare for ChromaDB storage
                        doc_id = f"{table}_{column}_{value}".replace(" ", "_").replace("/", "_")
                        ids.append(doc_id)
                        documents.append(str(value))
                        embedding_list.append(embedding)
                        metadatas.append({
                            'table': table,
                            'column': column,
                            'display_name': display,
                            'value': str(value)
                        })
                    except Exception as e:
                        logger.error(f"[VALUE-NORM] Error embedding '{value}': {e}")

            # Store in memory
            self.value_embeddings[key] = embeddings
            self.indexed_columns[key] = {
                'table': table,
                'column': column,
                'display_name': display,
                'values': values
            }

            # Store in ChromaDB
            if ids:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embedding_list,
                    metadatas=metadatas
                )
                logger.info(f"[VALUE-NORM] Indexed and stored {len(embeddings)} values for {display}")
            else:
                logger.warning(f"[VALUE-NORM] No valid embeddings created for {display}")

        except Exception as e:
            logger.error(f"[VALUE-NORM] Error indexing {display}: {e}")

    def normalize(
        self,
        table: str,
        column: str,
        user_input: str,
        threshold: float = 0.75
    ) -> Tuple[str, float, bool]:
        """
        Normalize user input to closest database value.

        Args:
            table: Table name
            column: Column name
            user_input: User's input value
            threshold: Minimum similarity score (0-1) for correction

        Returns:
            Tuple of (normalized_value, similarity_score, was_corrected)
        """
        key = f"{table}.{column}"

        # Check if column is indexed
        if key not in self.value_embeddings:
            return user_input, 0.0, False

        # Get embeddings for this column
        value_embeddings = self.value_embeddings[key]

        if not value_embeddings:
            return user_input, 0.0, False

        # Embed user input
        try:
            user_embedding = self.embedding_provider(user_input)
        except Exception as e:
            logger.error(f"[VALUE-NORM] Error embedding user input '{user_input}': {e}")
            return user_input, 0.0, False

        # Find closest match using cosine similarity
        best_match = None
        best_score = -1.0

        for value, value_embedding in value_embeddings.items():
            score = self._cosine_similarity(user_embedding, value_embedding)
            if score > best_score:
                best_score = score
                best_match = value

        # Decide whether to correct
        if best_match and best_score >= threshold:
            was_corrected = (best_match.lower() != user_input.lower())

            if was_corrected:
                logger.info(
                    f"[VALUE-NORM] Corrected '{user_input}' → '{best_match}' "
                    f"({column}, score: {best_score:.2f})"
                )

            return best_match, best_score, was_corrected
        else:
            logger.warning(
                f"[VALUE-NORM] No good match for '{user_input}' in {column} "
                f"(best: '{best_match}', score: {best_score:.2f})"
            )
            return user_input, best_score, False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def extract_and_normalize(self, query: str) -> Dict[str, Any]:
        """
        Extract entities from query and normalize them.

        Args:
            query: User's natural language query

        Returns:
            Dict with normalized entities and corrections made
        """
        corrections = []

        # Try to normalize common entities
        # This is a simple implementation - can be enhanced with NER

        # Check for state names
        for key, info in self.indexed_columns.items():
            if info['column'] == 'state':
                # Simple extraction - look for words that might be states
                words = query.split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        normalized, score, was_corrected = self.normalize(
                            info['table'],
                            info['column'],
                            word,
                            threshold=0.75
                        )
                        if was_corrected:
                            corrections.append({
                                'column': info['column'],
                                'original': word,
                                'normalized': normalized,
                                'score': score
                            })

        return {
            'corrections': corrections,
            'has_corrections': len(corrections) > 0
        }

    def get_valid_values(self, table: str, column: str) -> List[str]:
        """
        Get list of valid values for a column.

        Args:
            table: Table name
            column: Column name

        Returns:
            List of valid values
        """
        key = f"{table}.{column}"
        if key in self.indexed_columns:
            return self.indexed_columns[key]['values']
        return []


# Global instance
_value_normalizer = None


def get_value_normalizer() -> ValueNormalizer:
    """Get or create global value normalizer instance."""
    global _value_normalizer
    if _value_normalizer is None:
        _value_normalizer = ValueNormalizer()
    return _value_normalizer
