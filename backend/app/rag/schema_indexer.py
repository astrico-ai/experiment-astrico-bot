"""
Schema Indexer - Creates embeddings for database schemas.
Indexes tables, columns, and relationships for fast retrieval.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from ..config import settings

logger = logging.getLogger(__name__)


class SchemaIndexer:
    """
    Indexes database schemas into vector database for RAG retrieval.

    Features:
    - Table-level embeddings (coarse retrieval)
    - Column-level embeddings (fine-grained retrieval)
    - Relationship embeddings (join paths)
    - Semantic search capabilities
    """

    def __init__(self, metadata_dir: str = "metadata", persist_dir: str = ".chromadb"):
        """
        Initialize schema indexer.

        Args:
            metadata_dir: Directory containing metadata JSON files
            persist_dir: Directory to persist ChromaDB data
        """
        # Resolve metadata directory - check multiple locations
        metadata_path = Path(metadata_dir)

        # If relative path, try multiple locations
        if not metadata_path.is_absolute():
            # Try current directory
            if metadata_path.exists():
                self.metadata_dir = metadata_path
            # Try parent directory (for backend/app/rag structure)
            elif (Path.cwd().parent / metadata_dir).exists():
                self.metadata_dir = Path.cwd().parent / metadata_dir
            # Try project root (../../ from backend/)
            elif (Path(__file__).parent.parent.parent.parent / metadata_dir).exists():
                self.metadata_dir = Path(__file__).parent.parent.parent.parent / metadata_dir
            else:
                # Default to provided path
                self.metadata_dir = metadata_path
        else:
            self.metadata_dir = metadata_path

        self.persist_dir = persist_dir

        # Initialize embedding model based on configuration
        logger.info(f"[RAG] Loading embedding model: {settings.embedding_provider} - {settings.embedding_model}")

        self.embedding_provider = settings.embedding_provider
        self.embedding_dimensions = settings.embedding_dimensions

        if self.embedding_provider == "openai":
            # Use OpenRouter for OpenAI embeddings
            self.openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
                default_headers={
                    "HTTP-Referer": "https://experiment-bot.local",
                    "X-Title": "Experiment Bot"
                }
            )
            self.embedding_model_name = settings.embedding_model
            logger.info(f"[RAG] Using OpenAI embeddings via OpenRouter: {self.embedding_model_name} ({self.embedding_dimensions} dims) (App: Experiment Bot)")
        else:  # sentence-transformers
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info(f"[RAG] Using SentenceTransformer: {settings.embedding_model} ({self.embedding_dimensions} dims)")

        # Initialize ChromaDB with persistence
        logger.info("[RAG] Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_dir
        )

        # Create collections
        self.table_collection = self.client.get_or_create_collection(
            name="table_schemas",
            metadata={"description": "Table-level schema embeddings"}
        )

        self.column_collection = self.client.get_or_create_collection(
            name="column_schemas",
            metadata={"description": "Column-level schema embeddings"}
        )

        self.relationship_collection = self.client.get_or_create_collection(
            name="schema_relationships",
            metadata={"description": "Table relationship embeddings"}
        )

        logger.info("[RAG] Schema indexer initialized")

    def encode(self, text: str) -> List[float]:
        """
        Encode text to embedding vector using configured provider.

        Args:
            text: Text to encode

        Returns:
            List of floats representing the embedding vector
        """
        if self.embedding_provider == "openai":
            # Use OpenAI embeddings
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=text
            )
            return response.data[0].embedding
        else:
            # Use SentenceTransformer
            return self.embedding_model.encode(text).tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts to embedding vectors using configured provider.
        This is more efficient than calling encode() multiple times.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.embedding_provider == "openai":
            # OpenAI API supports batching up to 2048 texts
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=texts
            )
            # Return embeddings in the same order as input
            return [item.embedding for item in response.data]
        else:
            # SentenceTransformer also supports batch encoding
            embeddings = self.embedding_model.encode(texts)
            return [emb.tolist() for emb in embeddings]

    def index_all_schemas(self):
        """Index all schemas from metadata directory."""
        logger.info(f"[RAG] Indexing schemas from {self.metadata_dir}")

        # Find all JSON files
        json_files = list(self.metadata_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"[RAG] No metadata files found in {self.metadata_dir}")
            return

        for json_file in json_files:
            if json_file.name == "business_context.json":
                continue  # Skip non-schema files

            try:
                self._index_schema_file(json_file)
            except Exception as e:
                logger.error(f"[RAG] Error indexing {json_file}: {e}")

        logger.info(f"[RAG] Indexed {len(json_files)} schema files")

    def _index_schema_file(self, json_file: Path):
        """Index a single schema file."""
        with open(json_file, 'r') as f:
            schema = json.load(f)

        table_name = json_file.stem  # e.g., "sales_invoices_veedol"

        logger.info(f"[RAG] Indexing table: {table_name}")

        # 1. Index table-level info
        self._index_table(table_name, schema)

        # 2. Index column-level info
        self._index_columns(table_name, schema)

        # 3. Index relationships (foreign keys, common joins)
        self._index_relationships(table_name, schema)

    def _index_table(self, table_name: str, schema: Dict):
        """Index table-level metadata."""
        # Build table description for embedding
        columns = list(schema.keys())

        # Extract key columns
        key_columns = []
        for col_name, col_info in schema.items():
            if any(kw in col_name.lower() for kw in ['number', 'code', 'id', 'key']):
                key_columns.append(col_name)

        # Build searchable text
        table_text = f"""
        Table: {table_name}
        Columns: {', '.join(columns[:10])}
        Key columns: {', '.join(key_columns)}
        Total columns: {len(columns)}
        """

        # Create embedding
        embedding = self.encode(table_text)

        # Store in vector DB
        self.table_collection.add(
            ids=[table_name],
            embeddings=[embedding],
            documents=[table_text],
            metadatas=[{
                "table_name": table_name,
                "column_count": len(columns),
                "key_columns": ",".join(key_columns)
            }]
        )

    def _index_columns(self, table_name: str, schema: Dict):
        """Index column-level metadata."""
        for col_name, col_info in schema.items():
            # Build column description
            unit = col_info.get('unit', '')
            description = col_info.get('description', '')
            data_type = col_info.get('type', '')

            # Extract enum values from description if present (values in square brackets)
            enum_values = ""
            if '[' in description and ']' in description:
                start = description.find('[')
                end = description.find(']', start)
                enum_values = description[start+1:end]

            # Build searchable text with prominent enum values
            column_text = f"""
            Column: {table_name}.{col_name}
            Type: {data_type}
            Unit: {unit}
            Description: {description}
            Searchable as: {col_name}, {description}
            """

            # Add enum values prominently for better semantic matching
            if enum_values:
                column_text += f"\nContains values: {enum_values}\n"
                column_text += f"Match queries about: {enum_values.replace(',', ' or')}\n"

            # Create embedding
            embedding = self.encode(column_text)

            # Generate unique ID
            doc_id = f"{table_name}.{col_name}"

            # Store in vector DB
            self.column_collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[column_text],
                metadatas=[{
                    "table": table_name,
                    "column": col_name,
                    "type": data_type,
                    "unit": unit if unit else "none",
                    "description": description
                }]
            )

    def _index_relationships(self, table_name: str, schema: Dict):
        """Index table relationships (join keys, foreign keys)."""
        # Detect common join columns
        join_columns = []

        for col_name in schema.keys():
            # Common join patterns
            if any(pattern in col_name.lower() for pattern in [
                'customer_number', 'material_number', 'invoice',
                'billing_date', 'date', 'region', 'channel'
            ]):
                join_columns.append(col_name)

        if not join_columns:
            return  # No obvious join columns

        # Create relationship documents
        for join_col in join_columns:
            relationship_text = f"""
            Join key: {table_name}.{join_col}
            Can join with other tables on: {join_col}
            Used for: Linking {table_name} with related data
            """

            embedding = self.encode(relationship_text)

            doc_id = f"{table_name}.{join_col}.join"

            self.relationship_collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[relationship_text],
                metadatas=[{
                    "table": table_name,
                    "join_column": join_col,
                    "relationship_type": "potential_join"
                }]
            )

    def get_stats(self) -> Dict[str, int]:
        """Get indexing statistics."""
        return {
            "tables": self.table_collection.count(),
            "columns": self.column_collection.count(),
            "relationships": self.relationship_collection.count()
        }


# Global instance
schema_indexer = None

def get_schema_indexer() -> SchemaIndexer:
    """Get or create global schema indexer instance."""
    global schema_indexer
    if schema_indexer is None:
        schema_indexer = SchemaIndexer()
    return schema_indexer
