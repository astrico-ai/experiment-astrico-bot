# Advanced RAG System for Schema Retrieval

## Overview

This RAG (Retrieval Augmented Generation) system dramatically reduces LLM prompt size from 27k chars to 5-10k chars by retrieving only relevant database schemas for each query.

## Architecture

```
User Query
    ↓
Query Analyzer (extract intent, entities, dimensions)
    ↓
Schema Retriever (vector similarity search)
    ↓
Context Builder (assemble minimal prompt)
    ↓
LLM with focused context
```

## Components

### 1. Schema Indexer (`schema_indexer.py`)
- Indexes database schemas into ChromaDB vector database
- Three-level indexing:
  - **Table-level**: Coarse-grained retrieval
  - **Column-level**: Fine-grained attribute retrieval
  - **Relationship-level**: Join path detection
- Uses SentenceTransformer ('all-MiniLM-L6-v2') for embeddings
- Persistent storage in `.chromadb/` directory

### 2. Query Analyzer (`query_analyzer.py`)
- Extracts semantic intent from user queries
- Detects query type (simple, comparison, trend, forecast, analytical)
- Identifies entities (metrics, business terms)
- Extracts time information (quarters, years, months)
- Determines dimensions for grouping
- Builds optimized search keywords

### 3. Schema Retriever (`retriever.py`)
- Multi-stage retrieval pipeline:
  - **Stage 1**: Retrieve top-k relevant tables (default: 3)
  - **Stage 2**: Retrieve top-k relevant columns (default: 15)
  - **Stage 3**: Retrieve join relationships
- Returns relevance scores for each schema element

### 4. Context Builder (`context_builder.py`)
- Assembles minimal, focused prompts
- Includes:
  - Business context (always included)
  - Query intent
  - Relevant tables (top 3)
  - Relevant columns (top 15)
  - Join relationships (top 5)
  - SQL-specific rules
- Estimates token count

### 5. RAG Pipeline (`pipeline.py`)
- Orchestrates the complete flow
- Single entry point: `process_query(query)`
- Returns formatted context ready for LLM

## Installation

1. Install dependencies:
```bash
pip install chromadb sentence-transformers
```

2. Initialize RAG system (index all schemas):
```bash
python init_rag.py
```

This creates the vector database and indexes all schemas from the `metadata/` directory.

## Usage

### Standalone Usage

```python
from app.rag.pipeline import get_rag_pipeline

# Initialize pipeline
pipeline = get_rag_pipeline()

# Process a query
result = pipeline.process_query("What was total sales in FY24?")

# Access components
context = result['context']  # Formatted context for LLM
sql_rules = result['sql_rules']  # SQL-specific rules
metadata = result['metadata']  # Query analysis metadata

# Check stats
stats = pipeline.get_stats()
print(f"Indexed: {stats['indexed_tables']} tables, {stats['indexed_columns']} columns")
```

### Integration with ConversationManager

RAG is automatically enabled in ConversationManager by default:

```python
from app.chat.conversation_manager import ConversationManager

# RAG enabled by default
manager = ConversationManager(use_rag=True)

# Disable RAG (fallback to full schema)
manager = ConversationManager(use_rag=False)
```

When enabled, RAG automatically:
1. Analyzes each user query
2. Retrieves relevant schemas
3. Builds minimal context
4. Passes to LLM

## Performance Impact

### Before RAG
- System prompt: ~27,000 chars
- Token usage: ~6,750 tokens (at 4 chars/token)
- All schemas included (inefficient)

### After RAG
- System prompt: ~5,000-10,000 chars
- Token usage: ~1,250-2,500 tokens
- Only relevant schemas included
- **60-80% reduction in prompt tokens**

### Benefits
- Faster LLM responses (less to process)
- Lower API costs (fewer input tokens)
- Better focus (only relevant schemas)
- Scalable to 50+ datasets

## Testing

Run the test suite:
```bash
python test_rag.py
```

This tests:
- Pipeline initialization
- Query analysis
- Schema retrieval
- Context building
- Multiple query types

## Configuration

### Tuning Retrieval

Adjust retrieval parameters in `pipeline.py`:

```python
result = pipeline.process_query(
    query="...",
    top_k_tables=3,     # Number of tables to retrieve
    top_k_columns=15    # Number of columns to retrieve
)
```

### Tuning Embeddings

Modify embedding model in `schema_indexer.py`:

```python
# Current: all-MiniLM-L6-v2 (384 dims, fast)
self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Alternative: Better quality, slower
# self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
```

## Maintenance

### Re-indexing Schemas

When schemas are updated:
```bash
python init_rag.py
```

This will re-index all schemas from the metadata directory.

### Clearing Vector Database

To start fresh:
```bash
rm -rf .chromadb/
python init_rag.py
```

## Troubleshooting

### Issue: "No metadata files found"
**Solution**: Ensure JSON schema files exist in `metadata/` directory

### Issue: "ChromaDB initialization failed"
**Solution**: Check write permissions for `.chromadb/` directory

### Issue: "Import error for sentence_transformers"
**Solution**: Install with `pip install sentence-transformers`

### Issue: RAG returns irrelevant schemas
**Solution**:
- Check schema descriptions in metadata files
- Tune top_k parameters
- Add more descriptive column metadata

## Architecture Decisions

### Why ChromaDB?
- Lightweight (no separate server needed)
- Fast similarity search
- Persistent storage
- Easy integration

### Why all-MiniLM-L6-v2?
- Small size (384 dimensions)
- Fast inference
- Good balance of speed/quality
- Sufficient for schema matching

### Why three-level indexing?
- **Table-level**: Quick filtering of relevant tables
- **Column-level**: Precise attribute matching
- **Relationship-level**: Join path discovery
- Allows multi-stage retrieval (coarse → fine)

## Future Enhancements

1. **Query rewriting**: Expand user queries with synonyms
2. **Caching**: Cache frequently accessed schemas
3. **Feedback loop**: Learn from successful queries
4. **Cross-encoder reranking**: Improve retrieval quality
5. **Graph-based relationships**: Better join path detection
