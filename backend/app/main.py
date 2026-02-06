"""
FastAPI application entry point for Autonomous Insights Engine.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .api.chat_routes import router as chat_router
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Autonomous Insights Engine",
    description="LLM-powered autonomous data exploration for B2B sales data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["exploration"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 60)
    logger.info("Autonomous Insights Engine Starting...")
    logger.info(f"Model: {settings.llm_model}")
    logger.info(f"Reasoning Effort: {settings.reasoning_effort}")
    logger.info(f"Max Iterations: {settings.max_exploration_iterations}")
    logger.info("=" * 60)

    # Initialize RAG system
    try:
        from .rag.schema_indexer import get_schema_indexer
        logger.info("Initializing RAG system...")
        indexer = get_schema_indexer()

        # Check if schemas are already indexed
        table_count = indexer.table_collection.count()

        if table_count == 0:
            logger.info("No schemas found. Indexing schemas with OpenAI embeddings...")
            indexer.index_all_schemas()
            logger.info(f"RAG initialization complete: {indexer.table_collection.count()} tables indexed")
        else:
            logger.info(f"RAG already initialized: {table_count} tables found")
    except Exception as e:
        logger.error(f"RAG initialization failed: {e}", exc_info=True)

    # Initialize Value Normalizer
    try:
        from .rag.value_normalizer import get_value_normalizer
        from .rag.schema_indexer import get_schema_indexer

        logger.info("Initializing Value Normalizer...")
        normalizer = get_value_normalizer()
        indexer = get_schema_indexer()

        # Set embedding provider with batch support for faster processing
        normalizer.set_embedding_provider(indexer.encode, indexer.encode_batch)

        # Check if values are already indexed
        value_count = normalizer.collection.count()

        if value_count == 0:
            logger.info("No value embeddings found. Indexing categorical columns...")

            # Index high-priority categorical columns
            columns_to_index = [
                # Tier 1: Critical (≤5 values)
                ('material_master_veedol', 'company_group', 'Company Group'),
                ('material_master_veedol', 'base_unit_of_measure', 'Unit of Measure'),
                ('material_master_veedol', 'pack_type', 'Pack Type'),
                ('sales_invoices_veedol', 'distribution_channel', 'Distribution Channel'),
                ('sales_invoices_veedol', 'item_category', 'Item Category'),
                ('sales_invoices_veedol', 'billing_code', 'Billing Code'),

                # Tier 2: High priority (6-15 values)
                ('sales_invoices_veedol', 'billing_type', 'Billing Type'),
                ('customer_master_veedol', 'customer_group_report', 'Sales Channel'),
                ('material_master_veedol', 'segment_order', 'Product Segment'),
                ('material_master_veedol', 'product_category', 'Product Category'),
                ('material_master_veedol', 'product_vertical', 'Product Vertical'),
                ('budget_data_veedol', 'channel', 'Budget Channel'),
                ('customer_master_veedol', 'distribution_channel', 'Customer Channel'),
                ('customer_master_veedol', 'state', 'State'),
                ('customer_master_veedol', 'region', 'Region'),
            ]

            for table, column, display_name in columns_to_index:
                normalizer.index_column_values(table, column, display_name)

            logger.info(f"Value Normalizer initialization complete: {normalizer.collection.count()} values indexed")
        else:
            logger.info(f"Value Normalizer already initialized: {value_count} values found")
            # Load existing embeddings into memory
            normalizer.load_from_storage()

    except Exception as e:
        logger.error(f"Value Normalizer initialization failed: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Autonomous Insights Engine Shutting Down...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Autonomous Insights Engine",
        "version": "1.0.0",
        "description": "LLM-powered autonomous data exploration",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
