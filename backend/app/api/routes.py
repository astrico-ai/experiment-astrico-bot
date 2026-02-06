"""
FastAPI routes for data exploration API.
"""
import logging
from typing import Dict
from fastapi import APIRouter, HTTPException

from ..schemas.responses import ExplorationResponse, ExploreRequest, SchemaResponse
from ..schemas.metadata import MetadataLoader
from ..schemas.exploration import SchemaMetadata
from ..llm.explorer import DataExplorer
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state for loaded metadata
_loaded_metadata: Dict = {}
_loaded_business_context: str = ""


def get_or_load_metadata() -> tuple[Dict, str]:
    """Get loaded metadata or load if not already loaded."""
    global _loaded_metadata, _loaded_business_context

    if not _loaded_metadata:
        logger.info("Loading metadata from files...")
        _loaded_metadata = MetadataLoader.load_all_metadata()
        _loaded_business_context = MetadataLoader.load_business_context(
            settings.business_context_path
        )

    return _loaded_metadata, _loaded_business_context


@router.post("/explore", response_model=ExplorationResponse)
async def explore_data(request: ExploreRequest):
    """
    Run autonomous data exploration.

    Args:
        request: Exploration parameters

    Returns:
        ExplorationResponse: Exploration results with findings
    """
    try:
        # Load metadata
        metadata, business_context = get_or_load_metadata()

        if not metadata:
            raise HTTPException(
                status_code=400,
                detail="No metadata loaded. Please ensure metadata files exist in the metadata/ directory."
            )

        # Create schema metadata
        schema_metadata = SchemaMetadata(
            tables=metadata,
            business_context=business_context
        )

        # Create explorer
        explorer = DataExplorer(schema_metadata=schema_metadata)

        # Run exploration
        logger.info(f"Starting exploration (focus: {request.focus_area})")
        result = explorer.explore(
            focus_area=request.focus_area,
            max_iterations=request.num_hypotheses
        )

        # Convert to response
        return ExplorationResponse(**result.to_dict())

    except Exception as e:
        logger.error(f"Exploration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Exploration failed: {str(e)}")


@router.get("/explore/{run_id}")
async def get_exploration_result(run_id: str):
    """
    Get exploration result by run ID.

    Note: In MVP, this is not implemented (no persistence).
    Future: Store results in database and retrieve.
    """
    raise HTTPException(
        status_code=501,
        detail="Result persistence not implemented in MVP. Run /explore to get immediate results."
    )


@router.get("/schema", response_model=SchemaResponse)
async def get_schema():
    """
    Get loaded database schema and business context.

    Returns:
        SchemaResponse: Schema metadata
    """
    try:
        metadata, business_context = get_or_load_metadata()

        if not metadata:
            raise HTTPException(
                status_code=404,
                detail="No metadata loaded. Please ensure metadata files exist."
            )

        return SchemaResponse(
            tables=metadata,
            business_context=business_context
        )

    except Exception as e:
        logger.error(f"Failed to get schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schema")
async def reload_schema():
    """
    Reload schema metadata from files.

    Returns:
        dict: Status message
    """
    global _loaded_metadata, _loaded_business_context

    try:
        _loaded_metadata = MetadataLoader.load_all_metadata()
        _loaded_business_context = MetadataLoader.load_business_context(
            settings.business_context_path
        )

        logger.info("Schema metadata reloaded successfully")

        return {
            "status": "loaded",
            "tables": list(_loaded_metadata.keys()),
            "num_tables": len(_loaded_metadata)
        }

    except Exception as e:
        logger.error(f"Failed to reload schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": settings.llm_model,
        "reasoning_effort": settings.reasoning_effort
    }
