"""
Database metadata loader for dynamic schema loading.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MetadataLoader:
    """Loads database metadata from JSON files."""

    @staticmethod
    def load_table_metadata(file_path: str) -> Dict[str, dict]:
        """
        Load metadata for a single table from JSON file.

        Expected format:
        {
            "column_name": {
                "type": "varchar(50)",
                "unit": "₹" or null,
                "description": "Column description"
            }
        }

        Args:
            file_path: Path to JSON metadata file

        Returns:
            dict: Parsed metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded metadata from {file_path}")
                return data
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return {}

    @staticmethod
    def load_all_metadata(metadata_dir: str = "../metadata") -> Dict[str, List[dict]]:
        """
        Load metadata for all tables from a directory.

        Looks for files like:
        - metadata/primary_sales.json
        - metadata/customer_master.json
        - metadata/product_master.json

        Args:
            metadata_dir: Directory containing metadata JSON files (default: ../metadata from backend/)

        Returns:
            dict: Table name -> list of column metadata dicts
        """
        metadata_path = Path(metadata_dir)

        if not metadata_path.exists():
            logger.warning(f"Metadata directory not found: {metadata_dir}")
            return {}

        all_metadata = {}

        # Find all JSON files
        for json_file in metadata_path.glob("*.json"):
            table_name = json_file.stem  # filename without extension

            # Skip inspection metadata files (created by migration script)
            if table_name.startswith("inspect_"):
                logger.debug(f"Skipping inspection file: {json_file.name}")
                continue

            # Load metadata
            column_metadata = MetadataLoader.load_table_metadata(str(json_file))

            # Skip if empty or has wrong structure
            if not column_metadata or not isinstance(column_metadata, dict):
                logger.warning(f"Skipping invalid metadata file: {json_file.name}")
                continue

            # Convert to list format for schema context
            columns = []
            for col_name, col_info in column_metadata.items():
                # Handle case where col_info might be a string (old format)
                if isinstance(col_info, str):
                    columns.append({
                        "name": col_name,
                        "type": col_info,
                        "unit": None,
                        "description": ""
                    })
                elif isinstance(col_info, dict):
                    columns.append({
                        "name": col_name,
                        "type": col_info.get("type", "unknown"),
                        "unit": col_info.get("unit"),
                        "description": col_info.get("description", "")
                    })
                else:
                    logger.warning(f"Skipping invalid column {col_name} in {json_file.name}")
                    continue

            all_metadata[table_name] = columns
            logger.info(f"Loaded {len(columns)} columns for table: {table_name}")

        return all_metadata

    @staticmethod
    def load_business_context(file_path: str = "../metadata/business_context.txt") -> str:
        """
        Load business context from text file.

        Args:
            file_path: Path to business context file (default: ../metadata/business_context.txt from backend/)

        Returns:
            str: Business context text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                context = f.read()
                logger.info(f"Loaded business context from {file_path}")
                return context
        except FileNotFoundError:
            logger.warning(f"Business context file not found: {file_path}, using default")
            return MetadataLoader.get_default_business_context()

    @staticmethod
    def get_default_business_context() -> str:
        """Default business context if file not found."""
        return """
This is a B2B lubricant distribution company selling to:
- Auto workshops and service centers (AA-CP, AA-DD segments)
- OEM manufacturers (OEM-FF, OEM-FW, OEM-Customer)
- Institutional buyers (INST-CP, INST-DC)
- Export markets
- E-commerce channels

Key Business Priorities:
1. Customer retention - churn is expensive, monitor ordering patterns
2. Margin protection - contribution1 is the key metric
3. Volume growth in underpenetrated segments
4. Discount optimization - multiple discount types can stack

Sales Characteristics:
- Typical reorder cycle: 2-4 weeks for active customers
- Seasonality: Q4 typically strong (fleet maintenance season)
- Regional variation: Different regions have different product preferences
- Two business units: ENTI and TWOC (compare performance)

Key Signals to Watch:
- Order gap increases (churn risk)
- Margin compression (discount creep or COGS increase)
- Product mix shifts (customers switching grades)
- Return rates by batch/plant (quality issues)
- Geographic performance divergence
"""


# Example metadata structure for reference
EXAMPLE_METADATA = {
    "primary_sales": [
        {
            "name": "billing_date",
            "type": "date",
            "unit": "YYYY-MM-DD",
            "description": "Date of billing transaction"
        },
        {
            "name": "customer_name",
            "type": "varchar(100)",
            "unit": None,
            "description": "Name of the customer"
        },
        {
            "name": "gsv",
            "type": "numeric",
            "unit": "₹",
            "description": "Gross Sales Value before any discounts"
        },
        {
            "name": "contribution1",
            "type": "numeric",
            "unit": "₹",
            "description": "Key profitability metric (revenue - COGS - discounts)"
        }
    ]
}
