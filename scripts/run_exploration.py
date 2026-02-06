#!/usr/bin/env python3
"""
CLI script to run data exploration without the API.

Usage:
    python scripts/run_exploration.py
    python scripts/run_exploration.py --focus customer_churn
    python scripts/run_exploration.py --iterations 20
"""
import sys
import os
import argparse
import logging
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.schemas.metadata import MetadataLoader
from app.schemas.exploration import SchemaMetadata
from app.llm.explorer import DataExplorer
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print(" " * 20 + "AUTONOMOUS INSIGHTS ENGINE")
    print(" " * 15 + "LLM-Powered Data Exploration")
    print("=" * 80)
    print(f"Model: {settings.llm_model}")
    print(f"Reasoning Effort: {settings.reasoning_effort}")
    print("=" * 80)
    print()


def print_findings(result):
    """Print exploration findings in a nice format."""
    print("\n" + "=" * 80)
    print(f"EXPLORATION COMPLETE - Run ID: {result.run_id}")
    print("=" * 80)
    print(f"Iterations: {result.iterations}")
    print(f"Queries Executed: {result.total_queries_executed} "
          f"(✓ {result.successful_queries} / ✗ {result.failed_queries})")
    print(f"Duration: {(result.completed_at - result.started_at).total_seconds():.1f}s")

    if result.truncated:
        print("⚠️  Exploration was truncated (reached max iterations)")

    print()

    # Get interesting findings
    interesting_findings = [f for f in result.findings if f.is_interesting]

    if not interesting_findings:
        print("ℹ️  No interesting findings discovered.")
        return

    print(f"📊 DISCOVERED {len(interesting_findings)} INTERESTING FINDINGS")
    print("=" * 80)
    print()

    for i, finding in enumerate(interesting_findings, 1):
        # Determine priority emoji
        if finding.confidence == "high":
            priority = "🔴 HIGH PRIORITY"
        elif finding.confidence == "medium":
            priority = "🟡 MEDIUM PRIORITY"
        else:
            priority = "🟢 LOW PRIORITY"

        print(f"\n{i}. {priority} - Confidence: {finding.confidence.upper()}")
        print("-" * 80)

        if finding.insight:
            print(f"Finding: {finding.insight}")

        if finding.business_impact:
            print(f"\n💰 Business Impact: {finding.business_impact}")

        if finding.recommended_action:
            print(f"\n✅ Recommended Action: {finding.recommended_action}")

        if finding.supporting_data:
            print(f"\n📈 Supporting Data: {finding.supporting_data}")

        print()

    print("=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run autonomous data exploration"
    )
    parser.add_argument(
        "--focus",
        type=str,
        default=None,
        help="Focus area: customer_churn, margin_analysis, product_performance, regional_analysis"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=f"Max exploration iterations (default: {settings.max_exploration_iterations})"
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="metadata",
        help="Directory containing metadata JSON files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        print_banner()

        # Load metadata
        print("📂 Loading metadata...")
        metadata = MetadataLoader.load_all_metadata(args.metadata_dir)
        business_context = MetadataLoader.load_business_context()

        if not metadata:
            print("❌ Error: No metadata loaded. Please ensure metadata files exist.")
            sys.exit(1)

        print(f"✓ Loaded metadata for {len(metadata)} tables: {', '.join(metadata.keys())}")
        print()

        # Create schema metadata
        schema_metadata = SchemaMetadata(
            tables=metadata,
            business_context=business_context
        )

        # Create explorer
        print("🤖 Initializing explorer...")
        explorer = DataExplorer(schema_metadata=schema_metadata)
        print("✓ Explorer ready")
        print()

        # Run exploration
        print("🔍 Starting autonomous exploration...")
        if args.focus:
            print(f"   Focus area: {args.focus}")
        print()

        result = explorer.explore(
            focus_area=args.focus,
            max_iterations=args.iterations
        )

        # Print results
        print_findings(result)

        # Exit code based on success
        if result.error_message:
            print(f"\n❌ Error: {result.error_message}")
            sys.exit(1)
        else:
            print("\n✓ Exploration completed successfully")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Exploration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Exploration failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
