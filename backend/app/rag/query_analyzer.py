"""
Query Analyzer - Extracts semantic intent from user queries.
"""
import logging
import re
from typing import Dict, List, Set, Any
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries we can handle."""
    SIMPLE_METRIC = "simple_metric"  # "What was total sales?"
    COMPARISON = "comparison"  # "Compare Q1 vs Q2"
    TREND = "trend"  # "Show me sales trend"
    FORECAST = "forecast"  # "Predict next month"
    ANALYTICAL = "analytical"  # "Why did sales drop?"
    AGGREGATION = "aggregation"  # "Group by region"


class QueryAnalyzer:
    """
    Analyzes user queries to extract:
    1. Query type (simple, comparison, trend, forecast, analytical)
    2. Entities (tables, columns, metrics)
    3. Time ranges
    4. Dimensions (group by fields)
    5. Filters
    """

    def __init__(self):
        """Initialize query analyzer."""
        # Common metric patterns
        self.metric_keywords = {
            'sales', 'revenue', 'volume', 'quantity', 'amount', 'value',
            'profit', 'margin', 'growth', 'rate', 'percentage', 'budget',
            'target', 'achieved', 'actual', 'forecast', 'predicted'
        }

        # Dimension patterns
        self.dimension_keywords = {
            'region', 'channel', 'product', 'customer', 'material',
            'category', 'segment', 'division', 'territory', 'plant',
            'brand', 'type', 'grade'
        }

        # Time-related patterns
        self.time_keywords = {
            'month', 'quarter', 'year', 'ytd', 'mtd', 'daily', 'weekly',
            'monthly', 'quarterly', 'yearly', 'today', 'yesterday',
            'last', 'current', 'next', 'previous'
        }

        # Aggregation patterns
        self.aggregation_keywords = {
            'total', 'sum', 'average', 'mean', 'count', 'max', 'min',
            'highest', 'lowest', 'top', 'bottom'
        }

        logger.info("[RAG] Query analyzer initialized")

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query and extract semantic components.

        Args:
            query: User's natural language query

        Returns:
            Dict with extracted components
        """
        query_lower = query.lower()

        # Determine query type
        query_type = self._detect_query_type(query_lower)

        # Extract entities
        entities = self._extract_entities(query_lower)

        # Extract time components
        time_info = self._extract_time_info(query_lower)

        # Extract dimensions (group by candidates)
        dimensions = self._extract_dimensions(query_lower)

        # Extract aggregations
        aggregations = self._extract_aggregations(query_lower)

        # Build search keywords for schema retrieval
        search_keywords = self._build_search_keywords(
            query_lower, entities, dimensions, aggregations
        )

        result = {
            "original_query": query,
            "query_type": query_type.value,
            "entities": entities,
            "time_info": time_info,
            "dimensions": dimensions,
            "aggregations": aggregations,
            "search_keywords": search_keywords,
            "requires_forecast": self._requires_forecast(query_lower),
            "requires_insights": self._requires_insights(query_lower)
        }

        logger.info(f"[RAG] Analyzed query: type={query_type.value}, "
                   f"entities={len(entities)}, dimensions={len(dimensions)}")

        return result

    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query."""
        # Forecast indicators
        if any(kw in query for kw in ['predict', 'forecast', 'next', 'future', 'will be', 'projection']):
            return QueryType.FORECAST

        # Analytical indicators
        if any(kw in query for kw in ['why', 'reason', 'cause', 'explain', 'analyze', 'investigation']):
            return QueryType.ANALYTICAL

        # Comparison indicators
        if any(kw in query for kw in ['compare', 'versus', 'vs', 'difference', 'compared to', 'against']):
            return QueryType.COMPARISON

        # Trend indicators
        if any(kw in query for kw in ['trend', 'over time', 'pattern', 'evolution', 'progression']):
            return QueryType.TREND

        # Aggregation indicators (group by)
        if any(kw in query for kw in ['by region', 'by channel', 'by product', 'breakdown', 'distribution']):
            return QueryType.AGGREGATION

        # Default to simple metric query
        return QueryType.SIMPLE_METRIC

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract metrics and business entities from query."""
        entities = {
            "metrics": [],
            "business_terms": []
        }

        # Extract metrics
        for metric in self.metric_keywords:
            if metric in query:
                entities["metrics"].append(metric)

        # Extract business terms (could be table or column names)
        for term in query.split():
            # Look for potential business terms (capitalized, multi-word phrases)
            if term.istitle() or '_' in term:
                entities["business_terms"].append(term)

        return entities

    def _extract_time_info(self, query: str) -> Dict[str, Any]:
        """Extract time-related information."""
        time_info = {
            "has_time_filter": False,
            "time_keywords": [],
            "specific_periods": []
        }

        # Check for time keywords
        found_time = []
        for time_kw in self.time_keywords:
            if time_kw in query:
                found_time.append(time_kw)
                time_info["has_time_filter"] = True

        time_info["time_keywords"] = found_time

        # Extract specific periods (Q1, Q2, 2024, 2025, etc.)
        quarters = re.findall(r'q[1-4]', query, re.IGNORECASE)
        years = re.findall(r'20\d{2}', query)
        months = re.findall(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b', query)

        time_info["specific_periods"] = {
            "quarters": quarters,
            "years": years,
            "months": months
        }

        return time_info

    def _extract_dimensions(self, query: str) -> List[str]:
        """Extract dimension fields for grouping."""
        dimensions = []

        for dim in self.dimension_keywords:
            if dim in query:
                dimensions.append(dim)

        return dimensions

    def _extract_aggregations(self, query: str) -> List[str]:
        """Extract aggregation functions needed."""
        aggregations = []

        for agg in self.aggregation_keywords:
            if agg in query:
                aggregations.append(agg)

        return aggregations

    def _build_search_keywords(
        self,
        query: str,
        entities: Dict,
        dimensions: List[str],
        aggregations: List[str]
    ) -> str:
        """Build optimized search keywords for schema retrieval."""
        keywords = []

        # Add metrics
        keywords.extend(entities.get("metrics", []))

        # Add dimensions
        keywords.extend(dimensions)

        # Add aggregations
        keywords.extend(aggregations)

        # Add original query terms (filtered)
        query_terms = query.split()
        # Remove common stop words (expanded list)
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'what', 'how', 'show', 'me', 'get', 'find',
            'and', 'or', 'in', 'for', 'to', 'of', 'at', 'by', 'from', 'with', 'a', 'an'
        }

        # Keep ALL meaningful terms (no arbitrary limit)
        # Embeddings handle longer text well, and we don't want to drop important keywords
        meaningful_terms = [term for term in query_terms if term not in stop_words and len(term) > 2]
        keywords.extend(meaningful_terms)

        # Build search string
        search_string = " ".join(set(keywords))  # Remove duplicates

        return search_string

    def _requires_forecast(self, query: str) -> bool:
        """Check if query requires forecasting."""
        forecast_indicators = ['predict', 'forecast', 'next', 'future', 'projection', 'will be']
        return any(indicator in query for indicator in forecast_indicators)

    def _requires_insights(self, query: str) -> bool:
        """Check if query requires analytical insights."""
        insight_indicators = ['why', 'reason', 'cause', 'explain', 'analyze', 'understand']
        return any(indicator in query for indicator in insight_indicators)


# Global instance
query_analyzer = None

def get_query_analyzer() -> QueryAnalyzer:
    """Get or create global query analyzer instance."""
    global query_analyzer
    if query_analyzer is None:
        query_analyzer = QueryAnalyzer()
    return query_analyzer
