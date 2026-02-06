"""
Main data exploration engine using GPT-5.2 with reasoning and tool calling.
"""
import json
import logging
import re
from typing import Optional, List
from datetime import datetime

from .client import LLMClient
from .prompts import build_system_prompt, build_exploration_prompt
from ..db.executor import SQLExecutor
from ..schemas.exploration import ExplorationResult, SchemaMetadata
from ..schemas.finding import Finding
from ..config import settings

logger = logging.getLogger(__name__)


class DataExplorer:
    """
    Autonomous data exploration engine using GPT-5.2.

    Uses tool calling pattern where GPT-5.2 can:
    1. Reason about what to explore
    2. Execute SQL queries via execute_sql tool
    3. Analyze results and decide next steps
    4. Synthesize findings
    """

    def __init__(
        self,
        schema_metadata: SchemaMetadata,
        llm_client: Optional[LLMClient] = None,
        sql_executor: Optional[SQLExecutor] = None
    ):
        """
        Initialize explorer.

        Args:
            schema_metadata: Database schema and business context
            llm_client: LLM client (creates new if None)
            sql_executor: SQL executor (uses default if None)
        """
        self.schema_metadata = schema_metadata
        self.llm_client = llm_client or LLMClient()
        self.sql_executor = sql_executor or SQLExecutor()

        # Build system context
        self.system_context = build_system_prompt(
            schema_metadata.to_context_string()
        )

        logger.info("DataExplorer initialized")

    def explore(
        self,
        focus_area: Optional[str] = None,
        max_iterations: Optional[int] = None
    ) -> ExplorationResult:
        """
        Run autonomous exploration using GPT-5.2 with tool calling.

        Args:
            focus_area: Optional focus area (e.g., 'customer_churn', 'margin_analysis')
            max_iterations: Max exploration iterations (defaults to settings)

        Returns:
            ExplorationResult: Exploration results with findings
        """
        max_iter = max_iterations or settings.max_exploration_iterations
        result = ExplorationResult()

        try:
            logger.info(f"Starting exploration (focus: {focus_area}, max_iterations: {max_iter})")

            # Build initial prompt
            initial_prompt = build_exploration_prompt(
                focus_area=focus_area,
                max_iterations=max_iter
            )

            # Define SQL execution tool
            sql_tool = {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute a ClickHouse SELECT query and return results. Use this to explore the data and test hypotheses.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "ClickHouse SELECT query to execute. Must be valid ClickHouse SQL syntax."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation of what you're trying to discover with this query"
                            }
                        },
                        "required": ["query", "explanation"]
                    }
                }
            }

            # Initialize conversation
            messages = [
                {"role": "system", "content": self.system_context},
                {"role": "user", "content": initial_prompt}
            ]

            iteration = 0
            while iteration < max_iter:
                iteration += 1
                logger.info(f"Exploration iteration {iteration}/{max_iter}")

                # Call LLM with tools
                response = self.llm_client.chat(
                    messages=messages,
                    tools=[sql_tool]
                )

                message = response.choices[0].message

                # Check if model wants to use tools
                if message.tool_calls:
                    # Add assistant message with tool calls to history
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": message.tool_calls
                    })

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "execute_sql":
                            # Parse arguments
                            args = json.loads(tool_call.function.arguments)
                            query = args.get("query", "")
                            explanation = args.get("explanation", "No explanation provided")

                            logger.info(f"Executing query: {explanation}")
                            logger.debug(f"SQL: {query}")

                            # Execute query
                            result.total_queries_executed += 1
                            query_result = self.sql_executor.execute_query(query)

                            if query_result["success"]:
                                result.successful_queries += 1
                                logger.info(f"Query successful: {query_result['row_count']} rows")
                            else:
                                result.failed_queries += 1
                                logger.warning(f"Query failed: {query_result['error']}")

                            # Format result for LLM
                            formatted_result = self.sql_executor.format_result_for_llm(
                                query_result,
                                max_rows_display=30
                            )

                            # Add tool result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": formatted_result
                            })

                else:
                    # No tool calls - model is done exploring
                    logger.info("Exploration complete - model finished exploring")

                    # Parse findings from final response
                    findings = self._parse_findings_from_response(message.content)
                    result.findings = findings
                    result.iterations = iteration
                    result.completed_at = datetime.utcnow()

                    logger.info(f"Exploration complete: {len(findings)} total findings, "
                               f"{len([f for f in findings if f.is_interesting])} interesting")

                    return result

                # Check if context is getting too long (> 100K tokens ~= 75K words ~= 400K chars)
                total_chars = sum(len(str(m.get("content", ""))) for m in messages)
                if total_chars > 300000:
                    logger.info("Context getting large, compacting...")
                    messages = self.llm_client.compact_context(messages)

            # Reached max iterations
            logger.warning(f"Exploration truncated at {max_iter} iterations")
            result.truncated = True
            result.iterations = iteration
            result.completed_at = datetime.utcnow()

            # Try to get findings from last message
            if messages and messages[-1].get("role") == "assistant":
                findings = self._parse_findings_from_response(messages[-1].get("content", ""))
                result.findings = findings

            return result

        except Exception as e:
            logger.error(f"Exploration failed: {e}", exc_info=True)
            result.error_message = str(e)
            result.completed_at = datetime.utcnow()
            return result

    def _parse_findings_from_response(self, content: str) -> List[Finding]:
        """
        Parse findings from LLM's final response.

        Args:
            content: LLM response content

        Returns:
            List[Finding]: Parsed findings
        """
        findings = []

        try:
            # Look for the KEY FINDINGS section
            if "KEY FINDINGS" in content or "## KEY FINDINGS" in content:
                # Extract findings section
                findings_section = content

                # Try to parse structured findings
                # Pattern: Priority + Finding + Business Impact + Recommended Action + Confidence
                finding_blocks = re.split(r'\n(?=\*\*Priority Level\*\*:|🔴|🟡|🟢)', findings_section)

                finding_id = 0
                for block in finding_blocks:
                    if not block.strip():
                        continue

                    finding_id += 1

                    # Extract components
                    is_high_priority = '🔴' in block or 'HIGH' in block.upper()
                    confidence = "high" if "High" in block else ("low" if "Low" in block else "medium")

                    # Extract finding description
                    finding_match = re.search(r'\*\*Finding\*\*:?\s*(.+?)(?=\*\*|$)', block, re.DOTALL)
                    finding_desc = finding_match.group(1).strip() if finding_match else block[:200]

                    # Extract business impact
                    impact_match = re.search(r'\*\*Business Impact\*\*:?\s*(.+?)(?=\*\*|$)', block, re.DOTALL)
                    business_impact = impact_match.group(1).strip() if impact_match else None

                    # Extract recommended action
                    action_match = re.search(r'\*\*Recommended Action\*\*:?\s*(.+?)(?=\*\*|$)', block, re.DOTALL)
                    recommended_action = action_match.group(1).strip() if action_match else None

                    # Create finding
                    finding = Finding(
                        hypothesis_id=finding_id,
                        hypothesis_description=f"Exploration finding {finding_id}",
                        is_interesting=True,  # Assume all reported findings are interesting
                        insight=finding_desc,
                        business_impact=business_impact,
                        recommended_action=recommended_action,
                        confidence=confidence
                    )

                    findings.append(finding)
                    logger.debug(f"Parsed finding {finding_id}: {finding_desc[:100]}")

            # If no structured findings found, create a single finding with the content
            if not findings and content.strip():
                findings.append(Finding(
                    hypothesis_id=1,
                    hypothesis_description="General exploration",
                    is_interesting=True,
                    insight=content[:500] + ("..." if len(content) > 500 else ""),
                    confidence="medium"
                ))

        except Exception as e:
            logger.error(f"Error parsing findings: {e}")
            # Return at least something
            findings.append(Finding(
                hypothesis_id=1,
                hypothesis_description="Exploration completed",
                is_interesting=True,
                insight="Exploration completed but findings could not be parsed. See raw output.",
                confidence="low"
            ))

        return findings


# Factory function to create explorer with metadata
def create_explorer(
    metadata_dict: dict,
    business_context: str
) -> DataExplorer:
    """
    Create DataExplorer from metadata dictionary.

    Args:
        metadata_dict: Dictionary of table metadata
        business_context: Business context string

    Returns:
        DataExplorer: Configured explorer
    """
    schema_metadata = SchemaMetadata(
        tables=metadata_dict,
        business_context=business_context
    )

    return DataExplorer(schema_metadata=schema_metadata)
