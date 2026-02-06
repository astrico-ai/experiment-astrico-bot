"""
Conversation Manager for multi-turn NL-to-insights chat interface.
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..llm.client import LLMClient
from ..db.executor import SQLExecutor
from ..schemas.metadata import MetadataLoader
from ..config import settings
from ..ml.forecasting import forecasting_service
from ..rag.pipeline import get_rag_pipeline

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Single message in a conversation."""
    role: str  # 'user' | 'assistant' | 'tool'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Multi-turn conversation with message history."""
    conversation_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    state: str = "AWAITING_SQL"  # State machine: AWAITING_SQL | FORMATTING_RESPONSE
    last_sql_success: bool = False  # Track if last SQL succeeded
    turn_count: int = 0  # Track turn number for debugging


class ConversationManager:
    """
    Manages conversational NL-to-insights chat with GPT-5.2.

    Features:
    - Multi-turn conversations with context
    - Smart depth: simple answers for simple queries, deep-dive for "why" questions
    - Automatic root-cause analysis (no permission needed)
    - SQL transparency (show queries + execution times)
    - Clarification questions when ambiguous
    - In-memory storage
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        sql_executor: Optional[SQLExecutor] = None,
        schema_metadata: Optional[Dict[str, List[dict]]] = None,
        business_context: Optional[str] = None,
        use_rag: bool = True
    ):
        """
        Initialize ConversationManager.

        Args:
            llm_client: LLM client (defaults to global instance)
            sql_executor: SQL executor (defaults to new instance)
            schema_metadata: Database schema metadata (defaults to loading from files)
            business_context: Business domain knowledge (defaults to loading from file)
            use_rag: Whether to use RAG for schema retrieval (defaults to True)
        """
        from ..llm.client import llm_client as default_llm_client

        self.llm_client = llm_client or default_llm_client
        self.sql_executor = sql_executor or SQLExecutor()

        # Load metadata and business context
        self.schema_metadata = schema_metadata or MetadataLoader.load_all_metadata()
        self.business_context = business_context or MetadataLoader.load_business_context(
            settings.business_context_path
        )

        # In-memory conversation storage
        self.conversations: Dict[str, Conversation] = {}

        # RAG pipeline for schema retrieval
        self.use_rag = use_rag
        if self.use_rag:
            try:
                self.rag_pipeline = get_rag_pipeline()
                logger.info("ConversationManager initialized with RAG enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG pipeline: {e}. Falling back to full schema.")
                self.use_rag = False
                self.rag_pipeline = None
        else:
            self.rag_pipeline = None
            logger.info("ConversationManager initialized without RAG")

    def create_conversation(self) -> str:
        """
        Create a new conversation.

        Returns:
            conversation_id: Unique identifier for the conversation
        """
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = Conversation(conversation_id=conversation_id)
        logger.info(f"Created conversation: {conversation_id}")
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get an existing conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation object or None if not found
        """
        return self.conversations.get(conversation_id)

    def _detect_query_complexity(self, query: str) -> str:
        """
        Detect if a query is analytical (needs insights) or simple (just facts).

        Args:
            query: User's natural language question

        Returns:
            'analytical' or 'simple'
        """
        lower_query = query.lower()
        # Add spaces for word boundary matching
        padded_query = f" {lower_query} "

        # Strong analytical keywords that always indicate need for deeper insights
        strong_analytical_keywords = [
            'why', 'explain', 'analyze', 'investigate',
            'caused', 'reason', 'understand'
        ]

        # Weak analytical keywords that might be simple depending on context
        weak_analytical_keywords = [
            'how', 'trend', 'pattern', 'drop', 'spike', 'increase', 'decrease',
            'compare', 'comparison', 'versus', 'difference',
            'performance', 'anomaly', 'unusual', 'strange'
        ]

        # Simple keywords that indicate straightforward fact-finding
        simple_keywords = [
            'what is', 'show me', 'list', 'display', 'get',
            'total', 'count', 'sum', 'how many', 'how much'
        ]

        # Check for matches (whole word/phrase matching)
        strong_analytical_matches = [kw for kw in strong_analytical_keywords if f" {kw} " in padded_query]
        weak_analytical_matches = [kw for kw in weak_analytical_keywords if f" {kw} " in padded_query]
        simple_matches = [kw for kw in simple_keywords if f" {kw} " in padded_query]

        has_strong_analytical = len(strong_analytical_matches) > 0
        has_weak_analytical = len(weak_analytical_matches) > 0
        has_simple = len(simple_matches) > 0

        # Log detection details
        logger.info(f"[QUERY-DETECTION] Query: '{query}'")
        logger.info(f"[QUERY-DETECTION] Strong analytical matches: {strong_analytical_matches}")
        logger.info(f"[QUERY-DETECTION] Weak analytical matches: {weak_analytical_matches}")
        logger.info(f"[QUERY-DETECTION] Simple matches: {simple_matches}")

        # Priority logic:
        # 1. Strong analytical keywords always win (why, explain, etc.)
        if has_strong_analytical:
            logger.info(f"[QUERY-DETECTION] Result: ANALYTICAL (strong: {strong_analytical_matches})")
            return 'analytical'

        # 2. Simple keywords beat weak analytical (what is X vs Y = simple)
        if has_simple:
            logger.info(f"[QUERY-DETECTION] Result: SIMPLE (matched: {simple_matches})")
            return 'simple'

        # 3. Weak analytical keywords without simple context
        if has_weak_analytical:
            logger.info(f"[QUERY-DETECTION] Result: ANALYTICAL (weak: {weak_analytical_matches})")
            return 'analytical'

        # Default to simple if unsure (changed from analytical)
        logger.info(f"[QUERY-DETECTION] Result: SIMPLE (default - no strong signals)")
        return 'simple'

    async def chat(
        self,
        conversation_id: str,
        user_message: str,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Process a user message and return assistant response.

        Args:
            conversation_id: Conversation identifier
            user_message: User's natural language question
            max_iterations: Maximum tool-calling iterations

        Returns:
            dict with:
                - response: Assistant's natural language response
                - sql_executions: List of SQL queries executed with timing
                - complexity: Question complexity level
                - iterations: Number of iterations used
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")

        # Add user message to history
        conversation.messages.append(Message(
            role="user",
            content=user_message,
            timestamp=datetime.now()
        ))

        # STATE RESET: New user query → full schema context needed
        conversation.state = "AWAITING_SQL"
        conversation.last_sql_success = False
        conversation.turn_count += 1
        logger.info(f"[STATE-TRANSITION] New user query (turn {conversation.turn_count}) → AWAITING_SQL")

        # Analyze question complexity
        complexity = self._analyze_question_complexity(user_message)
        logger.info(f"Question complexity: {complexity}")

        # Build conversation messages for LLM
        messages = self._build_messages(conversation, complexity)

        # Tool-calling loop
        sql_executions = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{max_iterations}")

            try:
                response = self.llm_client.chat(
                    messages=messages,
                    tools=self._get_tool_definitions(),
                    reasoning_effort=settings.reasoning_effort
                )

                message = response.choices[0].message

                # Check if LLM wants to call tools
                if message.tool_calls:
                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    })

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        logger.info(f"Tool call: {tool_name}")

                        try:
                            args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse tool arguments: {e}")
                            tool_result = f"Error: Invalid tool arguments - {e}"
                        else:
                            # Execute the tool
                            tool_result = await self._execute_tool(
                                tool_name, args, conversation, sql_executions
                            )

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result)
                        })

                else:
                    # No more tool calls - LLM has final response
                    final_start = time.time()
                    final_response = message.content or "I couldn't generate a response."

                    logger.info(f"[TIMING] Preparing final response at {final_start:.3f}")

                    # Add assistant message to conversation history
                    conversation.messages.append(Message(
                        role="assistant",
                        content=final_response,
                        timestamp=datetime.now(),
                        metadata={"sql_executions": sql_executions}
                    ))

                    conversation.last_updated = datetime.now()

                    total_time = time.time() - loop_start
                    logger.info(f"[TIMING] Response complete. Total time: {total_time:.3f}s, Iterations: {iteration}, SQL queries: {len(sql_executions)}")

                    return {
                        "response": final_response,
                        "sql_executions": sql_executions,
                        "complexity": complexity,
                        "iterations": iteration
                    }

            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                return {
                    "response": f"I encountered an error: {str(e)}",
                    "sql_executions": sql_executions,
                    "complexity": complexity,
                    "iterations": iteration
                }

        # Max iterations reached
        logger.warning(f"Max iterations reached: {max_iterations}")
        return {
            "response": "I've reached my analysis limit. Please try rephrasing your question.",
            "sql_executions": sql_executions,
            "complexity": complexity,
            "iterations": iteration
        }

    async def chat_stream(
        self,
        conversation_id: str,
        user_message: str,
        max_iterations: int = 10
    ):
        """
        Process a user message with streaming updates.
        Yields events as they happen (approach first, then final answer).

        Args:
            conversation_id: Conversation identifier
            user_message: User's natural language question
            max_iterations: Maximum tool-calling iterations

        Yields:
            dict events with type and content
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")

        # Add user message to history
        conversation.messages.append(Message(
            role="user",
            content=user_message,
            timestamp=datetime.now()
        ))

        # STATE RESET: New user query → full schema context needed
        conversation.state = "AWAITING_SQL"
        conversation.last_sql_success = False
        conversation.turn_count += 1
        logger.info(f"[STATE-TRANSITION] New user query (turn {conversation.turn_count}) → AWAITING_SQL")

        # Analyze question complexity
        complexity = self._analyze_question_complexity(user_message)

        # Build conversation messages for LLM
        messages = self._build_messages(conversation, complexity)

        # Insights generation removed - all queries stream tokens immediately

        # Tool-calling loop
        sql_executions = []
        iteration = 0
        approach_sent = False
        start_time = time.time()

        logger.info(f"[TIMING-STREAM] Starting chat_stream at {start_time}")
        loop_start = time.time()

        logger.info(f"[TIMING] Starting chat_stream at {loop_start}")

        while iteration < max_iterations:
            iteration += 1
            iter_start = time.time()
            logger.info(f"[TIMING] Iteration {iteration} started at {iter_start:.3f}")

            try:
                llm_start = time.time()
                logger.info(f"[TIMING] Calling LLM (streaming) at {llm_start:.3f}")

                # Make ONE streaming call - handle both tool calls and content
                stream = self.llm_client.chat(
                    messages=messages,
                    tools=self._get_tool_definitions(),
                    reasoning_effort=settings.reasoning_effort,
                    stream=True
                )

                # Accumulate response as chunks arrive
                accumulated_tool_calls = []
                accumulated_content = ""
                first_token_time = None
                token_count = 0
                is_final_answer = False  # Track if this is final answer (no tools)

                for chunk in stream:
                    delta = chunk.choices[0].delta

                    # Check for tool calls in this chunk
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            # Extend list if needed
                            while len(accumulated_tool_calls) <= tc_delta.index:
                                accumulated_tool_calls.append({
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })

                            # Accumulate tool call data
                            if tc_delta.id:
                                accumulated_tool_calls[tc_delta.index]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    accumulated_tool_calls[tc_delta.index]["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    accumulated_tool_calls[tc_delta.index]["function"]["arguments"] += tc_delta.function.arguments

                    # Check for content in this chunk
                    if delta.content:
                        token = delta.content
                        accumulated_content += token
                        token_count += 1

                        # Track first token
                        if first_token_time is None:
                            first_token_time = time.time()
                            logger.info(f"[TIMING-STREAM] First token received in {first_token_time - llm_start:.3f}s")
                            is_final_answer = True  # If we're getting content, this is final answer

                        # Stream all tokens immediately
                        yield {
                            "type": "token",
                            "content": token
                        }

                llm_end = time.time()
                logger.info(f"[TIMING] LLM stream completed in {llm_end - llm_start:.3f}s")

                # Process what we accumulated
                if accumulated_tool_calls:
                    # We got tool calls - execute them
                    logger.info(f"[TIMING] Processing {len(accumulated_tool_calls)} tool calls")

                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"]
                                }
                            }
                            for tc in accumulated_tool_calls
                        ]
                    })

                    # Execute each tool call
                    for tool_call in accumulated_tool_calls:
                        tool_name = tool_call["function"]["name"]

                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError as e:
                            tool_result = f"Error: Invalid tool arguments - {e}"
                        else:
                            # STREAM APPROACH IMMEDIATELY
                            if tool_name == "describe_approach" and not approach_sent:
                                approach_start = time.time()
                                steps = args.get("steps", [])
                                approach_text = "📋 **How I'm calculating this:**\n\n"
                                for i, step in enumerate(steps, 1):
                                    approach_text += f"{i}. {step}\n"

                                # Send approach event immediately
                                logger.info(f"[TIMING] Yielding approach event at {approach_start:.3f}")
                                yield {
                                    "type": "approach",
                                    "content": approach_text
                                }
                                approach_sent = True
                                logger.info(f"[TIMING] Approach event yielded in {time.time() - approach_start:.3f}s")

                            # Execute the tool
                            tool_start = time.time()
                            logger.info(f"[TIMING] Executing tool '{tool_name}' at {tool_start:.3f}")
                            tool_result = await self._execute_tool(
                                tool_name, args, conversation, sql_executions
                            )
                            logger.info(f"[TIMING] Tool '{tool_name}' completed in {time.time() - tool_start:.3f}s")

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": str(tool_result)
                        })

                    # OPTIMIZATION: Rebuild system prompt if state changed to FORMATTING_RESPONSE
                    if conversation.state == "FORMATTING_RESPONSE":
                        # Check if we're currently using full prompt (contains database schema)
                        if len(messages) > 0 and messages[0]["role"] == "system" and "DATABASE SCHEMA" in messages[0]["content"]:
                            old_prompt_size = len(messages[0]["content"])
                            logger.info(f"[OPTIMIZATION] State is FORMATTING_RESPONSE - switching to minimal prompt")
                            minimal_prompt = self._build_minimal_prompt()
                            messages[0] = {"role": "system", "content": minimal_prompt}
                            new_prompt_size = len(minimal_prompt)
                            logger.info(f"[OPTIMIZATION] Reduced prompt from {old_prompt_size} chars (~{old_prompt_size//4} tokens) to {new_prompt_size} chars (~{new_prompt_size//4} tokens)")

                elif is_final_answer:
                    # We got final answer content
                    logger.info(f"[TIMING-STREAM] Final answer received. Total tokens: {token_count}")

                    final_response = accumulated_content or "I couldn't generate a response."

                    # Add assistant message to conversation history
                    conversation.messages.append(Message(
                        role="assistant",
                        content=final_response,
                        timestamp=datetime.now(),
                        metadata={"sql_executions": sql_executions}
                    ))

                    conversation.last_updated = datetime.now()

                    # Send answer immediately (insights generation removed)
                    answer_yield_time = time.time()
                    logger.info(f"[TIMING-STREAM] Yielding answer at {answer_yield_time:.3f}")
                    yield {
                        "type": "answer",
                        "content": final_response,
                        "sql_executions": sql_executions,
                        "iterations": iteration
                    }

                    total_time = time.time() - start_time
                    logger.info(f"[TIMING-STREAM] Complete. Total time: {total_time:.3f}s, Iterations: {iteration}")

                    return
                else:
                    # Empty response
                    logger.warning(f"[TIMING] LLM returned empty response")
                    yield {
                        "type": "error",
                        "content": "LLM returned empty response"
                    }
                    return

            except Exception as e:
                logger.error(f"Error in streaming chat loop: {e}")
                yield {
                    "type": "error",
                    "content": str(e)
                }
                return

        # Max iterations reached
        yield {
            "type": "answer",
            "content": "I've reached my analysis limit. Please try rephrasing your question.",
            "sql_executions": sql_executions,
            "iterations": iteration
        }

    async def generate_insights_stream(
        self,
        conversation_id: str,
        original_question: str,
        original_answer: str,
        mode: str = "quick",
        max_queries: int = None,
        max_iterations: int = None,
        timeout_seconds: int = None
    ):
        """
        Generate insights by analyzing the original answer with additional SQL queries.
        Streams insights as they are generated.

        Args:
            conversation_id: Conversation identifier
            original_question: User's original question
            original_answer: The answer that was provided
            mode: Insight mode - "quick" (30-45s, 3 queries) or "deep" (2-3min, 7 queries)
            max_queries: Maximum SQL queries allowed (overrides mode default)
            max_iterations: Maximum tool-calling iterations (overrides mode default)
            timeout_seconds: Maximum time allowed (overrides mode default)

        Yields:
            dict events with insights and progress updates
        """
        # Set defaults based on mode
        if mode == "quick":
            max_queries = max_queries or 5
            max_iterations = max_iterations or 10
            timeout_seconds = timeout_seconds or 60
            reasoning_effort = "low"
        else:  # deep
            max_queries = max_queries or 10
            max_iterations = max_iterations or 15
            timeout_seconds = timeout_seconds or 120
            reasoning_effort = "medium"
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")

        start_time = time.time()
        sql_executions = []
        iteration = 0
        force_synthesis = False  # Flag to disable tools after max queries

        logger.info(f"Starting {mode} insights generation (max_queries={max_queries}, timeout={timeout_seconds}s)")

        # Build insights system prompt
        insights_prompt = self._build_insights_system_prompt(mode=mode)
        messages = [
            {"role": "system", "content": insights_prompt},
            {"role": "user", "content": f"""
**Original Question:** {original_question}

**Answer Provided:** {original_answer}

Now analyze this answer intelligently and provide valuable insights. Execute additional SQL queries as needed to dig deeper.
"""}
        ]

        while iteration < max_iterations:
            iteration += 1

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(f"Insights generation timeout after {elapsed:.1f}s")
                yield {
                    "type": "insights",
                    "content": "⏱️ Analysis time limit reached. Here's what I found so far:\n\n" +
                               "Consider running insights again for deeper analysis."
                }
                return

            # Check query limit - but allow one more LLM call to synthesize
            if len(sql_executions) >= max_queries and not force_synthesis:
                logger.info(f"Max queries reached: {max_queries}. Requesting final synthesis...")
                # Add a message asking LLM to synthesize what it has found
                messages.append({
                    "role": "user",
                    "content": "You've executed the maximum number of queries. Please synthesize the insights from the data you've gathered so far. Do NOT call any more tools."
                })
                force_synthesis = True  # Next call will have no tools
                # Continue to get final response
                continue

            try:
                # If forcing synthesis, remove tools
                tools_to_use = None if force_synthesis else self._get_insights_tool_definitions()

                response = self.llm_client.chat(
                    messages=messages,
                    tools=tools_to_use,
                    reasoning_effort=reasoning_effort
                )

                message = response.choices[0].message

                # Check if LLM wants to call tools
                if message.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    })

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name

                        try:
                            args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            tool_result = f"Error: Invalid tool arguments - {e}"
                        else:
                            # Emit progress event before executing tool
                            if tool_name == "execute_sql" or tool_name == "compare_time_periods":
                                yield {
                                    "type": "progress",
                                    "content": f"Executing query {len(sql_executions) + 1}/{max_queries}...",
                                    "queries_completed": len(sql_executions),
                                    "total_queries": max_queries
                                }

                            # Execute the tool
                            tool_result = await self._execute_tool(
                                tool_name, args, conversation, sql_executions
                            )

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result)
                        })

                else:
                    # No more tool calls - LLM has final insights
                    final_insights = message.content or "No additional insights generated."

                    # Add to conversation history
                    conversation.messages.append(Message(
                        role="assistant",
                        content=f"💡 **Insights:**\n\n{final_insights}",
                        timestamp=datetime.now(),
                        metadata={"sql_executions": sql_executions, "type": "insights"}
                    ))

                    conversation.last_updated = datetime.now()

                    # Send final insights event
                    yield {
                        "type": "insights",
                        "content": final_insights,
                        "sql_executions": sql_executions,
                        "iterations": iteration
                    }

                    logger.info(f"Insights generated. Iterations: {iteration}, SQL queries: {len(sql_executions)}")
                    return

            except Exception as e:
                logger.error(f"Error in insights generation: {e}")
                yield {
                    "type": "error",
                    "content": str(e)
                }
                return

        # Max iterations reached - force synthesis
        logger.info(f"Max iterations reached: {max_iterations}. Forcing final synthesis...")

        # Add message asking LLM to synthesize without more queries
        messages.append({
            "role": "user",
            "content": "You've reached the iteration limit. Please synthesize the insights from the data you've gathered so far. Do NOT call any more tools - just provide your analysis based on what you've learned."
        })

        try:
            # Final call WITHOUT tools to force synthesis
            response = self.llm_client.chat(
                messages=messages,
                tools=None,  # No tools - just synthesize
                reasoning_effort=reasoning_effort
            )

            final_insights = response.choices[0].message.content or "No insights generated."

            yield {
                "type": "insights",
                "content": final_insights,
                "sql_executions": sql_executions,
                "iterations": iteration
            }
        except Exception as e:
            logger.error(f"Failed to synthesize insights: {e}")
            yield {
                "type": "insights",
                "content": "📊 Analysis complete with available data.",
                "sql_executions": sql_executions,
                "iterations": iteration
            }

    def _should_use_minimal_context(self, conversation: Conversation) -> bool:
        """
        Production-ready decision logic for context switching.

        SAFE CONDITIONS for minimal context (ALL must be true):
        1. State is FORMATTING_RESPONSE (not AWAITING_SQL)
        2. Last SQL execution succeeded (not failed)
        3. No new user message since state change

        Conservative: When in doubt, use FULL context.

        Returns:
            True if safe to use minimal context, False otherwise
        """
        # Safety check: Default to full context
        if conversation.state != "FORMATTING_RESPONSE":
            logger.info(f"[CONTEXT-DECISION] State={conversation.state} → FULL context")
            return False

        if not conversation.last_sql_success:
            logger.info(f"[CONTEXT-DECISION] Last SQL failed → FULL context")
            return False

        # Check if there's a new user message after we entered FORMATTING_RESPONSE state
        recent_user_messages = [m for m in conversation.messages[-5:] if m.role == "user"]
        if len(recent_user_messages) > 1:
            # Multiple user messages means follow-up query
            logger.info(f"[CONTEXT-DECISION] New user query detected → FULL context")
            conversation.state = "AWAITING_SQL"  # Reset state
            return False

        # All safety checks passed
        logger.info(f"[CONTEXT-DECISION] Safe to use MINIMAL context (formatting response)")
        return True

    def _build_messages(self, conversation: Conversation, complexity: str) -> List[Dict]:
        """
        Build message list for LLM including system prompt and history.
        Uses explicit state machine for production-ready context switching.

        Args:
            conversation: Conversation object with state tracking
            complexity: Question complexity level

        Returns:
            List of message dicts
        """
        logger.info(f"[STATE-MACHINE] Turn {conversation.turn_count}, State: {conversation.state}")

        # Production-ready decision: Should we use minimal context?
        use_minimal = self._should_use_minimal_context(conversation)

        if use_minimal:
            # SAFE: We have data, just formatting response
            logger.info(f"[CONTEXT-SWITCH] Using MINIMAL prompt (Turn {conversation.turn_count})")
            logger.info(f"[TOKEN-SAVINGS] Saving ~3,250 tokens on this turn")
            system_prompt = self._build_minimal_prompt()
        else:
            # NEED SCHEMA: Generating SQL or recovering from error
            logger.info(f"[CONTEXT-SWITCH] Using FULL prompt with RAG (Turn {conversation.turn_count})")

            # Get latest user message for RAG retrieval
            user_messages = [msg for msg in conversation.messages if msg.role == "user"]
            latest_user_query = user_messages[-1].content if user_messages else ""

            # Retrieve RAG context if enabled
            rag_context = None
            if latest_user_query:
                logger.info(f"[RAG-RETRIEVAL] Retrieving context for query: {latest_user_query[:100]}...")
                rag_context = self._get_rag_context(latest_user_query)
                if rag_context:
                    logger.info(f"[RAG-RETRIEVAL] Success! Retrieved {len(rag_context['metadata']['retrieved_tables'])} tables, {rag_context['metadata']['retrieved_columns_count']} columns")
                else:
                    logger.warning(f"[RAG-RETRIEVAL] Failed - will use full schema")

            system_prompt = self._build_system_prompt(complexity, rag_context)

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add last 50 messages from conversation history
        recent_messages = conversation.messages[-50:]
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        return messages

    def _get_rag_context(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get RAG-retrieved context for a query.

        Args:
            query: User's natural language query

        Returns:
            RAG context dict with minimal schema information
        """
        if not self.use_rag or not self.rag_pipeline:
            return None

        try:
            rag_result = self.rag_pipeline.process_query(query)
            logger.info(f"[RAG] Retrieved context: {rag_result['metadata']['estimated_tokens']} tokens")
            return rag_result
        except Exception as e:
            logger.error(f"[RAG] Error retrieving context: {e}", exc_info=True)
            return None

    def _build_minimal_prompt(self) -> str:
        """
        Build minimal system prompt for response formatting (when we already have data).
        This is used in Turn 2+ when schema context is not needed.
        """
        # Get current date context
        today = datetime.now()
        latest_data_date = self._get_latest_data_date()
        data_date = latest_data_date if latest_data_date else today

        # Ensure datetime object
        if data_date and not isinstance(data_date, datetime):
            data_date = datetime.combine(data_date, datetime.min.time())

        fy_year = self._get_current_fy_year(data_date)
        data_lag_days = (today - data_date).days if data_date else 0

        prompt = f"""You are a helpful data analyst assistant for a B2B lubricant distribution company.

**CURRENT DATE CONTEXT:**
- Latest data available: {data_date.strftime('%Y-%m-%d')} ({data_lag_days} days lag)
- Current Financial Year: FY{fy_year % 100}

**YOUR TASK:**
Format the query results conversationally and clearly.

**FORMATTING RULES:**
- Present volume in KL (Kilolitres)
  * If data is in Liters, it should already be converted to KL in SQL
  * Format: "X,XXX.XX KL" (e.g., "1,234.56 KL")
- Use friendly table names:
  * "Sales Data" not "sales_invoices_veedol"
  * "Customer Master" not "customer_master_veedol"
  * "Product Master" not "material_master_veedol"
- Present numbers with proper formatting and context
- Be concise but complete
- Include relevant insights when data shows interesting patterns

**RESPONSE FORMAT:**
📊 **Answer:**
[Direct answer with numbers and context]

💡 **Key Insights:** (only if relevant)
- [Insight 1]
- [Insight 2]

DO NOT repeat the calculation approach - it was already shown to the user."""

        logger.info(f"[SYSTEM-PROMPT] Using MINIMAL prompt: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        return prompt

    def _build_system_prompt(self, complexity: str, rag_context: Optional[Dict] = None) -> str:
        """
        Build system prompt for conversational assistant.

        Args:
            complexity: Question complexity level

        Returns:
            System prompt string
        """
        # Format schema information - use RAG if available, otherwise full schema
        if rag_context:
            # Use RAG-retrieved minimal schema
            schema_str = rag_context['context']
            sql_rules_str = rag_context['sql_rules']
            logger.info(f"[RAG] Using RAG context: {len(schema_str)} chars")
        else:
            # Fallback to full schema (old behavior)
            schema_json = {}
            for table_name, columns in self.schema_metadata.items():
                # Convert list back to dict format for clearer structure
                table_schema = {}
                for col in columns:
                    table_schema[col['name']] = {
                        "type": col['type'],
                        "unit": col.get('unit'),
                        "description": col.get('description', '')
                    }
                schema_json[table_name] = table_schema

            # Format as pretty JSON for LLM readability
            schema_str = json.dumps(schema_json, indent=2, ensure_ascii=False)
            sql_rules_str = ""  # Will use default rules below
            logger.info(f"[RAG] Using full schema: {len(schema_str)} chars")

        # Complexity-specific guidance
        if complexity == "deep_dive":
            complexity_guidance = """
This is a "WHY" question requiring deep investigation. You should:
- Execute 5-8 investigative queries to understand root causes
- Check multiple dimensions (time trends, segments, comparisons)
- Provide quantified root-cause analysis
- No need to ask permission for deep-dive - just do it
"""
        elif complexity == "simple":
            complexity_guidance = """
This is a straightforward question. You should:
- Execute 1-2 direct queries
- Provide a clear, concise answer
- Include relevant context if helpful
"""
        else:
            complexity_guidance = """
This question may require moderate investigation. You should:
- Execute 2-4 queries as needed
- Provide complete answer with supporting details
- Balance depth with conciseness
"""

        # Get current date context
        today = datetime.now()

        # Get latest data date from database (data may not be up to current date)
        latest_data_date = self._get_latest_data_date()
        data_date = latest_data_date if latest_data_date else today

        # Ensure both are datetime objects for comparison
        if data_date and not isinstance(data_date, datetime):
            data_date = datetime.combine(data_date, datetime.min.time())

        fy_year = self._get_current_fy_year(data_date)
        current_quarter = self._get_current_fy_quarter(data_date)
        fy_start_date = f"{fy_year}-04-01"
        fy_end_date = f"{fy_year + 1}-04-01"
        last_month_range = self._get_last_month_range(data_date)
        last_quarter_range = self._get_last_quarter_range(data_date)

        data_lag_days = (today - data_date).days if data_date else 0

        # Default SQL rules when RAG is not used
        default_sql_rules = """CRITICAL: Use the EXACT column names as they appear in the JSON keys above.
Example: For revenue, the column is "nsv1" - use this exact name in your SQL queries."""

        prompt = f"""You are a helpful data analyst assistant for a B2B lubricant distribution company.

**CURRENT DATE AND TIME CONTEXT:**
- Today's date: {today.strftime('%Y-%m-%d')} ({today.strftime('%B %d, %Y')})
- **Latest data available: {data_date.strftime('%Y-%m-%d')}** ({data_lag_days} days lag)
- Current Financial Year: FY{fy_year % 100} (runs from April 1, {fy_year} to March 31, {fy_year + 1})
- Current FY Quarter: Q{current_quarter} FY{fy_year % 100}

**CRITICAL: DATA CUTOFF**
- All queries must use '{data_date.strftime('%Y-%m-%d')}' as the end date, NOT today's date
- When user asks for "current" or "YTD", use data available up to {data_date.strftime('%Y-%m-%d')}
- Example: YTD query should be WHERE billing_date >= '{fy_start_date}' AND billing_date <= '{data_date.strftime('%Y-%m-%d')}'

**RELATIVE DATE REFERENCE GUIDE:**
When user says:
- "this year" or "YTD" → FY{fy_year % 100} = '{fy_start_date}' to '{data_date.strftime('%Y-%m-%d')}'
- "last year" → FY{(fy_year-1) % 100} = '{fy_year-1}-04-01' to '{fy_year}-04-01'
- "last month" → {last_month_range}
- "last quarter" → {last_quarter_range}
- "current month" → '{data_date.strftime('%Y-%m-01')}' to '{data_date.strftime('%Y-%m-%d')}'

**IMPORTANT:** Always assume Financial Year (FY) unless user explicitly says "calendar year" or "CY"

**FINANCIAL YEAR RULES:**
- FY runs April to March (Indian FY)
- FY 2024 = April 1, 2023 to March 31, 2024
- FY 2023 = April 1, 2022 to March 31, 2023
- Q1 = Apr-Jun, Q2 = Jul-Sep, Q3 = Oct-Dec, Q4 = Jan-Mar
- When user mentions "FY24" or "FY 2024", calculate: April 1, 2023 to March 31, 2024
- When user mentions "Q2 FY24", calculate: July 1, 2023 to September 30, 2023

BUSINESS CONTEXT:
{self.business_context}

{"DATABASE SCHEMA (RAG-Retrieved Relevant Schemas):" if rag_context else "DATABASE SCHEMA (ClickHouse - use EXACT column names from this JSON):"}
{"" if rag_context else "```json"}
{schema_str}
{"" if rag_context else "```"}

{sql_rules_str if rag_context else default_sql_rules}

YOUR ROLE:
- Answer user questions conversationally based on data
- Use business-friendly language (avoid technical jargon)
- Provide quantified insights (numbers, percentages)
- Build trust through transparency

{complexity_guidance}

TOOLS AVAILABLE:
1. describe_approach - Explain calculation steps to user (call together with execute_sql)
2. execute_sql - Query the database (call together with describe_approach to save time)
3. forecast_metric - Generate time series forecasts for future predictions (auto-handles seasonality)
4. analyze_result - Trigger deep-dive investigation (use for complex analysis)
5. search_history - Search conversation history for context
6. validate_value - Validate and correct user input values against database values (handles typos and variations)

**VALUE VALIDATION (IMPORTANT - Use BEFORE SQL):**
When user mentions specific categorical values (states, channels, product segments, etc.), call validate_value FIRST to correct typos:
- User says "MAHARASHTRA" → validate_value("customer_master_veedol", "state", "MAHARASHTRA") → corrects to "Maharashtra"
- User says "aftermarket" → validate_value("customer_master_veedol", "customer_group_report", "aftermarket") → corrects to "AFTER MARKET"
- User says "pcmo" → validate_value("material_master_veedol", "segment_order", "pcmo") → corrects to "PCMO"

Indexed columns for validation:
- customer_master_veedol: state, region, customer_group_report, distribution_channel
- material_master_veedol: segment_order, product_category, product_vertical, pack_type, base_unit_of_measure, company_group
- sales_invoices_veedol: distribution_channel, item_category, billing_code, billing_type
- budget_data_veedol: channel

Use the corrected values in WHERE clauses. This prevents empty results from typos/case mismatches.

IMPORTANT: Call describe_approach AND execute_sql TOGETHER in your first response to parallelize and reduce latency.

**FORECASTING CAPABILITIES:**
When user asks about future predictions, use forecast_metric tool:
- "What will volume be next month?" → forecast_metric(metric="volume", periods_ahead=1, granularity="month")
- "Are we on track to hit target?" → Get current + forecast future + compare to target
- "Forecast by region" → forecast_metric(..., breakdown_by="region")
- Automatically provides confidence intervals and seasonality adjustments
- Shows goal tracking if target value is provided

**CLARIFICATION QUESTIONS:**
If the user's question is ambiguous, just ask for clarification directly in your response - NO TOOL NEEDED.
Example: "I need clarification: Do you mean FY24 (Apr 2023-Mar 2024) or Calendar Year 2024?"

**TABLE NAME ALIASES** (use these in your descriptions, NOT technical names):
- sales_invoices_veedol → "Sales Data"
- customer_master_veedol → "Customer Master"
- material_master_veedol → "Product Master"

IMPORTANT RULES:
- Only use SELECT queries (no modifications allowed)
- **CRITICAL: NO SQL COMMENTS ALLOWED**
  * DO NOT use -- comments in SQL
  * DO NOT use /* */ comments in SQL
  * Write clean SQL without any comments
- **CRITICAL: ONE SQL STATEMENT ONLY**
  * DO NOT use semicolons (;) to separate multiple statements
  * Execute ONE query per execute_sql call
  * If you need to check multiple things, call execute_sql multiple times
  * Multiple statements are blocked for security
  * WRONG: SELECT * FROM a; SELECT * FROM b;
  * RIGHT: Call execute_sql twice, once for each query
- **CRITICAL: UNIT CONVERSION**
  * ALWAYS check the "unit" field in column metadata before calculations
  * When comparing/aggregating columns, they MUST be in the SAME unit
  * Convert units in SQL before comparison (e.g., Liters ÷ 1000 = Kilolitres, Grams ÷ 1000 = Kilograms)
  * Common conversions: L→KL (÷1000), g→kg (÷1000), m→km (÷1000), cm→m (÷100)
  * When presenting results, use the unit specified in metadata or convert to standard unit
- Prioritize volume metrics over value metrics (unless user asks for revenue/value)
- **CRITICAL: ALL VOLUME METRICS MUST BE NORMALIZED TO KL (KILOLITRES)**
  * Always present volume in KL, never in raw units
  * If volume is stored in liters, divide by 1000 to get KL
  * Format: "X,XXX.XX KL" (e.g., "1,234.56 KL")
  * Always include "KL" unit in your response
- **ClickHouse ROUND function:**
  * Use: ROUND(expression, 2)
  * For division: ROUND(a / b, 2) works directly
  * No type casting needed for ROUND
  * Example: ROUND(SUM(volume) / 1000, 2) for KL conversion
- Use contribution1 as the primary profitability metric
- Join tables using customer_number and material_number
- When ambiguous (e.g., "top customers"), ask for clarification
- Reference previous conversation context when relevant
- Be concise but complete - users value your time

⚠️ **CRITICAL: UNIT AWARENESS - READ BEFORE EVERY QUERY** ⚠️

Different tables store the SAME metric in DIFFERENT units. You MUST check units before writing SQL.

Common Unit Differences:
- sales_invoices_veedol.volume → Stored in LITERS → Must ÷1000 to convert to KL
- budget_data_veedol.volume → Stored in KILOLITRES → Already in KL, NO conversion

⚠️ WRONG (double conversion error):
  SELECT SUM(s.volume / 1000), SUM(b.volume / 1000)  -- ✗ Budget wrongly divided!

✓ CORRECT (respects different units):
  SELECT SUM(s.volume / 1000), SUM(b.volume)  -- ✓ Only sales converted

BEFORE writing SQL:
1. List all columns you'll use
2. Check "unit" field for EACH column in schema above
3. Apply conversion ONLY where needed
4. Declare your unit checks in describe_approach (first step)

**CRITICAL: WORKFLOW**

For EVERY data question, follow this exact workflow:

STEP 0: MANDATORY UNIT CHECK (do this mentally before any tool calls)
- Identify ALL columns you will use in your SQL query
- Check the "unit" field for EACH column in the schema above
- Determine which columns need unit conversion
- Plan your conversion strategy

Example mental checklist:
  ✓ sales_invoices_veedol.volume → unit: Liters → NEEDS CONVERSION: ÷1000 to get KL
  ✓ budget_data_veedol.volume → unit: Kilolitre → NO CONVERSION NEEDED (already KL)
  ✓ Both columns will be in KL after conversion for valid comparison

STEP 1: Call describe_approach AND execute_sql TOGETHER in the same iteration (parallel execution)
- describe_approach: Provide 3-5 simple steps explaining what you'll do
  * FIRST STEP MUST declare unit checks: "Checking units: [column]=Unit (conversion plan)"
  * Use friendly table names (Sales Data, Customer Master, Product Master)
  * Be specific about periods, filters, calculations
  * Example: "Checking units: sales volume=Liters (÷1000 to KL), budget volume=KL (no conversion)"
- execute_sql: Run the SQL query to get the data
- IMPORTANT: Call BOTH tools in your first response to save time
- The approach is IMMEDIATELY shown to user, so DO NOT repeat it in your final response

STEP 2: Format final response - ONLY the answer (approach already shown):

📊 **Answer:**
[Direct answer with numbers and context - use friendly table names]

💡 **Key Insights:** (optional, for complex queries only)
- [Insight 1]
- [Insight 2]

CRITICAL: DO NOT include "How I calculated this" section in final response - it was already shown to user via describe_approach tool.

EXAMPLE WORKFLOW:
User: "What is volume for current FY?"

Step 1 - Call BOTH tools in parallel (single response with multiple tool calls):
  Tool 1: describe_approach with steps:
    [
      "Checking units: sales volume=Liters (÷1000 to convert to KL)",
      "Identifying current Financial Year (FY24: April 1, 2023 to March 31, 2024)",
      "Retrieving all sales transactions from Sales Data",
      "Summing volume and converting to Kilolitres (KL)"
    ]
  Tool 2: execute_sql with query:
    SELECT SUM(volume / 1000) AS volume_kl FROM sales_invoices_veedol
    WHERE billing_date >= '2023-04-01' AND billing_date < '2024-04-01'

Step 2 - Final response (DO NOT repeat approach - already shown):
📊 **Answer:**
Total volume for FY24 is 1,234.57 KL across all regions and channels.

This represents sales from 6,535 active customers through 738,466 invoice line items from Sales Data.

EXAMPLE WORKFLOW 2 (Multiple units - Budget vs Sales):
User: "Compare sales vs budget volume for FY25"

Step 1 - Call BOTH tools in parallel:
  Tool 1: describe_approach with steps:
    [
      "Checking units: sales volume=Liters (÷1000 to KL), budget volume=Kilolitre (no conversion needed)",
      "Identifying FY25 period (April 1, 2025 to March 31, 2026)",
      "Retrieving actual sales from Sales Data and converting to KL",
      "Retrieving budget targets from Budget Data (already in KL)",
      "Calculating achievement percentage"
    ]
  Tool 2: execute_sql with query:
    SELECT
      SUM(s.volume / 1000) as actual_kl,  -- Sales in Liters, convert to KL
      SUM(b.volume) as budget_kl,         -- Budget already in KL, no conversion
      ROUND((SUM(s.volume / 1000) / SUM(b.volume) * 100)::numeric, 2) as achievement_pct
    FROM sales_invoices_veedol s
    JOIN budget_data_veedol b ON s.material_number = b.material_number
    WHERE s.billing_date >= '2025-04-01' AND b.date >= '2025-04-01'

Step 2 - Final response:
📊 **Answer:**
FY25 YTD: Achieved 89.4% of budget target (1,234 KL actual vs 1,381 KL budget).

(Note: The approach was already shown to the user via describe_approach tool, so we only send the answer here)"""

        # Log final system prompt details
        final_prompt_length = len(prompt)
        estimated_tokens = final_prompt_length // 4
        logger.info(f"[SYSTEM-PROMPT] Type: FULL (SQL generation)")
        logger.info(f"[SYSTEM-PROMPT] Total length: {final_prompt_length} chars (~{estimated_tokens} tokens)")
        logger.info(f"[SYSTEM-PROMPT] RAG used: {rag_context is not None}")
        if rag_context:
            logger.info(f"[SYSTEM-PROMPT] RAG context: {len(schema_str)} chars")
            logger.info(f"[SYSTEM-PROMPT] Retrieved tables: {rag_context['metadata']['retrieved_tables']}")
            logger.info(f"[SYSTEM-PROMPT] Retrieved columns: {rag_context['metadata']['retrieved_columns_count']}")

            # Calculate savings vs full schema
            full_schema_size = sum(len(json.dumps(cols)) for cols in self.schema_metadata.values())
            savings_pct = ((full_schema_size - len(schema_str)) / full_schema_size * 100)
            logger.info(f"[SYSTEM-PROMPT] Token savings vs full schema: {savings_pct:.1f}%")

        return prompt

    def _get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for LLM tool-calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "describe_approach",
                    "description": "ALWAYS call this FIRST before executing any SQL. Describe the calculation approach in 3-5 simple steps so user understands what you're doing. This gives immediate feedback while queries execute. CRITICAL: First step MUST declare unit checks for all columns you'll use.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of 3-5 simple steps in plain English describing how you'll calculate the answer. FIRST STEP MUST declare unit checks: 'Checking units: [column]=Unit (conversion plan)'. Use friendly table names (Sales Data, Customer Master, Product Master). Example: ['Checking units: sales volume=Liters (÷1000 to KL), budget volume=KL (no conversion)', 'Identifying current FY period', 'Retrieving sales transactions', 'Summing total volume in KL']"
                            }
                        },
                        "required": ["steps"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute a SELECT SQL query against the database. Returns query results with execution time. Use this AFTER describing approach.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The SQL SELECT query to execute"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation of what this query is checking (e.g., 'Finding total revenue for 2024')"
                            }
                        },
                        "required": ["query", "explanation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_result",
                    "description": "Trigger deeper investigation of a result. Use this when you need to understand WHY something happened. The system will automatically execute follow-up investigative queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "finding": {
                                "type": "string",
                                "description": "The finding that needs investigation (e.g., 'Margins dropped 4.3% in Q4')"
                            },
                            "dimensions_to_check": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Dimensions to investigate (e.g., ['discounts', 'product_mix', 'regional_performance'])"
                            }
                        },
                        "required": ["finding", "dimensions_to_check"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_history",
                    "description": "Search conversation history for relevant previous answers. Use this to maintain context in multi-turn conversations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keywords to search for in conversation history"
                            }
                        },
                        "required": ["keywords"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_schema",
                    "description": "Search database schema for columns matching a keyword. Use this when SQL fails with 'column does not exist' to find the correct column name. Example: if 'product_segment' fails, search for 'segment' to find 'segment_order'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_term": {
                                "type": "string",
                                "description": "Keyword to search for in column names and descriptions (e.g., 'segment', 'category', 'region')"
                            },
                            "table_name": {
                                "type": "string",
                                "description": "Optional: limit search to specific table (e.g., 'material_master_veedol')"
                            }
                        },
                        "required": ["search_term"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_value",
                    "description": "Validate and correct user input values against actual database values. Use this BEFORE generating SQL when user mentions specific values (states, channels, products, etc.) to handle typos and variations. Example: 'MAHARASHTRA' → 'Maharashtra', 'aftermarket' → 'AFTER MARKET'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table name (e.g., 'customer_master_veedol', 'material_master_veedol')"
                            },
                            "column": {
                                "type": "string",
                                "description": "Column name (e.g., 'state', 'customer_group_report', 'segment_order')"
                            },
                            "user_input": {
                                "type": "string",
                                "description": "User's input value that needs validation (e.g., 'MAHARASHTRA', 'pcmo', 'aftermarket')"
                            }
                        },
                        "required": ["table", "column", "user_input"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_time_periods",
                    "description": "Compare metrics across time periods (YoY, MoM, QoQ) with FY-aware date logic. Use this for consistent period comparisons.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "description": "Metric to compare: 'revenue', 'volume', 'invoices', 'customers', or column name",
                                "enum": ["revenue", "volume", "invoices", "customers"]
                            },
                            "period_type": {
                                "type": "string",
                                "description": "Type of comparison",
                                "enum": ["year_over_year", "month_over_month", "quarter_over_quarter"]
                            },
                            "filters": {
                                "type": "object",
                                "description": "Optional filters (region, product, table, date_column)",
                                "properties": {
                                    "region": {"type": "string"},
                                    "product": {"type": "string"},
                                    "table": {"type": "string"},
                                    "date_column": {"type": "string"}
                                }
                            }
                        },
                        "required": ["metric", "period_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "forecast_metric",
                    "description": "Generate time series forecast for a metric (volume, revenue, etc.). Use this when user asks about future predictions, 'next month/quarter', 'will we hit target', or trend projections. Automatically handles seasonality and provides confidence intervals.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "description": "Metric to forecast: 'volume', 'revenue', 'contribution1', etc.",
                                "enum": ["volume", "revenue", "contribution1", "nsv1", "gsv"]
                            },
                            "periods_ahead": {
                                "type": "integer",
                                "description": "Number of periods to forecast (e.g., 3 for next 3 months)",
                                "minimum": 1,
                                "maximum": 24
                            },
                            "granularity": {
                                "type": "string",
                                "description": "Time granularity for forecast",
                                "enum": ["day", "week", "month", "quarter"],
                                "default": "month"
                            },
                            "breakdown_by": {
                                "type": "string",
                                "description": "Optional: breakdown forecast by dimension (region, channel, material_group)",
                                "enum": ["region", "channel", "material_group"]
                            },
                            "target_value": {
                                "type": "number",
                                "description": "Optional: target/goal value for 'on track' analysis"
                            }
                        },
                        "required": ["metric", "periods_ahead"]
                    }
                }
            }
        ]

    def _get_insights_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for insights generation (subset of main tools)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute a SELECT SQL query to dig deeper into the data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The SQL SELECT query to execute"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation of what this query is checking"
                            }
                        },
                        "required": ["query", "explanation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_time_periods",
                    "description": "Compare metrics across time periods with FY-aware logic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "enum": ["revenue", "volume", "invoices", "customers"]
                            },
                            "period_type": {
                                "type": "string",
                                "enum": ["year_over_year", "month_over_month", "quarter_over_quarter"]
                            },
                            "filters": {
                                "type": "object",
                                "properties": {
                                    "region": {"type": "string"},
                                    "product": {"type": "string"}
                                }
                            }
                        },
                        "required": ["metric", "period_type"]
                    }
                }
            }
        ]

    def _build_insights_system_prompt(self, mode: str = "quick") -> str:
        """Build system prompt for insights generation."""
        # Format schema information as full JSON - no limits
        schema_json = {}
        for table_name, columns in self.schema_metadata.items():
            table_schema = {}
            for col in columns:  # All columns, no limit
                table_schema[col['name']] = {
                    "type": col['type'],
                    "unit": col.get('unit'),
                    "description": col.get('description', '')
                }
            schema_json[table_name] = table_schema

        schema_str = json.dumps(schema_json, indent=2, ensure_ascii=False)

        today = datetime.now()
        # Get latest data date (data may not be current)
        latest_data_date = self._get_latest_data_date()
        data_date = latest_data_date if latest_data_date else today

        # Ensure datetime object
        if data_date and not isinstance(data_date, datetime):
            data_date = datetime.combine(data_date, datetime.min.time())

        fy_year = self._get_current_fy_year(data_date)

        # Mode-specific guidance
        if mode == "quick":
            mode_guidance = """
**MODE: Quick Insights (45-60 seconds)**
- Execute up to 5 targeted queries
- Focus on high-level patterns and immediate red flags
- Provide concise, actionable insights
- Skip deep root-cause analysis"""
        else:  # deep
            mode_guidance = """
**MODE: Deep Analysis (90-120 seconds)**
- Execute up to 10 investigative queries
- Dig deep into root causes and drivers
- Analyze multiple dimensions (time, region, product, customer)
- Provide comprehensive insights with detailed breakdowns"""

        return f"""You are an expert business analyst specializing in B2B sales data insights.

{mode_guidance}

**CURRENT DATE CONTEXT:**
- Today: {today.strftime('%Y-%m-%d')}
- **Latest data available: {data_date.strftime('%Y-%m-%d')}**
- Current FY: FY{fy_year % 100} (April {fy_year} - March {fy_year + 1})

**CRITICAL:** Use '{data_date.strftime('%Y-%m-%d')}' as the latest date in all queries, NOT today's date.

**YOUR TASK:**
You've been given a question and an answer. Now generate valuable insights by:
1. Understanding what additional analysis would be most valuable
2. Executing SQL queries to dig deeper into the data
3. Finding patterns, drivers, anomalies, trends as relevant to the context
4. Providing actionable insights

**BUSINESS CONTEXT:**
{self.business_context}

**DATABASE SCHEMA (ClickHouse - use EXACT column names from JSON):**
```json
{schema_str}
```

CRITICAL: Use EXACT column names from the JSON above in your SQL queries.
Example: Revenue column is "nsv1", NOT "nsv" or "revenue".

**ANALYSIS GUIDELINES:**

Be context-aware and intelligent:
- If answer shows DECLINE → investigate drivers (which products/regions/customers?)
- If answer shows GROWTH → identify what's working (which segments are driving it?)
- If answer shows DISTRIBUTION → find key contributors (80/20 rule, top performers)
- If answer shows TRENDS → analyze trajectory and seasonality
- If answer shows ANOMALIES → investigate root causes

**TOOLS AVAILABLE:**
1. execute_sql - Run custom SQL queries for deeper analysis
2. compare_time_periods - Quick YoY, MoM, QoQ comparisons

**OUTPUT FORMAT:**
Provide insights as concise bullet points or short paragraphs:

💡 **Key Insights:**

- [Insight 1 with numbers]
- [Insight 2 with drivers/reasons]
- [Insight 3 with actionable recommendation]

**IMPORTANT:**
- **NO SQL COMMENTS ALLOWED** - Write clean SQL without -- or /* */ comments
- **ONE SQL STATEMENT ONLY** - No semicolons to separate statements. Call execute_sql multiple times if needed.
- **ALL VOLUME METRICS MUST BE NORMALIZED TO KL (KILOLITRES)**
  * Always present volume in KL, never in raw units
  * If volume is in liters, divide by 1000 to get KL
  * Format: "X,XXX.XX KL" (e.g., "1,234.56 KL")
- **ClickHouse ROUND function:**
  * Use: ROUND(expression, 2)
  * For division: ROUND(a / b, 2) works directly
  * No type casting needed
- Execute max 5-7 queries (you have limited time)
- Focus on what's most valuable given the context
- Provide quantified insights with numbers
- Be concise - users value your time
- Don't repeat what's already in the answer - add NEW insights"""

    def _analyze_question_complexity(self, question: str) -> str:
        """
        Analyze question complexity to determine response strategy.

        Args:
            question: User's question

        Returns:
            Complexity level: "simple" | "moderate" | "deep_dive"
        """
        question_lower = question.lower()

        # Deep-dive triggers
        deep_dive_keywords = ["why", "what caused", "reason", "explain", "investigate", "analyze"]
        if any(keyword in question_lower for keyword in deep_dive_keywords):
            return "deep_dive"

        # Simple question patterns
        simple_patterns = ["what is", "how many", "how much", "total", "sum", "count", "list", "show"]
        if any(pattern in question_lower for pattern in simple_patterns):
            return "simple"

        # Default to moderate
        return "moderate"

    def _get_latest_data_date(self) -> Optional[datetime]:
        """
        Query database to find the latest billing_date with data.
        Returns the most recent date where actual data exists.
        """
        try:
            query = """
            SELECT MAX(billing_date) as latest_date
            FROM sales_invoices_veedol
            WHERE billing_date IS NOT NULL
            """
            result = self.sql_executor.execute_query(query)
            rows = result.get("rows", [])

            if rows and rows[0].get("latest_date"):
                latest_date_str = rows[0]["latest_date"]
                # Parse the date string
                if isinstance(latest_date_str, str):
                    latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
                else:
                    latest_date = latest_date_str

                logger.info(f"[DATA-CUTOFF] Latest data date: {latest_date.strftime('%Y-%m-%d')}")
                return latest_date
            else:
                logger.warning("[DATA-CUTOFF] Could not determine latest data date, using current date")
                return None

        except Exception as e:
            logger.error(f"[DATA-CUTOFF] Error getting latest data date: {e}")
            return None

    def _get_current_fy_year(self, date: datetime) -> int:
        """Get current Financial Year year (e.g., 2024 for FY24)."""
        if date.month >= 4:  # Apr-Dec
            return date.year
        else:  # Jan-Mar (still in previous FY)
            return date.year - 1

    def _get_current_fy_quarter(self, date: datetime) -> int:
        """Get current FY quarter (1-4)."""
        month = date.month
        if month >= 4 and month <= 6:
            return 1
        elif month >= 7 and month <= 9:
            return 2
        elif month >= 10 and month <= 12:
            return 3
        else:  # Jan-Mar
            return 4

    def _get_last_month_range(self, today: datetime) -> str:
        """Get last month's date range."""
        last_month_end = today.replace(day=1)
        last_month_start = (last_month_end - timedelta(days=1)).replace(day=1)
        return f"'{last_month_start.strftime('%Y-%m-%d')}' to '{last_month_end.strftime('%Y-%m-%d')}'"

    def _get_last_quarter_range(self, today: datetime) -> str:
        """Get last FY quarter date range."""
        current_q = self._get_current_fy_quarter(today)
        fy_year = self._get_current_fy_year(today)

        # Map current quarter to last quarter's start month
        quarter_starts = {
            1: (fy_year - 1, 1),      # If Q1, last Q was Q4 of prev FY (Jan-Mar)
            2: (fy_year, 4),          # If Q2, last Q was Q1 (Apr-Jun)
            3: (fy_year, 7),          # If Q3, last Q was Q2 (Jul-Sep)
            4: (fy_year, 10)          # If Q4, last Q was Q3 (Oct-Dec)
        }

        start_year, start_month = quarter_starts[current_q]
        start = datetime(start_year, start_month, 1)
        # Add 3 months for end
        end_month = start_month + 3
        end_year = start_year
        if end_month > 12:
            end_month -= 12
            end_year += 1
        end = datetime(end_year, end_month, 1)

        return f"'{start.strftime('%Y-%m-%d')}' to '{end.strftime('%Y-%m-%d')}'"

    async def _execute_tool(
        self,
        tool_name: str,
        args: Dict,
        conversation: Conversation,
        sql_executions: List[Dict]
    ) -> str:
        """
        Execute a tool and return result.
        Also handles state transitions for production-ready context switching.

        Args:
            tool_name: Name of tool to execute
            args: Tool arguments
            conversation: Conversation object
            sql_executions: List to append SQL execution metadata

        Returns:
            Tool result as string
        """
        # Execute tool
        if tool_name == "describe_approach":
            result = await self._describe_approach_tool(args)
        elif tool_name == "execute_sql":
            result = await self._execute_sql_tool(args, sql_executions)
        elif tool_name == "compare_time_periods":
            result = await self._compare_time_periods_tool(args, sql_executions)
        elif tool_name == "forecast_metric":
            result = await self._forecast_metric_tool(args, sql_executions)
        elif tool_name == "analyze_result":
            result = await self._analyze_result_tool(args)
        elif tool_name == "search_history":
            result = await self._search_history_tool(args, conversation)
        elif tool_name == "search_schema":
            result = self._search_schema_tool(args)
        elif tool_name == "validate_value":
            result = self._validate_value_tool(args)
        else:
            result = f"Unknown tool: {tool_name}"

        # STATE TRANSITIONS: Update conversation state based on tool execution
        if tool_name == "execute_sql":
            # Check if SQL succeeded and returned data
            success = "Success" in str(result) or "executed in" in str(result)
            has_data = "rows" in str(result) and "0 rows" not in str(result)
            has_error = "Error" in str(result) or "failed" in str(result).lower()

            if success and has_data and not has_error:
                # SQL succeeded with data → Next turn can use minimal context
                conversation.state = "FORMATTING_RESPONSE"
                conversation.last_sql_success = True
                logger.info(f"[STATE-TRANSITION] execute_sql SUCCESS → FORMATTING_RESPONSE (minimal context on next turn)")
            else:
                # SQL failed or no data → Keep full context for retry/new query
                conversation.state = "AWAITING_SQL"
                conversation.last_sql_success = False
                logger.info(f"[STATE-TRANSITION] execute_sql FAILED/EMPTY → AWAITING_SQL (full context on next turn)")

        elif tool_name in ["analyze_result", "forecast_metric"]:
            # These tools might need follow-up SQL → Reset to AWAITING_SQL
            conversation.state = "AWAITING_SQL"
            logger.info(f"[STATE-TRANSITION] {tool_name} completed → AWAITING_SQL (might need more SQL)")

        # describe_approach and search_history don't change state

        return result

    async def _describe_approach_tool(self, args: Dict) -> str:
        """
        Describe calculation approach to user in simple steps.
        This is called FIRST to give immediate feedback.

        Args:
            args: Tool arguments with 'steps' array

        Returns:
            Formatted approach description
        """
        steps = args.get("steps", [])
        logger.info(f"Describing approach: {len(steps)} steps")

        if not steps:
            return "Proceeding with calculation..."

        approach = "📋 **How I'm calculating this:**\n\n"
        for i, step in enumerate(steps, 1):
            approach += f"{i}. {step}\n"

        approach += "\n✓ Approach confirmed. Executing queries now..."

        return approach

    async def _execute_sql_tool(self, args: Dict, sql_executions: List[Dict]) -> str:
        """
        Execute SQL query and track timing.

        Args:
            args: Tool arguments with 'query' and 'explanation'
            sql_executions: List to append execution metadata

        Returns:
            Formatted query result
        """
        query = args.get("query", "")
        explanation = args.get("explanation", "")

        logger.info(f"[SQL] Executing: {explanation}")
        logger.info(f"[SQL] Query:\n{query}")

        start_time = time.time()
        try:
            result = self.sql_executor.execute_query(query)
            execution_time_ms = int((time.time() - start_time) * 1000)

            row_count = len(result.get("rows", []))

            logger.info(f"[SQL] Success: {row_count} rows in {execution_time_ms}ms")

            # Log first few rows for verification
            if row_count > 0:
                preview_rows = result["rows"][:3]
                logger.info(f"[SQL] Sample data (first 3 rows):\n{json.dumps(preview_rows, indent=2, default=str)}")

            # Record execution metadata
            sql_executions.append({
                "query": query,
                "explanation": explanation,
                "execution_time_ms": execution_time_ms,
                "row_count": row_count,
                "success": True
            })

            # Format result for LLM
            if row_count == 0:
                return f"Query executed successfully in {execution_time_ms}ms but returned no rows."

            # Return first 100 rows to avoid token limits
            rows = result["rows"][:100]
            truncated = len(result["rows"]) > 100

            result_str = f"Query executed in {execution_time_ms}ms. Returned {row_count} rows"
            if truncated:
                result_str += " (showing first 100)"
            result_str += ":\n\n"
            result_str += json.dumps(rows, indent=2, default=str)

            return result_str

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            logger.error(f"SQL execution failed: {error_msg}")

            sql_executions.append({
                "query": query,
                "explanation": explanation,
                "execution_time_ms": execution_time_ms,
                "row_count": 0,
                "success": False,
                "error": error_msg
            })

            return f"Query failed after {execution_time_ms}ms. Error: {error_msg}"

    async def _compare_time_periods_tool(
        self,
        args: Dict,
        sql_executions: List[Dict]
    ) -> str:
        """
        Compare metrics across time periods with FY-aware logic.

        Args:
            args: Tool arguments with 'metric', 'period_type', 'filters'
            sql_executions: List to append SQL execution metadata

        Returns:
            Comparison results
        """
        metric = args.get("metric", "revenue")
        period_type = args.get("period_type", "year_over_year")  # year_over_year, month_over_month, quarter_over_quarter
        filters = args.get("filters", {})

        logger.info(f"Time period comparison: {metric} - {period_type}")

        today = datetime.now()
        fy_year = self._get_current_fy_year(today)

        # Build date ranges based on period type
        if period_type == "year_over_year":
            current_start = f"{fy_year}-04-01"
            current_end = today.strftime('%Y-%m-%d')
            previous_start = f"{fy_year-1}-04-01"
            previous_end = f"{fy_year}-04-01"
            period_label = f"FY{fy_year % 100} YTD vs FY{(fy_year-1) % 100} same period"

        elif period_type == "month_over_month":
            current_month_start = today.replace(day=1).strftime('%Y-%m-%d')
            current_month_end = today.strftime('%Y-%m-%d')
            last_month_end = today.replace(day=1)
            last_month_start = (last_month_end - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
            last_month_end = last_month_end.strftime('%Y-%m-%d')

            current_start = current_month_start
            current_end = current_month_end
            previous_start = last_month_start
            previous_end = last_month_end
            period_label = "Current month vs Last month"

        elif period_type == "quarter_over_quarter":
            current_q = self._get_current_fy_quarter(today)
            # Simplified quarter logic
            quarter_starts = {1: 4, 2: 7, 3: 10, 4: 1}
            current_start_month = quarter_starts[current_q]
            current_start = f"{fy_year if current_q != 4 else fy_year+1}-{current_start_month:02d}-01"
            current_end = today.strftime('%Y-%m-%d')

            prev_q = current_q - 1 if current_q > 1 else 4
            prev_fy = fy_year if current_q > 1 else fy_year - 1
            prev_start_month = quarter_starts[prev_q]
            previous_start = f"{prev_fy if prev_q != 4 else prev_fy+1}-{prev_start_month:02d}-01"
            # Previous quarter end is start of current quarter
            previous_end = current_start
            period_label = f"Q{current_q} FY{fy_year % 100} vs Q{prev_q} FY{prev_fy % 100}"

        else:
            return f"Unknown period_type: {period_type}"

        # Build SQL query
        table_name = filters.get("table", "sales_invoices_veedol")
        date_column = filters.get("date_column", "billing_date")

        # Determine metric SQL
        if metric == "revenue":
            metric_sql = "SUM(nsv1) as value"
        elif metric == "volume":
            metric_sql = "SUM(volume) as value"
        elif metric == "invoices":
            metric_sql = "COUNT(DISTINCT invoice_number) as value"
        elif metric == "customers":
            metric_sql = "COUNT(DISTINCT customer_number) as value"
        else:
            metric_sql = f"SUM({metric}) as value"

        # Additional filter conditions
        where_conditions = []
        if filters.get("region"):
            where_conditions.append(f"region = '{filters['region']}'")
        if filters.get("product"):
            where_conditions.append(f"material_number = '{filters['product']}'")

        where_clause = " AND " + " AND ".join(where_conditions) if where_conditions else ""

        query = f"""
        SELECT
            'Current Period' as period,
            {metric_sql}
        FROM {table_name}
        WHERE {date_column} >= '{current_start}'
          AND {date_column} < '{current_end}'
          {where_clause}

        UNION ALL

        SELECT
            'Previous Period' as period,
            {metric_sql}
        FROM {table_name}
        WHERE {date_column} >= '{previous_start}'
          AND {date_column} < '{previous_end}'
          {where_clause}
        """

        # Execute via SQL tool
        result = await self._execute_sql_tool(
            {"query": query, "explanation": f"Comparing {metric} for {period_label}"},
            sql_executions
        )

        return f"Period comparison ({period_label}):\n{result}"

    async def _forecast_metric_tool(self, args: Dict, sql_executions: List[Dict]) -> str:
        """
        Generate time series forecast for a metric.

        Args:
            args: Tool arguments with metric, periods_ahead, granularity, etc.
            sql_executions: List to append execution metadata

        Returns:
            Formatted forecast results
        """
        metric = args.get("metric", "volume")
        periods_ahead = args.get("periods_ahead", 3)
        granularity = args.get("granularity", "month")
        breakdown_by = args.get("breakdown_by")
        target_value = args.get("target_value")

        logger.info(f"[FORECAST] Forecasting {metric} for {periods_ahead} {granularity}s")

        try:
            # Step 1: Fetch historical data
            table_map = {
                "volume": "sales_invoices_veedol",
                "revenue": "sales_invoices_veedol",
                "nsv1": "sales_invoices_veedol",
                "gsv": "sales_invoices_veedol",
                "contribution1": "sales_invoices_veedol"
            }

            metric_column_map = {
                "volume": "volume",
                "revenue": "nsv1",
                "nsv1": "nsv1",
                "gsv": "gsv",
                "contribution1": "contribution1"
            }

            table_name = table_map.get(metric, "sales_invoices_veedol")
            metric_column = metric_column_map.get(metric, metric)

            # Determine aggregation granularity for historical data
            if granularity == "day":
                date_trunc = "billing_date"
                group_by = "billing_date"
            elif granularity == "week":
                date_trunc = "toStartOfWeek(billing_date)"
                group_by = date_trunc
            elif granularity == "month":
                date_trunc = "toStartOfMonth(billing_date)"
                group_by = date_trunc
            elif granularity == "quarter":
                date_trunc = "toStartOfQuarter(billing_date)"
                group_by = date_trunc
            else:
                date_trunc = "toStartOfMonth(billing_date)"
                group_by = date_trunc

            # Build query based on whether breakdown is requested
            if breakdown_by:
                query = f"""
                SELECT
                    toDate({group_by}) as date,
                    {breakdown_by},
                    SUM({metric_column}) as value
                FROM {table_name}
                WHERE billing_date >= today() - INTERVAL 3 YEAR
                GROUP BY {group_by}, {breakdown_by}
                ORDER BY date, {breakdown_by}
                """
            else:
                query = f"""
                SELECT
                    toDate({group_by}) as date,
                    SUM({metric_column}) as value
                FROM {table_name}
                WHERE billing_date >= today() - INTERVAL 3 YEAR
                GROUP BY {group_by}
                ORDER BY date
                """

            logger.info(f"[FORECAST] Fetching historical data...")

            # Execute query
            start_time = time.time()
            result = self.sql_executor.execute_query(query)
            execution_time_ms = int((time.time() - start_time) * 1000)

            historical_data = result.get("rows", [])
            logger.info(f"[FORECAST] Retrieved {len(historical_data)} historical data points in {execution_time_ms}ms")

            if len(historical_data) < 10:
                return f"❌ Insufficient historical data for forecasting. Found {len(historical_data)} data points, need at least 10. Please ensure there's enough historical data for {metric}."

            # Step 2: Generate forecast
            forecast_result = forecasting_service.forecast_metric(
                historical_data=historical_data,
                periods_ahead=periods_ahead,
                granularity=granularity,
                breakdown_by=breakdown_by,
                metric_name=metric
            )

            if not forecast_result.get("success"):
                error = forecast_result.get("error", "Unknown error")
                return f"❌ Forecasting failed: {error}"

            # Step 3: Format results
            if breakdown_by:
                # Breakdown forecast
                output = f"📈 **Forecast: {metric.upper()} by {breakdown_by.upper()}** (Next {periods_ahead} {granularity}s)\n\n"

                forecasts = forecast_result.get("forecasts", {})
                for dim_value, forecast_data in forecasts.items():
                    output += f"**{dim_value}:**\n"
                    for period in forecast_data[:5]:  # Show first 5 periods
                        output += f"  - {period['date']}: {period['forecast']:.2f} KL "
                        output += f"(Range: {period['lower_bound']:.2f} - {period['upper_bound']:.2f})\n"

                    total = sum([p['forecast'] for p in forecast_data])
                    output += f"  Total: {total:.2f} KL\n\n"

            else:
                # Single forecast
                forecast_data = forecast_result.get("forecast", [])
                total_forecast = forecast_result.get("total_forecast", 0)
                accuracy = forecast_result.get("accuracy_metrics", {})

                output = f"📈 **Forecast: {metric.upper()}** (Next {periods_ahead} {granularity}s)\n\n"

                # Show individual periods
                for period in forecast_data:
                    output += f"**{period['date']}:** {period['forecast']:.2f} KL\n"
                    output += f"  Range: {period['lower_bound']:.2f} - {period['upper_bound']:.2f} KL (80% confidence)\n\n"

                output += f"**Total Forecast:** {total_forecast:.2f} KL\n\n"

                # Add trend analysis to explain the forecast
                trend_info = forecast_result.get("trend_analysis", {})
                if trend_info:
                    yoy_growth = trend_info.get("yoy_growth_pct", 0)
                    recent_trend = trend_info.get("recent_trend", "stable")
                    recent_momentum = trend_info.get("recent_momentum_pct", 0)

                    output += f"📊 **Forecast Basis:**\n"
                    output += f"- Historical YoY Growth: {yoy_growth:+.1f}%\n"
                    output += f"- Recent Trend: {recent_trend.title()} ({recent_momentum:+.1f}%)\n"
                    output += f"- Seasonality: Adjusted for {metric} patterns\n\n"

                    # Explain what this means
                    if abs(yoy_growth) < 2 and recent_trend == "stable":
                        output += f"ℹ️ **Note:** Forecast reflects stable business with minimal growth trend. "
                        output += f"Values are primarily driven by seasonal patterns from previous years.\n\n"
                    elif yoy_growth > 5:
                        output += f"📈 **Note:** Forecast incorporates strong growth trend of {yoy_growth:.1f}% YoY.\n\n"
                    elif yoy_growth < -5:
                        output += f"📉 **Note:** Forecast reflects declining trend of {yoy_growth:.1f}% YoY.\n\n"

                # Add accuracy info
                if "mape" in accuracy:
                    output += f"🎯 **Model Accuracy:** ±{accuracy['mape']:.1f}% (based on {accuracy.get('sample_size', 0)} historical points)\n\n"

                # Step 4: Goal tracking analysis if target provided
                if target_value:
                    # Get current YTD value
                    today = datetime.now()
                    fy_year = self._get_current_fy_year(today)
                    fy_start = f"{fy_year}-04-01"

                    current_query = f"""
                    SELECT SUM({metric_column}) as current_value
                    FROM {table_name}
                    WHERE billing_date >= '{fy_start}'
                      AND billing_date <= today()
                    """

                    current_result = self.sql_executor.execute_query(current_query)
                    current_value = current_result.get("rows", [{}])[0].get("current_value", 0)

                    # Calculate periods
                    months_elapsed = (today.month - 4) % 12 + (12 if today.month < 4 else 0)
                    months_total = 12

                    # Analyze goal tracking
                    goal_analysis = forecasting_service.analyze_goal_tracking(
                        current_value=float(current_value or 0),
                        target_value=float(target_value),
                        forecast_final=float(current_value or 0) + total_forecast,
                        periods_total=months_total,
                        periods_elapsed=months_elapsed,
                        metric_name=metric
                    )

                    # Add goal tracking section
                    status_emoji = {
                        "on_track": "✅",
                        "likely": "🟢",
                        "at_risk": "⚠️",
                        "off_track": "🔴"
                    }

                    output += f"\n{status_emoji.get(goal_analysis['status'], '📊')} **Goal Tracking Analysis:**\n\n"
                    output += f"Current YTD: {goal_analysis['current_value']:.2f} KL\n"
                    output += f"Target: {goal_analysis['target_value']:.2f} KL\n"
                    output += f"Projected Final: {goal_analysis['forecast_final']:.2f} KL\n"
                    output += f"Expected Achievement: {goal_analysis['forecast_achievement_pct']:.1f}%\n\n"
                    output += f"**Status:** {goal_analysis['message']}\n\n"

                    if goal_analysis['status'] in ['at_risk', 'off_track']:
                        output += f"To hit target, you need to average {goal_analysis['required_run_rate']:.2f} KL/{granularity} "
                        output += f"for the remaining {goal_analysis['periods_remaining']} {granularity}s "
                        output += f"(vs. current rate of {goal_analysis['current_run_rate']:.2f} KL/{granularity}).\n"

            return output

        except Exception as e:
            logger.error(f"[FORECAST] Error: {e}")
            return f"❌ Forecasting error: {str(e)}"

    async def _analyze_result_tool(self, args: Dict) -> str:
        """
        Trigger deep-dive analysis (placeholder for MVP).

        Args:
            args: Tool arguments with 'finding' and 'dimensions_to_check'

        Returns:
            Instruction for LLM to proceed with investigation
        """
        finding = args.get("finding", "")
        dimensions = args.get("dimensions_to_check", [])

        logger.info(f"Deep-dive analysis triggered: {finding}")

        return f"""Deep-dive analysis mode activated.

Finding to investigate: {finding}

Suggested dimensions to check: {', '.join(dimensions)}

Proceed with executing investigative queries for each dimension. For each dimension:
1. Execute SQL query to analyze that aspect
2. Interpret the results
3. Quantify the impact

After checking all dimensions, synthesize findings into root-cause analysis."""

    async def _search_history_tool(self, args: Dict, conversation: Conversation) -> str:
        """
        Search conversation history for relevant context.

        Args:
            args: Tool arguments with 'keywords'
            conversation: Conversation object

        Returns:
            Relevant conversation history
        """
        keywords = args.get("keywords", [])
        logger.info(f"Searching history for: {keywords}")

        # Simple keyword matching (can be enhanced with embeddings later)
        relevant_messages = []
        for msg in conversation.messages[-20:]:  # Search last 20 messages
            if msg.role in ["user", "assistant"]:
                content_lower = msg.content.lower()
                if any(keyword.lower() in content_lower for keyword in keywords):
                    relevant_messages.append(f"{msg.role}: {msg.content[:200]}")

        if not relevant_messages:
            return "No relevant conversation history found."

        return "Relevant conversation history:\n\n" + "\n\n".join(relevant_messages)

    def _search_schema_tool(self, args: Dict) -> str:
        """
        Search database schema for columns matching a keyword.
        Use this when SQL fails with 'column does not exist' error.

        Args:
            args: Tool arguments with 'search_term' and optional 'table_name'

        Returns:
            Matching columns with table.column format and descriptions
        """
        search_term = args.get("search_term", "").lower()
        table_filter = args.get("table_name", "").lower()

        logger.info(f"Schema search: '{search_term}' in table '{table_filter or 'all'}'")

        matches = []

        for table_name, columns in self.schema_metadata.items():
            # Skip if table filter specified and doesn't match
            if table_filter and table_filter not in table_name.lower():
                continue

            for col in columns:
                col_name = col['name'].lower()
                col_desc = col.get('description', '').lower()

                # Match if search term in column name or description
                if search_term in col_name or search_term in col_desc:
                    matches.append({
                        "column": f"{table_name}.{col['name']}",
                        "type": col['type'],
                        "description": col.get('description', ''),
                        "unit": col.get('unit', '')
                    })

        if not matches:
            return f"No columns found matching '{search_term}'. Try a different search term or check the spelling."

        # Format results
        result = f"Found {len(matches)} column(s) matching '{search_term}':\n\n"
        for match in matches[:10]:  # Limit to top 10
            result += f"• **{match['column']}** ({match['type']})\n"
            if match['description']:
                result += f"  {match['description']}\n"
            if match['unit'] and match['unit'] != 'none':
                result += f"  Unit: {match['unit']}\n"
            result += "\n"

        if len(matches) > 10:
            result += f"... and {len(matches) - 10} more matches.\n"

        return result

    def _validate_value_tool(self, args: Dict) -> str:
        """
        Validate and correct user input against actual database values.
        Handles typos, case variations, and synonyms using embeddings.

        Args:
            args: Tool arguments with 'table', 'column', 'user_input'

        Returns:
            Corrected value or suggestion
        """
        table = args.get("table", "")
        column = args.get("column", "")
        user_input = args.get("user_input", "")

        logger.info(f"[VALUE-VALIDATE] Checking '{user_input}' for {table}.{column}")

        try:
            from ..rag.value_normalizer import get_value_normalizer
            normalizer = get_value_normalizer()

            # Quick check: Skip expensive embedding if it's an exact match (case-insensitive)
            valid_values = normalizer.get_valid_values(table, column)
            if valid_values:
                # Check for exact match (case-insensitive)
                user_input_lower = user_input.lower()
                for valid_value in valid_values:
                    if valid_value.lower() == user_input_lower:
                        logger.info(f"[VALUE-VALIDATE] Fast path: '{user_input}' is exact match")
                        return f"✓ Validated '{user_input}' as '{valid_value}' (exact match: 100%)."

            # Not an exact match, use embedding-based normalization
            normalized, score, was_corrected = normalizer.normalize(
                table, column, user_input, threshold=0.75
            )

            if was_corrected:
                return f"✓ Corrected '{user_input}' to '{normalized}' (confidence: {score:.0%}). Use this value in SQL."
            elif score > 0.5:
                return f"✓ Validated '{user_input}' as '{normalized}' (exact match: {score:.0%})."
            else:
                # Get valid values to show options
                if valid_values:
                    samples = valid_values[:5]
                    return f"⚠️ '{user_input}' not found in {column}. Valid options: {', '.join(samples)}{'...' if len(valid_values) > 5 else ''}"
                else:
                    return f"⚠️ Column {table}.{column} is not indexed for validation."

        except Exception as e:
            logger.error(f"[VALUE-VALIDATE] Error: {e}", exc_info=True)
            return f"Error validating value: {str(e)}"

    async def _clarify_question_tool(self, args: Dict) -> str:
        """
        Ask user for clarification.

        Args:
            args: Tool arguments with 'question' and optional 'options'

        Returns:
            Formatted clarification request
        """
        question = args.get("question", "")
        options = args.get("options", [])

        logger.info(f"Clarification requested: {question}")

        clarification = f"I need clarification: {question}"

        if options:
            clarification += "\n\nPlease specify:\n"
            for opt in options:
                clarification += f"- {opt}\n"

        return clarification
