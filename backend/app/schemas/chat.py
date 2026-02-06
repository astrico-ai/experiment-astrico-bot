"""
Pydantic schemas for chat API endpoints.
"""
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request to send a message in a conversation."""
    message: str = Field(..., description="User's natural language question")
    max_iterations: Optional[int] = Field(10, description="Maximum tool-calling iterations")


class SqlExecution(BaseModel):
    """Metadata about a SQL query execution."""
    query: str = Field(..., description="The SQL query that was executed")
    explanation: str = Field(..., description="Brief explanation of what this query checks")
    execution_time_ms: int = Field(..., description="Query execution time in milliseconds")
    row_count: int = Field(..., description="Number of rows returned")
    success: bool = Field(..., description="Whether the query succeeded")
    error: Optional[str] = Field(None, description="Error message if query failed")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    conversation_id: str = Field(..., description="Conversation identifier")
    response: str = Field(..., description="Assistant's natural language response")
    sql_executions: List[SqlExecution] = Field(
        default_factory=list,
        description="List of SQL queries executed with timing metadata"
    )
    complexity: Optional[str] = Field(None, description="Detected question complexity")
    iterations: int = Field(..., description="Number of tool-calling iterations used")


class CreateConversationResponse(BaseModel):
    """Response from conversation creation."""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    created_at: datetime = Field(..., description="Conversation creation timestamp")


class MessageSchema(BaseModel):
    """Schema for a single message in conversation history."""
    role: str = Field(..., description="Message role (user, assistant, tool)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., SQL executions)"
    )


class ConversationHistoryResponse(BaseModel):
    """Response with full conversation history."""
    conversation_id: str = Field(..., description="Conversation identifier")
    messages: List[MessageSchema] = Field(..., description="All messages in the conversation")
    created_at: datetime = Field(..., description="Conversation creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Total number of messages")


class DeleteConversationResponse(BaseModel):
    """Response from conversation deletion."""
    conversation_id: str = Field(..., description="Deleted conversation identifier")
    message: str = Field(..., description="Confirmation message")
