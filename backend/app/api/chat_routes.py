"""
FastAPI routes for conversational chat interface.
"""
import asyncio
import logging
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from datetime import datetime

from ..schemas.chat import (
    ChatRequest,
    ChatResponse,
    SqlExecution,
    CreateConversationResponse,
    ConversationHistoryResponse,
    MessageSchema,
    DeleteConversationResponse
)
from ..chat.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Global ConversationManager instance (singleton)
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """
    Get or create the global ConversationManager instance.

    Returns:
        ConversationManager: Singleton instance
    """
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
        logger.info("Initialized global ConversationManager")
    return _conversation_manager


@router.post(
    "/conversations",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
    description="Creates a new conversation and returns its unique identifier"
)
async def create_conversation():
    """
    Create a new conversation.

    Returns:
        CreateConversationResponse: Conversation ID and creation timestamp
    """
    manager = get_conversation_manager()
    conversation_id = manager.create_conversation()

    conversation = manager.get_conversation(conversation_id)

    return CreateConversationResponse(
        conversation_id=conversation_id,
        created_at=conversation.created_at
    )


@router.post(
    "/conversations/{conversation_id}/chat/stream",
    summary="Send a message in a conversation (streaming)",
    description="Send a natural language question and receive streaming updates"
)
async def chat_stream(conversation_id: str, request: ChatRequest):
    """
    Send a message and get streaming response from assistant.
    Sends approach first, then final answer.
    """
    manager = get_conversation_manager()

    # Check conversation exists
    conversation = manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )

    async def generate_response():
        try:
            # Process message with streaming
            async for event in manager.chat_stream(
                conversation_id=conversation_id,
                user_message=request.message,
                max_iterations=request.max_iterations
            ):
                # Send Server-Sent Event
                yield f"data: {json.dumps(event)}\n\n"
                # Force flush to ensure immediate delivery (typewriter effect)
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            error_event = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@router.post(
    "/conversations/{conversation_id}/chat",
    response_model=ChatResponse,
    summary="Send a message in a conversation",
    description="Send a natural language question and receive an answer with SQL transparency"
)
async def chat(conversation_id: str, request: ChatRequest):
    """
    Send a message and get response from assistant.

    Args:
        conversation_id: Conversation identifier
        request: Chat request with message and optional max_iterations

    Returns:
        ChatResponse: Assistant response with SQL execution metadata

    Raises:
        HTTPException 404: Conversation not found
        HTTPException 500: Internal error during processing
    """
    manager = get_conversation_manager()

    # Check conversation exists
    conversation = manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )

    try:
        # Process message
        result = await manager.chat(
            conversation_id=conversation_id,
            user_message=request.message,
            max_iterations=request.max_iterations
        )

        # Convert sql_executions to Pydantic models
        sql_executions = [
            SqlExecution(**execution)
            for execution in result.get("sql_executions", [])
        ]

        return ChatResponse(
            conversation_id=conversation_id,
            response=result["response"],
            sql_executions=sql_executions,
            complexity=result.get("complexity"),
            iterations=result["iterations"]
        )

    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
    description="Retrieve all messages in a conversation"
)
async def get_conversation_history(conversation_id: str):
    """
    Get full conversation history.

    Args:
        conversation_id: Conversation identifier

    Returns:
        ConversationHistoryResponse: Full conversation with all messages

    Raises:
        HTTPException 404: Conversation not found
    """
    manager = get_conversation_manager()

    conversation = manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )

    # Convert messages to Pydantic models
    messages = [
        MessageSchema(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            metadata=msg.metadata
        )
        for msg in conversation.messages
    ]

    return ConversationHistoryResponse(
        conversation_id=conversation.conversation_id,
        messages=messages,
        created_at=conversation.created_at,
        last_updated=conversation.last_updated,
        message_count=len(messages)
    )


@router.delete(
    "/conversations/{conversation_id}",
    response_model=DeleteConversationResponse,
    summary="Delete a conversation",
    description="Delete a conversation and its message history"
)
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation.

    Args:
        conversation_id: Conversation identifier

    Returns:
        DeleteConversationResponse: Confirmation message

    Raises:
        HTTPException 404: Conversation not found
    """
    manager = get_conversation_manager()

    conversation = manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}"
        )

    # Delete from in-memory storage
    del manager.conversations[conversation_id]

    logger.info(f"Deleted conversation: {conversation_id}")

    return DeleteConversationResponse(
        conversation_id=conversation_id,
        message="Conversation deleted successfully"
    )


