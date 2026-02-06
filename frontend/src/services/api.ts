/**
 * API client for chat backend
 */
import axios from 'axios';
import {
  ChatRequest,
  ChatResponse,
  CreateConversationResponse,
  ConversationHistoryResponse,
} from '../types/chat';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1/chat';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Create a new conversation
 */
export const createConversation = async (): Promise<CreateConversationResponse> => {
  const response = await apiClient.post<CreateConversationResponse>('/conversations');
  return response.data;
};

/**
 * Send a message in a conversation (streaming)
 */
export const sendChatMessageStream = async (
  conversationId: string,
  message: string,
  onApproach: (approach: string) => void,
  onAnswer: (answer: string) => void,
  onError: (error: string) => void,
  onToken?: (token: string) => void,
  maxIterations?: number
): Promise<void> => {
  const request: ChatRequest = {
    message,
    max_iterations: maxIterations,
  };

  const response = await fetch(
    `${API_BASE_URL}/conversations/${conversationId}/chat/stream`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    }
  );

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error('No response body');
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);

            if (event.type === 'approach') {
              onApproach(event.content);
            } else if (event.type === 'token') {
              // Token-level streaming for final answer
              if (onToken) {
                onToken(event.content);
              }
            } else if (event.type === 'answer') {
              onAnswer(event.content);
            } else if (event.type === 'error') {
              onError(event.content);
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
};

/**
 * Send a message in a conversation (non-streaming fallback)
 */
export const sendChatMessage = async (
  conversationId: string,
  message: string,
  maxIterations?: number
): Promise<ChatResponse> => {
  const request: ChatRequest = {
    message,
    max_iterations: maxIterations,
  };
  const response = await apiClient.post<ChatResponse>(
    `/conversations/${conversationId}/chat`,
    request
  );
  return response.data;
};

/**
 * Get conversation history
 */
export const getConversationHistory = async (
  conversationId: string
): Promise<ConversationHistoryResponse> => {
  const response = await apiClient.get<ConversationHistoryResponse>(
    `/conversations/${conversationId}`
  );
  return response.data;
};

/**
 * Delete a conversation
 */
export const deleteConversation = async (conversationId: string): Promise<void> => {
  await apiClient.delete(`/conversations/${conversationId}`);
};
