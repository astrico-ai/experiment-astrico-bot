/**
 * TypeScript interfaces for chat functionality
 */

export interface SqlExecution {
  query: string;
  explanation: string;
  execution_time_ms: number;
  row_count: number;
  success: boolean;
  error?: string;
}

export interface Message {
  role: 'user' | 'assistant' | 'tool';
  content: string;
  timestamp?: Date;
  metadata?: {
    sql_executions?: SqlExecution[];
    type?: string; // 'insights' or other message types
  };
}

export interface ChatRequest {
  message: string;
  max_iterations?: number;
}

export interface ChatResponse {
  conversation_id: string;
  response: string;
  sql_executions: SqlExecution[];
  complexity?: string;
  iterations: number;
}

export interface CreateConversationResponse {
  conversation_id: string;
  created_at: string;
}

export interface ConversationHistoryResponse {
  conversation_id: string;
  messages: Message[];
  created_at: string;
  last_updated: string;
  message_count: number;
}
