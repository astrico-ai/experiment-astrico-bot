/**
 * useChat hook for managing chat state and interactions
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { Message, SqlExecution } from '../types/chat';
import { createConversation, sendChatMessageStream } from '../services/api';

export const useChat = () => {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Token queue for typewriter effect
  const tokenQueueRef = useRef<string[]>([]);
  const isProcessingQueueRef = useRef(false);
  const assistantMessageIndexRef = useRef(-1);

  /**
   * Process token queue with artificial delay for typewriter effect
   */
  const processTokenQueue = useCallback(() => {
    if (isProcessingQueueRef.current || tokenQueueRef.current.length === 0) {
      return;
    }

    isProcessingQueueRef.current = true;

    const processNext = () => {
      if (tokenQueueRef.current.length === 0) {
        isProcessingQueueRef.current = false;
        return;
      }

      const token = tokenQueueRef.current.shift()!;

      setMessages((prev) => {
        const updated = [...prev];
        const idx = assistantMessageIndexRef.current;
        if (idx >= 0 && idx < updated.length) {
          const currentContent = updated[idx].content;
          const newContent = currentContent === '⏳ Analyzing data...' ? token : currentContent + token;
          updated[idx] = {
            ...updated[idx],
            content: newContent,
          };
        }
        return updated;
      });

      // Continue processing with delay (20ms per token for smooth animation)
      setTimeout(processNext, 20);
    };

    processNext();
  }, []);

  /**
   * Initialize conversation on mount
   */
  useEffect(() => {
    const initConversation = async () => {
      try {
        const response = await createConversation();
        setConversationId(response.conversation_id);
      } catch (err) {
        setError('Failed to create conversation');
        console.error('Error creating conversation:', err);
      }
    };

    initConversation();
  }, []);

  /**
   * Send a message (with streaming)
   */
  const sendMessage = useCallback(
    async (message: string) => {
      if (!conversationId) {
        setError('No active conversation');
        return;
      }

      // Add user message immediately
      const userMessage: Message = {
        role: 'user',
        content: message,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      setError(null);

      // Placeholder for assistant message (will be updated)
      let assistantMessageIndex = -1;

      try {
        // Call streaming API
        await sendChatMessageStream(
          conversationId,
          message,
          // On approach (shows immediately as first message)
          (approach: string) => {
            const approachMessage: Message = {
              role: 'assistant',
              content: approach,
              timestamp: new Date(),
            };
            // Add approach message
            setMessages((prev) => [...prev, approachMessage]);

            // Add a placeholder "thinking" message that will be replaced by the answer
            const thinkingMessage: Message = {
              role: 'assistant',
              content: '⏳ Analyzing data...',
              timestamp: new Date(),
            };
            setMessages((prev) => {
              assistantMessageIndex = prev.length;
              return [...prev, thinkingMessage];
            });
          },
          // On answer (replace the thinking message with complete answer)
          (answer: string) => {
            setMessages((prev) => {
              const updated = [...prev];
              if (assistantMessageIndex >= 0 && assistantMessageIndex < updated.length) {
                // Replace thinking/streaming message with final answer
                updated[assistantMessageIndex] = {
                  role: 'assistant',
                  content: answer,
                  timestamp: new Date(),
                };
              } else {
                // Fallback: add as new message
                updated.push({
                  role: 'assistant',
                  content: answer,
                  timestamp: new Date(),
                });
              }
              return updated;
            });
            setIsLoading(false);
          },
          // On error
          (errorMsg: string) => {
            setError(errorMsg);
            const errorMessage: Message = {
              role: 'assistant',
              content: `Error: ${errorMsg}`,
              timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
            setIsLoading(false);
          },
          // On token (add to queue for typewriter effect)
          (token: string) => {
            assistantMessageIndexRef.current = assistantMessageIndex;
            tokenQueueRef.current.push(token);
            processTokenQueue();
          }
        );
      } catch (err: any) {
        setError(err.message || 'Failed to send message');
        console.error('Error sending message:', err);

        const errorMessage: Message = {
          role: 'assistant',
          content: `Error: ${err.message || 'Failed to send message'}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
        setIsLoading(false);
      }
    },
    [conversationId]
  );

  /**
   * Clear conversation (create new one)
   */
  const clearConversation = useCallback(async () => {
    try {
      const response = await createConversation();
      setConversationId(response.conversation_id);
      setMessages([]);
      setError(null);
    } catch (err) {
      setError('Failed to create new conversation');
      console.error('Error creating conversation:', err);
    }
  }, []);

  return {
    conversationId,
    messages,
    isLoading,
    error,
    sendMessage,
    clearConversation,
  };
};
