/**
 * MessageList - Display list of messages
 */
import React, { useEffect, useRef } from 'react';
import { Message } from '../../types/chat';
import MessageItem from './MessageItem';
import './MessageList.css';

interface MessageListProps {
  messages: Message[];
  isLoading?: boolean;
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  isLoading = false
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="message-list">
      {messages.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">💬</div>
          <h3>Data Assistant</h3>
          <p>Ask me anything about your sales data</p>
          <div className="empty-state-divider"></div>
          <div className="example-questions">
            <p>Example questions:</p>
            <ul>
              <li>What are sales for Q4?</li>
              <li>Compare budget vs actuals</li>
              <li>Show top customers</li>
              <li>Why did margins drop?</li>
            </ul>
          </div>
        </div>
      ) : (
        <div className="message-list-content">
          {messages.map((message, index) => (
            <MessageItem
              key={index}
              message={message}
            />
          ))}
          {isLoading && (
            <div className="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          )}
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;
