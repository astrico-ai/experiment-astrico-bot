/**
 * MessageItem - Display a single message bubble
 */
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Message } from '../../types/chat';
import './MessageItem.css';

interface MessageItemProps {
  message: Message;
}

const MessageItem: React.FC<MessageItemProps> = ({
  message
}) => {
  const isUser = message.role === 'user';

  return (
    <div className={`message-item ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="message-bubble">
        <div className="message-content">
          {isUser ? (
            // User messages: simple text
            message.content.split('\n').map((line, index) => (
              <p key={index}>{line}</p>
            ))
          ) : (
            // Assistant messages: render markdown with table support
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          )}
        </div>
      </div>
      {message.timestamp && (
        <div className="message-timestamp">
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

export default MessageItem;
