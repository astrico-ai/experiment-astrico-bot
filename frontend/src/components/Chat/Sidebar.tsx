/**
 * Sidebar - Conversation history and new chat button
 */
import React from 'react';
import './Sidebar.css';

interface Conversation {
  id: string;
  title: string;
  timestamp: Date;
}

interface SidebarProps {
  conversations?: Conversation[];
  activeConversationId?: string;
  onNewChat?: () => void;
  onSelectConversation?: (id: string) => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  conversations = [],
  activeConversationId,
  onNewChat,
  onSelectConversation,
  isCollapsed = false,
  onToggleCollapse
}) => {
  const formatTimestamp = (date: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffHours < 1) {
      const diffMins = Math.floor(diffMs / (1000 * 60));
      return diffMins < 1 ? 'Just now' : `${diffMins} min ago`;
    } else if (diffHours < 24) {
      return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const truncateTitle = (title: string, maxLength: number = 25): string => {
    return title.length > maxLength ? `${title.substring(0, maxLength)}...` : title;
  };

  if (isCollapsed) {
    return (
      <div className="sidebar collapsed">
        <button className="toggle-button" onClick={onToggleCollapse} title="Expand sidebar">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 18L15 12L9 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <button className="new-chat-button" onClick={onNewChat}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 5V19M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          New Chat
        </button>
        <button className="toggle-button" onClick={onToggleCollapse} title="Collapse sidebar">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M15 18L9 12L15 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>

      <div className="sidebar-divider"></div>

      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div className="empty-conversations">
            <p>No conversations yet</p>
          </div>
        ) : (
          conversations.map((conversation) => (
            <button
              key={conversation.id}
              className={`conversation-item ${activeConversationId === conversation.id ? 'active' : ''}`}
              onClick={() => onSelectConversation?.(conversation.id)}
            >
              <div className="conversation-icon">💬</div>
              <div className="conversation-details">
                <div className="conversation-title">{truncateTitle(conversation.title)}</div>
                <div className="conversation-timestamp">{formatTimestamp(conversation.timestamp)}</div>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
};

export default Sidebar;
