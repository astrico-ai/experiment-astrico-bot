/**
 * SqlQueryDisplay - Shows SQL queries with execution metadata
 */
import React from 'react';
import { SqlExecution } from '../../types/chat';
import './SqlQueryDisplay.css';

interface SqlQueryDisplayProps {
  executions: SqlExecution[];
}

const SqlQueryDisplay: React.FC<SqlQueryDisplayProps> = ({ executions }) => {
  if (!executions || executions.length === 0) {
    return null;
  }

  const getTimeColor = (ms: number): string => {
    if (ms < 500) return 'fast';
    if (ms < 2000) return 'medium';
    return 'slow';
  };

  return (
    <div className="sql-executions">
      {executions.map((execution, index) => (
        <div key={index} className="sql-execution">
          <div className="sql-header">
            <span className="sql-explanation">{execution.explanation}</span>
          </div>
          <pre className="sql-query">
            <code>{execution.query}</code>
          </pre>
          <div className="sql-metadata">
            <span className={`sql-time ${getTimeColor(execution.execution_time_ms)}`}>
              ⏱️ {execution.execution_time_ms}ms
            </span>
            <span className="sql-rows">📊 {execution.row_count} rows</span>
            <span className={`sql-status ${execution.success ? 'success' : 'error'}`}>
              {execution.success ? '✓' : '✗'}
            </span>
          </div>
          {execution.error && (
            <div className="sql-error">
              Error: {execution.error}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default SqlQueryDisplay;
