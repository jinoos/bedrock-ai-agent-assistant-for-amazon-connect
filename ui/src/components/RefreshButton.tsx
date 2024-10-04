// src/components/RefreshButton.tsx
import React from 'react';

interface RefreshButtonProps {
  onRefresh: () => void;
}

const RefreshButton = ({ onRefresh }: RefreshButtonProps) => {
  return (
    <button
      onClick={onRefresh}
      style={{
        marginLeft: '10px',
        background: 'none',
        border: 'none',
        cursor: 'pointer',
        fontSize: '16px'
      }}
    >
      🔄
    </button>
  );
};

export default RefreshButton;
