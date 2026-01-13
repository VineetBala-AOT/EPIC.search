import React, { useState } from 'react';
import { Box, Typography, IconButton } from '@mui/material';
import { ThumbUp, ThumbDown } from '@mui/icons-material';
import FeedbackModal from './FeedbackModal';

interface FeedbackControlProps {
  sessionId?: string;
}

const FeedbackControl: React.FC<FeedbackControlProps> = ({ sessionId }) => {
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackType, setFeedbackType] = useState<'up' | 'down' | null>(null);

  const handleFeedbackClick = (type: 'up' | 'down') => {
    setFeedbackType(type);
    setFeedbackOpen(true);
  };

  return (
    <>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 1,
          py: 1,
          borderRadius: '6px',
          backgroundColor: '#FEF8E8',
          flex: 1,
          minHeight: '36px',
          maxWidth: '400px',
        }}
      >
        {/* Feedback Text */}
        <Typography
          sx={{
            fontSize: '0.875rem',
            color: '#374151',
            fontWeight: 400,
            flex: 1,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          Please provide us your feedback on this search.
        </Typography>

        {/* Positive Feedback */}
        <IconButton
          size="small"
          onClick={() => handleFeedbackClick('up')}
          sx={{
            p: 1,
            '&:hover': { backgroundColor: '#F3F4F6' },
            flexShrink: 0,
          }}
        >
          <ThumbUp sx={{ fontSize: 20, color: '#4B5563' }} />
        </IconButton>

        {/* Negative Feedback */}
        <IconButton
          size="small"
          onClick={() => handleFeedbackClick('down')}
          sx={{
            p: 1,
            '&:hover': { backgroundColor: '#F3F4F6' },
            flexShrink: 0,
          }}
        >
          <ThumbDown sx={{ fontSize: 20, color: '#4B5563' }} />
        </IconButton>
      </Box>

      {/* Feedback Modal */}
      {feedbackType && (
        <FeedbackModal
          sessionId={sessionId}
          isOpen={feedbackOpen}
          onClose={() => setFeedbackOpen(false)}
          feedbackType={feedbackType}
        />
      )}
    </>
  );
};

export default FeedbackControl;
