import { useState, useEffect, useRef } from 'react';
import { askQuestion } from '../utils/api';
import SuggestedQuestions from './SuggestedQuestions';
import MarkdownMessage from './MarkdownMessage';

const API_BASE_URL = process.env.REACT_APP_API_URL;

function ChatPanel({ 
  chatMessages, 
  setChatMessages, 
  suggestedQuestions, 
  isGeneratingQuestions, 
  lastDocumentInfo 
}) {
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatMessagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const toInlineSnippet = (text, maxLength = 190) => {
    if (!text) return '';
    const singleLine = text.replace(/\s+/g, ' ').trim();
    if (singleLine.length <= maxLength) {
      return singleLine;
    }
    return `${singleLine.slice(0, maxLength - 1)}…`;
  };

  const buildAssistantMessage = (result) => {
    const answer = result?.answer?.trim() || 'I could not generate an answer.';
    const sources = Array.isArray(result?.sources) ? result.sources : [];

    if (sources.length === 0) {
      return answer;
    }

    const seenReferences = new Set();
    const sourceLines = [];
    const referenceLines = [];

    for (const source of sources) {
      const fileName = source?.document_name || 'Unknown file';
      const excerpt = toInlineSnippet(source?.content || '');
      const chunkIndex = Number.isInteger(source?.chunk_index) ? source.chunk_index : null;
      const pageNumber = Number.isInteger(source?.page_number) ? source.page_number : null;
      const scoreValue = typeof source?.score === 'number' ? source.score.toFixed(3) : null;
      const pdfUrl = source?.pdf_url;

      const referenceKey = `${fileName}:${pageNumber ?? 'no-page'}`;
      if (!excerpt || seenReferences.has(referenceKey)) {
        continue;
      }

      seenReferences.add(referenceKey);
      sourceLines.push(`> "${excerpt}"`);

      const referenceParts = [`\`${fileName}\``];
      if (chunkIndex !== null) {
        referenceParts.push(`chunk ${chunkIndex}`);
      }
      if (scoreValue) {
        referenceParts.push(`score ${scoreValue}`);
      }
      if (pageNumber !== null) {
        referenceParts.push(`page ${pageNumber}`);
      }
      if (pdfUrl && pageNumber !== null && API_BASE_URL) {
        const pageLink = `${API_BASE_URL}${pdfUrl}#page=${pageNumber}`;
        referenceParts.push(`[open page](${pageLink})`);
      }

      referenceLines.push(`- ${referenceParts.join(' · ')}`);

      if (sourceLines.length === 2) {
        break;
      }
    }

    if (sourceLines.length === 0) {
      return answer;
    }

    return [
      "### Evidence from your files",
      ...sourceLines,
      '',
      '### Quick explanation',
      answer,
      '',
      '### References',
      ...referenceLines
    ].join('\n');
  };

  const handleSendMessage = async (questionToSend) => {
    // Allow passing question directly (for suggested questions) or use current input
    const question = questionToSend || chatInput;
    
    if (question.trim() && !loading) {
      const userMessage = {
        id: Date.now(),
        content: question,
        sender: 'user',
        timestamp: new Date().toLocaleString()
      };
      
      setChatMessages([...chatMessages, userMessage]);
      setChatInput(''); // Clear input
      setLoading(true);

      try {
        // Call real API
        const result = await askQuestion(question);
        
        const botResponse = {
          id: Date.now() + 1,
          content: buildAssistantMessage(result),
          sender: 'bot',
          timestamp: new Date().toLocaleString()
        };
        setChatMessages(prev => [...prev, botResponse]);
      } catch (error) {
        console.error('Error asking question:', error);
        const errorResponse = {
          id: Date.now() + 1,
          content: 'An error occurred while processing your request. Please ensure you have uploaded documents and try again.',
          sender: 'bot',
          timestamp: new Date().toLocaleString()
        };
        setChatMessages(prev => [...prev, errorResponse]);
      } finally {
        setLoading(false);
      }
    }
  };

  // Handler for when a suggested question is clicked
  const handleQuestionClick = (question) => {
    // Directly send the question
    handleSendMessage(question);
  };

  return (
    <div className="right-panel">
      <div className="chat-header">Chat</div>
      <div className="chat-messages">
        {chatMessages.map(msg => (
          <div key={msg.id} className={`chat-message ${msg.sender}`}>
            <div className="message-content">
              {msg.sender === 'bot' ? (
                <MarkdownMessage content={msg.content} />
              ) : (
                msg.content
              )}
            </div>
            <div className="message-timestamp">{msg.timestamp}</div>
          </div>
        ))}
        <div ref={chatMessagesEndRef} />
      </div>
      <div className="chat-input-container">
        <SuggestedQuestions 
          questions={suggestedQuestions} 
          onQuestionClick={handleQuestionClick}
          isGenerating={isGeneratingQuestions}
          lastDocumentInfo={lastDocumentInfo}
        />
        <div className="input-row">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !loading && handleSendMessage()}
            placeholder="Type a message..."
            className="chat-input"
            disabled={loading}
          />
          <button onClick={() => handleSendMessage()} className="send-btn" disabled={loading}>
            {loading ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatPanel;

