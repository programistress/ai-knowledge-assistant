import { useState, useEffect, useRef } from 'react';
import { askQuestion } from '../utils/api';
import SuggestedQuestions from './SuggestedQuestions';
import MarkdownMessage from './MarkdownMessage';
import { suggestedQuestions } from '../utils/preloadedDataset';

function ChatPanel({ chatMessages, setChatMessages }) {
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatMessagesEndRef = useRef(null);

  const scrollToBottom = () => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

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
          content: result.answer,
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

