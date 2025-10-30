import { useState, useEffect, useRef } from 'react';
import { askQuestion } from '../utils/api';

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

  const handleSendMessage = async () => {
    if (chatInput.trim() && !loading) {
      const userMessage = {
        id: Date.now(),
        content: chatInput,
        sender: 'user',
        timestamp: new Date().toLocaleString()
      };
      
      setChatMessages([...chatMessages, userMessage]);
      const question = chatInput;
      setChatInput('');
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
          content: 'Oopsie! Something went wrong. Please make sure you have uploaded documents first, then try again.',
          sender: 'bot',
          timestamp: new Date().toLocaleString()
        };
        setChatMessages(prev => [...prev, errorResponse]);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="right-panel">
      <div className="chat-header">Chat</div>
      <div className="chat-messages">
        {chatMessages.map(msg => (
          <div key={msg.id} className={`chat-message ${msg.sender}`}>
            <div className="message-content">{msg.content}</div>
            <div className="message-timestamp">{msg.timestamp}</div>
          </div>
        ))}
        <div ref={chatMessagesEndRef} />
      </div>
      <div className="chat-input-container">
        <input
          type="text"
          value={chatInput}
          onChange={(e) => setChatInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Type a message..."
          className="chat-input"
          disabled={loading}
        />
        <button onClick={handleSendMessage} className="send-btn" disabled={loading}>
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default ChatPanel;

