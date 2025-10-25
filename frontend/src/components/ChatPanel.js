import { useState, useEffect, useRef } from 'react';

function ChatPanel({ chatMessages, setChatMessages }) {
  const [chatInput, setChatInput] = useState('');
  const chatMessagesEndRef = useRef(null);

  const scrollToBottom = () => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const handleSendMessage = () => {
    if (chatInput.trim()) {
      const userMessage = {
        id: Date.now(),
        content: chatInput,
        sender: 'user',
        timestamp: new Date().toLocaleString()
      };
      
      setChatMessages([...chatMessages, userMessage]);
      setChatInput('');

      // Simulate bot response
      setTimeout(() => {
        const botResponse = {
          id: Date.now() + 1,
          content: 'I\'ve processed your query. Based on your knowledge base, I can help you find relevant information from your stored notes and documents. What specific information are you looking for?',
          sender: 'bot',
          timestamp: new Date().toLocaleString()
        };
        setChatMessages(prev => [...prev, botResponse]);
      }, 800);
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
        />
        <button onClick={handleSendMessage} className="send-btn">
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatPanel;

