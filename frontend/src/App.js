import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [showEntryForm, setShowEntryForm] = useState(false);
  const [entryTitle, setEntryTitle] = useState('');
  const [entryText, setEntryText] = useState('');
  const [entries, setEntries] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([
    {
      id: 1,
      content: 'Hello! I\'m your AI Knowledge Assistant. I can help you retrieve information from your saved notes and documents. How can I assist you today?',
      sender: 'bot',
      timestamp: new Date().toLocaleString()
    }
  ]);
  
  const chatMessagesEndRef = useRef(null);

  const scrollToBottom = () => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const handleSubmitEntry = () => {
    if (entryTitle.trim() && entryText.trim()) {
      setEntries([...entries, {
        id: Date.now(),
        type: 'text',
        title: entryTitle,
        content: entryText,
        timestamp: new Date().toLocaleString()
      }]);
      setEntryTitle('');
      setEntryText('');
      setShowEntryForm(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setEntries([...entries, {
        id: Date.now(),
        type: 'file',
        title: file.name,
        content: file.name,
        timestamp: new Date().toLocaleString()
      }]);
    }
  };

  const handleDeleteEntry = (id) => {
    setEntries(entries.filter(entry => entry.id !== id));
  };

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
    <div className="App">
      <div className="left-panel">
        <div className="entry-section">
          <button 
            className="new-entry-btn"
            onClick={() => setShowEntryForm(!showEntryForm)}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M10 4V16M4 10H16" stroke="currentColor" strokeWidth="2" strokeLinecap="square"/>
            </svg>
            New Entry
          </button>

          {showEntryForm && (
            <div className="entry-form">
              <input
                type="text"
                value={entryTitle}
                onChange={(e) => setEntryTitle(e.target.value)}
                placeholder="Enter note title..."
                className="entry-title-input"
              />
              <textarea
                value={entryText}
                onChange={(e) => setEntryText(e.target.value)}
                placeholder="Type your entry here..."
                className="entry-textarea"
              />
              <div className="form-actions">
                <button onClick={handleSubmitEntry} className="submit-btn">
                  Submit
                </button>
                <label className="upload-btn">
                  Upload Document
                  <input 
                    type="file" 
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                  />
                </label>
              </div>
            </div>
          )}
        </div>

        {entries.length > 0 && (
          <div className="entries-list">
            {entries.map(entry => (
              <div key={entry.id} className="entry-item">
                <div className="entry-header">
                  <div className="entry-type">{entry.type === 'file' ? 'Document' : 'Note'}</div>
                  <button 
                    className="delete-btn"
                    onClick={() => handleDeleteEntry(entry.id)}
                    title="Delete"
                  >
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                      <path d="M4 4L12 12M12 4L4 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                  </button>
                </div>
                <div className="entry-content">{entry.title}</div>
                <div className="entry-timestamp">{entry.timestamp}</div>
              </div>
            ))}
          </div>
        )}
      </div>

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
    </div>
  );
}

export default App;
