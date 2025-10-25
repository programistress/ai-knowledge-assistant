import { useState } from 'react';
import './App.css';
import DataUploadPanel from './components/DataUploadPanel';
import ChatPanel from './components/ChatPanel';

function App() {
  const [entries, setEntries] = useState([]);
  const [chatMessages, setChatMessages] = useState([
    {
      id: 1,
      content: 'Hello! I\'m your AI Knowledge Assistant. I can help you retrieve information from your saved notes and documents. How can I assist you today?',
      sender: 'bot',
      timestamp: new Date().toLocaleString()
    }
  ]);

  return (
    <div className="App">
      <DataUploadPanel entries={entries} setEntries={setEntries} />
      <ChatPanel chatMessages={chatMessages} setChatMessages={setChatMessages} />
    </div>
  );
}

export default App;
