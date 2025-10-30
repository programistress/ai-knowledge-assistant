import { useEffect, useState } from 'react';
import './App.css';
import DataUploadPanel from './components/DataUploadPanel';
import ChatPanel from './components/ChatPanel';
import { getDocuments } from './utils/api';

function App() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [chatMessages, setChatMessages] = useState([
    {
      id: 1,
      content: 'Hello! I\'m your AI Knowledge Assistant. I can help you find information from your notes and documents. How can I assist you today?',
      sender: 'bot',
      timestamp: new Date().toLocaleString()
    }
  ]);

  useEffect(() => {
    async function fetchDocuments() {
      try {
        const result = await getDocuments();
        const transformedDocs = result.documents.map(doc => ({
          id: doc.document_id,
          title: doc.document_name,
        }));
        setEntries(transformedDocs);
      } catch (error) {
        console.error('Failed to fetch documents:', error);
      } finally {
        setLoading(false);
      }
    }
    
    fetchDocuments();
  }, []);

  return (
    <div className="App">
      <h1 className="app-title">AI Knowledge Assistant</h1>
      <div className="app-container">
        <DataUploadPanel 
          entries={entries} 
          setEntries={setEntries}
          />
        <ChatPanel 
          chatMessages={chatMessages} 
          setChatMessages={setChatMessages} 
        />
      </div>
    </div>
  );
}

export default App;
