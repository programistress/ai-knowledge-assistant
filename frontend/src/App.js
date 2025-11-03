import { useEffect, useState } from 'react';
import './App.css';
import DataUploadPanel from './components/DataUploadPanel';
import ChatPanel from './components/ChatPanel';
import { getDocuments, initializeDataset } from './utils/api';

function App() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [initStatus, setInitStatus] = useState('Checking dataset...');
  const [chatMessages, setChatMessages] = useState([
    {
      id: 1,
      content: 'Welcome to the DSA Knowledge Assistant! This demo is pre-loaded with comprehensive Data Structures & Algorithms content. Ask me anything about algorithms, data structures, problem-solving patterns, and more!',
      sender: 'bot',
      timestamp: new Date().toLocaleString()
    }
  ]);

  useEffect(() => {
    async function initializeApp() {
      try {
        setInitStatus('Checking for existing data...');
        
        // Check if data already exists
        const existingDocs = await getDocuments();
        
        if (existingDocs.documents && existingDocs.documents.length > 0) {
          // Data already exists
          setInitStatus('Loading existing documents...');
          const transformedDocs = existingDocs.documents.map(doc => ({
            id: doc.document_id,
            title: doc.document_name,
          }));
          setEntries(transformedDocs);
          setInitStatus('Ready!');
        } else {
          // No data found - initialize dataset
          setInitStatus('Initializing DSA dataset (this may take a minute)...');
          const result = await initializeDataset();
          
          setInitStatus(`Loaded ${result.documents_uploaded} documents with ${result.total_chunks} chunks!`);
          
          // Fetch the newly uploaded documents
          const updatedDocs = await getDocuments();
          const transformedDocs = updatedDocs.documents.map(doc => ({
            id: doc.document_id,
            title: doc.document_name,
          }));
          setEntries(transformedDocs);
          
          // Update welcome message
          setChatMessages([{
            id: 1,
            content: `Dataset initialized successfully! ${result.documents_uploaded} DSA documents are now loaded and ready. Ask me anything about algorithms, data structures, problem-solving patterns, and more!`,
            sender: 'bot',
            timestamp: new Date().toLocaleString()
          }]);
          
          setInitStatus('Ready!');
        }
      } catch (error) {
        console.error('Initialization error:', error);
        setInitStatus('Backend unavailable - Please start the backend server');
        setChatMessages([{
          id: 1,
          content: 'Unable to connect to backend. Please ensure the backend server is running at http://localhost:8000',
          sender: 'bot',
          timestamp: new Date().toLocaleString()
        }]);
      } finally {
        setLoading(false);
      }
    }
    
    initializeApp();
  }, []);

  return (
    <div className="App">
      <div className="app-header">
        <h1 className="app-title">AI Knowledge Assistant</h1>
        <p className="app-description">Intelligent document management and information retrieval system</p>
        <p className="app-demo-badge">Demo: Pre-loaded with DSA (Data Structures & Algorithms) Knowledge Base</p>
        {loading && (
          <div className="init-status">
            <div className="spinner"></div>
            <span>{initStatus}</span>
          </div>
        )}
      </div>
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
