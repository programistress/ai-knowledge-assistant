import { useEffect, useState } from 'react';
import './App.css';
import DataUploadPanel from './components/DataUploadPanel';
import ChatPanel from './components/ChatPanel';
import { getDocuments } from './utils/api';

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

  // Suggested questions state
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [isGeneratingQuestions, setIsGeneratingQuestions] = useState(false);
  const [lastDocumentInfo, setLastDocumentInfo] = useState(null);

  useEffect(() => {
    async function loadDocuments() {
      try {
        setInitStatus('Loading documents...');
        // !!! need to change for deployment where its stored
        const existingDocs = await getDocuments();
        
        if (existingDocs.documents && existingDocs.documents.length > 0) {
          const transformedDocs = existingDocs.documents.map(doc => ({
            id: doc.document_id,
            title: doc.document_name,
            type: doc.file_type || 'note',
            pdfUrl: doc.pdf_url || null
          }));
          setEntries(transformedDocs);
        }
        setInitStatus('Ready!');
      } catch (error) {
        console.error('Load error:', error);
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
    
    loadDocuments();
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
          suggestedQuestions={suggestedQuestions}
          setSuggestedQuestions={setSuggestedQuestions}
          setIsGeneratingQuestions={setIsGeneratingQuestions}
          setLastDocumentInfo={setLastDocumentInfo}
        />
        <ChatPanel 
          chatMessages={chatMessages} 
          setChatMessages={setChatMessages}
          suggestedQuestions={suggestedQuestions}
          isGeneratingQuestions={isGeneratingQuestions}
          lastDocumentInfo={lastDocumentInfo}
        />
      </div>
    </div>
  );
}

export default App;
