import { useState } from 'react';
import { extractTextFromDocument, isAllowedFileFormat } from '../utils/documentExtractor';

function DataUploadPanel({ entries, setEntries }) {
  const [showEntryForm, setShowEntryForm] = useState(false);
  const [entryTitle, setEntryTitle] = useState('');
  const [entryText, setEntryText] = useState('');
  const [uploading, setUploading] = useState(false);

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

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const allowedFormats = ['pdf', 'doc', 'docx'];

    if (!isAllowedFileFormat(file, allowedFormats)) {
      alert('Please upload only PDF or DOC/DOCX files');
      e.target.value = ''; // Reset file input
      return;
    }

    setUploading(true);

    try {
      const extractedText = await extractTextFromDocument(file);

      console.log('Extracted text:', extractedText);

      // Add the entry with extracted text
      setEntries([...entries, {
        id: Date.now(),
        type: 'file',
        title: file.name,
        content: extractedText,
        timestamp: new Date().toLocaleString()
      }]);

      alert('Document uploaded and text extracted successfully!');
    } catch (error) {
      console.error('Error processing file:', error);
      alert('Failed to extract text from document: ' + error.message);
    } finally {
      setUploading(false);
      e.target.value = ''; // Reset file input
    }
  };

  const handleDeleteEntry = (id) => {
    setEntries(entries.filter(entry => entry.id !== id));
  };

  return (
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
              <label className={`upload-btn ${uploading ? 'uploading' : ''}`}>
                {uploading ? 'Uploading...' : 'Upload Document'}
                <input 
                  type="file" 
                  accept=".pdf,.doc,.docx"
                  onChange={handleFileUpload}
                  disabled={uploading}
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
  );
}

export default DataUploadPanel;

