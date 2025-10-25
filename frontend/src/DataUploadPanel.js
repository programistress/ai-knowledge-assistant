import { useState } from 'react';

function DataUploadPanel({ entries, setEntries }) {
  const [showEntryForm, setShowEntryForm] = useState(false);
  const [entryTitle, setEntryTitle] = useState('');
  const [entryText, setEntryText] = useState('');

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
  );
}

export default DataUploadPanel;

