import { useState } from 'react';
import { extractTextFromDocument, isAllowedFileFormat } from '../utils/documentExtractor';
import { deleteDocument, uploadDocument } from '../utils/api';

function DataUploadPanel({ entries, setEntries }) {
  const [showEntryForm, setShowEntryForm] = useState(false);
  const [entryTitle, setEntryTitle] = useState('');
  const [entryText, setEntryText] = useState('');
  const [uploading, setUploading] = useState(false);

  const handleSubmitEntry = async () => {
    if (entryTitle.trim() && entryText.trim() && !uploading) {
      setUploading(true);
      try {
        // Upload text note to backend
        const documentId = `note_${Date.now()}`;
        await uploadDocument(documentId, entryTitle, entryText);

        setEntries([...entries, {
          id: documentId,
          title: entryTitle,
        }]);
        
        setEntryTitle('');
        setEntryText('');
        setShowEntryForm(false);
        console.log('Note saved successfully!');
      } catch (error) {
        console.error('Error saving note:', error);
        alert('Failed to save note: ' + error.message);
      } finally {
        setUploading(false);
      }
    }
  };


  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const allowedFormats = ['pdf', 'doc', 'docx'];

    if (!isAllowedFileFormat(file, allowedFormats)) {
      alert('Please upload only PDF or DOC/DOCX files');
      e.target.value = ''; // reset file input
      return;
    }

    setUploading(true);

    try {
      const extractedText = await extractTextFromDocument(file);
      console.log('Extracted text:', extractedText);

       // upload to backend
      const documentId = `doc_${Date.now()}`;
      await uploadDocument(documentId, file.name, extractedText);

      setEntries([...entries, {
        id: documentId,
        title: file.name,
      }]);

      console.log('Document uploaded and text extracted successfully!');
    } catch (error) {
      console.error('Error processing file:', error);
      console.log('Failed to process file:' + error.message);
    } finally {
      setUploading(false);
      e.target.value = ''; // Reset file input
    }
  };

  const handleDeleteEntry = async (id) => {
  try {
    await deleteDocument(id);
    setEntries(entries.filter(entry => entry.id !== id));
  } catch (error) {
    console.error('Failed to delete document:', error);
    alert('Failed to delete document');
  }
};

  return (
    <div className="left-panel">
      <div className="panel-header">My Knowledge Base</div>
      <div className="entry-section">
        <button 
          className="new-entry-btn"
          onClick={() => setShowEntryForm(!showEntryForm)}
        >
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
              disabled={uploading}
            />
            <textarea
              value={entryText}
              onChange={(e) => setEntryText(e.target.value)}
              placeholder="Type your entry here..."
              className="entry-textarea"
              disabled={uploading}
            />
            <div className="form-actions">
              <button 
                onClick={handleSubmitEntry} 
                className="submit-btn"
                disabled={uploading}
              >
                {uploading ? (
                  <>
                    <span className="spinner"></span>
                    Uploading...
                  </>
                ) : (
                  'Submit'
                )}
              </button>
              <label className={`upload-btn ${uploading ? 'uploading' : ''}`}> 
              Upload Document
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

      {entries.length > 0 ? (
        <div className="entries-list">
          {entries.map(entry => (
            <div key={entry.id} className="entry-item">
              <div className="entry-header">
                <div className="entry-type">Entry</div>
                <button 
                  className="delete-btn"
                  onClick={() => handleDeleteEntry(entry.id)}
                  title="Delete"
                >
                  ×
                </button>
              </div>
              <div className="entry-content">{entry.title}</div>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-text">No entries yet!</div>
          <div className="empty-subtext">Click "New Entry" above to get started.</div>
        </div>
      )}
    </div>
  );
}

export default DataUploadPanel;

