import { useState } from 'react';
import { extractTextFromDoc, isAllowedFileFormat } from '../utils/documentExtractor';
import { deleteDocument, uploadDocument, uploadPdfDocument, generateSuggestedQuestions } from '../utils/api';

function DataUploadPanel({ 
  entries, 
  setEntries, 
  setSuggestedQuestions, 
  setIsGeneratingQuestions, 
  setLastDocumentInfo 
}) {
  const QUESTION_CONTEXT_MAX_CHARS = 3000;
  const [showEntryForm, setShowEntryForm] = useState(false);
  const [entryTitle, setEntryTitle] = useState('');
  const [entryText, setEntryText] = useState('');
  const [uploading, setUploading] = useState(false);

  // Helper function to generate questions after successful upload
  const generateQuestionsForDocument = async (documentId, documentName, content) => {
    const startedAt = performance.now();
    const excerpt = content.slice(0, QUESTION_CONTEXT_MAX_CHARS);
    console.log(`[question-generation] started for "${documentName}" with ${excerpt.length} chars`);

    try {
      setIsGeneratingQuestions(true);
      const apiStartedAt = performance.now();
      const result = await generateSuggestedQuestions(documentId, documentName, excerpt);
      const apiDuration = ((performance.now() - apiStartedAt) / 1000).toFixed(2);
      console.log(`[question-generation] API call completed in ${apiDuration}s`);
      
      if (result.success && result.suggested_questions) {
        setSuggestedQuestions(result.suggested_questions);
        setLastDocumentInfo({
          documentId: result.document_id,
          documentName: result.document_name
        });
        console.log(`[question-generation] generated ${result.suggested_questions.length} questions`);
      } else {
        console.log('[question-generation] completed without suggested questions');
      }
    } catch (error) {
      console.error('[question-generation] failed:', error);
      // Don't show error to user as this is a nice-to-have feature
    } finally {
      setIsGeneratingQuestions(false);
      const totalDuration = ((performance.now() - startedAt) / 1000).toFixed(2);
      console.log(`[question-generation] finished in ${totalDuration}s`);
    }
  };

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
        
        // Generate suggested questions based on the uploaded note
        await generateQuestionsForDocument(documentId, entryTitle, entryText);
        
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
    const overallStartedAt = performance.now();
    console.log(`[upload] started for "${file.name}" (${file.size} bytes)`);

    const allowedFormats = ['pdf', 'doc', 'docx'];

    if (!isAllowedFileFormat(file, allowedFormats)) {
      alert('Please upload only PDF or DOC/DOCX files');
      e.target.value = ''; // reset file input
      return;
    }

    setUploading(true);

    try {
      const documentId = `doc_${Date.now()}`;
      const uploadStartedAt = performance.now();
      const fileExtension = file.name.split('.').pop().toLowerCase();
      let questionSourceText = '';

      if (fileExtension === 'pdf') {
        const uploadResult = await uploadPdfDocument(documentId, file.name, file);
        questionSourceText = uploadResult.content || '';
      } else {
        const extractedText = await extractTextFromDoc(file);
        await uploadDocument(documentId, file.name, extractedText);
        questionSourceText = extractedText;
      }

      const uploadDuration = ((performance.now() - uploadStartedAt) / 1000).toFixed(2);
      console.log(`[upload] backend upload/indexing completed in ${uploadDuration}s`);

      setEntries([...entries, {
        id: documentId,
        title: file.name,
      }]);

      // Generate suggested questions in background so upload flow is not blocked
      if (questionSourceText) {
        void generateQuestionsForDocument(documentId, file.name, questionSourceText);
        console.log('[upload] background question generation started');
      }

      const totalDuration = ((performance.now() - overallStartedAt) / 1000).toFixed(2);
      console.log(`[upload] completed in ${totalDuration}s`);
    } catch (error) {
      const totalDuration = ((performance.now() - overallStartedAt) / 1000).toFixed(2);
      console.error(`[upload] failed after ${totalDuration}s:`, error);
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

