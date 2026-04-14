
function SuggestedQuestions({ questions, onQuestionClick, isGenerating, lastDocumentInfo }) {
  const SHORT_QUESTION_MAX_LENGTH = 58;

  const getDisplayQuestion = (question) => {
    if (!question) return '';
    const cleaned = question.replace(/^\d+[).\s-]*/, '').trim();
    if (cleaned.length <= SHORT_QUESTION_MAX_LENGTH) {
      return cleaned;
    }
    return `${cleaned.slice(0, SHORT_QUESTION_MAX_LENGTH - 1)}…`;
  };

  // Don't render the component at all if no questions and not generating
  if (!isGenerating && questions.length === 0) {
    return null;
  }

  // Determine the label text based on whether questions are being generated
  const getLabelText = () => {
    if (isGenerating) {
      return "Preparing quick prompts...";
    }
    
    if (lastDocumentInfo) {
      return `Try one about "${lastDocumentInfo.documentName}"`;
    }
    
    return "Quick prompts";
  };

  return (
    <div className="suggested-questions">
      <div className="suggested-questions-label">
        {getLabelText()}
        {isGenerating && <span className="generating-spinner">⟳</span>}
      </div>
      <div className="questions-container">
        {isGenerating ? (
          // Show placeholder buttons while generating
          [1, 2, 3].map((index) => (
            <div
              key={`placeholder-${index}`}
              className="question-bubble placeholder"
            >
              Prompt {index}
            </div>
          ))
        ) : (
          // Show actual questions
          questions.map((question, index) => (
            <button
              key={index}
              className="question-bubble"
              onClick={() => onQuestionClick(question)}
              title="Click to ask this question"
            >
              {getDisplayQuestion(question)}
            </button>
          ))
        )}
      </div>
    </div>
  );
}

export default SuggestedQuestions;

