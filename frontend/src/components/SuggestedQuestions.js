
function SuggestedQuestions({ questions, onQuestionClick }) {
  return (
    <div className="suggested-questions">
      <div className="suggested-questions-label">Suggested questions:</div>
      <div className="questions-container">
        {questions.map((question, index) => (
          <button
            key={index}
            className="question-bubble"
            onClick={() => onQuestionClick(question)}
            title="Click to ask this question"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

export default SuggestedQuestions;

