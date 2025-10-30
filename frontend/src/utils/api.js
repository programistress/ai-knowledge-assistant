
const API_URL = process.env.REACT_APP_API_URL

// helper function to make API requests
async function apiRequest(endpoint, options = {}) {
  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'API request failed');
    }
    return data;

  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    throw error;
  }
}

export async function uploadDocument(documentId, documentName, content) {
  return apiRequest('/upload', {
    method: 'POST',
    body: JSON.stringify({
      document_id: documentId,
      document_name: documentName,
      content: content,
    }),
  });
}

export async function askQuestion(question, topK = 3) {
  return apiRequest('/query', {
    method: 'POST',
    body: JSON.stringify({
      question: question,
      top_k: topK,
    }),
  });
}

// get documents from the database
export async function getDocuments() {
  return apiRequest('/documents', {
    method: 'GET',
  });
}

// delete a document by its ID
export async function deleteDocument(documentId) {
  return apiRequest(`/documents/${documentId}`, {
    method: 'DELETE',
  });
}

// check if the backend is running
export async function checkHealth() {
  return apiRequest('/health', {
    method: 'GET',
  });
}

