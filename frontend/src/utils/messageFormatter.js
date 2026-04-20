const API_BASE_URL = process.env.REACT_APP_API_URL;

export function toInlineSnippet(text, maxLength = 190) {
  if (!text) return '';
  const singleLine = text.replace(/\s+/g, ' ').trim();
  if (singleLine.length <= maxLength) {
    return singleLine;
  }
  return `${singleLine.slice(0, maxLength - 1)}…`;
}

export function buildAssistantMessage(result) {
  const answer = result?.answer?.trim() || 'I could not generate an answer.';
  const sources = Array.isArray(result?.sources) ? result.sources : [];

  if (sources.length === 0) {
    return answer;
  }

  const seenReferences = new Set();
  const sourceLines = [];
  const referenceLines = [];

  for (const source of sources) {
    const fileName = source?.document_name || 'Unknown file';
    const excerpt = toInlineSnippet(source?.content || '');
    const chunkIndex = Number.isInteger(source?.chunk_index) ? source.chunk_index : null;
    const pageNumber = Number.isInteger(source?.page_number) ? source.page_number : null;
    const scoreValue = typeof source?.score === 'number' ? source.score.toFixed(3) : null;
    const pdfUrl = source?.pdf_url;

    const referenceKey = `${fileName}:${pageNumber ?? 'no-page'}`;
    if (!excerpt || seenReferences.has(referenceKey)) {
      continue;
    }

    seenReferences.add(referenceKey);
    sourceLines.push(`> "${excerpt}"`);

    const referenceParts = [`\`${fileName}\``];
    if (chunkIndex !== null) {
      referenceParts.push(`chunk ${chunkIndex}`);
    }
    if (scoreValue) {
      referenceParts.push(`score ${scoreValue}`);
    }
    if (pageNumber !== null) {
      referenceParts.push(`page ${pageNumber}`);
    }
    if (pdfUrl && pageNumber !== null && API_BASE_URL) {
      const pageLink = `${API_BASE_URL}${pdfUrl}#page=${pageNumber}`;
      referenceParts.push(`[open page](${pageLink})`);
    }

    referenceLines.push(`- ${referenceParts.join(' · ')}`);

    if (sourceLines.length === 2) {
      break;
    }
  }

  if (sourceLines.length === 0) {
    return answer;
  }

  return [
    "### Evidence from your files",
    ...sourceLines,
    '',
    '### Quick explanation',
    answer,
    '',
    '### References',
    ...referenceLines
  ].join('\n');
}
