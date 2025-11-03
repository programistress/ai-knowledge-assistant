import pdfToText from 'react-pdftotext';
import mammoth from 'mammoth';

export const extractTextFromPDF = async (file) => {
  try {
    const text = await pdfToText(file);
    return text;
  } catch (error) {
    console.error("Failed to extract text from PDF:", error);
    throw new Error("Failed to extract text from PDF");
  }
};


export const extractTextFromDoc = async (file) => {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    return result.value;
  } catch (error) {
    console.error("Failed to extract text from DOC:", error);
    throw new Error("Failed to extract text from DOC file");
  }
};


export const extractTextFromDocument = async (file) => {
  const fileExtension = file.name.split('.').pop().toLowerCase();
  
  if (fileExtension === 'pdf') {
    return await extractTextFromPDF(file);
  } else if (fileExtension === 'doc' || fileExtension === 'docx') {
    return await extractTextFromDoc(file);
  } else {
    throw new Error(`Unsupported file format: ${fileExtension}`);
  }
};


export const isAllowedFileFormat = (file, allowedFormats = ['pdf', 'doc', 'docx']) => {
  const fileExtension = file.name.split('.').pop().toLowerCase();
  return allowedFormats.includes(fileExtension);
};
