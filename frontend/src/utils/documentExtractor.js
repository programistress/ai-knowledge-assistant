import mammoth from 'mammoth';

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

export const isAllowedFileFormat = (file, allowedFormats = ['pdf', 'doc', 'docx']) => {
  const fileExtension = file.name.split('.').pop().toLowerCase();
  return allowedFormats.includes(fileExtension);
};
