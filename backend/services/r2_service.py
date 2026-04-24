"""
handles communication with r2 storage in cloudflare
"""
import os
import boto3
from botocore.config import Config
from pathlib import Path

_client = None


def get_r2_client():
    """get or create the r2 client singleton"""
    global _client
    if _client is None:
        _client = boto3.client(
            's3',
            endpoint_url=os.getenv('R2_ENDPOINT'),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
    return _client


def get_bucket_name() -> str:
    return os.getenv('R2_BUCKET_NAME', 'ai-assistant-pdfs')


def get_public_url() -> str:
    return os.getenv('R2_PUBLIC_URL', '').rstrip('/')


def upload_pdf(document_id: str, file_path: Path) -> str:
    """
    upload a pdf file to r2
    
    Args:
        document_id: Unique identifier for the document
        file_path: local path to the pdf file
        
    Returns:
        public url of the uploaded file
    """
    client = get_r2_client()
    key = f"pdfs/{document_id}.pdf"
    
    client.upload_file(
        str(file_path),
        get_bucket_name(),
        key,
        ExtraArgs={'ContentType': 'application/pdf'}
    )
    
    return f"{get_public_url()}/{key}"


def upload_pdf_from_bytes(document_id: str, file_bytes: bytes) -> str:
    """
    Upload PDF bytes directly to R2.
    
    Args:
        document_id: Unique identifier for the document
        file_bytes: PDF file content as bytes
        
    Returns:
        Public URL of the uploaded file
    """
    client = get_r2_client()
    key = f"pdfs/{document_id}.pdf"
    
    client.put_object(
        Bucket=get_bucket_name(),
        Key=key,
        Body=file_bytes,
        ContentType='application/pdf'
    )
    
    return f"{get_public_url()}/{key}"


def delete_pdf(document_id: str) -> bool:
    """
    Delete a PDF from R2.
    
    Args:
        document_id: Unique identifier for the document
        
    Returns:
        True if deletion was successful
    """
    try:
        client = get_r2_client()
        key = f"pdfs/{document_id}.pdf"
        
        client.delete_object(
            Bucket=get_bucket_name(),
            Key=key
        )
        return True
    except Exception as e:
        print(f"[r2] Error deleting {document_id}: {e}")
        return False


def pdf_exists(document_id: str) -> bool:
    """Check if a PDF exists in R2."""
    try:
        client = get_r2_client()
        key = f"pdfs/{document_id}.pdf"
        
        client.head_object(Bucket=get_bucket_name(), Key=key)
        return True
    except client.exceptions.ClientError:
        return False


def get_pdf_url(document_id: str) -> str:
    """Get the public URL for a PDF."""
    return f"{get_public_url()}/pdfs/{document_id}.pdf"
