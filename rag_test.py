import requests
import os
from pathlib import Path
import time
import logging
import pdfplumber

# Setup logging
logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base configuration
BASE_URL = "http://127.0.0.1:8000"
PDF_PATH = Path("D:/personal_projects/banking_rag/test_errors.pdf")  # Updated to test PDF
TIMEOUT = 60
DELETE_FILES = False  # Set to True to delete FAISS index after testing

def check_health():
    """Check API health."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        response.raise_for_status()
        logger.info(f"Health check: {response.json()}")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Health check failed: {str(e)}")
        raise

def upload_pdf():
    """Upload test_errors.pdf."""
    try:
        if not PDF_PATH.exists():
            raise FileNotFoundError(f"PDF not found at {PDF_PATH}")
        with pdfplumber.open(PDF_PATH) as pdf:
            if not any(page.extract_text() for page in pdf.pages):
                raise ValueError(f"PDF at {PDF_PATH} contains no extractable text")
        with open(PDF_PATH, "rb") as f:
            files = {"file": (PDF_PATH.name, f, "application/pdf")}
            response = requests.post(f"{BASE_URL}/upload", files=files, timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Upload response: {result}")
            return result["file_id"]
    except (requests.RequestException, ValueError, FileNotFoundError) as e:
        logger.error(f"Upload failed: {str(e)}, Server response: {getattr(e, 'response', None) and e.response.text}")
        raise

def analyze_document(file_id: str):
    """Analyze document for errors."""
    try:
        payload = {"file_id": file_id}
        response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Analyze response: {result}")
        return result
    except requests.RequestException as e:
        logger.error(f"Analyze failed for file_id {file_id}: {str(e)}, Server response: {e.response.text if e.response else 'No response'}")
        raise

def verify_report(report_path: str):
    """Verify the generated PDF report exists."""
    try:
        report_path = Path(report_path)
        if report_path.exists():
            logger.info(f"Report found at {report_path}")
            return True
        else:
            logger.error(f"Report not found at {report_path}")
            return False
    except Exception as e:
        logger.error(f"Report verification failed: {str(e)}")
        raise

def delete_document(file_id: str):
    """Delete document from FAISS."""
    try:
        response = requests.delete(f"{BASE_URL}/delete/{file_id}", timeout=TIMEOUT)
        response.raise_for_status()
        logger.info(f"Delete response: {response.json()}")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Delete failed for file_id {file_id}: {str(e)}, Server response: {e.response.text if e.response else 'No response'}")
        raise

def main():
    """Run end-to-end test."""
    logger.info("Starting RAG system test...")
    check_health()
    file_id = upload_pdf()
    result = analyze_document(file_id)
    report_path = result["report_path"]
    if verify_report(report_path):
        logger.info("PDF report verified successfully")
    else:
        raise FileNotFoundError(f"PDF report verification failed at {report_path}")
    if DELETE_FILES:
        delete_document(file_id)
        logger.info("FAISS index deleted successfully")
    else:
        logger.info("Skipping FAISS index deletion, files preserved")
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        exit(1)