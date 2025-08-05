from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# File paths
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"
LOG_FILE = BASE_DIR / "app.log"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH.mkdir(exist_ok=True)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in .env or environment.")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Key banking terms for analysis
KEY_TERMS = ["KYC", "AML", "Fraud Detection", "Transaction Monitoring", "Compliance"]

# Report configuration
REPORT_FONT = "Helvetica"
REPORT_TITLE = "Banking Document Analysis Report"