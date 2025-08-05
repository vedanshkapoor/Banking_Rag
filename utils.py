import os
import re
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH
from fpdf import FPDF
import pdfplumber

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            logging.warning(f"No text extracted from {pdf_path}")
            return None
        logging.info(f"Successfully extracted text from {pdf_path}")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return None

def clean_text(text):
    """Clean extracted text by removing extra whitespace and special characters."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """Split text into chunks using LangChain's text splitter."""
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Text split into {len(chunks)} chunks")
    return chunks

async def extract_pdf_content(pdf_path):
    """Extract text from PDF and return as Document."""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return []
    cleaned_text = clean_text(text)
    return [Document(page_content=cleaned_text, metadata={"source": str(pdf_path), "file_id": str(pdf_path.stem)})]

def chunk_documents(docs):
    """Chunk documents into smaller pieces."""
    if not docs:
        return []
    text = "\n".join(doc.page_content for doc in docs)
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
    return [Document(page_content=chunk, metadata=docs[0].metadata) for chunk in chunks]

def initialize_vector_store():
    """Initialize FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_texts([""], embeddings)

def index_documents(vector_store, chunks, file_id):
    """Index document chunks in FAISS."""
    for chunk in chunks:
        chunk.metadata["file_id"] = file_id
    vector_store.add_documents(chunks)
    index_path = FAISS_INDEX_PATH / f"{file_id}.faiss"
    vector_store.save_local(str(index_path))
    logging.info(f"Indexed {len(chunks)} chunks for file_id: {file_id}")

def load_vector_store(file_id):
    """Load FAISS vector store for file_id."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index_path = FAISS_INDEX_PATH / f"{file_id}.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)

def generate_pdf_report(errors, report_content, output_path):
    """Generate PDF report from markdown content."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, report_content)
    pdf.output(str(output_path))
    logging.info(f"Generated PDF report at {output_path}")