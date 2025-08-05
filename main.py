import logging
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
import aiofiles
import asyncio
from services import build_graph
from utils import extract_pdf_content, chunk_documents, initialize_vector_store, index_documents
from config import TEMP_DIR, LOG_FILE, LOG_FORMAT, LOG_LEVEL, FAISS_INDEX_PATH

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.info("Starting FastAPI server...")

# Initialize FastAPI app
app = FastAPI(title="Banking Document Error Detection RAG System")

# Pydantic models
class QueryInput(BaseModel):
    file_id: str
    query: str = "Analyze document for technical errors related to key banking terms."

class QueryResponse(BaseModel):
    file_id: str
    errors: list
    report_path: str

# Initialize LangGraph
try:
    graph = build_graph()
    logger.info("LangGraph initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LangGraph: {str(e)}", exc_info=True)
    raise

async def cleanup_file(file_path):
    """Clean up temporary file."""
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {str(e)}")

@app.post("/upload", response_model=dict)
async def upload_file(file: UploadFile = File(...)):
    """Upload and index PDF document."""
    if not file.filename.endswith(".pdf"):
        logger.error(f"Invalid file format uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_id = str(uuid.uuid4())
    file_path = TEMP_DIR / f"{file_id}.pdf"
    try:
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            if not content or len(content) < 100:
                logger.error(f"Empty or invalid file uploaded for file_id: {file_id}")
                raise HTTPException(status_code=400, detail="Uploaded file is empty or too small.")
            await f.write(content)
        if not file_path.exists() or file_path.stat().st_size < 100:
            logger.error(f"PDF file not saved correctly for file_id: {file_id}")
            raise HTTPException(status_code=400, detail="Failed to save PDF file.")
        logger.info(f"Processing file_id: {file_id}")
        docs = await extract_pdf_content(file_path)
        if not docs:
            logger.error(f"No content extracted from {file_path}")
            raise HTTPException(status_code=400, detail="No content could be extracted from the PDF.")
        chunks = await asyncio.to_thread(chunk_documents, docs)
        if not chunks:
            logger.error(f"No chunks created for file_id: {file_id}")
            raise HTTPException(status_code=400, detail="No document chunks created.")
        for attempt in range(3):
            try:
                vector_store = await asyncio.to_thread(initialize_vector_store)
                await asyncio.to_thread(index_documents, vector_store, chunks, file_id)
                break
            except Exception as e:
                logger.warning(f"Indexing attempt {attempt + 1} failed for file_id {file_id}: {str(e)}")
                if attempt == 2:
                    raise
                await asyncio.sleep(1)
        logger.info(f"Uploaded and indexed file_id: {file_id}")
        return {"file_id": file_id, "message": "Document uploaded and indexed successfully."}
    except Exception as e:
        logger.error(f"Upload failed for file_id {file_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        await cleanup_file(file_path)

@app.post("/analyze", response_model=QueryResponse)
async def analyze_document(query: QueryInput):
    """Analyze document for errors and generate report."""
    try:
        state = {
            "file_id": query.file_id,
            "query": query.query,
            "context": [],
            "errors": [],
            "report": "",
            "report_path": ""
        }
        result = await graph.ainvoke(state)
        logger.info(f"Analysis completed for file_id: {query.file_id}")
        return QueryResponse(
            file_id=query.file_id,
            errors=result["errors"],
            report_path=result["report_path"]
        )
    except Exception as e:
        logger.error(f"Analysis failed for {query.file_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{file_id}")
async def delete_document(file_id: str):
    """Delete document from FAISS."""
    try:
        index_path = FAISS_INDEX_PATH / f"{file_id}.faiss"
        if index_path.exists():
            index_path.unlink()
        logger.info(f"Deleted document for file_id: {file_id}")
        return JSONResponse(content={"message": f"Document {file_id} deleted successfully."})
    except Exception as e:
        logger.error(f"Deletion failed for {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check API health."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Failed to start FastAPI server: {str(e)}", exc_info=True)
        raise