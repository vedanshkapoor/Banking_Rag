import logging
import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from typing import List, Dict
from pydantic import BaseModel
from langchain_core.documents import Document
from utils import load_vector_store, generate_pdf_report
from config import GROQ_API_KEY, GROQ_MODEL, KEY_TERMS, FAISS_INDEX_PATH, TEMP_DIR

# Setup logging
logger = logging.getLogger(__name__)

class State(BaseModel):
    file_id: str
    query: str
    context: List[Document]
    errors: List[Dict[str, str]]
    report: str
    report_path: str

def retrieve(state: State) -> Dict:
    """Retrieve relevant chunks from FAISS."""
    try:
        index_path = FAISS_INDEX_PATH / f"{state.file_id}.faiss"
        if not index_path.exists():
            logger.error(f"FAISS index not found for file_id: {state.file_id} at {index_path}")
            raise FileNotFoundError(f"FAISS index not found for file_id: {state.file_id}")
        query = f"Identify technical errors related to: {', '.join(KEY_TERMS)}"
        vector_store = load_vector_store(state.file_id)
        retrieved_docs = vector_store.similarity_search(query, k=5, filter={"file_id": state.file_id})
        if not retrieved_docs:
            logger.warning(f"No relevant documents retrieved for file_id: {state.file_id}")
        logger.info(f"Retrieved {len(retrieved_docs)} chunks for file_id: {state.file_id}")
        return {"context": retrieved_docs}
    except Exception as e:
        logger.error(f"Retrieval failed for {state.file_id}: {str(e)}", exc_info=True)
        raise

def detect_errors(state: State) -> Dict:
    """Detect errors using Groq API."""
    try:
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
        prompt = PromptTemplate.from_template(
            """Analyze the following document chunks for technical errors related to these banking terms: {terms}.
            Identify inaccuracies, missing details, inconsistencies, or ambiguous definitions.
            Return a JSON list of errors, where each error is an object with:
            - term: The banking term involved (string)
            - error: Description of the error (string)
            - location: Approximate location, e.g., page number or section (string)
            Example: [{{ "term": "KYC", "error": "Missing verification process", "location": "Section 3" }}]
            If no errors are found, return an empty list: []
            Context: {context}
            Output ONLY valid JSON, enclosed in square brackets."""
        )
        context = "\n\n".join(doc.page_content for doc in state.context)
        if not context.strip():
            logger.warning(f"No context available for file_id: {state.file_id}")
            return {"errors": []}
        messages = prompt.format_prompt(terms=", ".join(KEY_TERMS), context=context)
        response = llm.invoke(messages)
        if not response.content:
            logger.error(f"Empty response from Groq API for file_id: {state.file_id}")
            raise ValueError("Empty response from Groq API")
        json_str = response.content.strip()
        try:
            errors = json.loads(json_str)
            if not isinstance(errors, list):
                logger.error(f"Invalid response format from Groq API for file_id: {state.file_id}: {json_str}")
                raise ValueError("Response is not a JSON list")
        except json.JSONDecodeError:
            json_match = re.match(r'\[\s*(?:\{.*?\}\s*,\s*)*\{.*?\}\s*\]|\[\]', json_str, re.DOTALL)
            if not json_match:
                logger.error(f"No valid JSON list found in response for file_id: {state.file_id}: {json_str}")
                raise ValueError("No valid JSON list found in response")
            json_str = json_match.group(0)
            errors = json.loads(json_str)
        for error in errors:
            if not isinstance(error, dict) or not all(key in error for key in ["term", "error", "location"]):
                logger.error(f"Malformed error object in response for file_id: {state.file_id}: {error}")
                raise ValueError("Malformed error object in response")
        logger.info(f"Detected {len(errors)} errors for file_id: {state.file_id}")
        return {"errors": errors}
    except Exception as e:
        logger.error(f"Error detection failed for {state.file_id}: {str(e)}", exc_info=True)
        raise

def generate_report(state: State) -> Dict:
    """Generate fixing report and PDF."""
    try:
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
        prompt = PromptTemplate.from_template(
            """Based on the following errors in a banking document, generate a concise fixing report in markdown:
            Errors: {errors}
            Provide:
            - Summary of errors
            - Recommended fixes for each error
            - Conclusion emphasizing compliance and accuracy
            Keep the report under 500 words. If no errors, provide a brief report stating the document is compliant."""
        )
        errors_str = "\n".join([f"{e['term']}: {e['error']} (Location: {e['location']})" for e in state.errors]) if state.errors else "No errors detected."
        messages = prompt.format_prompt(errors=errors_str)
        response = llm.invoke(messages)
        if not response.content:
            logger.error(f"Empty report response from Groq API for file_id: {state.file_id}")
            raise ValueError("Empty report response from Groq API")
        report_content = response.content
        report_path = TEMP_DIR / f"report_{state.file_id}.pdf"
        generate_pdf_report(state.errors, report_content, report_path)
        logger.info(f"Generated report for file_id: {state.file_id}")
        return {"report": report_content, "report_path": str(report_path)}
    except Exception as e:
        logger.error(f"Report generation failed for {state.file_id}: {str(e)}", exc_info=True)
        raise

def build_graph():
    """Build LangGraph workflow."""
    workflow = StateGraph(State)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("detect_errors", detect_errors)
    workflow.add_node("generate_report", generate_report)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "detect_errors")
    workflow.add_edge("detect_errors", "generate_report")
    return workflow.compile()