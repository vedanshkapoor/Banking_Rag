# Banking Document Error Detection RAG System

## Overview

The **Banking Document Error Detection Retrieval-Augmented Generation (RAG) System** is a robust, modular application developed by Vedansh Kapoor under the supervision of Mr. Jay Kumar at **Tech Mahindra**. This system automates the analysis of banking documents to identify technical errors related to compliance terms such as Know Your Customer (KYC), Anti-Money Laundering (AML), Fraud Detection, Transaction Monitoring, and Compliance. Leveraging modern technologies including **FastAPI**, **LangChain**, **FAISS**, and the **Groq API**, the system processes PDF documents, extracts and indexes text, detects errors using large language models, and generates actionable PDF reports. Designed for scalability and reliability, it addresses critical compliance needs in the banking sector, reducing manual review time and ensuring regulatory adherence.

This project was developed to streamline Tech Mahindra’s compliance operations, offering a maintainable, enterprise-grade solution with robust error handling and comprehensive testing capabilities.

## Features

- **Automated Error Detection**: Identifies inaccuracies, missing details, or ambiguous definitions in banking documents using a RAG approach.
- **Modular Architecture**: Organized into distinct modules (`config.py`, `utils.py`, `service.py`, `main.py`) for maintainability and scalability.
- **High-Performance API**: Built with FastAPI for asynchronous endpoints supporting file upload, analysis, deletion, and health checks.
- **Efficient Document Processing**: Uses LangChain for text extraction and chunking, and FAISS for vector-based document retrieval.
- **Generative AI Integration**: Leverages the Groq API for error detection and report generation.
- **Robust Error Handling**: Manages file upload, PDF processing, vector store, API, and report generation errors with detailed logging.
- **Comprehensive Testing**: Includes unit tests and a local testing script (`rag_test.py`) for end-to-end validation with various PDF types.

## Prerequisites

- **Python 3.8+**
- **Groq API Key**: Obtain from [xAI API](https://x.ai/api).
- **PDF Test Files**: Valid PDFs, empty PDFs, scanned PDFs, and corrupted PDFs for testing.
- **System Dependencies**: Ensure `Tesseract-OCR` is installed for potential OCR support (optional).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd banking-rag-system
