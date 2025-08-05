from fpdf import FPDF
from pathlib import Path

# Ensure the output directory exists
output_dir = Path("D:/personal_projects/banking_rag")
output_dir.mkdir(exist_ok=True)

# Create a test PDF with intentional errors
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

# Sample content with errors
content = """
Banking Document: Compliance and Monitoring Procedures

Section 1: KYC Process
The KYC process is implemented for all customers. Verification is done but not specified how.

Section 2: AML Policy
AML checks are performed daily. Transactions above $10,000 require manual review, but small transactions are ignored, which contradicts standard AML protocols.

Section 3: Fraud Detection
Fraud Detection uses AI. No further details provided.

Section 4: Transaction Monitoring
Transaction Monitoring is active, but only for accounts with balances over $50,000, missing coverage for smaller accounts.

Section 5: Compliance
Compliance is ensured, but no regulatory framework is mentioned, which is ambiguous.
"""

pdf.multi_cell(0, 10, content)
output_path = output_dir / "test_errors.pdf"
pdf.output(str(output_path))

print(f"Test PDF created at {output_path}")