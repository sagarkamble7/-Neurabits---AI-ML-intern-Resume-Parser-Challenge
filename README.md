# -Neurabits---AI-ML-intern-Resume-Parser-Challenge

This project is a Streamlit web application built on top of a custom resume parsing engine.
It allows users to upload a PDF resume and view the extracted information in a structured JSON format.

Approach
The user uploads a resume (PDF format) through the Streamlit interface.

The file is temporarily saved and passed into a ResumeParser class.

The parser extracts key fields such as:

Name

Email

Phone Number

LinkedIn Profile

Skills

Education

Work Experience

Certifications

Projects

The structured data is displayed as JSON on the web interface.

An optional expandable section shows detailed parsing results along with confidence scores.

Libraries and Tools Used
Streamlit – Building the web interface.

PyMuPDF (fitz) – Extracting text directly from PDFs.

spaCy – Natural language processing, primarily for named entity recognition.

pytesseract – Optical character recognition (OCR) for image-based PDFs.

scikit-learn – Utility functions for text processing and similarity calculations.

NLTK – Tokenization and stopwords handling.

dateutil – Parsing and normalizing dates.

Pillow (PIL) – Image processing from PDF pages.

NumPy and pandas – General data handling.

logging – Error logging and process tracking.

Assumptions and Limitations
Assumptions
Resumes are written primarily in English.

The uploaded document is in PDF format only.

The resume structure contains standard sections such as Skills, Education, and Experience.

Limitations
Highly graphical or non-standard resumes may not be parsed accurately even with OCR fallback.

Extracted information depends heavily on the formatting and clarity of the document.

Confidence scores are heuristic-based and are approximate.

Batch processing of multiple resumes simultaneously is not yet supported.

Parsing focuses mainly on essential sections; less common sections (e.g., awards, hobbies) are not extracted.
