import re
import json
import os
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime
import spacy
import logging
from dateutil import parser as date_parser
import pytesseract
from PIL import Image
import io
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading Spacy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class ResumeParser:
    def __init__(self):
        # Common skills list - can be expanded
        self.common_skills = [
            # Programming Languages
            "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "php", "swift",
            "kotlin", "rust", "scala", "perl", "html", "css", "sql", "r", "matlab", "bash", "shell",
            
            # Frameworks & Libraries
            "react", "angular", "vue", "django", "flask", "spring", "node.js", "express", 
            "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "kubernetes", "docker",
            
            # Database
            "mysql", "postgresql", "mongodb", "oracle", "sql server", "redis", "elasticsearch",
            
            # Cloud & DevOps
            "aws", "azure", "gcp", "jenkins", "gitlab ci", "github actions", "terraform", "ansible",
            
            # Tools & Others
            "git", "jira", "figma", "sketch", "photoshop", "illustrator", "tableau", "power bi",
            "machine learning", "deep learning", "nlp", "agile", "scrum", "kanban", "ci/cd"
        ]
        
        # Education related keywords
        self.education_keywords = [
            "education", "academic", "university", "college", "school", "institute", 
            "bachelor", "master", "phd", "doctorate", "degree", "bs", "ms", "ba", "ma", 
            "b.tech", "m.tech", "b.e.", "m.e.", "b.sc", "m.sc", "mba"
        ]
        
        # Experience related keywords
        self.experience_keywords = [
            "experience", "work", "employment", "job", "career", "professional", 
            "position", "role", "responsibility", "company", "corporation", "firm", 
            "organization", "employer"
        ]
        
        # Certification related keywords
        self.certification_keywords = [
            "certification", "certificate", "certified", "credential", "license", 
            "accreditation", "diploma"
        ]
        
        # Project related keywords
        self.project_keywords = [
            "project", "development", "implementation", "application", "system", 
            "solution", "initiative", "team project", "individual project", "created", 
            "developed", "designed", "built", "architected"
        ]
        
        # Regex patterns
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'(\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}'
        self.linkedin_pattern = r'(https?://)?(www\.)?linkedin\.com/in/[\w-]+/?'
        self.date_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)[a-z]*\.?\s+\d{4}\s*[-–—]\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december|present|current)[a-z]*\.?\s+\d{0,4}'
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF, with fallback to OCR if needed"""
        doc = fitz.open(pdf_path)
        text = ""
        images = []
        
        # First try direct text extraction
        for page in doc:
            text += page.get_text()
            
            # If the page has very little text, it might be image-based
            if len(page.get_text().strip()) < 50:
                # Get images from the page
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        
        # If text is very short, try OCR on extracted images
        if len(text.strip()) < 200 and images:
            logger.info("Text extraction yielded limited results. Attempting OCR...")
            ocr_text = ""
            for img in images:
                try:
                    ocr_text += pytesseract.image_to_string(img)
                except Exception as e:
                    logger.error(f"OCR error: {e}")
            
            # If OCR found more text, use it
            if len(ocr_text.strip()) > len(text.strip()):
                logger.info("Using OCR results instead of direct text extraction")
                text = ocr_text
        
        return text

    def normalize_date(self, date_str):
        """Normalize date formats to YYYY-MM format"""
        if not date_str or date_str.lower() in ['present', 'current']:
            return date_str
        
        try:
            parsed_date = date_parser.parse(date_str, fuzzy=True)
            return parsed_date.strftime("%Y-%m")
        except Exception:
            return date_str  # Return original if parsing fails

    def normalize_duration(self, duration_str):
        """Normalize duration strings"""
        if not duration_str:
            return None
            
        # Convert the duration string to lowercase for easier matching
        duration_lower = duration_str.lower()
        
        # Extract dates using regex
        matches = re.findall(self.date_pattern, duration_lower, re.IGNORECASE)
        
        if matches:
            # Extract the first and second parts of the matched duration
            start_date = matches[0][0] + ' ' + matches[0][1].strip()
            
            # The end date might be 'present' or a date
            end_part = matches[0][2] + (' ' + matches[0][3].strip() if matches[0][3].strip() else '')
            if 'present' in end_part:
                end_date = 'Present'
            else:
                end_date = end_part
                
            # Normalize dates
            norm_start = self.normalize_date(start_date)
            norm_end = 'Present' if end_date == 'Present' else self.normalize_date(end_date)
            
            return f"{norm_start} - {norm_end}"
        
        return duration_str  # Return original if parsing fails

    def calculate_confidence(self, field_value, field_type):
        """Calculate confidence score for an extracted field"""
        if field_value is None or field_value == "" or (isinstance(field_value, list) and len(field_value) == 0):
            return 0.0
            
        confidence = 0.5  # Default mid-point
        
        # Specific rules for different field types
        if field_type == "email":
            if re.match(self.email_pattern, field_value):
                confidence = 0.95
        elif field_type == "phone":
            if re.match(self.phone_pattern, field_value):
                confidence = 0.90
        elif field_type == "linkedin":
            if re.match(self.linkedin_pattern, field_value):
                confidence = 0.95
        elif field_type == "name":
            # Check if it appears to be a name (2+ words, proper case, etc.)
            words = field_value.split()
            if len(words) >= 2 and all(w.istitle() for w in words):
                confidence = 0.85
            else:
                confidence = 0.6
        elif field_type == "skills":
            if isinstance(field_value, list) and len(field_value) > 0:
                # Higher confidence with more identified skills
                confidence = min(0.9, 0.5 + (len(field_value) * 0.02))
        elif field_type == "education" or field_type == "experience":
            if isinstance(field_value, list) and len(field_value) > 0:
                # Check if required fields are populated
                complete_entries = 0
                for entry in field_value:
                    if all(entry.get(k) for k in entry.keys()):
                        complete_entries += 1
                confidence = min(0.9, 0.6 + (complete_entries / len(field_value) * 0.3))
        elif field_type == "certifications" or field_type == "projects":
            if isinstance(field_value, list) and len(field_value) > 0:
                confidence = 0.7
        
        return round(confidence, 2)
        
    def extract_name(self, text):
        """Extract full name using NLP"""
        first_lines = text.split('\n')[:3]  # Usually name is at the top
        
        # Try to find a name in the first few lines
        for line in first_lines:
            line = line.strip()
            if line and len(line) < 50:  # Names are typically short
                doc = nlp(line)
                # Look for a person name
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        return ent.text
                
                # If no entity found but the line looks like a name (2-3 words, proper case)
                words = line.split()
                if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                    return line
        
        # Backup: just take the first line if it's short
        for line in first_lines:
            if line and 2 <= len(line.split()) <= 4:
                return line.strip()
        
        return None
        
    def extract_contact_info(self, text):
        """Extract email, phone, and LinkedIn profile"""
        email = None
        phone = None
        linkedin = None
        
        # Find email
        email_matches = re.findall(self.email_pattern, text)
        if email_matches:
            email = email_matches[0]
            
        # Find phone number
        phone_matches = re.findall(self.phone_pattern, text)
        if phone_matches:
            phone = phone_matches[0]
            
        # Find LinkedIn URL
        linkedin_matches = re.findall(self.linkedin_pattern, text, re.IGNORECASE)
        if linkedin_matches:
            linkedin = linkedin_matches[0]
            
        return email, phone, linkedin
        
    def extract_skills(self, text):
        """Extract skills from text"""
        skills = []
        text_lower = text.lower()
        
        # Check for common skill keywords
        for skill in self.common_skills:
            # Match whole words only (with word boundaries)
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                skills.append(skill)
                
        # Check for skill sections
        skill_section_pattern = r'(?:skills|technical skills|core competencies|proficiencies)(?::|.{0,10})\s*([\s\S]*?)\n\n'
        skill_sections = re.findall(skill_section_pattern, text_lower, re.IGNORECASE)
        
        if skill_sections:
            for section in skill_sections:
                # Split by commas, bullets, or new lines
                items = re.split(r'[,•\n]', section)
                for item in items:
                    item = item.strip().lower()
                    if item and len(item) > 2 and item not in skills:
                        # Filter out noise - items that are too long or too short
                        if 2 < len(item) < 25:
                            skills.append(item)
        
        return list(set(skills))  # Remove duplicates
        
    def extract_education(self, text):
        """Extract education details"""
        education = []
        
        # Identify education section
        education_pattern = r'(?:education|academic|qualification)(?::|.{0,10})\s*([\s\S]*?)(?:\n\n|\Z)'
        education_sections = re.findall(education_pattern, text, re.IGNORECASE)
        
        if not education_sections:
            # Try another approach - look for degree keywords
            lines = text.split('\n')
            degree_pattern = r'\b(?:bachelor|master|phd|doctorate|bs|ms|ba|ma|b\.tech|m\.tech|mba)\b'
            education_lines = []
            
            capturing = False
            for i, line in enumerate(lines):
                if re.search(r'\beducation\b', line, re.IGNORECASE):
                    capturing = True
                    education_lines.append(line)
                    continue
                    
                if capturing:
                    if i < len(lines) - 1 and not line.strip() and not lines[i + 1].strip():
                        capturing = False  # Two empty lines might indicate end of section
                    elif re.search(r'\b(?:experience|work|employment)\b', line, re.IGNORECASE):
                        capturing = False  # New section started
                    else:
                        education_lines.append(line)
            
            education_text = '\n'.join(education_lines)
            education_sections = [education_text] if education_text.strip() else []
        
        for section in education_sections:
            # Split into entries (typically separated by newlines)
            entries = re.split(r'\n{2,}', section)
            
            for entry in entries:
                if not entry.strip():
                    continue
                    
                # Look for degree, institution and years
                degree = None
                institution = None
                year = None
                
                # Extract degree
                degree_patterns = [
                    r'\b(?:Bachelor|Master|PhD|Doctorate|BS|MS|BA|MA|B\.Tech|M\.Tech|MBA)[^,\n]*(?:of|in)[^,\n]*',
                    r'\b(?:Bachelor|Master|PhD|Doctorate|BS|MS|BA|MA|B\.Tech|M\.Tech|MBA)[^,\n]*'
                ]
                
                for pattern in degree_patterns:
                    degree_match = re.search(pattern, entry, re.IGNORECASE)
                    if degree_match:
                        degree = degree_match.group(0).strip()
                        break
                
                # Extract institution
                institute_indicators = [
                    r'(?:University|College|Institute|School) of [A-Z][a-z]+',
                    r'[A-Z][a-z]+ (?:University|College|Institute|School)',
                    r'[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)+ (?:University|College|Institute|School)'
                ]
                
                for pattern in institute_indicators:
                    inst_match = re.search(pattern, entry)
                    if inst_match:
                        institution = inst_match.group(0).strip()
                        break
                
                # If we didn't find an institution yet, try NER
                if not institution:
                    doc = nlp(entry)
                    for ent in doc.ents:
                        if ent.label_ == "ORG" and len(ent.text) > 3:
                            institution = ent.text
                            break
                
                # Extract year
                year_match = re.search(r'(?:19|20)\d{2}(?:\s*-\s*(?:19|20)\d{2}|(?:\s*-\s*Present)?)', entry)
                if year_match:
                    year = year_match.group(0).strip()
                
                if degree or institution:  # At least one field must be present
                    education.append({
                        "degree": degree,
                        "institution": institution,
                        "year": year
                    })
        
        return education
        
    def extract_experience(self, text):
        """Extract work experience details"""
        experience = []
        
        # Identify experience section
        exp_pattern = r'(?:experience|work history|employment|professional experience)(?::|.{0,10})\s*([\s\S]*?)(?:\n\n\n|\Z)'
        exp_sections = re.findall(exp_pattern, text, re.IGNORECASE)
        
        if not exp_sections:
            # Try another approach - look for company/job patterns
            lines = text.split('\n')
            exp_lines = []
            
            capturing = False
            for i, line in enumerate(lines):
                if re.search(r'\b(?:experience|work history|employment)\b', line, re.IGNORECASE):
                    capturing = True
                    exp_lines.append(line)
                    continue
                    
                if capturing:
                    if i < len(lines) - 1 and not line.strip() and not lines[i + 1].strip():
                        capturing = False  # Two empty lines might indicate end of section
                    elif re.search(r'\b(?:education|skills|projects|certifications)\b', line, re.IGNORECASE):
                        capturing = False  # New section started
                    else:
                        exp_lines.append(line)
            
            exp_text = '\n'.join(exp_lines)
            exp_sections = [exp_text] if exp_text.strip() else []
        
        for section in exp_sections:
            # Split into entries (typically separated by multiple newlines)
            entries = re.split(r'\n{2,}', section)
            
            current_company = None
            current_title = None
            current_duration = None
            current_description = []
            
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue
                
                lines = entry.split('\n')
                first_line = lines[0].strip()
                
                # Check if this looks like a new job entry
                is_new_job = False
                
                # Look for company patterns: capital letters followed by location or a date pattern
                company_pattern = r'([A-Z][A-Za-z0-9\s&,.\'()-]+)(?:,\s*[A-Za-z]+|\s+\d{1,2}/\d{1,2}|\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))'
                company_match = re.search(company_pattern, first_line)
                
                # Look for date patterns that often accompany job entries
                date_match = re.search(self.date_pattern, entry, re.IGNORECASE)
                
                # Look for title patterns: Title at Company or Company - Title
                title_patterns = [
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)\s+(?:at|@|,)\s+([A-Z][A-Za-z0-9\s&,.\'()-]+)',
                    r'([A-Z][A-Za-z0-9\s&,.\'()-]+)\s+[-–—]\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)'
                ]
                
                title_company = None
                for pattern in title_patterns:
                    match = re.search(pattern, first_line)
                    if match:
                        title_company = match
                        break
                
                # If we have strong indicators of a new job entry, process the previous one if it exists
                if (company_match or date_match or title_company) and (company_match or title_company):
                    is_new_job = True
                    
                    # Save the previous job entry if there is one
                    if current_company or current_title:
                        experience.append({
                            "company": current_company,
                            "title": current_title,
                            "duration": current_duration,
                            "description": "\n".join(current_description) if current_description else None
                        })
                        
                        # Reset for new entry
                        current_company = None
                        current_title = None
                        current_duration = None
                        current_description = []
                    
                    # Extract company and title from new entry
                    if title_company:
                        if len(title_company.groups()) >= 2:
                            # Check pattern to determine which is title and which is company
                            if '@' in title_company.group(0) or 'at' in title_company.group(0):
                                current_title = title_company.group(1).strip()
                                current_company = title_company.group(2).strip()
                            else:
                                current_company = title_company.group(1).strip()
                                current_title = title_company.group(2).strip()
                    elif company_match:
                        current_company = company_match.group(1).strip()
                        # Try to find title nearby
                        for line in lines[:2]:  # Check first two lines
                            title_indicators = ['manager', 'director', 'engineer', 'developer', 'analyst', 'specialist', 'coordinator']
                            for indicator in title_indicators:
                                if indicator in line.lower() and line != first_line:
                                    current_title = line.strip()
                                    break
                    
                    # Extract duration
                    if date_match:
                        current_duration = date_match.group(0)
                    
                    # The rest is description
                    if is_new_job:
                        # Skip the title/company line
                        description_lines = lines[1:] if lines else []
                    else:
                        description_lines = lines
                        
                    current_description = [line.strip() for line in description_lines if line.strip()]
                else:
                    # If not a new job, add to current description
                    current_description.append(entry)
            
            # Don't forget to add the last job entry
            if current_company or current_title:
                experience.append({
                    "company": current_company,
                    "title": current_title,
                    "duration": self.normalize_duration(current_duration) if current_duration else None,
                    "description": "\n".join(current_description) if current_description else None
                })
        
        return experience
        
    def extract_certifications(self, text):
        """Extract certifications"""
        certifications = []
        
        # Look for certifications section
        cert_pattern = r'(?:certification|certificate|credential)s?(?::|.{0,10})\s*([\s\S]*?)(?:\n\n\n|\Z)'
        cert_sections = re.findall(cert_pattern, text, re.IGNORECASE)
        
        if cert_sections:
            for section in cert_sections:
                # Split by lines, bullets, or commas
                items = re.split(r'[\n•,]', section)
                for item in items:
                    item = item.strip()
                    if item and len(item) > 5:
                        # Filter out noise
                        if 5 < len(item) < 100:  # Reasonable length for certification name
                            certifications.append(item)
        else:
            # If no specific section, look for certificate keywords
            cert_keywords = ['certified', 'certification', 'certificate', 'credential']
            lines = text.split('\n')
            
            for line in lines:
                if any(keyword in line.lower() for keyword in cert_keywords):
                    # Check that it's not just mentioning certifications in general
                    if len(line.strip()) > 10 and len(line.strip()) < 100:
                        certifications.append(line.strip())
        
        return list(set(certifications))  # Remove duplicates
        
    def extract_projects(self, text):
        """Extract projects"""
        projects = []
        
        # Look for projects section
        project_pattern = r'(?:project|portfolio)s?(?::|.{0,10})\s*([\s\S]*?)(?:\n\n\n|\Z)'
        project_sections = re.findall(project_pattern, text, re.IGNORECASE)
        
        if project_sections:
            for section in project_sections:
                # Try to identify individual projects (usually start with a title)
                entries = re.split(r'\n{2,}', section)
                
                for entry in entries:
                    entry = entry.strip()
                    if entry:
                        # Take the first line as title, rest as description
                        lines = entry.split('\n')
                        title = lines[0].strip()
                        
                        # Only include if it looks like a real project (not too short or just a category)
                        if len(title) > 3 and not title.lower().endswith(':') and title.lower() != 'projects':
                            projects.append(title)
        
        return projects
        
    def parse_resume(self, pdf_path):
        """Parse resume and return structured data with confidence scores"""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text or len(text) < 100:
                logger.error(f"Failed to extract sufficient text from {pdf_path}")
                return {
                    "error": "Could not extract sufficient text from the PDF"
                }
            
            # Extract all information
            name = self.extract_name(text)
            email, phone, linkedin = self.extract_contact_info(text)
            skills = self.extract_skills(text)
            education = self.extract_education(text)
            experience = self.extract_experience(text)
            certifications = self.extract_certifications(text)
            projects = self.extract_projects(text)
            
            # Build result with confidence scores
            result = {
                "name": {
                    "value": name,
                    "confidence": self.calculate_confidence(name, "name")
                },
                "email": {
                    "value": email,
                    "confidence": self.calculate_confidence(email, "email")
                },
                "phone": {
                    "value": phone,
                    "confidence": self.calculate_confidence(phone, "phone")
                },
                "linkedin": {
                    "value": linkedin,
                    "confidence": self.calculate_confidence(linkedin, "linkedin")
                },
                "skills": {
                    "value": skills,
                    "confidence": self.calculate_confidence(skills, "skills")
                },
                "education": {
                    "value": education,
                    "confidence": self.calculate_confidence(education, "education")
                },
                "experience": {
                    "value": experience,
                    "confidence": self.calculate_confidence(experience, "experience")
                },
                "certifications": {
                    "value": certifications,
                    "confidence": self.calculate_confidence(certifications, "certifications")
                },
                "projects": {
                    "value": projects,
                    "confidence": self.calculate_confidence(projects, "projects")
                }
            }
            
            # For output format without confidence scores
            clean_result = {
                "name": name or None,
                "email": email or None,
                "phone": phone or None,
                "linkedin": linkedin or None,
                "skills": skills or [],
                "education": education or [],
                "experience": experience or [],
                "certifications": certifications or [],
                "projects": projects or []
            }
            
            # Log missing fields
            for field, value in clean_result.items():
                if value is None or (isinstance(value, list) and len(value) == 0):
                    logger.warning(f"Missing field in resume: {field}")
            
            return {
                "detailed": result,  # With confidence scores
                "parsed": clean_result  # Clean format for output
            }
            
        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            return {
                "error": f"Failed to parse resume: {str(e)}"
            }

# Main execution block for running the script directly
if __name__ == "__main__":
    parser = ResumeParser()
    
    # Check if filename is provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py <path_to_resume.pdf>")
        sys.exit(1)
    
    # Use the provided filename instead of hardcoded path
    pdf_path = sys.argv[1]
    
    try:
        result = parser.parse_resume(pdf_path)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            # Print basic information
            print("\n=== BASIC INFORMATION ===")
            if result["parsed"]["name"]:
                print(f"Name: {result['parsed']['name']}")
            if result["parsed"]["email"]:
                print(f"Email: {result['parsed']['email']}")
            if result["parsed"]["phone"]:
                print(f"Phone: {result['parsed']['phone']}")
            
            # Print number of entries found
            print(f"Skills found: {len(result['parsed']['skills'])}")
            print(f"Education entries: {len(result['parsed']['education'])}")
            print(f"Experience entries: {len(result['parsed']['experience'])}")
            
            # Save cleaned result to JSON
            output_file = os.path.splitext(os.path.basename(pdf_path))[0] + "_parsed.json"
            with open(output_file, "w") as f:
                json.dump(result["parsed"], f, indent=2)
            
            print(f"Complete parsed resume saved to {output_file}")
            
    except Exception as e:
        print(f"Error: {str(e)}")