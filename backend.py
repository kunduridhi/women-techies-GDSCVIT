import os
import spacy
import PyPDF2
import docx
import re
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Initialize Flask App
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract key details from text
def extract_resume_details(text):
    doc = nlp(text)
    skills = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
    experience = re.findall(r'\b\d+ years?\b', text, re.IGNORECASE)
    education = re.findall(r'\b(Bachelor|Master|PhD)\b', text, re.IGNORECASE)
    return {
        "skills": list(set(skills)),
        "experience": experience,
        "education": education
    }

# Function to match resumes with job descriptions
def match_resumes_with_job(resume_texts, job_description):
    vectorizer = TfidfVectorizer()
    texts = resume_texts + [job_description]
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])
    return similarity_scores.flatten()

# API Endpoint for Resume Upload
@app.route("/upload", methods=["POST"])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['resume']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    text = extract_text_from_pdf(file_path) if file.filename.endswith(".pdf") else extract_text_from_docx(file_path)
    details = extract_resume_details(text)
    
    return jsonify({"resume_details": details})

# API Endpoint for Job Matching
@app.route("/match", methods=["POST"])
def match_resumes():
    data = request.json
    resumes = data.get("resumes", [])
    job_desc = data.get("job_description", "")
    
    scores = match_resumes_with_job(resumes, job_desc)
    return jsonify({"match_scores": scores.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
