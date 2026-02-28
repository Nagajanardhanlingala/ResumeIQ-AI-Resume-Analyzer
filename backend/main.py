from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pdfplumber
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="AI Resume Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Common English stopwords
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "he","him","his","she","her","hers","it","its","they","them","their","what",
    "which","who","whom","this","that","these","those","am","is","are","was","were",
    "be","been","being","have","has","had","do","does","did","will","would","shall",
    "should","may","might","must","can","could","a","an","the","and","but","if","or",
    "because","as","until","while","of","at","by","for","with","about","against",
    "between","through","during","before","after","above","below","to","from","up",
    "down","in","out","on","off","over","under","then","once","here","there","when",
    "where","why","how","all","both","each","few","more","most","other","some","such",
    "no","nor","not","only","own","same","so","than","too","very","just","don","into",
    "also","any","etc","via","per","use","used","using","make","made","work","working"
}

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

def extract_keywords(text: str, top_n: int = 30) -> list[str]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=200)
    try:
        tfidf = vectorizer.fit_transform([text])
        scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_scores[:top_n] if score > 0]
    except:
        return text.split()[:top_n]

def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            raise ValueError("No text found in PDF")
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {str(e)}")

def generate_suggestions(missing_keywords: list[str], match_score: float) -> list[str]:
    suggestions = []
    
    # Group missing keywords into categories
    tech_tools = [k for k in missing_keywords if any(t in k for t in 
        ['python','java','sql','aws','docker','react','node','git','linux','cloud',
         'machine learning','deep learning','tensorflow','pytorch','kubernetes','api',
         'javascript','typescript','mongodb','redis','graphql','rest','agile','ci/cd'])]
    
    soft_skills = [k for k in missing_keywords if any(s in k for s in 
        ['communication','leadership','teamwork','management','collaboration','analytical',
         'problem solving','critical thinking'])]
    
    experience_terms = [k for k in missing_keywords if any(e in k for e in 
        ['experience','years','senior','junior','lead','architect','design','develop'])]
    
    if match_score < 40:
        suggestions.append("📌 Your resume needs significant alignment with this job. Consider a targeted rewrite.")
    elif match_score < 60:
        suggestions.append("📌 Moderate match — focus on adding missing keywords naturally throughout your resume.")
    elif match_score < 80:
        suggestions.append("📌 Good match! A few targeted additions can push you into the top-candidate tier.")
    else:
        suggestions.append("🏆 Excellent match! Fine-tune with specific metrics and achievements to stand out.")

    if tech_tools:
        tools_str = ', '.join(tech_tools[:4])
        suggestions.append(f"🛠️ Add technical skills: Mention {tools_str} in your skills section or project descriptions.")
    
    if soft_skills:
        soft_str = ', '.join(soft_skills[:3])
        suggestions.append(f"💬 Highlight soft skills: Incorporate {soft_str} with concrete examples.")
    
    if experience_terms:
        exp_str = ', '.join(experience_terms[:3])
        suggestions.append(f"📋 Align experience language: Use phrases like '{experience_terms[0]}' to mirror job requirements.")
    
    remaining = [k for k in missing_keywords if k not in tech_tools + soft_skills + experience_terms]
    if remaining[:4]:
        kw_str = ', '.join(remaining[:4])
        suggestions.append(f"🔑 Include domain keywords: {kw_str} — weave these into your work history naturally.")
    
    suggestions.append("📊 Quantify your achievements with metrics (e.g., 'Reduced load time by 40%', 'Led a team of 5').")
    
    return suggestions[:6]

@app.get("/")
def serve_frontend():
    return FileResponse("../frontend/index.html")

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    # Validate file type
    if not resume.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    
    # Read and extract PDF text
    file_bytes = await resume.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    
    resume_text = extract_pdf_text(file_bytes)
    
    # Preprocess
    clean_resume = preprocess_text(resume_text)
    clean_jd = preprocess_text(job_description)
    
    # TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        match_score = round(float(similarity) * 100, 1)
    except Exception:
        raise HTTPException(status_code=500, detail="Error computing similarity.")
    
    # Extract keywords
    resume_keywords = set(extract_keywords(clean_resume, 40))
    jd_keywords = set(extract_keywords(clean_jd, 40))
    
    matched_keywords = list(resume_keywords & jd_keywords)[:15]
    missing_keywords = list(jd_keywords - resume_keywords)[:20]
    
    suggestions = generate_suggestions(missing_keywords, match_score)
    
    return {
        "match_score": match_score,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords[:12],
        "suggestions": suggestions,
        "resume_word_count": len(resume_text.split()),
        "jd_word_count": len(job_description.split())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
