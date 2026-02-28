# 🚀 ResumeIQ – AI Resume Analyzer

<img width="1890" height="862" alt="image" src="https://github.com/user-attachments/assets/41898942-b10c-4616-8e68-ce419464fbe2" />

ResumeIQ is an AI-powered web application that analyzes a resume against a given job description and provides:

- 📊 Match Score (percentage)
- 🔍 Keyword Matching Analysis
- ❌ Missing Skills Identification
- 💡 Actionable Improvement Suggestions

This project demonstrates backend API development combined with NLP-based similarity scoring in a clean and minimal UI.

---

## 🧠 Features

- Upload resume in PDF format
- Paste any job description
- Automatic text extraction from PDF
- TF-IDF based text vectorization
- Cosine similarity scoring
- Missing keyword detection
- Improvement suggestion engine
- Clean, minimal, and responsive UI
- REST API architecture

---

## 🏗 Architecture Overview

Frontend (HTML, CSS, JavaScript)  
⬇  
FastAPI Backend (Python)  
⬇  
NLP Engine (TF-IDF + Cosine Similarity)  

---

## 🛠 Tech Stack

### 🔹 Backend
- Python
- FastAPI
- scikit-learn
- PyPDF2 / pdfplumber

### 🔹 Frontend
- HTML
- CSS
- JavaScript

### 🔹 AI / NLP
- TF-IDF Vectorizer
- Cosine Similarity
- Text Preprocessing (stopword removal, normalization)

---

## 📂 Project Structure

```bash
ai-resume-analyzer/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── venv/
│
├── frontend/
│   ├── index.html
│  
└── README.md
```

---

## ⚙️ How to Run Locally

### 🔹 1. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python main.py
```

Backend runs on:

```
http://localhost:8000
```

If using FastAPI with Uvicorn:

```bash
uvicorn main:app --reload
```

---

### 🔹 2. Frontend Setup

```bash
cd frontend
python -m http.server 3000
```

Open in browser:

```
http://localhost:3000
```

---

## 📊 How Match Score is Calculated

1. Resume text is extracted from the uploaded PDF  
2. Job description text is cleaned and preprocessed  
3. Both texts are converted into TF-IDF vectors  
4. Cosine similarity is computed between vectors  
5. Similarity score is scaled into a percentage  

---

## 🧪 Example Workflow

1. Upload a resume (PDF)
2. Paste a job description
3. Click **Analyze**
4. View:
   - Match Score
   - Matched Keywords
   - Missing Skills
   - Improvement Suggestions

---

## 🔐 Error Handling

- Invalid file format detection
- Empty input validation
- Graceful API error responses
- Loading state during analysis

---

## 🚀 Future Improvements

- Deploy backend on AWS / Render
- Add JWT-based authentication
- Store analysis history in database
- Replace TF-IDF with BERT embeddings
- Add skill categorization model
- Add Docker support
- Add analytics dashboard

---

## 🎯 Why This Project Matters

This project demonstrates:

- Backend REST API development
- AI/NLP integration in production-style system
- Text similarity modeling
- Real-world HR tech use case
- Clean project structuring and documentation

---

## 👨‍💻 Author

Naga Janardhan Lingala  
Aspiring AI + Backend Engineer  

---
