# 🧠 AI-Powered ATS Resume Ranking System

An enterprise-grade, high-fidelity web application built with **Streamlit** and **Python** that leverages machine learning to evaluate alignment between candidate resumes and specific job requirements.

## 🚀 Live Demo
🔗 **Try the live application here:** [Insert your Streamlit app link here]

## ✨ Features
* **Multi-Format Document Parsing:** Dynamically extracts text from `.pdf`, `.docx`, and `.txt` files flawlessly.
* **Vector Math Backend:** Uses `scikit-learn`'s **TF-IDF Vectorizer** and **Cosine Similarity** to calculate authentic mathematical text alignment scores.
* **Deep Keyword Insights:** Generates interactive tag profiles showcasing **Matched Core Competencies** and **Missing High-Value Keywords** to help candidates optimize their reach.
* **High-Fidelity Cyber UI:** Custom-built permanent dark theme optimized for mobile and desktop screens with custom glowing layout elements.

## 🧮 How It Works



1. The text corpus is tokenized and stripped of standard English stop words.
2. A TF-IDF vector matrix maps the unique statistical weights of industry keywords.
3. Cosine similarity calculates the directional angle between the job description vector and the resume vector to produce an exact percentage match.

## ⚙️ Tech Stack & Installation
* **Backend:** Python 3.11+, Scikit-Learn
* **Frontend UI:** Streamlit (Custom HTML/CSS injection)
* **Document Processing:** PyPDF, Docx2txt

```bash
git clone [https://github.com/Tommyjah/FUTURE_ML_03.git](https://github.com/Tommyjah/FUTURE_ML_03.git)
cd FUTURE_ML_03
pip install -r requirements.txt
python -m streamlit run app.py