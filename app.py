import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 1. Page Configuration
st.set_page_config(
    page_title="AI-Powered ATS Ranker Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced CSS Injector for the Custom Dark Cyber Theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1E293B;
    }
    .sidebar-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
    }
    .banner-container {
        background: linear-gradient(90deg, #0093E9 0%, #80D0C7 50%, #9B51E0 100%);
        padding: 35px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 147, 233, 0.2);
    }
    .banner-title {
        color: #FFFFFF !important;
        font-size: 42px !important;
        font-weight: 800 !important;
        margin-bottom: 10px;
    }
    .banner-subtitle {
        color: #E2E8F0 !important;
        font-size: 18px !important;
        font-style: italic;
    }
    .stTextArea textarea {
        background-color: #05070F !important;
        color: #F8FAFC !important;
        border: 2px solid #0093E9 !important;
        border-image: linear-gradient(to right, #0093E9, #9B51E0) 1 !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #05070F !important;
        border: 2px dashed #9B51E0 !important;
        border-radius: 12px !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0093E9 0%, #9B51E0 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100% !important;
        height: 55px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(155, 81, 224, 0.4);
    }
    [data-testid="stMetric"] {
        background-color: #05070F !important;
        border: 1px solid #1E293B !important;
        padding: 20px !important;
        border-radius: 12px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 55px !important;
        font-weight: 800 !important;
        color: #00FFCC !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    }
    /* Visual UI Pills/Tags for Keyword Dashboard */
    .keyword-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin: 4px;
    }
    .pill-matched {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
    }
    .pill-missing {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
    }
    h1, h2, h3, h4, h5, h6, p, span { color: #F8FAFC !important; }
    </style>
""", unsafe_allow_html=True)

# Helper function to dynamically extract text based on file format
def extract_text_from_resume(file):
    file_type = file.name.split('.')[-1].lower()
    if file_type == "txt":
        return file.read().decode("utf-8")
    elif file_type == "pdf":
        import pypdf
        pdf_reader = pypdf.PdfReader(file)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text: extracted_text += text + "\n"
        return extracted_text
    elif file_type == "docx":
        import docx2txt
        return docx2txt.process(file)
    return None

# Helper algorithm to analyze vocabulary differences
def analyze_keywords(job, resume):
    # Standard cleanup regex
    words_job = set(re.findall(r'\b[a-zA-Z]{3,15}\b', job.lower()))
    words_resume = set(re.findall(r'\b[a-zA-Z]{3,15}\b', resume.lower()))
    
    # Use TF-IDF vocabulary to grab meaningful non-stopword tokens
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        vectorizer.fit([job])
        important_keywords = set(vectorizer.get_feature_names_out())
    except:
        important_keywords = words_job
        
    matched = sorted(list(important_keywords.intersection(words_resume)))
    missing = sorted(list(important_keywords.difference(words_resume)))
    return matched[:15], missing[:15] # Top 15 samples each for tight UI styling

# 3. Sidebar Configuration
with st.sidebar:
    st.markdown("<div style='text-align: center;'><img src='https://cdn-icons-png.flaticon.com/512/8816/8816405.png' width='90' style='filter: drop-shadow(0px 0px 8px #0093E9);'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: 10px;'>ATS Dashboard Control</h2>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-card'><h4 style='color: #0093E9 !important; margin-top:0;'>How it works:</h4><ol style='padding-left: 20px; font-size: 14px; line-height: 1.6;'><li><b>Paste</b> target job requirement text.</li><li><b>Upload</b> CV/Resume (PDF, DOCX, TXT).</li><li><b>Analyze</b> vocabulary distribution mappings.</li></ol></div>", unsafe_allow_html=True)
    st.write("---")
    st.caption("Task 3 - Machine Learning ATS Ranking System")

# 4. Main Banner Component
st.markdown("<div class='banner-container'><div class='banner-title'>🧠 AI-Powered Applicant Tracking System</div><div class='banner-subtitle'>Optimize your recruitment pipeline with mathematical text alignment mapping.</div></div>", unsafe_allow_html=True)

# 5. Two Column Input Interface Layout
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📋 Target Job Description")
    job_text = st.text_area("Paste requirements here...", height=280, label_visibility="collapsed")
with col2:
    st.markdown("### 📄 Candidate Resume")
    uploaded_file = st.file_uploader("Drop candidate file here", type=["pdf", "docx", "txt"], label_visibility="collapsed")

st.write("")

# 6. Execution and Analytics Matching Blocks
if st.button("🚀 Execute Machine Learning Match Assessment"):
    if job_text and uploaded_file:
        with st.spinner("Processing TF-IDF Vectorizer calculations..."):
            try:
                resume_text = extract_text_from_resume(uploaded_file)
                
                if not resume_text or len(resume_text.strip()) == 0:
                    st.error("❌ Extraction Failed: Could not extract readable text from the uploaded file.")
                else:
                    # Run standard TF-IDF and Cosine algorithms
                    text_corpus = [job_text, resume_text]
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(text_corpus)
                    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                    match_score = float(similarity_matrix[0][0]) * 100
                    
                    st.write("---")
                    st.markdown("## 📊 Matching Metrics Result")
                    
                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        st.metric(label="Overall Match Score", value=f"{match_score:.1f}%")
                    with res_col2:
                        if match_score >= 75:
                            st.balloons()
                            st.success("### 🔥 Strong Candidate Match!\nThe skills and terminology found in the resume heavily align with your target profile.")
                        elif match_score >= 40:
                            st.info("### 👍 Competitive Match\nThe candidate shares solid foundational overlap, but might be missing critical secondary keywords.")
                        else:
                            st.warning("### ⚠️ Low Match Discrepancy\nSignificant terminology mismatch detected. The resume lacks key core requirements outlined in the description.")
                    
                    # NEW UPGRADE: Extraction of Keyword Analytics Mappings
                    matched_words, missing_words = analyze_keywords(job_text, resume_text)
                    
                    st.write("")
                    st.markdown("## 🔍 Deep Keyword Profile Insights")
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        st.markdown("#### ✅ Matched Competencies")
                        if matched_words:
                            pills_html = "".join([f"<span class='keyword-pill pill-matched'>{w}</span>" for w in matched_words])
                            st.markdown(pills_html, unsafe_allow_html=True)
                        else:
                            st.caption("No significant overlapping keywords discovered.")
                            
                    with analysis_col2:
                        st.markdown("#### ❌ Missing Target Keywords")
                        if missing_words:
                            pills_html = "".join([f"<span class='keyword-pill pill-missing'>{w}</span>" for w in missing_words])
                            st.markdown(pills_html, unsafe_allow_html=True)
                        else:
                            st.caption("Perfect keyword coverage! No major gaps found.")
                    
                    st.write("")
                    with st.expander("⚙️ View Raw Extracted Metadata"):
                        st.write(f"**Resume Data Size:** {len(resume_text)} characters")
                        st.write(f"**Job Description Data Size:** {len(job_text)} characters")
                        
            except Exception as e:
                st.error(f"An unexpected data processing fault occurred: {e}")
    else:
        st.error("❌ Missing Data: Please ensure both input panels are filled out before execution.")