import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration (Must be first)
st.set_page_config(
    page_title="AI-Powered ATS Ranker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced CSS Injector for the Custom Dark Cyber Theme
st.markdown("""
    <style>
    /* Force main container background to dark */
    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1E293B;
    }
    
    /* Neon Glowing Sidebar Container for How it works */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom Neon Gradient Header Banner */
    .banner-container {
        background: linear-gradient(90deg, #0093E9 0%, #80D0C7 50%, #9B51E0 100%);
        padding: 35px;
        border-radius: 12px;
        text-align: left;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 147, 233, 0.2);
    }
    .banner-title {
        color: #FFFFFF !important;
        font-size: 42px !important;
        font-weight: 800 !important;
        font-family: 'Inter', sans-serif;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .banner-subtitle {
        color: #E2E8F0 !important;
        font-size: 18px !important;
        font-style: italic;
        font-weight: 300;
    }

    /* Style Text Areas with Cyber Glowing Borders */
    .stTextArea textarea {
        background-color: #05070F !important;
        color: #F8FAFC !important;
        border: 2px solid #0093E9 !important;
        border-image: linear-gradient(to right, #0093E9, #9B51E0) 1 !important;
        border-radius: 8px !important;
    }
    
    /* Style Drag-and-Drop Uploader Box */
    [data-testid="stFileUploadDropzone"] {
        background-color: #05070F !important;
        border: 2px dashed #9B51E0 !important;
        border-radius: 12px !important;
    }
    
    /* Custom Action Button Styling */
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
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 147, 233, 0.6);
    }
    
    /* Metric Card Customization */
    [data-testid="stMetric"] {
        background-color: #05070F !important;
        border: 1px solid #1E293B !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: inset 0 0 10px rgba(0, 147, 233, 0.2);
    }
    [data-testid="stMetricValue"] {
        font-size: 55px !important;
        font-weight: 800 !important;
        color: #00FFCC !important; /* Cyberpunk Neon Teal */
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }
    
    /* Fix built-in header text color visibility */
    h1, h2, h3, h4, h5, h6, p, span {
        color: #F8FAFC !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration
with st.sidebar:
    # Futuristic Glowing Icon
    st.markdown("""
    <div style='text-align: center;'>
        <img src="https://cdn-icons-png.flaticon.com/512/8816/8816405.png" width="90" style="filter: drop-shadow(0px 0px 8px #0093E9);">
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: 10px;'>ATS Dashboard Control</h2>", unsafe_allow_html=True)
    
    # Styled text container cards
    st.markdown("""
    <div class="sidebar-card">
        <h4 style='color: #0093E9 !important; margin-top:0;'>How it works:</h4>
        <ol style='padding-left: 20px; font-size: 14px; line-height: 1.6;'>
            <li><b>Paste</b> the target job description.</li>
            <li><b>Upload</b> the candidate's resume (.txt format).</li>
            <li><b>Run Analysis</b> to execute the TF-IDF Vectorizer matching algorithm.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("---")
    st.caption("Task 3 - Machine Learning ATS Ranking System")

# 4. Main Banner Component
st.markdown("""
    <div class="banner-container">
        <div class="banner-title">🧠 AI-Powered Applicant Tracking System</div>
        <div class="banner-subtitle">Optimize your recruitment pipeline with mathematical text alignment mapping.</div>
    </div>
""", unsafe_allow_html=True)

# 5. Two Column Input Interface Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Target Job Description")
    job_text = st.text_area("Paste requirements, key skills, and responsibilities here...", height=280, label_visibility="collapsed")

with col2:
    st.markdown("### 📄 Candidate Resume")
    # UPGRADED: Now accepts PDF, DOCX, and TXT files!
    uploaded_file = st.file_uploader("Drop candidate file here (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], label_visibility="collapsed")

st.write("")
st.write("")

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
            if text:
                extracted_text += text + "\n"
        return extracted_text
        
    elif file_type == "docx":
        import docx2txt
        return docx2txt.process(file)
        
    return None

# 6. Execution and Analytics Matching Blocks
if st.button("🚀 Execute Machine Learning Match Assessment"):
    if job_text and uploaded_file:
        with st.spinner("Processing TF-IDF Vectorizer calculations..."):
            try:
                # UPGRADED: Extracting text using our new robust multi-format function
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
                    
                    with st.expander("🔍 View Raw Extracted Metadata"):
                        st.write(f"**Resume Data Size:** {len(resume_text)} characters")
                        st.write(f"**Job Description Data Size:** {len(job_text)} characters")
                        
            except Exception as e:
                st.error(f"An unexpected data processing fault occurred: {e}")
    else:
        st.error("❌ Missing Data: Please ensure both input panels are filled out before execution.")