import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration (Adds a custom tab title and browser icon)
st.set_page_config(
    page_title="AI-Powered ATS Ranker",
    page_icon="💼",
    layout="wide"
)

# 2. Inject Custom CSS for styling tweaks
st.markdown("""
    <style>
    /* Change the color of the main metrics card */
    [data-testid="stMetricValue"] {
        font-size: 45px;
        font-weight: bold;
        color: #1E3A8A; /* Deep blue corporate color */
    }
    /* Style the analyze button */
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        width: 100%;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Customization
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921226.png", width=80) # Generic professional avatar/logo
    st.title("ATS Dashboard Control")
    st.markdown("### How it works:")
    st.info("""
    1. **Paste** the target job description.
    2. **Upload** the candidate's resume (.txt format).
    3. **Run Analysis** to execute the TF-IDF Vectorizer matching algorithm.
    """)
    st.write("---")
    st.caption("Task 3 - Machine Learning ATS Ranking System")

# 4. Main Page Header
st.markdown("# 🧠 AI-Powered Applicant Tracking System")
st.markdown("##### *Optimize your recruitment pipeline with mathematical text alignment mapping.*")
st.write("---")

# 5. Two Column Layout for Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Target Job Description")
    job_text = st.text_area("Paste requirements, key skills, and responsibilities here...", height=250)

with col2:
    st.subheader("📄 Candidate Resume")
    uploaded_file = st.file_uploader("Drop candidate file here (.TXT only)", type=["txt"])

st.write("---")

# 6. Action and Results UI Block
if st.button("🚀 Execute Machine Learning Match Assessment"):
    if job_text and uploaded_file:
        with st.spinner("Processing TF-IDF Vectorizer calculations..."):
            try:
                # Read file
                resume_text = uploaded_file.read().decode("utf-8")
                
                # Math Processing
                text_corpus = [job_text, resume_text]
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(text_corpus)
                similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                match_score = float(similarity_matrix[0][0]) * 100
                
                # Display Customized Results Area
                st.markdown("### 📊 Matching Metrics Result")
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    # Displays the score prominently
                    st.metric(label="Overall Match Score", value=f"{match_score:.1f}%")
                
                with res_col2:
                    # Customized feedback card based on score thresholds
                    if match_score >= 75:
                        st.balloons()
                        st.success("### 🔥 Strong Candidate Match!\nThe skills and terminology found in the resume heavily align with your target profile.")
                    elif match_score >= 40:
                        st.info("### 👍 Competitive Match\nThe candidate shares solid foundational overlap, but might be missing critical secondary keywords.")
                    else:
                        st.warning("### ⚠️ Low Match Discrepancy\nSignificant terminology mismatch detected. The resume lacks key core requirements outlined in the description.")
                
                # Expander box for granular details
                with st.expander("🔍 View Raw Extracted Metadata"):
                    st.write("**Processed Document Size:**")
                    st.write(f"Resume Character Length: {len(resume_text)} characters")
                    st.write(f"Job Description Character Length: {len(job_text)} characters")
                    
            except Exception as e:
                st.error(f"An unexpected data processing fault occurred: {e}")
    else:
        st.error("❌ Missing Data: Please ensure both input panels are filled out before execution.")