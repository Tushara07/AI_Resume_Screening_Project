import streamlit as st
import pandas as pd
import joblib
import PyPDF2
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

jobs = pd.read_csv(os.path.join(BASE_DIR, "data", "jobs_cleaned.csv"))
model = joblib.load(os.path.join(BASE_DIR, "model", "suitability_model.pkl"))
st.set_page_config(layout="wide", page_title="AI Resume Screening")

# ---------- LOAD DATA ----------

jobs['full_jd'] = (
    jobs['Job Title'] + " " +
    jobs['Category'] + " " +
    jobs['Education Requirement'].astype(str) + " " +
    jobs['Experience Years'].astype(str) + " " +
    jobs['Required Skills']
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# ---------- UI ----------
st.title("AI Resume Screening and Job Recommendation Dashboard")

st.sidebar.header("Candidate Input")
uploaded_file = st.sidebar.file_uploader("Upload Resume PDF", type=["pdf"])
analyze = st.sidebar.button("Analyze Resume")

if analyze:

    if uploaded_file is None:
        st.warning("Upload resume PDF first")
        st.stop()

    with st.spinner("Analyzing resume..."):

        resume_text = extract_text_from_pdf(uploaded_file)
        resume_clean = clean_text(resume_text)

        tfidf = TfidfVectorizer(max_features=5000)
        vectors = tfidf.fit_transform([resume_clean] + jobs['full_jd'].tolist())

        similarity_scores = cosine_similarity(vectors[0], vectors[1:])[0] * 100

        jobs['score'] = similarity_scores
        top_jobs = jobs.sort_values(by='score', ascending=False).head(5)

        best_job = top_jobs.iloc[0]['Job Title']
        best_score = top_jobs.iloc[0]['score']
        best_skills = top_jobs.iloc[0]['Required Skills']

        resume_set = set(resume_clean.split())
        job_set = set(clean_text(best_skills).split())

        matched = resume_set.intersection(job_set)
        missing = job_set - resume_set

        skill_percent = (len(matched) / len(job_set)) * 100 if len(job_set) > 0 else 0

        decision = model.predict([[best_score, skill_percent]])[0]

    st.subheader("Best Recommended Job")
    st.success(best_job)

    # KPI ROW
    c1, c2, c3 = st.columns(3)
    c1.metric("ATS Score", f"{best_score:.2f}%")
    c2.metric("Skill Match", f"{skill_percent:.2f}%")
    c3.metric("Missing Skills", len(missing))

    st.divider()

    # CENTER GAUGE
    center = st.columns([1,2,1])[1]

    with center:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=best_score,
            title={'text': "ATS Score"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#2E86C1"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # SKILLS
    s1, s2 = st.columns(2)

    with s1:
        st.subheader("Matched Skills")
        st.write(", ".join(matched) if matched else "No match")

    with s2:
        st.subheader("Missing Skills")
        st.write(", ".join(missing))

    st.divider()

    if decision == 1:
        st.success("Candidate Suitable")
    else:
        st.error("Candidate Not Suitable")

    st.divider()

    st.subheader("Other Recommended Jobs")

    rank_table = top_jobs[['Job Title', 'score']].reset_index(drop=True)

    rank_table.index = rank_table.index + 1

    rank_table.columns = ['Job Role', 'Match Score']

    rank_table['Match Score'] = rank_table['Match Score'].round(2).astype(str) + '%'

    st.table(rank_table)
