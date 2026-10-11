
import os
import re
import joblib
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer & Job Recommendation System")
st.caption(
    "Resume classification • Fuzzy skill matching • "
    "Job recommendations • Uncertainty estimation"
)

# -------------------------------
# 2. LOAD SAVED MODEL FILES
# -------------------------------
MODEL_DIR = "resume_analyzer_model"

required_files = [
    "calibrated_svm.pkl",
    "tfidf_vectorizer.pkl",
    "job_vectorizer.pkl",
    "sample_jobs.csv"
]

missing = [
    f for f in required_files
    if not os.path.exists(os.path.join(MODEL_DIR, f))
]

if missing:
    st.error(
        "Missing model files: " + ", ".join(missing)
        + ". Check your resume_analyzer_model folder."
    )
    st.stop()

model = joblib.load(
    os.path.join(MODEL_DIR, "calibrated_svm.pkl")
)
resume_vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)
job_vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "job_vectorizer.pkl")
)
jobs = pd.read_csv(
    os.path.join(MODEL_DIR, "sample_jobs.csv")
)

# -------------------------------
# 3. TEXT PREPROCESSING
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------
# 4. SKILL ALIASES
# -------------------------------
SKILL_ALIASES = {
    "python": ["python", "python programming"],
    "sql": ["sql", "mysql", "postgresql", "sqlite"],
    "pandas": ["pandas"],
    "power bi": ["power bi", "powerbi"],
    "statistics": ["statistics", "statistical analysis"],
    "data analysis": ["data analysis", "data analytics"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "numpy": ["numpy"],
    "tensorflow": ["tensorflow"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "javascript": ["javascript", "js"],
    "react": ["react", "react.js", "reactjs"],
    "git": ["git", "github"],
    "cybersecurity": ["cybersecurity", "cyber security"],
    "network security": ["network security"],
    "linux": ["linux"],
    "risk assessment": ["risk assessment"],
    "fastapi": ["fastapi"],
    "node.js": ["node.js", "nodejs"],
    "mongodb": ["mongodb", "mongo db"],
    "api development": ["api development", "rest api", "apis"]
}

def extract_skills(text):
    text = clean_text(text)
    found = set()

    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, text):
                found.add(skill)
                break

    return found

def fuzzy_skill_match(resume_skills, required_skills):
    matched = []
    missing_skills = []

    for skill in required_skills:
        if skill in resume_skills:
            matched.append(skill)
        else:
            best = process.extractOne(
                skill,
                list(resume_skills),
                scorer=fuzz.ratio
            ) if resume_skills else None

            if best and best[1] >= 88:
                matched.append(skill)
            else:
                missing_skills.append(skill)

    coverage = (
        len(matched) / len(required_skills) * 100
        if required_skills else 0
    )

    return matched, missing_skills, coverage

# -------------------------------
# 5. RESUME INPUT
# -------------------------------
st.subheader("1. Enter your resume")

resume_text = st.text_area(
    "Paste resume text here",
    height=250,
    placeholder=(
        "Paste your education, skills, projects, "
        "certifications and work experience..."
    )
)

analyze = st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
)

if analyze:
    if len(resume_text.strip()) < 50:
        st.warning(
            "Please enter more resume information "
            "(at least 50 characters)."
        )
        st.stop()

    cleaned_resume = clean_text(resume_text)

    # -------------------------------
    # 6. PREDICT RESUME CATEGORY
    # -------------------------------
    resume_vector = resume_vectorizer.transform(
        [cleaned_resume]
    )
    predicted_category = model.predict(
        resume_vector
    )[0]

    probabilities = model.predict_proba(resume_vector)[0]
    sorted_probs = sorted(probabilities, reverse=True)

    top_probability = float(sorted_probs[0])
    second_probability = (
        float(sorted_probs[1])
        if len(sorted_probs) > 1 else 0.0
    )
    margin = top_probability - second_probability

    st.subheader("2. Resume classification")

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Category", str(predicted_category))
    col2.metric("Top Model Estimate", f"{top_probability:.1%}")
    col3.metric("Top-two Margin", f"{margin:.1%}")

    if top_probability < 0.50 or margin < 0.10:
        st.warning(
            "Uncertain prediction: consider reviewing the "
            "resume or comparing other categories."
        )
    else:
        st.success(
            "The model has a clearer leading category, "
            "but this is not a hiring probability."
        )

    st.caption(
        "Model estimates depend on the training data and calibration. "
        "They do not represent the probability of getting a job."
    )

    # -------------------------------
    # 7. EXTRACT RESUME SKILLS
    # -------------------------------
    resume_skills = extract_skills(resume_text)

    st.subheader("3. Skills detected")

    if resume_skills:
        st.write(", ".join(sorted(resume_skills)))
    else:
        st.info(
            "No skills from the current skill dictionary were detected. "
            "Try including specific skills in your resume."
        )

    # -------------------------------
    # 8. JOB RECOMMENDATIONS
    # -------------------------------
    st.subheader("4. Job recommendations")

    required_columns = [
        "Job Title", "Required Skills", "Description"
    ]

    if not all(col in jobs.columns for col in required_columns):
        st.error(
            "The job catalogue does not have the expected columns: "
            + ", ".join(required_columns)
        )
        st.stop()

    job_texts = (
        jobs["Job Title"].fillna("").astype(str)
        + " "
        + jobs["Required Skills"].fillna("").astype(str)
        + " "
        + jobs["Description"].fillna("").astype(str)
    ).tolist()

    # Use the saved job vectorizer for text similarity.
    resume_for_jobs = job_vectorizer.transform(
        [cleaned_resume]
    )
    job_matrix = job_vectorizer.transform(job_texts)

    similarities = cosine_similarity(
        resume_for_jobs, job_matrix
    )[0]

    results = []

    for index, row in jobs.iterrows():
        raw_skills = str(row["Required Skills"])
        required_skills = [
            s.strip().lower()
            for s in raw_skills.split(",")
            if s.strip()
        ]

        # Support the original space-separated catalogue too.
        if len(required_skills) == 1 and " " in required_skills[0]:
            required_skills = [
                s for s in required_skills[0].split()
                if s
            ]

        matched, missing_skills, coverage = fuzzy_skill_match(
            resume_skills, required_skills
        )

        text_score = float(similarities[index]) * 100

        # Combined score is a prototype ranking score, not a probability.
        combined_score = 0.60 * text_score + 0.40 * coverage

        results.append({
            "Job Title": row["Job Title"],
            "Text Similarity (%)": round(text_score, 2),
            "Skill Coverage (%)": round(coverage, 2),
            "Combined Ranking Score (%)": round(combined_score, 2),
            "Matched Skills": ", ".join(matched) or "None detected",
            "Skills to Develop": ", ".join(missing_skills) or "None detected"
        })

    result_df = pd.DataFrame(results).sort_values(
        "Combined Ranking Score (%)",
        ascending=False
    )

    for rank, (_, job) in enumerate(
        result_df.iterrows(), start=1
    ):
        with st.expander(
            f"#{rank} {job['Job Title']} — "
            f"{job['Combined Ranking Score (%)']:.2f}% ranking score",
            expanded=(rank == 1)
        ):
            a, b = st.columns(2)

            a.metric(
                "Text Similarity",
                f"{job['Text Similarity (%)']:.2f}%"
            )
            b.metric(
                "Skill Coverage",
                f"{job['Skill Coverage (%)']:.2f}%"
            )

            st.write("**Matched skills:**", job["Matched Skills"])
            st.write(
                "**Skills to develop:**",
                job["Skills to Develop"]
            )

    st.subheader("5. Export results")

    st.download_button(
        "Download job recommendations as CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="job_recommendations.csv",
        mime="text/csv"
    )

    st.caption(
        "Recommendations use the included sample job catalogue. "
        "Fuzzy matching and ranking are experimental and should be "
        "validated before real recruitment use."
    )
