# AI Resume Analyzer & Job Recommendation System

An AI-powered tool that analyzes resumes, classifies them into industry categories, extracts skills, and recommends matching jobs with fuzzy skill matching and uncertainty estimation.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## Features

- **Resume Classification**: Predicts resume domain category using a calibrated SVM model.
- **Skill Extraction & Matching**: Extracts key technical skills and matches them against job requirements.
- **Job Recommendations**: Computes text similarity and skill coverage to rank recommendations.
- **Export Results**: Download recommended job matches as CSV.

## Project Structure

```text
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── resume_analyzer_model/      # Pre-trained models and dataset
│   ├── calibrated_svm.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── job_vectorizer.pkl
│   └── sample_jobs.csv
└── README.md
```

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```
