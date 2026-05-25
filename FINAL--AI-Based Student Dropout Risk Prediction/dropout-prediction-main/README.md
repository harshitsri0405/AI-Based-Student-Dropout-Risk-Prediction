# NoDrop  
### AI-Powered Student Retention Intelligence  
**Team Codex**

---

## Overview

NoDrop is an AI-driven student dropout risk prediction platform designed to help engineering institutions proactively identify at-risk students and take timely intervention actions.

The system combines machine learning, feature engineering, explainability (SHAP), and institutional analytics into a clean, decision-support dashboard built with Streamlit.

---

## Problem Statement

Many engineering colleges face increasing student dropout rates due to:

- Academic underperformance  
- Attendance issues  
- Financial stress  
- Backlogs accumulation  

Institutions often react too late because they lack predictive insights.

NoDrop solves this by providing:

- Individual student risk prediction  
- Institutional-level risk intelligence  
- AI explainability for transparent decision-making  
- Structured intervention recommendations  

---

## Key Features

### 1. Individual Risk Analysis
- Real-time dropout probability prediction
- Severity classification (Low / Moderate / High Risk)
- AI confidence score
- SHAP-based feature contribution visualization
- Structured intervention recommendations

### 2. Bulk Institutional Intelligence
- CSV upload for batch analysis
- Risk categorization
- Executive summary generation
- Risk distribution visualization
- Top 10 high-risk students identification
- Downloadable full results

### 3. Explainable AI
- SHAP-based decision flow visualization
- Transparent feature impact analysis
- Confidence scoring mechanism

---

## Tech Stack

- Python
- Streamlit
- XGBoost
- Scikit-learn
- SHAP
- Plotly
- Pandas / NumPy
- Joblib

---

## Project Structure
dropout-prediction/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── models/
│ ├── xgb_dropout_model.pkl
│ └── feature_cols.pkl
│
└── src/
├── config.py
├── features/
├── explainability/
└── models/

---

## Installation & Setup (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dropout-prediction.git
cd dropout-prediction
2. Create Virtual Environment
python -m venv venv
Activate:
Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the Application
streamlit run app.py
Deployment
This application is deployed using Streamlit Cloud.
Live Demo:
https://your-app-link.streamlit.app
Input Features Used
The model uses engineered and raw features including:
CGPA
Attendance Percentage
Total Backlogs
Fee Defaults
Core Subject Average
Semesters Completed
Education Loan Status
Family Income Category
Engineered features include:
CGPA × Attendance
Backlog × Fee Default interaction
Academic-Financial Risk Score
Normalized Backlog
High Risk Flag
Model Details
Algorithm: XGBoost Classifier
Output: Probability of student dropout
Interpretation: SHAP TreeExplainer
Prediction Type: Binary classification (Dropout vs Continue)
AI Confidence Score
Confidence score is derived from the consistency and magnitude of SHAP value distribution to provide additional interpretability for institutional stakeholders.
Use Cases
Academic risk monitoring
Financial aid prioritization
Early intervention systems
Strategic institutional planning
Retention analytics dashboards
Future Enhancements
Real-time database integration
Role-based access control
Longitudinal tracking dashboard
Multi-institution benchmarking
Intervention outcome feedback loop
Authors
Team Codex
NoDrop — AI for Student Retention
License
This project is developed for academic and hackathon demonstration purposes.

---

# Optional Enhancements (If You Want to Impress Judges)

You can add:

- Architecture diagram image
- 2 screenshots of UI
- Model performance metrics (Accuracy, AUC, F1)
- Short 4-line elevator pitch at the top

If you'd like, I can also generate:

- A polished 60-second pitch script  
- A short LinkedIn-ready project description  
- A one-page technical summary for judges  

You’re now ready to push and deploy.