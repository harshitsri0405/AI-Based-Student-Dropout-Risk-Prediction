import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NoDrop | AI Student Retention Platform",
    layout="wide"
)

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: white;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

h1 { font-weight: 600; }
h2 { font-weight: 600; }

.badge {
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 600;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
    <div>
        <h1 style="margin:0;">NoDrop</h1>
        <div style="color:#6b7280;font-size:14px;">
        AI-Powered Student Retention Intelligence
        </div>
    </div>
    <div style="font-weight:600;color:#111827;">
        Team Codex
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = joblib.load("models/xgb_dropout_model.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer(df):
    df["CGPA_x_Attendance"] = df["CGPA"] * df["Attendance_Pct"] / 100
    df["Backlog_x_FeeDefault"] = df["Total_Backlogs"] * (df["Fee_Defaults"] + 1)
    df["CGPA_x_CoreAvg"] = df["CGPA"] * df["Core_Subject_Average"]
    df["AttendanceLow"] = (df["Attendance_Pct"] < 60).astype(int)
    df["HighRiskFlag"] = (
        (df["CGPA"] < 5.5) &
        (df["Total_Backlogs"] >= 3) &
        (df["Fee_Defaults"] >= 1)
    ).astype(int)
    df["AcademicFinanceRisk"] = (
        (df["CGPA"] < 6.0).astype(int) +
        (df["Fee_Defaults"] > 0).astype(int) +
        (df["Total_Backlogs"] > 2).astype(int)
    )
    df["SemProgress"] = df["Semesters_Completed"] / 8
    df["NormBacklog"] = df["Total_Backlogs"] / (df["Semesters_Completed"] + 1)
    return df

# ─────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["Individual Analysis", "Bulk Intelligence"])

# ─────────────────────────────────────────────
# INDIVIDUAL ANALYSIS
# ─────────────────────────────────────────────
if page == "Individual Analysis":

    st.subheader("Student Risk Analysis")

    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
    attendance = st.slider("Attendance %", 0, 100, 75)
    backlogs = st.number_input("Backlogs", 0, 15, 0)
    fee_defaults = st.number_input("Fee Defaults", 0, 10, 0)
    core_avg = st.slider("Core Subject Avg", 0.0, 10.0, 7.0, 0.1)
    sem_completed = st.slider("Semesters Completed", 1, 8, 4)
    education_loan = st.selectbox("Education Loan", [0,1])
    family_income = st.selectbox("Family Income Category", [0,1,2])

    input_dict = {
        "CGPA": cgpa,
        "Attendance_Pct": attendance,
        "Total_Backlogs": backlogs,
        "Fee_Defaults": fee_defaults,
        "Core_Subject_Average": core_avg,
        "Semesters_Completed": sem_completed,
        "Education_Loan": education_loan,
        "Family_Income_Category": family_income
    }

    df_input = engineer(pd.DataFrame([input_dict]))
    df_input = df_input[feature_cols]

    final_prob = model.predict_proba(df_input)[0][1]
    placeholder = st.empty()

    for val in np.linspace(0, final_prob, 25):
        placeholder.metric("Predicted Dropout Probability", f"{val*100:.2f}%")
        time.sleep(0.01)

    prob = final_prob

    # Severity Badge
    if prob < 0.4:
        badge_color = "#16a34a"
        severity = "LOW RISK"
    elif prob < 0.7:
        badge_color = "#ca8a04"
        severity = "MODERATE RISK"
    else:
        badge_color = "#dc2626"
        severity = "HIGH RISK"

    st.markdown(
        f"<div class='badge' style='background:{badge_color};display:inline-block;'>{severity}</div>",
        unsafe_allow_html=True
    )

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_input)
    shap_vals = shap_values.values[:,:,1] if len(shap_values.values.shape)==3 else shap_values.values

    contrib = pd.DataFrame({
        "Feature": feature_cols,
        "Impact": shap_vals[0]
    }).sort_values("Impact", key=abs, ascending=False).head(6)

    st.markdown("### AI Decision Flow")

    fig = go.Figure(go.Sankey(
        node=dict(label=list(contrib["Feature"])+["Risk Output"], color="#111827"),
        link=dict(
            source=list(range(len(contrib))),
            target=[len(contrib)]*len(contrib),
            value=contrib["Impact"].abs(),
            color=["#dc2626" if v>0 else "#2563eb" for v in contrib["Impact"]]
        )
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Confidence
    confidence = 1 - (np.std(shap_vals) / (np.mean(np.abs(shap_vals)) + 1e-6))
    confidence = max(0, min(confidence, 1))
    st.metric("AI Confidence Score", f"{confidence*100:.1f}%")

    # Structured Recommendation Block
    st.markdown("### Recommended Action Plan")

    if prob < 0.4:
        st.success("Maintain monitoring. No immediate intervention required.")
    elif prob < 0.7:
        st.warning("""
        • Academic mentoring session  
        • Attendance review meeting  
        • Early performance checkpoint  
        """)
    else:
        st.error("""
        • Immediate academic counseling  
        • Financial aid assessment  
        • Faculty intervention scheduling  
        • Weekly progress tracking  
        """)

# ─────────────────────────────────────────────
# BULK INTELLIGENCE
# ─────────────────────────────────────────────
elif page == "Bulk Intelligence":

    st.subheader("Institutional Risk Intelligence")

    uploaded = st.file_uploader("Upload Student CSV", type=["csv"])

    if uploaded:

        bulk_df = engineer(pd.read_csv(uploaded))
        preds = model.predict_proba(bulk_df[feature_cols])[:,1]
        bulk_df["Dropout_Risk"] = preds

        bulk_df["Risk_Category"] = pd.cut(
            bulk_df["Dropout_Risk"],
            bins=[0,0.4,0.7,1],
            labels=["Low","Moderate","High"]
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", len(bulk_df))
        col2.metric("High Risk Students", (bulk_df["Risk_Category"]=="High").sum())
        col3.metric("Average Institutional Risk", f"{bulk_df['Dropout_Risk'].mean()*100:.1f}%")

        # Auto Executive Summary
        st.markdown("### Executive Summary")

        high_pct = (bulk_df["Risk_Category"]=="High").mean()*100

        if high_pct > 30:
            summary = "High institutional risk detected. Immediate strategic intervention recommended."
        elif high_pct > 15:
            summary = "Moderate institutional risk. Targeted intervention advisable."
        else:
            summary = "Institutional risk levels within manageable range."

        st.info(summary)

        # Pie
        st.markdown("### Risk Distribution")
        fig_pie = px.pie(
            bulk_df,
            names="Risk_Category",
            color="Risk_Category",
            color_discrete_map={
                "Low":"#16a34a",
                "Moderate":"#ca8a04",
                "High":"#dc2626"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Histogram
        st.markdown("### Risk Probability Distribution")
        fig_hist = px.histogram(bulk_df, x="Dropout_Risk", nbins=30)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Top Risk
        st.markdown("### Top 10 High-Risk Students")
        st.dataframe(
            bulk_df.sort_values("Dropout_Risk", ascending=False).head(10),
            use_container_width=True
        )

        # Full Table
        st.markdown("### Complete Dataset")
        st.dataframe(bulk_df, height=400, use_container_width=True)

        csv = bulk_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Full Results", csv, "NoDrop_results.csv")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;color:#9ca3af;font-size:13px;">
NoDrop © 2026 | Built by Team Codex | AI for Student Retention
</div>
""", unsafe_allow_html=True)