"""
╔══════════════════════════════════════════════════════════════════╗
║       CAMPUS PLACEMENT PREDICTOR — ML DASHBOARD APP             ║
║       Built with Streamlit | scikit-learn | plotly               ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    pip install streamlit pandas numpy scikit-learn plotly shap xgboost imbalanced-learn
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ──────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CampusAI — Placement Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg:         #060610;
    --surface:    #0d0d1a;
    --card:       rgba(255,255,255,0.04);
    --card-solid: #111122;
    --border:     rgba(255,255,255,0.08);
    --border-hov: rgba(99,102,241,0.5);
    --indigo:     #6366f1;
    --violet:     #8b5cf6;
    --cyan:       #06b6d4;
    --emerald:    #10b981;
    --rose:       #f43f5e;
    --amber:      #f59e0b;
    --text:       #f1f5f9;
    --muted:      #64748b;
    --muted2:     #94a3b8;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--indigo); border-radius: 4px; }

/* ── Layout ── */
.main .block-container { padding: 1.8rem 2.5rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: .9rem !important;
    transition: background .15s !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(99,102,241,0.12) !important; }
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio input:checked + div { background: rgba(99,102,241,0.2) !important; border-left: 3px solid var(--indigo) !important; }

/* ── Typography ── */
h1 { font-family: 'Space Grotesk', sans-serif !important; font-size: 2.4rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; line-height: 1.15 !important; }
h2 { font-family: 'Space Grotesk', sans-serif !important; font-size: 1.5rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
h3 { font-family: 'Space Grotesk', sans-serif !important; font-size: 1.05rem !important; font-weight: 600 !important; color: var(--cyan) !important; }
p, li, label { font-size: .92rem !important; line-height: 1.65 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.3rem !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color .2s, transform .2s !important;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-hov) !important;
    transform: translateY(-2px) !important;
}
[data-testid="metric-container"] label {
    color: var(--muted2) !important;
    font-size: .72rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #fff !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: .78rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
    color: var(--muted2) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: .65rem 1.1rem !important;
    border-radius: 0 !important;
    transition: color .15s !important;
}
[data-testid="stTabs"] button:hover { color: var(--text) !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--indigo) !important;
    border-bottom: 2px solid var(--indigo) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--indigo), var(--violet)) !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .65rem 1.8rem !important;
    letter-spacing: .01em !important;
    transition: opacity .15s, transform .15s !important;
}
.stButton > button:hover { opacity: .85 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Form inputs ── */
[data-testid="stSlider"] > div > div > div { background: var(--indigo) !important; }
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: .9rem !important;
    transition: border-color .15s !important;
}
.stSelectbox div[data-baseweb="select"] > div:hover,
.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: var(--indigo) !important;
}

/* ── Cards ── */
.campus-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color .25s, transform .25s;
}
.campus-card:hover {
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-2px);
}

/* ── Accent left-border card ── */
.accent-card {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 3px solid var(--indigo);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: .75rem;
}

/* ── Prediction result ── */
.predict-placed {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.4);
    color: #6ee7b7;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.35rem;
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    text-align: center;
    letter-spacing: -0.01em;
}
.predict-notplaced {
    background: rgba(244,63,94,0.12);
    border: 1px solid rgba(244,63,94,0.4);
    color: #fda4af;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.35rem;
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    text-align: center;
    letter-spacing: -0.01em;
}

/* ── Score ring number ── */
.score-ring {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    text-align: center;
    line-height: 1;
    letter-spacing: -0.03em;
}

/* ── Pill badge ── */
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.pill-indigo { background: rgba(99,102,241,0.15); color: #a5b4fc; }
.pill-emerald { background: rgba(16,185,129,0.15); color: #6ee7b7; }
.pill-rose { background: rgba(244,63,94,0.15); color: #fda4af; }
.pill-amber { background: rgba(245,158,11,0.15); color: #fcd34d; }

/* ── Section label ── */
.section-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .5rem;
    display: block;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stExpander"]:hover { border-color: rgba(99,102,241,0.35) !important; }
[data-testid="stExpanderDetails"] { padding: .75rem 1.1rem !important; }

/* ── Form container ── */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 14px !important; overflow: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DATA LOADING & CACHING
# ══════════════════════════════════════════════════════════════════
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#cbd5e1", size=12),
    margin=dict(l=20, r=20, t=44, b=20),
    title_font=dict(family="Space Grotesk", size=15, color="#f1f5f9"),
)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("campus_placement_data.csv")
    df["placed"] = df["placed"].astype(int)
    df["salary_lpa"] = pd.to_numeric(df["salary_lpa"], errors="coerce")
    return df

@st.cache_resource(show_spinner=False)
def train_models(df):
    cat_cols = ["gender", "city_tier", "ssc_board", "hsc_board",
                "hsc_stream", "degree_field", "specialization"]
    num_cols = ["age", "ssc_percentage", "hsc_percentage", "degree_percentage",
                "mba_percentage", "technical_skills_score", "soft_skills_score",
                "aptitude_score", "communication_score", "work_experience_months",
                "internships_count", "projects_count", "certifications_count",
                "leadership_roles", "extracurricular_activities", "backlogs"]

    df2 = df.copy()
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df2[c] = le.fit_transform(df2[c].astype(str))
        encoders[c] = le

    feature_cols = num_cols + cat_cols
    X = df2[feature_cols].fillna(df2[feature_cols].median(numeric_only=True))
    y = df2["placed"]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=6, random_state=42,
                                           use_label_encoder=False, eval_metric="logloss")

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": m,
            "accuracy": accuracy_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
            "report": classification_report(y_test, preds, output_dict=True),
            "cm": confusion_matrix(y_test, preds),
            "fpr": roc_curve(y_test, proba)[0],
            "tpr": roc_curve(y_test, proba)[1],
            "proba": proba,
            "y_test": y_test.values,
        }

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    return results, best_name, scaler, encoders, feature_cols, X_train, y_train

# ══════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem;'>
      <p style='font-family:Space Grotesk,sans-serif;font-size:1.3rem;font-weight:700;color:#f1f5f9;margin:0;letter-spacing:-0.02em'>Campus<span style='color:#6366f1'>AI</span></p>
      <p style='font-size:.75rem;color:#64748b;margin:4px 0 0;letter-spacing:.08em;text-transform:uppercase;font-weight:600'>Placement Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0 0 1rem'>", unsafe_allow_html=True)
    nav = st.radio("Navigate", [
        "🏠  Overview",
        "📊  EDA Explorer",
        "🤖  Model Arena",
        "🔮  Predict Student",
        "💡  Smart Insights",
        "🎯  Student Ranker",
    ], label_visibility="collapsed")
    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:.25rem 0'>
      <p style='font-size:.72rem;color:#475569;margin:0;line-height:1.9'>
        <span style='color:#6366f1;font-weight:600'>100,000</span> students &nbsp;·&nbsp; 25 features<br>
        <span style='color:#6366f1;font-weight:600'>4</span> ML models &nbsp;·&nbsp; Real-time prediction
      </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
with st.spinner("Loading data..."):
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("⚠️  `campus_placement_data.csv` not found in the same directory as app.py.")
        st.stop()

with st.spinner("Training models (cached after first run)..."):
    results, best_name, scaler, encoders, feature_cols, X_train, y_train = train_models(df)

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
if nav == "🏠  Overview":
    st.markdown("""
    <div style='margin-bottom:2rem'>
      <p class='section-label'>Dashboard</p>
      <h1 style='margin:0 0 .4rem'>Campus<span style='color:#6366f1'>AI</span> &mdash; Placement Intelligence</h1>
      <p style='color:#64748b;font-size:.95rem;margin:0'>End-to-end ML dashboard for campus recruitment analytics &nbsp;·&nbsp; 100K students</p>
    </div>
    """, unsafe_allow_html=True)

    placed = df["placed"].sum()
    not_placed = len(df) - placed
    placement_rate = placed / len(df) * 100
    avg_sal = df[df["placed"] == 1]["salary_lpa"].mean()
    max_sal = df["salary_lpa"].max()
    median_sal = df[df["placed"] == 1]["salary_lpa"].median()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students", f"{len(df):,}")
    c2.metric("Placed", f"{placed:,}", delta=f"+{placement_rate:.1f}%")
    c3.metric("Placement Rate", f"{placement_rate:.1f}%")
    c4.metric("Avg Salary (LPA)", f"₹{avg_sal:.2f}")
    c5.metric("Top Salary (LPA)", f"₹{max_sal:.2f}")

    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        fig = go.Figure(go.Pie(
            labels=["Placed", "Not Placed"],
            values=[placed, not_placed],
            hole=0.65,
            marker_colors=["#6366f1", "#1e1e3a"],
            marker_line=dict(color="rgba(0,0,0,0)", width=0),
            textfont_size=13,
        ))
        fig.add_annotation(text=f"<b>{placement_rate:.0f}%</b><br><span style='font-size:11px'>Placed</span>",
                           showarrow=False, font=dict(size=22, color="#f1f5f9"))
        fig.update_layout(title="Placement Split", **PLOTLY_THEME,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sal_df = df[df["placed"] == 1]
        fig2 = px.histogram(sal_df, x="salary_lpa", nbins=60,
                            color_discrete_sequence=["#6366f1"],
                            title="Salary Distribution — Placed Students")
        fig2.update_traces(marker_line_width=0, opacity=0.85)
        fig2.add_vline(x=avg_sal, line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"Avg ₹{avg_sal:.1f}L", annotation_font_color="#f59e0b")
        fig2.add_vline(x=median_sal, line_dash="dot", line_color="#10b981",
                       annotation_text=f"Med ₹{median_sal:.1f}L", annotation_font_color="#10b981")
        fig2.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    # Best model callout
    br = results[best_name]
    st.markdown(f"""
    <div class="campus-card" style="border-color:rgba(99,102,241,0.45);margin-top:.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem">
      <div>
        <span class='section-label'>Best Performing Model</span>
        <p style='font-family:Space Grotesk,sans-serif;font-size:1.25rem;font-weight:700;margin:0;color:#f1f5f9'>{best_name}</p>
      </div>
      <div style='display:flex;gap:2rem;flex-wrap:wrap'>
        <div style='text-align:center'>
          <p style='color:#64748b;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600;margin:0'>Accuracy</p>
          <p style='font-family:Space Grotesk,sans-serif;font-size:1.6rem;font-weight:700;color:#6366f1;margin:0'>{br['accuracy']*100:.2f}%</p>
        </div>
        <div style='text-align:center'>
          <p style='color:#64748b;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600;margin:0'>ROC-AUC</p>
          <p style='font-family:Space Grotesk,sans-serif;font-size:1.6rem;font-weight:700;color:#10b981;margin:0'>{br['roc_auc']:.4f}</p>
        </div>
        <div style='text-align:center'>
          <p style='color:#64748b;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600;margin:0'>F1 Score</p>
          <p style='font-family:Space Grotesk,sans-serif;font-size:1.6rem;font-weight:700;color:#f59e0b;margin:0'>{br['report']['1']['f1-score']:.4f}</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════
elif nav == "📊  EDA Explorer":
    st.markdown("<p class='section-label'>Exploration</p>", unsafe_allow_html=True)
    st.markdown("# 📊 EDA Explorer")
    st.markdown("<p style='color:#64748b;font-size:.92rem;margin-bottom:1.5rem'>Interactive exploratory data analysis — drill into any feature</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Academic Scores", "Skills Profile", "Demographics", "Correlations"])

    with tab1:
        score_cols = ["ssc_percentage", "hsc_percentage", "degree_percentage", "aptitude_score"]
        col = st.selectbox("Select score", score_cols, key="eda1")
        fig = px.box(df, x="placed", y=col, color="placed",
                     color_discrete_map={0: "#f43f5e", 1: "#6366f1"},
                     labels={"placed": "Placement Status"},
                     title=f"{col.replace('_', ' ').title()} vs Placement")
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

        # Placed vs not violin
        fig2 = px.violin(df, x=df["placed"].map({1: "Placed", 0: "Not Placed"}),
                         y=col, color=df["placed"].map({1: "Placed", 0: "Not Placed"}),
                         color_discrete_map={"Placed": "#6366f1", "Not Placed": "#f43f5e"},
                         box=True, title="Distribution Violin")
        fig2.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        skill_cols = ["technical_skills_score", "soft_skills_score", "communication_score"]
        placed_df = df[df["placed"] == 1][skill_cols].mean().reset_index()
        placed_df.columns = ["Skill", "Score"]
        notplaced_df = df[df["placed"] == 0][skill_cols].mean().reset_index()
        notplaced_df.columns = ["Skill", "Score"]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Placed", x=placed_df["Skill"], y=placed_df["Score"],
                             marker_color="#6366f1"))
        fig.add_trace(go.Bar(name="Not Placed", x=notplaced_df["Skill"], y=notplaced_df["Score"],
                             marker_color="#f43f5e"))
        fig.update_layout(barmode="group", title="Average Skills: Placed vs Not Placed", **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

        # Radar
        categories = ["Technical", "Soft Skills", "Communication", "Aptitude"]
        vals_placed = [
            df[df["placed"]==1]["technical_skills_score"].mean(),
            df[df["placed"]==1]["soft_skills_score"].mean(),
            df[df["placed"]==1]["communication_score"].mean(),
            df[df["placed"]==1]["aptitude_score"].mean(),
        ]
        vals_not = [
            df[df["placed"]==0]["technical_skills_score"].mean(),
            df[df["placed"]==0]["soft_skills_score"].mean(),
            df[df["placed"]==0]["communication_score"].mean(),
            df[df["placed"]==0]["aptitude_score"].mean(),
        ]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=vals_placed + [vals_placed[0]],
                                        theta=categories + [categories[0]],
                                        fill="toself", name="Placed", line_color="#6366f1"))
        fig_r.add_trace(go.Scatterpolar(r=vals_not + [vals_not[0]],
                                        theta=categories + [categories[0]],
                                        fill="toself", name="Not Placed", line_color="#f43f5e"))
        fig_r.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)"),
                             title="Skills Radar", **PLOTLY_THEME)
        st.plotly_chart(fig_r, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            gender_placed = df.groupby(["gender", "placed"]).size().reset_index(name="count")
            fig = px.bar(gender_placed, x="gender", y="count", color=gender_placed["placed"].map({1:"Placed",0:"Not Placed"}),
                         color_discrete_map={"Placed":"#6366f1","Not Placed":"#f43f5e"},
                         barmode="group", title="Gender vs Placement")
            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            tier_rate = df.groupby("city_tier")["placed"].mean().reset_index()
            fig2 = px.bar(tier_rate, x="city_tier", y="placed",
                          color="placed", color_continuous_scale=["#f43f5e","#f59e0b","#6366f1"],
                          title="Placement Rate by City Tier")
            fig2.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig2, use_container_width=True)

        stream_rate = df.groupby("hsc_stream")["placed"].mean().sort_values(ascending=False).reset_index()
        fig3 = px.bar(stream_rate, x="hsc_stream", y="placed",
                      color="placed", color_continuous_scale=["#f43f5e","#6366f1"],
                      title="Placement Rate by HSC Stream")
        fig3.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        num_df = df[["ssc_percentage","hsc_percentage","degree_percentage",
                     "technical_skills_score","soft_skills_score","aptitude_score",
                     "communication_score","work_experience_months","salary_lpa","placed"]].corr()
        fig = px.imshow(num_df, color_continuous_scale="RdBu_r", text_auto=".2f",
                        title="Feature Correlation Heatmap")
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ARENA
# ══════════════════════════════════════════════════════════════════
elif nav == "🤖  Model Arena":
    st.markdown("<p class='section-label'>Machine Learning</p>", unsafe_allow_html=True)
    st.markdown("# 🤖 Model Arena")
    st.markdown("<p style='color:#64748b;font-size:.92rem;margin-bottom:1.5rem'>Compare all classifiers — metrics, ROC curves, confusion matrices</p>", unsafe_allow_html=True)

    # Summary table
    rows = []
    for name, r in results.items():
        rpt = r["report"]
        rows.append({
            "Model": name,
            "Accuracy": f"{r['accuracy']*100:.2f}%",
            "ROC-AUC": f"{r['roc_auc']:.4f}",
            "Precision(1)": f"{rpt['1']['precision']:.4f}",
            "Recall(1)": f"{rpt['1']['recall']:.4f}",
            "F1(1)": f"{rpt['1']['f1-score']:.4f}",
        })
    summary = pd.DataFrame(rows)
    st.dataframe(summary.style.highlight_max(
        subset=["Accuracy","ROC-AUC","F1(1)"], color="#1a3a2a", axis=0),
        use_container_width=True)

    st.markdown("---")

    # ROC curves
    fig_roc = go.Figure()
    colors = ["#6366f1", "#f59e0b", "#f43f5e", "#8b5cf6"]
    for i, (name, r) in enumerate(results.items()):
        fig_roc.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"],
                                      name=f"{name} (AUC={r['roc_auc']:.3f})",
                                      line=dict(color=colors[i % len(colors)], width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color="#7a7a9a"), name="Random"))
    fig_roc.update_layout(title="ROC Curves — All Models",
                           xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate", **PLOTLY_THEME)
    st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion matrices
    st.markdown("### Confusion Matrices")
    cols = st.columns(len(results))
    for i, (name, r) in enumerate(results.items()):
        with cols[i]:
            cm = r["cm"]
            fig_cm = px.imshow(cm, text_auto=True,
                               color_continuous_scale=["#0a0a0f","#6366f1"],
                               labels=dict(x="Predicted", y="Actual"),
                               x=["Not Placed","Placed"], y=["Not Placed","Placed"],
                               title=name)
            fig_cm.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig_cm, use_container_width=True)

    # Feature Importance (best model)
    st.markdown(f"### 🏆 Feature Importance — {best_name}")
    best_model = results[best_name]["model"]
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=True).tail(15)
        fig_fi = px.bar(fi, x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale=["#f43f5e","#f59e0b","#6366f1"],
                        title=f"Top 15 Features — {best_name}")
        fig_fi.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT STUDENT
# ══════════════════════════════════════════════════════════════════
elif nav == "🔮  Predict Student":
    st.markdown("<p class='section-label'>Prediction Engine</p>", unsafe_allow_html=True)
    st.markdown("# 🔮 Predict Student Placement")
    st.markdown("<p style='color:#64748b;font-size:.92rem;margin-bottom:1.5rem'>Enter a student profile for instant placement prediction + salary estimate</p>", unsafe_allow_html=True)

    chosen_model = st.selectbox("Select Model", list(results.keys()), index=list(results.keys()).index(best_name))
    model = results[chosen_model]["model"]

    with st.form("predict_form"):
        st.markdown("### 📋 Academic Background")
        c1, c2, c3 = st.columns(3)
        with c1:
            ssc = st.slider("SSC %", 30.0, 100.0, 70.0, 0.1)
            hsc = st.slider("HSC %", 30.0, 100.0, 70.0, 0.1)
            degree = st.slider("Degree %", 30.0, 100.0, 65.0, 0.1)
        with c2:
            mba = st.slider("MBA %", 0.0, 100.0, 60.0, 0.1)
            aptitude = st.slider("Aptitude Score", 0.0, 100.0, 60.0, 0.5)
            age = st.number_input("Age", 18, 30, 22)
        with c3:
            gender = st.selectbox("Gender", ["Male", "Female"])
            city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
            hsc_stream = st.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])

        st.markdown("### 🎯 Skills & Experience")
        c4, c5, c6 = st.columns(3)
        with c4:
            tech = st.slider("Technical Skills (1-10)", 1.0, 10.0, 6.0, 0.1)
            soft = st.slider("Soft Skills (1-10)", 1.0, 10.0, 6.0, 0.1)
            comm = st.slider("Communication (1-10)", 1.0, 10.0, 6.0, 0.1)
        with c5:
            internships = st.number_input("Internships", 0, 10, 1)
            projects = st.number_input("Projects", 0, 20, 2)
            certs = st.number_input("Certifications", 0, 20, 2)
        with c6:
            work_exp = st.number_input("Work Experience (months)", 0, 60, 6)
            leadership = st.number_input("Leadership Roles", 0, 10, 1)
            extracurr = st.number_input("Extracurricular Activities", 0, 30, 5)
            backlogs = st.number_input("Backlogs", 0, 10, 0)

        st.markdown("### 🏫 Board & Field")
        c7, c8, c9 = st.columns(3)
        with c7:
            ssc_board = st.selectbox("SSC Board", ["CBSE", "State", "ICSE"])
        with c8:
            hsc_board = st.selectbox("HSC Board", ["CBSE", "State", "ICSE"])
        with c9:
            degree_field = st.selectbox("Degree Field", ["Engineering", "Business", "Arts", "Science", "Commerce"])
            specialization = st.selectbox("Specialization", ["None", "Mkt&HR", "Mkt&Fin", "Finance", "HR"])

        submitted = st.form_submit_button("🚀 Predict Now")

    if submitted:
        input_dict = {
            "age": age, "ssc_percentage": ssc, "hsc_percentage": hsc,
            "degree_percentage": degree, "mba_percentage": mba,
            "technical_skills_score": tech, "soft_skills_score": soft,
            "aptitude_score": aptitude, "communication_score": comm,
            "work_experience_months": work_exp, "internships_count": internships,
            "projects_count": projects, "certifications_count": certs,
            "leadership_roles": leadership, "extracurricular_activities": extracurr,
            "backlogs": backlogs,
            "gender": gender, "city_tier": city_tier, "ssc_board": ssc_board,
            "hsc_board": hsc_board, "hsc_stream": hsc_stream,
            "degree_field": degree_field, "specialization": specialization,
        }

        row = []
        for col_name in feature_cols:
            val = input_dict.get(col_name, 0)
            if isinstance(val, str):
                le = encoders.get(col_name)
                if le:
                    try:
                        val = le.transform([val])[0]
                    except ValueError:
                        val = 0
            row.append(val)

        row_sc = scaler.transform([row])
        pred = model.predict(row_sc)[0]
        prob = model.predict_proba(row_sc)[0][1]

        st.markdown("---")
        st.markdown("### 🔮 Prediction Result")
        r1, r2, r3 = st.columns([1,1,1])
        with r1:
            if pred == 1:
                st.markdown(f'<div class="predict-placed">✅ PLACED<br><span style="font-size:1rem">Confidence: {prob*100:.1f}%</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="predict-notplaced">❌ NOT PLACED<br><span style="font-size:1rem">Confidence: {(1-prob)*100:.1f}%</span></div>', unsafe_allow_html=True)

        with r2:
            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Placement Probability", "font": {"color": "#e8e8f0"}},
                number={"suffix": "%", "font": {"color": "#6366f1", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7a7a9a"},
                    "bar": {"color": "#6366f1" if prob > 0.5 else "#f43f5e"},
                    "bgcolor": "rgba(15,15,30,0.8)",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(244,63,94,0.08)"},
                        {"range": [50, 100], "color": "rgba(99,102,241,0.08)"},
                    ],
                    "threshold": {"line": {"color": "#f59e0b", "width": 2}, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=250, **PLOTLY_THEME)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r3:
            if pred == 1:
                # Estimate salary
                placed_df = df[df["placed"] == 1]
                similar = placed_df[
                    (placed_df["degree_percentage"].between(degree-10, degree+10)) &
                    (placed_df["technical_skills_score"].between(tech-1.5, tech+1.5))
                ]
                est_sal = similar["salary_lpa"].mean() if len(similar) > 0 else placed_df["salary_lpa"].mean()
                st.markdown(f"""
                <div class="campus-card" style="border-color:#ffb347;text-align:center">
                <p style='color:#94a3b8;font-size:.8rem;margin:0'>ESTIMATED SALARY</p>
                <p style='font-family:Syne;font-size:2.5rem;font-weight:800;color:#ffb347;margin:0'>₹{est_sal:.2f} LPA</p>
                <p style='color:#94a3b8;font-size:.75rem'>Based on similar placed students</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Improvement suggestions
                st.markdown("""
                <div class="campus-card" style="border-color:rgba(244,63,94,0.4)">
                <h3 style='color:#f43f5e'>Improve Your Profile</h3>
                <ul style='color:#94a3b8;font-size:.9rem'>
                <li>Increase internships & projects</li>
                <li>Boost technical & communication skills</li>
                <li>Clear backlogs if any</li>
                <li>Add certifications</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 5 — SMART INSIGHTS (UNIQUE FEATURE)
# ══════════════════════════════════════════════════════════════════
elif nav == "💡  Smart Insights":
    st.markdown("<p class='section-label'>Analytics</p>", unsafe_allow_html=True)
    st.markdown("# 💡 Smart Insights Engine")
    st.markdown("<p style='color:#64748b;font-size:.92rem;margin-bottom:1.5rem'>AI-driven patterns, salary drivers, and placement success factors</p>", unsafe_allow_html=True)

    # Salary drivers
    st.markdown("### 💰 What Drives Salary?")
    placed_df = df[df["placed"] == 1].copy()
    salary_cols = ["ssc_percentage","hsc_percentage","degree_percentage",
                   "technical_skills_score","soft_skills_score","aptitude_score",
                   "communication_score","work_experience_months","internships_count",
                   "projects_count","certifications_count"]
    corr_sal = placed_df[salary_cols + ["salary_lpa"]].corr()["salary_lpa"].drop("salary_lpa").sort_values()
    fig_sal = px.bar(corr_sal, orientation="h",
                     color=corr_sal.values, color_continuous_scale=["#f43f5e","#888","#6366f1"],
                     title="Salary Correlation with Features")
    fig_sal.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_sal, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Top Salary Profiles (Top 1%)")
    top1 = placed_df[placed_df["salary_lpa"] >= placed_df["salary_lpa"].quantile(0.99)]
    avg_top = top1[["technical_skills_score","soft_skills_score","aptitude_score","communication_score",
                    "degree_percentage","work_experience_months"]].mean()
    avg_all = placed_df[["technical_skills_score","soft_skills_score","aptitude_score","communication_score",
                          "degree_percentage","work_experience_months"]].mean()
    comp = pd.DataFrame({"Top 1%": avg_top, "All Placed": avg_all})
    fig_comp = px.bar(comp, barmode="group", color_discrete_sequence=["#f59e0b","#6366f1"],
                      title="Top 1% Earners vs Average Placed Students")
    fig_comp.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Placement Rate Trends")
    c1, c2 = st.columns(2)
    with c1:
        # Internships
        df["intern_group"] = pd.cut(df["internships_count"], bins=[-1,0,1,2,3,10],
                                     labels=["0","1","2","3","4+"])
        intern_rate = df.groupby("intern_group", observed=True)["placed"].mean().reset_index()
        fig_i = px.line(intern_rate, x="intern_group", y="placed", markers=True,
                        color_discrete_sequence=["#6366f1"],
                        title="Placement Rate by Internship Count",
                        labels={"intern_group":"Internships","placed":"Rate"})
        fig_i.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_i, use_container_width=True)

    with c2:
        # Backlog effect
        bl_rate = df.groupby("backlogs")["placed"].mean().reset_index().head(8)
        fig_bl = px.bar(bl_rate, x="backlogs", y="placed",
                        color="placed", color_continuous_scale=["#6366f1","#f43f5e"],
                        title="Placement Rate vs Number of Backlogs")
        fig_bl.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_bl, use_container_width=True)

    # Scatter: degree % vs salary
    st.markdown("### 🎯 Degree % vs Salary (Interactive)")
    sample = placed_df.sample(min(3000, len(placed_df)), random_state=42)
    fig_sc = px.scatter(sample, x="degree_percentage", y="salary_lpa",
                        color="technical_skills_score",
                        color_continuous_scale="Turbo",
                        size="work_experience_months",
                        size_max=14,
                        hover_data=["gender","city_tier","hsc_stream"],
                        title="Degree % vs Salary (bubble size = work experience)",
                        opacity=0.7)
    fig_sc.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 6 — STUDENT RANKER (UNIQUE FEATURE)
# ══════════════════════════════════════════════════════════════════
elif nav == "🎯  Student Ranker":
    st.markdown("<p class='section-label'>Unique Feature</p>", unsafe_allow_html=True)
    st.markdown("# 🎯 Student Ranker")
    st.markdown("<p style='color:#64748b;font-size:.92rem;margin-bottom:1.5rem'>Compute a 0-100 Placement Score, peer percentile rank, and personalised action plan</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        r_ssc = st.slider("SSC %", 30.0, 100.0, 68.0, 0.5)
        r_hsc = st.slider("HSC %", 30.0, 100.0, 68.0, 0.5)
        r_deg = st.slider("Degree %", 30.0, 100.0, 65.0, 0.5)
        r_tech = st.slider("Technical Skills", 1.0, 10.0, 6.5, 0.1)
        r_soft = st.slider("Soft Skills", 1.0, 10.0, 6.5, 0.1)
    with c2:
        r_comm = st.slider("Communication", 1.0, 10.0, 6.5, 0.1)
        r_apt = st.slider("Aptitude Score", 0.0, 100.0, 60.0, 1.0)
        r_intern = st.number_input("Internships", 0, 10, 1)
        r_proj = st.number_input("Projects", 0, 20, 3)
        r_backlog = st.number_input("Backlogs (penalises score)", 0, 10, 0)
        r_exp = st.number_input("Work Experience (months)", 0, 60, 6)



    # ── Fast vectorized placement score ──────────────────────────
    SCORE_FEATS = {
        "degree_percentage": 0.15,
        "technical_skills_score": 0.20,
        "soft_skills_score": 0.12,
        "communication_score": 0.12,
        "aptitude_score": 0.15,
        "internships_count": 0.10,
        "projects_count": 0.08,
        "work_experience_months": 0.05,
        "ssc_percentage": 0.015,
        "hsc_percentage": 0.015,
    }

    def compute_score_fast(vals_dict, backlog, df_ref):
        total, total_w = 0.0, 0.0
        for feat, w in SCORE_FEATS.items():
            pct = float(np.searchsorted(np.sort(df_ref[feat].values), vals_dict[feat])) / len(df_ref) * 100
            total += pct * w
            total_w += w
        score = total / total_w
        return max(0.0, score - min(backlog * 5, 30))

    @st.cache_data(show_spinner=False)
    def precompute_all_scores(_df):
        vals = {
            "degree_percentage": _df["degree_percentage"].values,
            "technical_skills_score": _df["technical_skills_score"].values,
            "soft_skills_score": _df["soft_skills_score"].values,
            "communication_score": _df["communication_score"].values,
            "aptitude_score": _df["aptitude_score"].values,
            "internships_count": _df["internships_count"].values,
            "projects_count": _df["projects_count"].values,
            "work_experience_months": _df["work_experience_months"].values,
            "ssc_percentage": _df["ssc_percentage"].values,
            "hsc_percentage": _df["hsc_percentage"].values,
        }
        sorted_vals = {k: np.sort(v) for k, v in vals.items()}
        total_w = sum(SCORE_FEATS.values())
        scores = np.zeros(len(_df))
        for feat, w in SCORE_FEATS.items():
            pcts = np.searchsorted(sorted_vals[feat], vals[feat]) / len(_df) * 100
            scores += pcts * w
        scores = scores / total_w
        penalties = np.minimum(_df["backlogs"].values * 5, 30)
        return np.maximum(scores - penalties, 0)

    if st.button("📊 Compute Placement Score"):
        with st.spinner("Calculating score..."):
            vals_dict = {
                "degree_percentage": r_deg,
                "technical_skills_score": r_tech,
                "soft_skills_score": r_soft,
                "communication_score": r_comm,
                "aptitude_score": r_apt,
                "internships_count": float(r_intern),
                "projects_count": float(r_proj),
                "work_experience_months": float(r_exp),
                "ssc_percentage": r_ssc,
                "hsc_percentage": r_hsc,
            }
            score = compute_score_fast(vals_dict, r_backlog, df)
            all_scores = precompute_all_scores(df)
            percentile = float(np.mean(all_scores <= score) * 100)

        col_s, col_p = st.columns(2)
        with col_s:
            color = "#6366f1" if score >= 65 else "#f59e0b" if score >= 45 else "#f43f5e"
            st.markdown(f"""
            <div class="campus-card" style="border-color:{color};text-align:center">
            <p style='color:#94a3b8;font-size:.85rem;margin:0;letter-spacing:.1em'>PLACEMENT SCORE</p>
            <p class='score-ring' style='color:{color}'>{score:.1f}<span style='font-size:1.5rem'>/100</span></p>
            <p style='color:#94a3b8;font-size:.85rem'>{"🟢 Strong Profile" if score>=65 else "🟡 Average Profile" if score>=45 else "🔴 Needs Improvement"}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_p:
            st.markdown(f"""
            <div class="campus-card" style="border-color:#ffb347;text-align:center">
            <p style='color:#94a3b8;font-size:.85rem;margin:0;letter-spacing:.1em'>PEER PERCENTILE</p>
            <p class='score-ring' style='color:#ffb347'>{percentile:.0f}<span style='font-size:1.5rem'>%ile</span></p>
            <p style='color:#94a3b8;font-size:.85rem'>Better than {percentile:.0f}% of all students</p>
            </div>
            """, unsafe_allow_html=True)

        # Score gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 65, "increasing": {"color": "#6366f1"}, "decreasing": {"color": "#f43f5e"}},
            title={"text": "vs Target Score (65)", "font": {"color": "#e8e8f0"}},
            number={"font": {"color": "#f59e0b", "size": 48}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 45], "color": "rgba(244,63,94,0.08)"},
                    {"range": [45, 65], "color": "rgba(245,158,11,0.08)"},
                    {"range": [65, 100], "color": "rgba(99,102,241,0.08)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 3}, "value": 65},
            },
        ))
        fig_g.update_layout(height=280, **PLOTLY_THEME)
        st.plotly_chart(fig_g, use_container_width=True)

        # Action Plan
        st.markdown("### 🗺️ Personalised Action Plan")
        suggestions = []
        if r_tech < 7: suggestions.append(("📘 Technical Skills", f"Your score: {r_tech:.1f}/10. Target 7.5+. Take online courses on DSA, system design, or relevant tech stack."))
        if r_intern < 2: suggestions.append(("🏢 Internships", f"You have {r_intern} internship(s). Aim for 2+. Platforms: Internshala, LinkedIn, AngelList."))
        if r_proj < 3: suggestions.append(("💻 Projects", f"Only {r_proj} project(s). Add 3+ GitHub projects with real-world impact."))
        if r_apt < 65: suggestions.append(("🧠 Aptitude", f"Score: {r_apt:.0f}/100. Target 70+. Practice on IndiaBix, PrepInsta."))
        if r_backlog > 0: suggestions.append(("⚠️ Backlogs", f"{r_backlog} active backlog(s). Clear them ASAP — major red flag for recruiters."))
        if r_comm < 6.5: suggestions.append(("🗣️ Communication", f"Score: {r_comm:.1f}/10. Join public speaking clubs, record yourself, practice GDs."))

        if suggestions:
            for title_s, desc in suggestions:
                with st.expander(title_s):
                    st.markdown(f"<p style='color:#94a3b8'>{desc}</p>", unsafe_allow_html=True)
        else:
            st.success("🎉 Excellent profile! You're well-positioned for campus placements.")

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<p style='color:#94a3b8;text-align:center;font-size:.8rem'>"
    "CampusAI &nbsp;·&nbsp; Campus Placement Intelligence &nbsp;·&nbsp; Streamlit + scikit-learn + plotly"
    "</p>",
    unsafe_allow_html=True
)
