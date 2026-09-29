import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ORANGE, CHOC, DARKCHOC, CREAM, PEACH = "#FF8C1A", "#5C3317", "#3B1F0E", "#FFF3E6", "#FFD9B3"
CMAP = LinearSegmentedColormap.from_list("choc", [DARKCHOC, "#8B4513", "#D2691E", ORANGE, PEACH])

st.set_page_config(page_title="SaYoPillow Stress Dashboard", page_icon="🧡", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #FFF3E6 0%, #FFE0C2 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #3B1F0E 0%, #5C3317 100%); }
[data-testid="stSidebar"] * { color: #FFD9B3 !important; }
.hero { background: linear-gradient(90deg, #3B1F0E 0%, #8B4513 50%, #FF8C1A 100%);
        padding: 22px 28px; border-radius: 16px; }
.hero h1 { color: #FFF3E6 !important; margin: 0; } .hero p { color: #FFD9B3 !important; margin: 4px 0 0 0; }
[data-testid="stMetric"] { background: #FF8C1A; padding: 14px; border-radius: 12px; border-bottom: 6px solid #3B1F0E; }
[data-testid="stMetric"] * { color: #3B1F0E !important; }
[data-testid="stColumn"]:nth-child(even) [data-testid="stMetric"],
[data-testid="column"]:nth-child(even) [data-testid="stMetric"] { background: #5C3317; border-bottom: 6px solid #FF8C1A; }
[data-testid="stColumn"]:nth-child(even) [data-testid="stMetric"] *,
[data-testid="column"]:nth-child(even) [data-testid="stMetric"] * { color: #FFE0C2 !important; }
h2, h3 { color: #5C3317; }
</style>
""", unsafe_allow_html=True)

def paint(fig, *axes):
    fig.patch.set_facecolor(CREAM)
    for ax in axes:
        ax.set_facecolor("#FFF9F2")
        for s in ax.spines.values():
            s.set_color(CHOC)

FILENAME = "SaYoPillow.csv"
KAGGLE_SLUG = "laavanya/human-stress-detection-in-and-through-sleep"  # verify on kaggle.com if it fails

def get_csv(filename, slug):
    for p in [Path(filename), Path("data") / filename]:
        if p.exists():
            return p
    try:
        import kagglehub
        folder = Path(kagglehub.dataset_download(slug))
        hits = list(folder.rglob(filename)) or list(folder.rglob("*.csv"))
        if hits:
            return hits[0]
    except Exception:
        pass
    up = st.sidebar.file_uploader(f"Upload {filename}", type="csv")
    if up is None:
        st.info(f"Couldn't auto-download the data. Upload **{filename}** in the sidebar to continue.")
        st.stop()
    Path("data").mkdir(exist_ok=True)
    out = Path("data") / filename
    out.write_bytes(up.getvalue())
    return out

LABELS = {0: 'Low/Normal', 1: 'Medium-Low', 2: 'Medium', 3: 'Medium-High', 4: 'High'}

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'sr.1': 'sleep_hours', 'sr': 'snoring_range', 'rr': 'respiration_rate',
                            't': 'body_temp', 'lm': 'limb_movement', 'bo': 'blood_oxygen',
                            'rem': 'eye_movement', 'hr': 'heart_rate', 'sl': 'stress_level'})
    df['stress_label'] = df['stress_level'].map(LABELS)
    return df

df = load_data(get_csv(FILENAME, KAGGLE_SLUG))

@st.cache_resource
def train_models(df):
    X, y = df.drop(columns=['stress_level', 'stress_label']), df['stress_level']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, random_state=42).fit(sc.transform(Xtr), ytr)
    rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
    return (rf, accuracy_score(yte, lr.predict(sc.transform(Xte))),
            accuracy_score(yte, rf.predict(Xte)), X.columns.tolist())

rf, acc_lr, acc_rf, feats = train_models(df)

# ---------- SIDEBAR = LIVE CALCULATOR ----------
st.sidebar.markdown("## 🎚️ Tonight's Readings")
vals = {
    'snoring_range': st.sidebar.slider("Snoring Range (dB)", 45.0, 100.0, 60.0),
    'respiration_rate': st.sidebar.slider("Respiration Rate (bpm)", 15.0, 30.0, 20.0),
    'body_temp': st.sidebar.slider("Body Temperature (°F)", 85.0, 99.0, 92.0),
    'limb_movement': st.sidebar.slider("Limb Movement Rate", 5.0, 20.0, 10.0),
    'blood_oxygen': st.sidebar.slider("Blood Oxygen (%)", 85.0, 100.0, 98.0),
    'eye_movement': st.sidebar.slider("Eye Movement (REM)", 60.0, 120.0, 90.0),
    'sleep_hours': st.sidebar.slider("Sleep Hours", 2.0, 10.0, 7.0),
    'heart_rate': st.sidebar.slider("Heart Rate (bpm)", 50.0, 100.0, 72.0),
}
probs = rf.predict_proba(pd.DataFrame([vals])[feats])[0]
pred = int(rf.classes_[np.argmax(probs)])
score = int(100 - pred / 4 * 100)

st.markdown("""<div class="hero"><h1>🧡🍫 SaYoPillow: Sleep & Stress Dashboard</h1>
<p>Move the sliders in the sidebar — your predicted stress state updates live.</p></div>""", unsafe_allow_html=True)
st.write("")

# ---------- FUTURE (live result) ----------
left, right = st.columns([2, 3])
with left:
    bg = {0: "#FFD9B3", 1: "#FFB366", 2: "#FF8C1A", 3: "#8B4513", 4: "#3B1F0E"}[pred]
    fg = "#3B1F0E" if pred <= 2 else "#FFF3E6"
    st.markdown(f'<div style="background:{bg};color:{fg};padding:28px;border-radius:18px;text-align:center;'
                f'border:4px solid {DARKCHOC}"><div>Predicted stress state</div>'
                f'<div style="font-size:40px;font-weight:800">{LABELS[pred]}</div>'
                f'<div style="font-size:15px">Sleep quality score: {score}/100</div></div>', unsafe_allow_html=True)
    st.progress(score)
    tip = ("🍫 Wind down earlier and aim for longer sleep." if pred >= 3 else
           "🧡 Keep an eye on your recovery routine." if pred == 2 else "🌿 Healthy resting pattern.")
    st.caption(tip)
with right:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar([LABELS[int(c)] for c in rf.classes_], probs * 100,
           color=[ORANGE if i % 2 == 0 else CHOC for i in range(len(probs))], edgecolor=DARKCHOC)
    ax.set_ylabel("Probability %"); ax.set_title("How confident is the model?")
    ax.tick_params(axis='x', rotation=20); paint(fig, ax); plt.tight_layout()
    st.pyplot(fig)

st.subheader("🔎 Your readings vs the typical sleeper")
z = pd.Series({k: (v - df[k].mean()) / df[k].std() for k, v in vals.items()})
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.barh(z.index, z.values, color=[ORANGE if v >= 0 else CHOC for v in z.values], edgecolor=DARKCHOC)
ax.axvline(0, color=DARKCHOC); ax.set_xlabel("Standard deviations from average (orange = above, chocolate = below)")
paint(fig, ax); plt.tight_layout(); st.pyplot(fig)

# ---------- PAST ----------
st.subheader("📜 The dataset so far")
m1, m2, m3 = st.columns(3)
m1.metric("Sleep Records", f"{len(df):,}")
m2.metric("Avg Heart Rate", f"{df['heart_rate'].mean():.1f} bpm")
m3.metric("Avg Sleep Hours", f"{df['sleep_hours'].mean():.1f} hrs")

p1, p2 = st.columns(2)
with p1:
    dist = df['stress_label'].value_counts().reindex(list(LABELS.values()))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(dist, labels=dist.index, colors=[PEACH, "#FFB366", ORANGE, "#8B4513", DARKCHOC],
           wedgeprops=dict(width=0.5, edgecolor="white"), textprops=dict(fontsize=8, color=DARKCHOC))
    fig.patch.set_facecolor(CREAM); ax.set_title("Stress level mix", color=CHOC)
    st.pyplot(fig)
with p2:
    corr = df.drop(columns=['stress_label']).corr()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(corr, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax); ax.set_title("Signal correlations", color=CHOC); fig.patch.set_facecolor(CREAM)
    plt.tight_layout(); st.pyplot(fig)

# ---------- PRESENT ----------
with st.expander("⚙️ Model performance & feature importance", expanded=False):
    a, b = st.columns(2)
    a.metric("Logistic Regression Accuracy", f"{acc_lr*100:.1f}%")
    b.metric("Random Forest Accuracy", f"{acc_rf*100:.1f}%")
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(imp.index, imp.values, color=[ORANGE if i % 2 else CHOC for i in range(len(imp))], edgecolor=DARKCHOC)
    paint(fig, ax); plt.tight_layout(); st.pyplot(fig)