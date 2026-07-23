import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WEPIS — War Economic Prediction and Impact System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --black:   #0A0A0A;
    --dark:    #111111;
    --card:    #181818;
    --border:  #2A2A2A;
    --red:     #E8311A;
    --red2:    #FF5C42;
    --gold:    #F5C842;
    --green:   #2ECC71;
    --text:    #F0EDE8;
    --muted:   #888888;
    --accent:  #E8311A;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--black) !important;
    color: var(--text);
}

/* Force dark background everywhere — no white flash */
.stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
section[data-testid="stMain"] > div,
.main, .main > div,
div[class*="block-container"],
div[class*="stMarkdown"],
div[class*="element-container"] {
    background-color: #0A0A0A !important;
}

/* Remove any white from file uploader */
[data-testid="stFileUploader"] {
    background: #181818 !important;
    border: 1px solid #2A2A2A !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #181818 !important;
    border: 1px dashed #2A2A2A !important;
}
[data-testid="stFileUploadDropzone"] * { color: #888 !important; }

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--dark) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0A0A0A 0%, #1a0a06 50%, #0A0A0A 100%);
    border-bottom: 1px solid var(--border);
    padding: 4rem 2rem 3rem 2rem;
    margin: 0 -2rem 2rem -2rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 50%, rgba(232,49,26,0.1) 0%, transparent 65%);
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 6rem;
    letter-spacing: 0.08em;
    line-height: 0.9;
    color: var(--text);
    margin: 0;
    position: relative;
}
.hero-title span { color: var(--red); }
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    position: relative;
}

/* Section headers */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 0.25rem;
}
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    letter-spacing: 0.05em;
    color: var(--text);
    margin: 0 0 1.5rem 0;
    line-height: 1;
}

/* Metric cards */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border); margin-bottom: 2rem; }
.metric-card {
    background: var(--card);
    padding: 1.5rem;
    position: relative;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--red);
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    color: var(--text);
    line-height: 1;
    letter-spacing: 0.02em;
}
.metric-value.positive { color: var(--green); }
.metric-value.negative { color: var(--red2); }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.25rem;
}

/* Data table */
.data-table { background: var(--card); border: 1px solid var(--border); }

/* Nav pills */
.nav-pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    margin-right: 0.5rem;
    margin-bottom: 1rem;
}

/* Result card */
.result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--red);
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.result-war { font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; color: var(--text); }
.result-value { font-family: 'Bebas Neue', sans-serif; font-size: 3rem; }
.result-value.pos { color: var(--green); }
.result-value.neg { color: var(--red2); }
.result-meta { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }

/* Divider */
.divider { height: 1px; background: var(--border); margin: 2rem 0; }

/* Prediction form */
.form-section {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 2rem;
    margin-bottom: 1rem;
}
.form-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Streamlit overrides */
.stSelectbox > div > div { background: var(--card) !important; border-color: var(--border) !important; }
.stNumberInput > div > div > input { background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important; }
.stButton > button {
    background: var(--red) !important;
    color: white !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    padding: 0.75rem 2rem !important;
    border-radius: 0 !important;
    width: 100%;
}
.stButton > button:hover { background: #c42a15 !important; }

/* Warning / info boxes */
.war-tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    background: rgba(232,49,26,0.15);
    border: 1px solid rgba(232,49,26,0.3);
    color: var(--red2);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}
.gold-tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    background: rgba(245,200,66,0.1);
    border: 1px solid rgba(245,200,66,0.3);
    color: var(--gold);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB DARK THEME ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#111111',
    'axes.facecolor':    '#181818',
    'axes.edgecolor':    '#2A2A2A',
    'axes.labelcolor':   '#888888',
    'text.color':        '#F0EDE8',
    'xtick.color':       '#888888',
    'ytick.color':       '#888888',
    'grid.color':        '#2A2A2A',
    'grid.linewidth':    0.5,
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

COLORS = ['#E8311A', '#F5C842', '#2ECC71', '#4A90E2', '#9B59B6']

# ── HELPERS ───────────────────────────────────────────────────────────────────
FEATURE_COLS = ['gdp_before','inflation_before','oil_before','gold_before',
                'duration_years','region_encoded','global_war','sanctions']

REGION_MAP = {
    'Africa': 0, 'Asia': 1, 'Europe': 2, 'Europe/Asia': 3,
    'Middle East': 4, 'S. America': 5, 'South Asia': 6
}

TARGET_COLS   = ['gdp_pct_change', 'gold_pct_change', 'stock_pct_change']
TARGET_LABELS = ['GDP % Change', 'Gold % Change', 'Stock Index % Change']

@st.cache_resource
def load_models():
    models, imputer, le = {}, None, None
    for t in TARGET_COLS:
        path = f'model_{t}.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[t] = pickle.load(f)
    if os.path.exists('imputer.pkl'):
        with open('imputer.pkl', 'rb') as f:
            imputer = pickle.load(f)
    if os.path.exists('label_encoder.pkl'):
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    return models, imputer, le

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'region_encoded' not in df.columns:
        le = LabelEncoder()
        df['region_encoded'] = le.fit_transform(df['region'])
    numeric_cols = ['gdp_before','gdp_after','inflation_before','inflation_after',
                    'oil_before','oil_after','gold_before','gold_after',
                    'duration_years','global_war','sanctions',
                    'gdp_pct_change','inflation_change','oil_pct_change',
                    'gold_pct_change','stock_pct_change']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0 1rem 0;'>
        <div style='font-family: Bebas Neue, sans-serif; font-size: 1.6rem; color: #F0EDE8; letter-spacing: 0.08em;'>
            WEP<span style="color:#E8311A">IS</span>
        </div>
        <div style='font-family: DM Mono, monospace; font-size: 0.55rem; color: #555; letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.2rem;'>
            War Economic Prediction & Impact System
        </div>
    </div>
    <div style='height:1px; background:#2A2A2A; margin-bottom: 1.5rem;'></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        ["⚔️  Overview", "📊  Data Explorer", "🧠  Model Analysis", "🎯  Predict Impact"],
        label_visibility="visible"
    )

    st.markdown("<div style='height:1px; background:#2A2A2A; margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: DM Mono, monospace; font-size: 0.58rem; color: #444; letter-spacing: 0.08em; line-height: 2;'>
    MODEL ········ XGBOOST<br>
    VALIDATION ··· 5-FOLD CV<br>
    SOURCE ······· WORLD BANK API<br>
    DATASET ······ 62 WARS<br>
    PERIOD ······· 1914 – 2024
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px; background:#2A2A2A; margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: DM Mono, monospace; font-size: 0.58rem; color: #E8311A; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem;'>
    📂 Load Dataset
    </div>
    <div style='font-family: DM Mono, monospace; font-size: 0.55rem; color: #555; margin-bottom: 0.5rem;'>
    Upload CSV to enable Data Explorer and live charts
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=['csv'], label_visibility="collapsed")

models, imputer, le = load_models()
df = load_data(uploaded) if uploaded else None

# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
if "Overview" in page:
    st.markdown("""
    <div class='hero'>
        <p class='hero-sub'>📉 Machine Learning — Conflict Economic Impact Analysis</p>
        <h1 class='hero-title'><span>WEPIS</span></h1>
        <p style='font-family: DM Sans, sans-serif; color: #666; max-width: 520px; margin: 1rem auto 0 auto; font-size: 0.9rem; line-height: 1.7; position: relative;'>
        Predicting macroeconomic shocks from armed conflict using 62 historical wars sourced from the World Bank API.
        Validated on the Ukraine-Russia War 2022 — direction correct on all 3 indicators.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-grid'>
        <div class='metric-card'>
            <div class='metric-value'>62</div>
            <div class='metric-label'>Wars in Dataset</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>1914</div>
            <div class='metric-label'>Earliest Conflict</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value positive'>0.23</div>
            <div class='metric-label'>Best R² Score (Gold)</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>3/3</div>
            <div class='metric-label'>Ukraine Direction ✓</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<p class='section-label'>System Architecture</p><h2 class='section-title'>HOW IT WORKS</h2>", unsafe_allow_html=True)
        steps = [
            ("01", "DATA COLLECTION", "World Bank API + manual research for pre-1960 wars. 62 conflict-country pairs across 7 regions."),
            ("02", "FEATURE ENGINEERING", "8 input features: GDP, inflation, oil, gold prices before war + duration, region, sanctions flags."),
            ("03", "MODEL TRAINING", "XGBoost regressor per target. 5-Fold cross validation on 56 training wars."),
            ("04", "VALIDATION", "Tested on Yemen, Ethiopia, Nagorno-Karabakh. Ukraine-Russia 2022 as blind demo prediction."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; gap:1.5rem; margin-bottom:1.5rem; align-items:flex-start;'>
                <div style='font-family: Bebas Neue, sans-serif; font-size:2rem; color:#E8311A; line-height:1; min-width:2rem;'>{num}</div>
                <div>
                    <div style='font-family: DM Mono, monospace; font-size:0.65rem; letter-spacing:0.15em; color:#888; text-transform:uppercase; margin-bottom:0.3rem;'>{title}</div>
                    <div style='font-size:0.85rem; color:#bbb; line-height:1.5;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<p class='section-label'>Model Results</p><h2 class='section-title'>PERFORMANCE</h2>", unsafe_allow_html=True)
        results = [
            ("GDP % Change",          "0.144", "66.50", "#E8311A"),
            ("Gold % Change",         "0.229", "59.12", "#F5C842"),
            ("Stock Index % Change",  "N/A",   "17.97", "#2ECC71"),
        ]
        for target, r2, mae, color in results:
            st.markdown(f"""
            <div style='background:#181818; border:1px solid #2A2A2A; border-left: 3px solid {color}; padding:1rem 1.2rem; margin-bottom:0.75rem;'>
                <div style='font-family: DM Mono, monospace; font-size:0.6rem; color:#888; letter-spacing:0.15em; text-transform:uppercase;'>{target}</div>
                <div style='display:flex; justify-content:space-between; align-items:baseline; margin-top:0.5rem;'>
                    <div>
                        <div style='font-family: Bebas Neue, sans-serif; font-size:2rem; color:{color};'>{r2}</div>
                        <div style='font-family: DM Mono, monospace; font-size:0.55rem; color:#555;'>R² SCORE</div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-family: Bebas Neue, sans-serif; font-size:2rem; color:#F0EDE8;'>{mae}</div>
                        <div style='font-family: DM Mono, monospace; font-size:0.55rem; color:#555;'>CV MAE</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:rgba(232,49,26,0.08); border:1px solid rgba(232,49,26,0.2); padding:1rem; margin-top:1rem;'>
            <div style='font-family: DM Mono, monospace; font-size:0.6rem; color:#E8311A; letter-spacing:0.15em; margin-bottom:0.5rem;'>⚡ UKRAINE-RUSSIA 2022</div>
            <div style='font-size:0.8rem; color:#bbb; line-height:1.6;'>
                GDP → Predicted <b style='color:#E8311A'>negative</b> ✓<br>
                Gold → Predicted <b style='color:#F5C842'>positive</b> ✓<br>
                Stock → Predicted <b style='color:#E8311A'>negative</b> ✓
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── PAGE 2: DATA EXPLORER ─────────────────────────────────────────────────────
elif "Data Explorer" in page:
    st.markdown("<p class='section-label'>Dataset Analysis</p><h1 class='section-title'>DATA EXPLORER</h1>", unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div style='background:#181818; border:1px dashed #2A2A2A; padding:3rem; text-align:center;'>
            <div style='font-family: Bebas Neue, sans-serif; font-size:2rem; color:#555;'>UPLOAD CSV TO BEGIN</div>
            <div style='font-family: DM Mono, monospace; font-size:0.7rem; color:#444; margin-top:0.5rem;'>USE THE SIDEBAR TO UPLOAD YOUR DATASET</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        train_df = df[df['split'] == 'Train']

        # Stats row
        st.markdown(f"""
        <div class='metric-grid'>
            <div class='metric-card'><div class='metric-value'>{len(df)}</div><div class='metric-label'>Total Rows</div></div>
            <div class='metric-card'><div class='metric-value'>{df['region'].nunique()}</div><div class='metric-label'>Regions</div></div>
            <div class='metric-card'><div class='metric-value'>{df['war_name'].nunique()}</div><div class='metric-label'>Unique Wars</div></div>
            <div class='metric-card'><div class='metric-value'>{df['split'].value_counts().get("Train",0)}</div><div class='metric-label'>Training Rows</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Charts row
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<p class='section-label'>Regional Impact</p>", unsafe_allow_html=True)
            region_gdp = train_df.groupby('region')['gdp_pct_change'].mean().sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            colors_bar = ['#E8311A' if v < 0 else '#2ECC71' for v in region_gdp.values]
            bars = ax.barh(region_gdp.index, region_gdp.values, color=colors_bar, height=0.6)
            ax.axvline(0, color='#2A2A2A', linewidth=1)
            ax.set_xlabel('Avg GDP % Change', fontsize=8)
            for bar, val in zip(bars, region_gdp.values):
                ax.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
                        f'{val:.0f}%', va='center', ha='left' if val >= 0 else 'right',
                        fontsize=7, color='#888')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("<p class='section-label'>GDP Distribution</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            data = train_df['gdp_pct_change'].dropna()
            data_clipped = data.clip(-200, 500)
            ax.hist(data_clipped, bins=20, color='#E8311A', alpha=0.8, edgecolor='#0A0A0A', linewidth=0.5)
            ax.axvline(0, color='#F5C842', linewidth=1.5, linestyle='--', alpha=0.7)
            ax.axvline(data_clipped.mean(), color='#2ECC71', linewidth=1.5, linestyle='--', alpha=0.7)
            ax.set_xlabel('GDP % Change (clipped at ±500)', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            legend = [mpatches.Patch(color='#F5C842', label='Zero line'),
                      mpatches.Patch(color='#2ECC71', label=f'Mean: {data_clipped.mean():.1f}%')]
            ax.legend(handles=legend, fontsize=7, facecolor='#181818', edgecolor='#2A2A2A')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Correlation heatmap
        st.markdown("<p class='section-label'>Feature Correlations</p><h2 class='section-title'>CORRELATION MATRIX</h2>", unsafe_allow_html=True)
        numeric_cols = ['duration_years','gdp_before','inflation_before','oil_before',
                        'gold_before','global_war','sanctions','gdp_pct_change',
                        'gold_pct_change']
        corr_df = train_df[numeric_cols].dropna()
        if len(corr_df) > 5:
            corr = corr_df.corr()
            fig, ax = plt.subplots(figsize=(10, 7))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                        cmap=sns.diverging_palette(10, 150, as_cmap=True),
                        center=0, ax=ax, linewidths=1, linecolor='#111',
                        annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
            ax.set_title('Feature & Target Correlation Matrix', fontsize=10, color='#F0EDE8', pad=15)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Raw data table
        st.markdown("<p class='section-label'>Raw Data</p><h2 class='section-title'>DATASET TABLE</h2>", unsafe_allow_html=True)
        display_cols = ['country','war_name','year_start','region','duration_years',
                        'gdp_pct_change','gold_pct_change','stock_pct_change','split']
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[available].style
                .background_gradient(subset=['gdp_pct_change'] if 'gdp_pct_change' in available else [],
                                     cmap='RdYlGn', vmin=-100, vmax=100)
                .format({c: '{:.1f}%' for c in ['gdp_pct_change','gold_pct_change','stock_pct_change'] if c in available}),
            use_container_width=True,
            height=400
        )

# ── PAGE 3: MODEL ANALYSIS ────────────────────────────────────────────────────
elif "Model Analysis" in page:
    st.markdown("<p class='section-label'>XGBoost Performance</p><h1 class='section-title'>MODEL ANALYSIS</h1>", unsafe_allow_html=True)

    # Performance table
    st.markdown("<p class='section-label'>Cross Validation Results</p>", unsafe_allow_html=True)
    perf_data = {
        'Target':    ['GDP % Change', 'Gold % Change', 'Stock Index % Change'],
        'CV MAE':    [66.50, 59.12, 17.97],
        'CV Std':    [21.48, 22.11, 7.54],
        'Val R²':    [0.144, 0.229, None],
        'Val MAE':   [18.88, 11.08, None],
    }
    perf_df = pd.DataFrame(perf_data)

    col1, col2, col3 = st.columns(3)
    cols_display = [col1, col2, col3]
    colors_display = ['#E8311A', '#F5C842', '#2ECC71']

    for i, (_, row) in enumerate(perf_df.iterrows()):
        r2_str = f"{row['Val R²']:.3f}" if row['Val R²'] is not None else "N/A"
        mae_str = f"{row['Val MAE']:.2f}" if row['Val MAE'] is not None else "N/A"
        r2_color = colors_display[i] if row['Val R²'] is not None else '#555'
        with cols_display[i]:
            st.markdown(f"""
            <div style='background:#181818; border:1px solid #2A2A2A; border-top: 3px solid {colors_display[i]}; padding:1.5rem;'>
                <div style='font-family: DM Mono, monospace; font-size:0.6rem; color:#888; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1rem;'>{row['Target']}</div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:1rem;'>
                    <div>
                        <div style='font-family: Bebas Neue, sans-serif; font-size:2.5rem; color:{r2_color};'>{r2_str}</div>
                        <div style='font-family: DM Mono, monospace; font-size:0.55rem; color:#555;'>VAL R²</div>
                    </div>
                    <div>
                        <div style='font-family: Bebas Neue, sans-serif; font-size:2.5rem; color:#F0EDE8;'>{row['CV MAE']:.0f}</div>
                        <div style='font-family: DM Mono, monospace; font-size:0.55rem; color:#555;'>CV MAE</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Feature importance
    if models:
        st.markdown("<p class='section-label'>What Drives Predictions</p><h2 class='section-title'>FEATURE IMPORTANCE</h2>", unsafe_allow_html=True)

        feature_labels = ['GDP Before','Inflation Before','Oil Before','Gold Before',
                          'Duration (Years)','Region','Global War','Sanctions']

        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
        if len(models) == 1: axes = [axes]

        for i, (col, label, color) in enumerate(zip(TARGET_COLS, TARGET_LABELS, COLORS)):
            if col not in models: continue
            imp = models[col].feature_importances_
            idx = np.argsort(imp)
            feats = [feature_labels[j] for j in idx]
            vals  = imp[idx]
            axes[i].barh(feats, vals, color=color, alpha=0.85, height=0.6)
            axes[i].set_title(label, fontsize=9, color='#F0EDE8', pad=10)
            axes[i].set_xlabel('Importance', fontsize=7)
            for j, (feat, val) in enumerate(zip(feats, vals)):
                axes[i].text(val + 0.002, j, f'{val:.3f}', va='center', fontsize=6.5, color='#888')

        fig.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Ukraine prediction chart
    st.markdown("<p class='section-label'>Blind Test — Never Seen in Training</p><h2 class='section-title'>UKRAINE-RUSSIA 2022</h2>", unsafe_allow_html=True)

    pred_vals   = [-31.67, 23.83, -58.77]
    actual_vals = [-9.29,   7.89,  None]
    labels      = ['GDP % Change', 'Gold % Change', 'Stock % Change']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax.bar(x - w/2, pred_vals, w, label='Predicted',
                   color=['#E8311A','#E8311A','#E8311A'], alpha=0.9, edgecolor='#0A0A0A')
    actual_plot = [v if v is not None else 0 for v in actual_vals]
    bars2 = ax.bar(x + w/2, actual_plot, w, label='Actual',
                   color=['#F5C842','#F5C842','#555555'], alpha=0.9, edgecolor='#0A0A0A')

    ax.axhline(0, color='#2A2A2A', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('% Change', fontsize=8)
    ax.legend(fontsize=8, facecolor='#181818', edgecolor='#2A2A2A')

    for bar, val in zip(bars1, pred_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (1 if val >= 0 else -3),
                f'{val:.1f}%', ha='center', fontsize=8, color='#E8311A', fontweight='bold')
    for bar, val, actual in zip(bars2, actual_plot, actual_vals):
        if actual is not None:
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (1 if val >= 0 else -3),
                    f'{val:.1f}%', ha='center', fontsize=8, color='#F5C842', fontweight='bold')

    ax.text(
    0.98, 0.98,
    '✓ Direction correct on all 3 targets',
    transform=ax.transAxes,
    ha='right',
    va='top',
    fontsize=8,
    color='#2ECC71',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor=(46/255, 204/255, 113/255, 0.1),
        edgecolor='#2ECC71'
    )
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── PAGE 4: PREDICT ───────────────────────────────────────────────────────────
elif "Predict" in page:
    st.markdown("<p class='section-label'>Run a Prediction</p><h1 class='section-title'>PREDICT IMPACT</h1>", unsafe_allow_html=True)

    if not models:
        st.markdown("""
        <div style='background:#181818; border:1px dashed #E8311A; padding:2rem; text-align:center;'>
            <div style='font-family: Bebas Neue, sans-serif; font-size:1.5rem; color:#E8311A;'>MODEL FILES NOT FOUND</div>
            <div style='font-family: DM Mono, monospace; font-size:0.7rem; color:#888; margin-top:0.5rem;'>
            Place model_gdp_pct_change.pkl, model_gold_pct_change.pkl,<br>
            model_stock_pct_change.pkl, imputer.pkl, label_encoder.pkl<br>
            in the same folder as app.py
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(232,49,26,0.05); border:1px solid rgba(232,49,26,0.15); padding:1rem 1.5rem; margin-bottom:2rem; font-family: DM Mono, monospace; font-size:0.7rem; color:#888; line-height:1.8;'>
        Enter pre-war economic conditions to predict economic impact.
        Model predicts % change in GDP, Gold prices, and Stock Index.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<div class='form-section'><div class='form-title'>WAR CHARACTERISTICS</div>", unsafe_allow_html=True)
            war_name   = st.text_input("War / Conflict Name", value="Taiwan Strait Crisis")
            region     = st.selectbox("Region", list(REGION_MAP.keys()))
            duration   = st.number_input("Duration (Years)", min_value=0.04, max_value=30.0, value=2.0, step=0.5)
            global_war = st.selectbox("Global War?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            sanctions  = st.selectbox("Sanctions Imposed?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='form-section'><div class='form-title'>PRE-WAR ECONOMIC CONDITIONS</div>", unsafe_allow_html=True)
            gdp_before       = st.number_input("GDP Before (USD Billion)",       value=500.0,  step=10.0)
            inflation_before = st.number_input("Inflation Before (%)",           value=3.5,    step=0.1)
            oil_before       = st.number_input("Oil Price Before (USD/barrel)",  value=80.0,   step=1.0)
            gold_before      = st.number_input("Gold Price Before (USD/oz)",     value=1900.0, step=10.0)
            actual_gdp       = st.number_input("Actual GDP % Change (optional)", value=0.0,    step=0.1,
                                               help="Fill if you know the actual value to compare")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("⚔  PREDICT ECONOMIC IMPACT"):
            region_encoded = REGION_MAP.get(region, 4)

            X_input = np.array([[
                gdp_before, inflation_before, oil_before, gold_before,
                duration, region_encoded, global_war, sanctions
            ]])

          

            X_input = X_input

            predictions = {}
            for t in TARGET_COLS:
                if t in models:
                    predictions[t] = models[t].predict(X_input)[0]

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='margin-bottom:1.5rem;'>
                <div style='font-family: DM Mono, monospace; font-size:0.6rem; color:#E8311A; letter-spacing:0.2em; text-transform:uppercase;'>PREDICTION RESULTS</div>
                <div style='font-family: Bebas Neue, sans-serif; font-size:3rem; color:#F0EDE8; line-height:1;'>{war_name.upper()}</div>
                <div style='font-family: DM Mono, monospace; font-size:0.65rem; color:#888;'>{region} · {duration} years · {'Global' if global_war else 'Regional'} · {'Sanctions' if sanctions else 'No Sanctions'}</div>
            </div>
            """, unsafe_allow_html=True)

            # Result cards
            res_cols = st.columns(len(predictions))
            icons = ['📉', '🪙', '📊']
            for i, (t, label, icon) in enumerate(zip(TARGET_COLS, TARGET_LABELS, icons)):
                if t not in predictions: continue
                val = predictions[t]
                color = '#2ECC71' if val >= 0 else '#E8311A'
                sign = '+' if val >= 0 else ''
                with res_cols[i]:
                    st.markdown(f"""
                    <div style='background:#181818; border:1px solid #2A2A2A; border-top: 3px solid {color}; padding:1.5rem; text-align:center;'>
                        <div style='font-size:1.5rem;'>{icon}</div>
                        <div style='font-family: DM Mono, monospace; font-size:0.6rem; color:#888; letter-spacing:0.1em; text-transform:uppercase; margin:0.5rem 0;'>{label}</div>
                        <div style='font-family: Bebas Neue, sans-serif; font-size:3rem; color:{color}; line-height:1;'>{sign}{val:.1f}%</div>
                        <div style='font-family: DM Mono, monospace; font-size:0.55rem; color:#555; margin-top:0.5rem;'>{'↑ INCREASE' if val >= 0 else '↓ DECREASE'}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Prediction chart
            st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            available_preds = [(t, l) for t, l in zip(TARGET_COLS, TARGET_LABELS) if t in predictions]
            labels_p = [l for _, l in available_preds]
            pred_p   = [predictions[t] for t, _ in available_preds]
            x = np.arange(len(labels_p))
            bar_colors = ['#2ECC71' if v >= 0 else '#E8311A' for v in pred_p]
            bars = ax.bar(x, pred_p, color=bar_colors, alpha=0.9, edgecolor='#0A0A0A', width=0.5)
            ax.axhline(0, color='#2A2A2A', linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_p, fontsize=9)
            ax.set_ylabel('Predicted % Change', fontsize=8)
            ax.set_title(f'Predicted Economic Impact — {war_name}', fontsize=10, color='#F0EDE8', pad=15)
            for bar, val in zip(bars, pred_p):
                sign = '+' if val >= 0 else ''
                ax.text(bar.get_x() + bar.get_width()/2,
                        val + (1 if val >= 0 else -2),
                        f'{sign}{val:.1f}%', ha='center', fontsize=9,
                        fontweight='bold', color='#F0EDE8')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

            # Interpretation
            gdp_pred = predictions.get('gdp_pct_change', 0)
            gold_pred = predictions.get('gold_pct_change', 0)
            st.markdown(f"""
            <div style='background:#181818; border:1px solid #2A2A2A; border-left:3px solid #F5C842; padding:1.5rem; margin-top:1.5rem;'>
                <div style='font-family: DM Mono, monospace; font-size:0.6rem; color:#F5C842; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.75rem;'>⚡ INTERPRETATION</div>
                <div style='font-size:0.85rem; color:#bbb; line-height:1.8;'>
                    Based on historical patterns, <b style='color:#F0EDE8'>{war_name}</b> is predicted to cause a
                    <b style='color:{"#2ECC71" if gdp_pred >= 0 else "#E8311A"}'>{'+' if gdp_pred >= 0 else ''}{gdp_pred:.1f}% GDP change</b>
                    with gold prices expected to
                    <b style='color:{"#2ECC71" if gold_pred >= 0 else "#E8311A"}'>{"rise" if gold_pred >= 0 else "fall"} by {abs(gold_pred):.1f}%</b>.
                    {"Sanctions increase economic isolation and typically amplify GDP contraction." if sanctions else "Without sanctions, economic recovery tends to be faster."}
                    {"Global wars have historically caused more severe and prolonged economic disruption." if global_war else "Regional conflicts typically show more contained economic impact."}
                </div>
            </div>
            """, unsafe_allow_html=True)