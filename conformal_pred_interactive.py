import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
from scipy.stats import t
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interactive Conformal Prediction",
    page_icon="",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, .metric-label {
    font-family: 'Space Mono', monospace;
}

.stApp {
    background: #f7f7f5;
    color: #1a1a1a;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e0dedd;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #666360 !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-box {
    background: #ffffff;
    border: 1px solid #e0dedd;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-box .value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1;
}

.metric-box .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #999795;
    margin-top: 0.4rem;
}

.coverage-good .value  { color: #1a7a4a; }
.coverage-warn .value  { color: #a06010; }
.coverage-bad .value   { color: #b03030; }

/* Remove Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Tighten sidebar padding */
section[data-testid="stSidebar"] > div:first-child { padding: 2rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Data & model definitions ──────────────────────────────────────────────────
noise_distributions = {
    "Gaussian":             lambda n: np.random.normal(0, 1, n),
    "Heavy-tailed (t3)":    lambda n: t(df=3).rvs(n),
    "Skewed (Exponential)": lambda n: np.random.exponential(1, n) - 1,
    "Poisson":              lambda n: np.random.poisson(5, n),
    "Uniform":              lambda n: np.random.uniform(-1, 0, n),
}

model_names = [
    "Linear Regression",
    "Ridge Regression",
    "Random Forest",
    "Gradient Boosting",
    "Support Vector Regression",
    "KNN",
]

def get_model(name):
    return {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "Random Forest":    RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
        "upport Vector Regression":    SVR(),
        "KNN": KNeighborsRegressor(n_neighbors=20),
    }[name]

# ── Cached computation ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_and_fit(n, model_name, dist_name):
    np.random.seed(42)
    X = np.linspace(-2, 2, n).reshape(-1, 1)
    noise = noise_distributions[dist_name](n)
    y = 2 * X.reshape(-1) + noise

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
    X_cal, X_test, y_cal, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = get_model(model_name)
    model.fit(X_train, y_train)

    cal_pred  = model.predict(X_cal)
    scores    = np.abs(y_cal - cal_pred)
    test_pred = model.predict(X_test)

    X_grid    = np.linspace(-2, 2, 300).reshape(-1, 1)
    mean_grid = model.predict(X_grid)

    return dict(scores=scores, test_pred=test_pred, y_test=y_test,
                mean_grid=mean_grid, X_grid=X_grid, X_test=X_test)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("Controls")
    st.markdown("---")

    n = st.slider("Sample size  (n)", min_value=100, max_value=1000, value=1000, step=100)
    alpha = st.slider("Alpha  (α)", min_value=0.05, max_value=0.30, value=0.10, step=0.05,
                      format="%.2f")
    model_name = st.selectbox("Model", model_names, index=0)
    dist_name  = st.selectbox("Noise distribution", list(noise_distributions.keys()), index=0)

    st.markdown("---")
    st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#aaa8a5;line-height:1.7;'>
</div>
""", unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("<p style='font-size:30px;'>Interactive Conformal Prediction</p>")
st.markdown(
    "<p style='color:#999795;font-size:0.9rem;margin-top:-0.5rem;'>Split conformal intervals · marginal coverage guarantee</p>",
    unsafe_allow_html=True,
)

with st.spinner("Fitting model…"):
    data = generate_and_fit(n, model_name, dist_name)

scores    = data["scores"]
test_pred = data["test_pred"]
y_test    = data["y_test"]
mean_grid = data["mean_grid"]
X_grid    = data["X_grid"]
X_test    = data["X_test"]

# Conformal quantile
n_cal   = len(scores)
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q       = np.quantile(scores, min(q_level, 1.0))

lower   = test_pred - q
upper   = test_pred + q
covered = (y_test >= lower) & (y_test <= upper)
coverage = np.mean(covered) * 100
target   = (1 - alpha) * 100

# ── Metric row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

def metric_card(col, value, label, css_class=""):
    col.markdown(
        f'<div class="metric-box {css_class}">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

cov_class = ("coverage-good" if abs(coverage - target) < 3
             else "coverage-warn" if abs(coverage - target) < 8
             else "coverage-bad")

metric_card(col1, f"{coverage:.1f}%", "Empirical Coverage", cov_class)
metric_card(col2, f"{target:.0f}%",   "Target (1−α)")
metric_card(col3, f"{q:.3f}",         "Interval Half-width (q)")
metric_card(col4, f"{n_cal}",         "Calibration points")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  "#f7f7f5",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#cccac8",
    "axes.labelcolor":   "#444240",
    "xtick.color":       "#888683",
    "ytick.color":       "#888683",
    "text.color":        "#1a1a1a",
    "grid.color":        "#e8e6e3",
    "grid.linewidth":    0.7,
    "font.family":       "monospace",
})

fig, ax = plt.subplots(figsize=(11, 6))

# Conformal band
ax.fill_between(
    X_grid.ravel(),
    mean_grid - q,
    mean_grid + q,
    alpha=0.18,
    color="#2563eb",
    label="Conformal interval",
    zorder=1,
)

# Prediction line
ax.plot(X_grid, mean_grid, color="#2563eb", linewidth=1.8, label="Prediction", zorder=2)

# Scatter — covered
ax.scatter(
    X_test[covered], y_test[covered],
    s=18, alpha=0.65, color="#16a34a",
    linewidths=0, label=f"Covered ({covered.sum()})",
    zorder=3,
)

# Scatter — not covered
ax.scatter(
    X_test[~covered], y_test[~covered],
    s=45, marker="x", color="#dc2626",
    linewidths=1.4, label=f"Not covered ({(~covered).sum()})",
    zorder=4,
)

ax.set_xlim(-2.2, 2.2)
ax.set_xlabel("x", labelpad=8, fontsize=10)
ax.set_ylabel("y", labelpad=8, fontsize=10)
ax.set_title(
    f"{model_name}  ·  {dist_name}  ·  n={n}  ·  α={alpha}",
    fontsize=10, color="#888683", pad=12,
)
ax.grid(True, zorder=0)

leg = ax.legend(
    framealpha=0, labelcolor="#444240",
    fontsize=9, loc="upper left",
)

st.pyplot(fig, use_container_width=True)
plt.close(fig)
