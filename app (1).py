import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Parkinson's Disease Detector",
    page_icon="🧠",
    layout="wide"
)

# Load model & scaler
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Feature names (exact order from dataset)
FEATURES = [
    ("MDVP:Fo(Hz)",       "Average vocal fundamental frequency",        100.0,  270.0,  154.2),
    ("MDVP:Fhi(Hz)",      "Maximum vocal fundamental frequency",        100.0,  600.0,  197.1),
    ("MDVP:Flo(Hz)",      "Minimum vocal fundamental frequency",         65.0,  240.0,  116.3),
    ("MDVP:Jitter(%)",    "Jitter - variation in frequency",              0.0,    0.03,   0.006),
    ("MDVP:Jitter(Abs)",  "Jitter (absolute)",                           0.0,    0.0003, 0.00004),
    ("MDVP:RAP",          "Relative amplitude perturbation",             0.0,    0.02,   0.003),
    ("MDVP:PPQ",          "Pitch period perturbation quotient",          0.0,    0.02,   0.003),
    ("Jitter:DDP",        "Average absolute difference of differences",  0.0,    0.07,   0.009),
    ("MDVP:Shimmer",      "Shimmer - variation in amplitude",            0.0,    0.12,   0.029),
    ("MDVP:Shimmer(dB)",  "Shimmer in dB",                               0.0,    1.3,    0.282),
    ("Shimmer:APQ3",      "3-point amplitude perturbation quotient",     0.0,    0.06,   0.015),
    ("Shimmer:APQ5",      "5-point amplitude perturbation quotient",     0.0,    0.08,   0.018),
    ("MDVP:APQ",          "Amplitude perturbation quotient",             0.0,    0.14,   0.024),
    ("Shimmer:DDA",       "Average absolute diff between amplitudes",    0.0,    0.17,   0.044),
    ("NHR",               "Noise-to-harmonics ratio",                    0.0,    0.31,   0.024),
    ("HNR",               "Harmonics-to-noise ratio",                    8.0,   34.0,   21.9),
    ("RPDE",              "Recurrence period density entropy",            0.25,   0.69,   0.498),
    ("DFA",               "Detrended fluctuation analysis",               0.57,   0.83,   0.718),
    ("spread1",           "Nonlinear measure of frequency variation",    -7.96,  -2.43,  -5.68),
    ("spread2",           "Nonlinear measure of frequency variation",     0.0,    0.45,   0.227),
    ("D2",                "Correlation dimension",                        1.42,   3.67,   2.38),
    ("PPE",               "Pitch period entropy",                         0.04,   0.53,   0.206),
]

# ── Header ──────────────────────────────────────────────────────────────
st.title("🧠 Parkinson's Disease Detection System")
st.markdown("**ML-based voice analysis using Radial Basis SVM** | Accuracy: 89.7%")
st.markdown("---")

# ── Sidebar: sample data ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Options")
    st.markdown("**Load sample data:**")
    
    if st.button("🟢 Load Healthy Sample"):
        st.session_state["sample"] = [
            119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370, 0.00554, 0.01109,
            0.04374, 0.426, 0.02182, 0.03130, 0.02971, 0.06545, 0.02211, 21.033,
            0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654
        ]

    if st.button("🔴 Load Parkinson's Sample"):
        st.session_state["sample"] = [
            197.076, 206.896, 192.055, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498,
            0.01098, 0.097, 0.00563, 0.00680, 0.00802, 0.01689, 0.00339, 26.775,
            0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569
        ]

    st.markdown("---")
    st.info("💡 Tip: Load a sample to auto-fill values, then click Predict.")

# ── Input form ────────────────────────────────────────────────────────────
sample = st.session_state.get("sample", None)

st.subheader("📋 Enter Voice Features")
st.caption("Adjust the sliders or type values directly. Default values are dataset averages.")

inputs = []
cols_per_row = 3

for i in range(0, len(FEATURES), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        idx = i + j
        if idx >= len(FEATURES):
            break
        name, desc, min_val, max_val, default = FEATURES[idx]
        val = sample[idx] if sample else default
        with col:
            entered = st.number_input(
                label=name,
                min_value=float(min_val - abs(min_val) * 0.5),
                max_value=float(max_val + abs(max_val) * 0.5),
                value=float(val),
                format="%.6f",
                help=desc,
                key=f"feat_{idx}"
            )
            inputs.append(entered)

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

if predict_clicked:
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("📊 Results")

    r1, r2 = st.columns(2)

    with r1:
        if prediction == 0:
            st.success("## ✅ Healthy")
            st.metric("Confidence", f"{probability[0]*100:.1f}%")
            st.markdown("The voice patterns suggest **no signs** of Parkinson's Disease.")
        else:
            st.error("## ⚠️ Parkinson's Detected")
            st.metric("Confidence", f"{probability[1]*100:.1f}%")

            # Severity estimation
            HNR  = inputs[15]
            NHR  = inputs[14]
            RPDE = inputs[16]

            if HNR > 25 and NHR < 0.02 and RPDE < 0.4:
                severity, color = "Mild 🟡", "🟡"
            elif HNR > 20 and NHR < 0.04 and RPDE < 0.5:
                severity, color = "Moderate 🟠", "🟠"
            else:
                severity, color = "Severe 🔴", "🔴"

            st.warning(f"🩺 Estimated Severity: **{severity}**")

    with r2:
        st.markdown("**Key Voice Indicators:**")
        st.markdown(f"- HNR (Harmonics-to-Noise): `{inputs[15]:.3f}`")
        st.markdown(f"- NHR (Noise-to-Harmonics): `{inputs[14]:.6f}`")
        st.markdown(f"- RPDE (Recurrence Density): `{inputs[16]:.4f}`")
        st.markdown(f"- PPE (Pitch Period Entropy): `{inputs[21]:.4f}`")

    st.caption("⚠️ This tool is for educational/research purposes only. Always consult a medical professional.")
