import os
import gc
from io import BytesIO
from typing import Dict, Optional

import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import torch

# -------------------------------------------------------------------
# Session state flags
# -------------------------------------------------------------------
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "run_requested" not in st.session_state:
    st.session_state.run_requested = False

if "gemini_error" not in st.session_state:
    st.session_state.gemini_error = None

# -------------------------------------------------------------------
# Page config & global styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Glaucoma Assessment – Research Prototype",
    page_icon="👁️",
    layout="wide",
)

# 🔑 Load GOOGLE_API_KEY from Streamlit secrets if present
if "GOOGLE_API_KEY" in st.secrets and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Enhanced Clinical-themed dark UI (Design 2: Dark Medical Dashboard style)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'DM Sans', sans-serif;
    }
    
    .main {
        padding-top: 0.5rem;
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
    }
    
    .stApp {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
    }
    
    .gl-card {
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(2, 6, 23, 0.95) 100%);
        border: 1px solid rgba(20, 184, 166, 0.2);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 40px rgba(20, 184, 166, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .gl-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        max-width: 980px;
        line-height: 1.6;
    }
    
    .gl-section-title {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        color: #e2e8f0;
        font-weight: 600;
    }
    
    .gl-disclaimer {
        font-size: 0.85rem;
        color: #64748b;
        padding: 1rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 0.5rem;
        border-left: 3px solid #0d9488;
        margin-top: 1rem;
    }
    
    .risk-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        gap: 0.5rem;
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(20, 184, 166, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    
    .risk-mod {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(234, 179, 8, 0.1) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fbbf24;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.15);
    }
    
    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(248, 113, 113, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #f87171;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);
    }
    
    .gl-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%);
        border: 1px solid rgba(20, 184, 166, 0.3);
        color: #14b8a6;
        font-weight: 500;
    }
    
    .gl-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .gl-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 24px rgba(20, 184, 166, 0.3);
    }
    
    .gl-metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
    
    .gl-metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    
    .gl-metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .gl-image-preview {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(2, 6, 23, 0.95) 100%);
        border: 1px solid rgba(20, 184, 166, 0.2);
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 40px rgba(20, 184, 166, 0.05);
        overflow: hidden;
        max-width: 480px;
        margin: 0 auto;
    }
    
    .gl-image-caption {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(51, 65, 85, 0.5);
    }
    
    [data-testid="stImage"] img {
        border-radius: 0.75rem;
        display: block;
        margin: 0 auto;
        object-fit: contain;
        max-height: 420px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%) !important;
        color: #020617 !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 8px 24px rgba(20, 184, 166, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(20, 184, 166, 0.4) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 0.75rem;
        padding: 0.25rem;
        gap: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(20, 184, 166, 0.2) !important;
        color: #14b8a6 !important;
    }
    
    .stFileUploader {
        background: rgba(15, 23, 42, 0.5);
        border: 2px dashed rgba(20, 184, 166, 0.3);
        border-radius: 1rem;
        padding: 1rem;
    }
    
    .stExpander {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 0.75rem;
    }
    
    h1, h2, h3, h4 {
        color: #f1f5f9 !important;
    }
    
    p, span, div {
        color: #cbd5e1;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #14b8a6;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .model-agreement-bar {
        display: flex;
        gap: 4px;
        margin-top: 0.5rem;
    }
    
    .agreement-segment {
        flex: 1;
        height: 6px;
        border-radius: 3px;
        background: rgba(51, 65, 85, 0.5);
    }
    
    .agreement-segment.active {
        background: linear-gradient(90deg, #14b8a6, #06b6d4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Manual lazy loaders using session_state (for instant spinner)
# -------------------------------------------------------------------
def ensure_ensemble() -> object:
    """
    Create the GlaucomaEnsemble once and keep it in session_state.
    The heavy constructor is wrapped by the caller's spinner.
    """
    if "ensemble" not in st.session_state:
        from backend.inference_core import GlaucomaEnsemble
        st.session_state.ensemble = GlaucomaEnsemble()
    return st.session_state.ensemble


def ensure_gemini() -> Optional[object]:
    """
    Create GeminiExplainer once and keep it in session_state.
    """
    if "gemini" in st.session_state:
        return st.session_state.gemini

    try:
        from backend.gemini_explainer import GeminiExplainer
        explainer = GeminiExplainer()
        st.session_state.gemini = explainer
        st.session_state.gemini_error = None
        return explainer
    except Exception as e:
        st.session_state.gemini_error = str(e)
        st.session_state.gemini = None
        return None


# Paths
BASE_DIR = os.path.dirname(__file__)
GRADCAM_DIR = os.path.join(BASE_DIR, "gradcams")
os.makedirs(GRADCAM_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def build_text_report(ensemble_result: Dict, per_model: Dict) -> str:
    prob_normal, prob_glaucoma = ensemble_result["probs"]
    pred_label = ensemble_result["pred_label"]

    lines = [
        "═══════════════════════════════════════════════════════════════",
        "         GLAUCOMA ASSESSMENT – RESEARCH REPORT",
        "═══════════════════════════════════════════════════════════════",
        "",
        f"  Overall Assessment:    {pred_label.upper()}",
        f"  Normal Probability:    {prob_normal:.3f}",
        f"  Glaucoma Probability:  {prob_glaucoma:.3f}",
        "",
        "───────────────────────────────────────────────────────────────",
        "  MODEL AGREEMENT DETAILS",
        "───────────────────────────────────────────────────────────────",
    ]

    for name, info in per_model.items():
        p0, p1 = info["probs"]
        lines.append(
            f"  • {name}: {info['pred_label']} "
            f"(Normal: {p0:.3f}, Glaucoma: {p1:.3f})"
        )

    lines.extend(
        [
            "",
            "───────────────────────────────────────────────────────────────",
            "  DISCLAIMER",
            "───────────────────────────────────────────────────────────────",
            "  This output is generated by a research prototype and is",
            "  NOT a clinical diagnosis. Clinical judgement and",
            "  comprehensive ophthalmic examination remain essential.",
            "",
            "═══════════════════════════════════════════════════════════════",
        ]
    )

    return "\n".join(lines)


def build_pdf(report_text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    for line in report_text.split("\n"):
        c.setFont("Helvetica", 10)
        c.drawString(40, y, line)
        y -= 16
        if y < 50:
            c.showPage()
            y = height - 50

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


def risk_label(prob_glaucoma: float) -> str:
    if prob_glaucoma < 0.20:
        cls = "risk-low"
        txt = "✓ Low Suspicion"
    elif prob_glaucoma < 0.50:
        cls = "risk-mod"
        txt = "⚠ Moderate Suspicion"
    else:
        cls = "risk-high"
        txt = "⚠ High Suspicion"
    return f'<span class="risk-pill {cls}">{txt}</span>'


def agreement_bar(pred_label: str, per_model: Dict) -> str:
    total = len(per_model)
    agree = sum(1 for info in per_model.values() if info["pred_label"] == pred_label)

    segments = ""
    for i in range(total):
        active = "active" if i < agree else ""
        segments += f'<div class="agreement-segment {active}"></div>'

    return f"""
    <p style="color: #94a3b8; margin-bottom: 0.25rem;">{agree} of {total} models agree</p>
    <div class="model-agreement-bar">{segments}</div>
    """


# -------------------------------------------------------------------
# Main header
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="gl-header">
        <div class="gl-icon">👁️</div>
        <div>
            <span class="gl-badge">IAPI-RL • Research Prototype</span>
            <h1 style="margin: 0.25rem 0 0 0; font-size: 1.75rem;">Glaucoma Assessment</h1>
        </div>
        <div style="margin-left: auto;" class="status-indicator">
            <div class="status-dot"></div>
            <span style="color: #64748b; font-size: 0.8rem;">Active</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="gl-subtitle">
    AI-powered analysis of retinal fundus images for glaucoma detection research.
    This interface is intended <strong>solely for research and educational purposes</strong>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# -------------------------------------------------------------------
# Upload + preview layout
# -------------------------------------------------------------------
left_col, right_col = st.columns([1.05, 1.45])

with left_col:
    st.markdown("#### 📁 Upload Fundus Image")
    uploaded_file = st.file_uploader(
        "Select a macula-centred retinal photograph",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    # ⚙️ Advanced options (heavy steps)
    st.markdown("##### ⚙️ Advanced options")
    generate_gradcam = st.checkbox(
        "Generate Grad-CAM heatmaps (slower)", value=True
    )
    generate_gemini = st.checkbox(
        "Generate AI narrative explanation (slower)",
        value=True,
        key="generate_gemini",
    )

    def request_run():
        # Called on button click; must be light.
        if not st.session_state.is_processing:
            st.session_state.run_requested = True

    analyze_button = st.button(
        "🔬 Run Assessment",
        type="primary",
        use_container_width=True,
        on_click=request_run,
        disabled=st.session_state.is_processing,
    )

    status_placeholder = st.empty()

with right_col:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="gl-image-preview">', unsafe_allow_html=True)
        st.image(image, width=420)
        st.markdown(
            '<p class="gl-image-caption">📷 Uploaded fundus image</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="gl-image-preview" style="height: 320px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                <div style="text-align: center;">
                    <p style="font-size: 3.5rem; margin: 0; opacity: 0.6;">👁️</p>
                    <p style="color: #64748b; margin-top: 1rem; font-size: 0.95rem;">Upload an image to begin analysis</p>
                    <p style="color: #475569; font-size: 0.8rem; margin-top: 0.5rem;">Supported: JPG, JPEG, PNG</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------------
# Run models and render tabs
# -------------------------------------------------------------------
if uploaded_file is not None and st.session_state.run_requested:
    st.session_state.is_processing = True

    try:
        # 🔹 Step 0: ensure models are loaded (first time only)
        with status_placeholder.container():
            with st.spinner("🧠 Loading models (first time may take a bit)…"):
                ensemble = ensure_ensemble()
                gemini = ensure_gemini() if generate_gemini else None

        # 🔹 Step 1: prediction
        with status_placeholder.container():
            with st.spinner("🔄 Analysing image with ensemble…"):
                results = ensemble.predict(image)
                ensemble_result = results["ensemble"]
                per_model = results["per_model"]

        # 🔹 Step 2: Grad-CAM (optional) – now only on ResNet50, lighter
        cam_paths = {}
        if generate_gradcam:
            with status_placeholder.container():
                with st.spinner("🧠 Generating Grad-CAM heatmaps…"):
                    try:
                        cam_paths = ensemble.gradcam_for_cnn(image, GRADCAM_DIR)
                    except Exception as e:
                        cam_paths = {}
                        st.warning(
                            f"Grad-CAM generation failed and was skipped: {e}"
                        )

        # Small cleanup after heavy steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 🔹 Step 3: Gemini narrative (optional)
        text_report = build_text_report(ensemble_result, per_model)
        gemini_report = None
        if generate_gemini and gemini is not None:
            with status_placeholder.container():
                with st.spinner("💬 Generating AI narrative explanation…"):
                    try:
                        gemini_report = gemini.explain(
                            ensemble_result,
                            per_model,
                            cam_paths,
                            vit_rollout_path="",
                        )
                    except Exception as e:
                        st.session_state.gemini_error = str(e)
                        st.warning(f"Gemini explanation failed: {e}")

        status_placeholder.empty()

        prob_normal, prob_glaucoma = ensemble_result["probs"]
        pred_label = ensemble_result["pred_label"]

        tab_summary, tab_gradcam, tab_report = st.tabs(
            ["📊 Clinical Summary", "🔍 Grad-CAM Visualization", "📄 Report & PDF"]
        )

        # ----- Summary tab -----
        with tab_summary:
            st.markdown('<div class="gl-card">', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                    <div class="gl-metric-card">
                        <div class="gl-metric-value" style="color: #14b8a6;">{prob_normal:.1%}</div>
                        <div class="gl-metric-label">Normal</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="gl-metric-card">
                        <div class="gl-metric-value" style="color: #f87171;">{prob_glaucoma:.1%}</div>
                        <div class="gl-metric-label">Glaucoma</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                agree_count = sum(
                    1 for info in per_model.values()
                    if info["pred_label"] == pred_label
                )
                st.markdown(
                    f"""
                    <div class="gl-metric-card">
                        <div class="gl-metric-value">{agree_count}/{len(per_model)}</div>
                        <div class="gl-metric-label">Agreement</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<h3 class="gl-section-title">Overall Assessment</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(risk_label(prob_glaucoma), unsafe_allow_html=True)
            st.markdown(
                agreement_bar(pred_label, per_model), unsafe_allow_html=True
            )

            with st.expander("📈 View detailed model probabilities"):
                for name, info in per_model.items():
                    p0, p1 = info["probs"]
                    st.markdown(
                        f"**{name}** – {info['pred_label']} "
                        f"(Normal: `{p0:.3f}`, Glaucoma: `{p1:.3f}`)"
                    )

            # 🔹 AI Narrative section (always visible)
            st.markdown(
                '<h3 class="gl-section-title">AI Narrative Explanation</h3>',
                unsafe_allow_html=True,
            )

            if gemini_report:
                st.write(gemini_report)
            else:
                st.info(
                    "Narrative explanation unavailable or skipped. "
                    "Enable it in Advanced options to generate."
                )
                if generate_gemini and st.session_state.gemini_error:
                    st.caption(
                        "Technical note: Gemini could not be used because of: "
                        f"`{st.session_state.gemini_error}`"
                    )

            st.markdown(
                """
                <p class="gl-disclaimer">
                ⚠️ This summary reflects model behaviour on this image alone and does not replace
                clinical examination, visual field testing, or OCT imaging.
                </p>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # ----- Grad-CAM tab -----
        with tab_gradcam:
            st.markdown('<div class="gl-card">', unsafe_allow_html=True)
            st.markdown(
                '<h3 class="gl-section-title">Attention Heatmaps</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <p class="gl-subtitle">
                These Grad-CAM overlays highlight regions that contributed most strongly to the CNN prediction.
                Warmer colors indicate higher attention.
                </p>
                """,
                unsafe_allow_html=True,
            )

            if cam_paths:
                cols = st.columns(len(cam_paths))
                for i, (name, path) in enumerate(cam_paths.items()):
                    with cols[i]:
                        st.image(
                            path,
                            caption=f"{name}",
                            use_container_width=True,
                        )
            else:
                st.warning(
                    "Grad-CAM generation was disabled or failed. "
                    "Enable it in Advanced options before running the assessment."
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ----- Report tab -----
        with tab_report:
            st.markdown('<div class="gl-card">', unsafe_allow_html=True)
            st.markdown(
                '<h3 class="gl-section-title">Full Text Report</h3>',
                unsafe_allow_html=True,
            )
            st.code(text_report, language=None)

            st.markdown(
                '<h3 class="gl-section-title">Export Options</h3>',
                unsafe_allow_html=True,
            )
            pdf_bytes = build_pdf(text_report)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="glaucoma_assessment_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.markdown(
                """
                <p class="gl-disclaimer">
                When sharing this report, clearly label it as output from a research prototype,
                not a clinical diagnostic device.
                </p>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

    finally:
        # ✅ Always reset + clean up, even on error
        st.session_state.is_processing = False
        st.session_state.run_requested = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

elif uploaded_file is None and not st.session_state.run_requested:
    st.info("👆 Upload a fundus photograph to begin the assessment.")
