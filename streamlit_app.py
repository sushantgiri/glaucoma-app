import os
import gc
from io import BytesIO
from typing import Dict, Optional

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
import torch
import numpy as np

from backend.report_pdf import generate_report_pdf_like_template


# ---------------------------
# Cached singletons (FAST)
# ---------------------------
@st.cache_resource
def load_model_cached():
    from backend.inference_core import DenseNet121Predictor
    return DenseNet121Predictor()


@st.cache_resource
def load_gemini_cached():
    from backend.gemini_explainer import GeminiExplainer
    return GeminiExplainer()


@st.cache_resource
def load_segmenter_cached():
    from backend.inference_core import DiscCupSegmenter
    return DiscCupSegmenter()


def main():
    # -------------------------------------------------------------------
    # Session state flags (stable UI)
    # -------------------------------------------------------------------
    st.session_state.setdefault("is_processing", False)
    st.session_state.setdefault("run_requested", False)
    st.session_state.setdefault("gemini_error", None)
    st.session_state.setdefault("od_error", None)

    # persist results
    st.session_state.setdefault("has_result", False)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_cam_path", None)
    st.session_state.setdefault("last_uploaded_name", None)
    st.session_state.setdefault("last_gemini_report", None)
    st.session_state.setdefault("last_od_out", None)

    # UI defaults
    st.session_state.setdefault("gradcam_width", 520)
    st.session_state.setdefault("od_overlay_width", 520)
    st.session_state.setdefault("od_view_mode", "Disc/Cup overlay (contours + fill)")

    # navigation/animation state
    st.session_state.setdefault("_scroll_to_results", False)

    # strict tab pinning state
    st.session_state.setdefault("_active_tab_target", None)
    st.session_state.setdefault("_tab_click_token", None)

    # -------------------------------------------------------------------
    # Page config
    # -------------------------------------------------------------------
    st.set_page_config(
        page_title="Glaucoma Assessment – Research Prototype",
        page_icon="👁️",
        layout="wide",
    )

    # -------------------------------------------------------------------
    # Secrets -> env (safe)
    # -------------------------------------------------------------------
    try:
        if "GOOGLE_API_KEY" in st.secrets and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        if "HF_TOKEN" in st.secrets and not os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    except Exception:
        pass

    # -------------------------------------------------------------------
    # Enhanced Professional Styling
    # -------------------------------------------------------------------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-primary: #020617;
            --bg-secondary: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --primary: #14b8a6;
            --primary-glow: #2dd4bf;
            --primary-dark: #0d9488;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border: rgba(51, 65, 85, 0.5);
            --border-primary: rgba(20, 184, 166, 0.2);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --shadow-glow: 0 0 40px rgba(20, 184, 166, 0.15);
            --shadow-lg: 0 20px 50px rgba(0, 0, 0, 0.4);
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important; }

        .main, .stApp, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%) !important;
        }

        .block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }

        .main .stMarkdown,
        .main .stMarkdown p,
        .main .stMarkdown li,
        .main .stCaption,
        .main [data-testid="stText"] {
            color: var(--text-secondary) !important;
        }

        .glass-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid var(--border-primary);
            border-radius: 1rem;
            padding: 1.75rem;
            box-shadow: var(--shadow-lg), var(--shadow-glow);
            backdrop-filter: blur(12px);
            transition: var(--transition);
        }

        .glass-card:hover {
            border-color: rgba(20, 184, 166, 0.35);
            box-shadow: var(--shadow-lg), 0 0 60px rgba(20, 184, 166, 0.2);
        }

        .header-container {
            display: flex; align-items: flex-start; gap: 1.25rem;
            margin-bottom: 1.5rem; animation: slideUp 0.6s ease-out;
        }

        .header-icon {
            width: 56px; height: 56px; border-radius: 16px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.75rem; box-shadow: 0 8px 24px rgba(20, 184, 166, 0.3);
            flex-shrink: 0;
        }

        .badge-research {
            display: inline-flex; align-items: center;
            padding: 0.4rem 0.85rem; border-radius: 999px;
            font-size: 0.65rem !important; font-weight: 600 !important;
            letter-spacing: 0.12em; text-transform: uppercase;
            background: linear-gradient(135deg, rgba(20, 184, 166, 0.12) 0%, rgba(6, 182, 212, 0.06) 100%);
            border: 1px solid rgba(20, 184, 166, 0.35);
            color: var(--primary) !important; margin-bottom: 0.5rem;
        }

        .header-title {
            font-size: 2rem !important; font-weight: 700 !important;
            color: var(--text-primary) !important;
            margin: 0.25rem 0 0 0 !important;
            line-height: 1.2 !important; letter-spacing: -0.03em !important;
        }

        .header-subtitle {
            color: var(--text-secondary) !important;
            font-size: 1.05rem !important; max-width: 900px;
            line-height: 1.7 !important; margin-top: 1rem !important;
        }

        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, var(--border) 50%, transparent 100%);
            margin: 2rem 0;
        }

        .section-title {
            display: flex; align-items: center; gap: 0.6rem;
            font-size: 1.1rem !important; font-weight: 600 !important;
            color: var(--text-primary) !important; margin-bottom: 1rem !important;
        }

        .section-title-icon { color: var(--primary) !important; font-size: 1.25rem; }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem; margin-bottom: 1.5rem;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.4) 100%);
            border: 1px solid var(--border);
            border-radius: 0.875rem;
            padding: 1.25rem;
            text-align: center;
            transition: var(--transition);
        }

        .metric-card:hover { border-color: var(--border-primary); transform: translateY(-2px); }

        .metric-value {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            line-height: 1.2 !important;
            margin-bottom: 0.25rem !important;
        }

        .metric-value.success { color: var(--success) !important; }
        .metric-value.danger { color: var(--danger) !important; }

        .metric-label {
            font-size: 0.7rem !important;
            color: var(--text-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600 !important;
        }

        .risk-pill {
            display: inline-flex; align-items: center; gap: 0.5rem;
            padding: 0.6rem 1.25rem; border-radius: 999px;
            font-size: 0.8rem !important; font-weight: 700 !important;
            letter-spacing: 0.06em; text-transform: uppercase;
        }

        .risk-low {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(20, 184, 166, 0.08) 100%);
            border: 1px solid rgba(16, 185, 129, 0.45);
            color: #34d399 !important;
        }

        .risk-mod {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(234, 179, 8, 0.08) 100%);
            border: 1px solid rgba(245, 158, 11, 0.45);
            color: #fbbf24 !important;
        }

        .risk-high {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(248, 113, 113, 0.08) 100%);
            border: 1px solid rgba(239, 68, 68, 0.5);
            color: #f87171 !important;
        }

        .image-preview {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid var(--border-primary);
            border-radius: 1rem;
            overflow: hidden;
            transition: var(--transition);
        }

        .image-caption {
            text-align: center;
            color: var(--text-muted) !important;
            font-size: 0.875rem !important;
            padding: 0.875rem;
            border-top: 1px solid var(--border);
            display: flex; align-items: center; justify-content: center; gap: 0.5rem;
        }

        .empty-state {
            min-height: 320px;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            text-align: center; padding: 2rem;
        }

        .empty-icon { font-size: 4rem; opacity: 0.5; margin-bottom: 1rem; }
        .empty-title { font-size: 1.25rem !important; font-weight: 600 !important; color: var(--text-primary) !important; }
        .empty-text { color: var(--text-muted) !important; font-size: 0.95rem !important; max-width: 320px; }

        .stButton > button {
            background: linear-gradient(135deg, #0ea5a4 0%, #0d9488 100%) !important;
            font-weight: 700 !important; font-size: 0.95rem !important;
            border: none !important; padding: 0.875rem 2rem !important;
            border-radius: 0.875rem !important;
            box-shadow: 0 8px 24px rgba(20, 184, 166, 0.25) !important;
        }

        .stButton > button, .stButton > button * { color: #ffffff !important; }
        .stButton > button:hover { transform: translateY(-2px) !important; }

        [data-testid="stFileUploader"] {
            background: rgba(30, 41, 59, 0.5) !important;
            border: 2px dashed var(--border) !important;
            border-radius: 1rem !important;
            padding: 1.5rem !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(30, 41, 59, 0.5) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0.875rem !important;
            padding: 0.375rem !important;
            gap: 0.375rem !important;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: 0.625rem !important;
            padding: 0.75rem 1.25rem !important;
            font-weight: 700 !important;
            border: none !important;
        }

        .stTabs [aria-selected="true"] { background: var(--primary) !important; color: #041014 !important; }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

        /* Sliders */
        .stSlider label { color: var(--text-secondary) !important; }

        /* Helper anchor */
        #results-anchor { position: relative; top: -12px; }

        @keyframes slideUp { from { opacity: 0; transform: translateY(20px);} to { opacity: 1; transform: translateY(0);} }
        @keyframes fadeIn { from { opacity: 0;} to { opacity: 1;} }

        .animate-slide-up { animation: slideUp 0.5s ease-out; }
        .animate-fade-in { animation: fadeIn 0.4s ease-out; }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def content_bbox_from_original(pil_img: Image.Image, thr: int = 10, margin: int = 12):
        arr = np.asarray(pil_img.convert("RGB"))
        mask = (arr[..., 0] > thr) | (arr[..., 1] > thr) | (arr[..., 2] > thr)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            w, h = pil_img.size
            return (0, 0, w, h)

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(arr.shape[1], x1 + margin)
        y1 = min(arr.shape[0], y1 + margin)
        return (x0, y0, x1, y1)

    def crop_same_area_and_square(pil_img: Image.Image, bbox, out_size: int = 512) -> Image.Image:
        cropped = pil_img.convert("RGB").crop(bbox)
        resample = getattr(Image, "Resampling", Image).LANCZOS
        return ImageOps.pad(
            cropped,
            (out_size, out_size),
            method=resample,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )

    def risk_label(prob_glaucoma: float) -> str:
        if prob_glaucoma < 0.20:
            cls = "risk-low"
            icon = "✓"
            txt = "Low Suspicion"
        elif prob_glaucoma < 0.50:
            cls = "risk-mod"
            icon = "⚠"
            txt = "Moderate Suspicion"
        else:
            cls = "risk-high"
            icon = "⚠"
            txt = "High Suspicion"
        return f'<span class="risk-pill {cls}">{icon} {txt}</span>'

    # ---- strict tab pinning helpers ----
    def pin_tab(tab_text: str):
        st.session_state["_active_tab_target"] = tab_text

    def slider_width(
        label: str,
        key: str,
        tab_target: str,
        default: int = 520,
        min_w: int = 260,
        max_w: int = 1100,
        step: int = 20,
    ) -> int:
        def _on_change():
            pin_tab(tab_target)
            # IMPORTANT: do not scroll on widget changes
            st.session_state["_scroll_to_results"] = False

        return int(
            st.slider(
                label,
                min_value=min_w,
                max_value=max_w,
                value=int(st.session_state.get(key, default)),
                step=step,
                key=key,
                on_change=_on_change,
            )
        )

    # -------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------
    st.markdown(
        """
        <div class="header-container">
            <div class="header-icon">👁️</div>
            <div class="header-content">
                <span class="badge-research">IAPI-RL • Research Prototype</span>
                <h1 class="header-title">Glaucoma Assessment</h1>
            </div>
        </div>
        <p class="header-subtitle">
            AI-powered analysis of retinal fundus images for glaucoma detection research.
            This interface is intended <strong>solely for research and educational purposes</strong>.
        </p>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # Upload + preview layout
    # -------------------------------------------------------------------
    left_col, right_col = st.columns([1.05, 1.45], gap="large")

    with left_col:
        st.markdown(
            '<div class="section-title"><span class="section-title-icon">📁</span> Upload Fundus Image</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Select a macula-centred retinal photograph",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
            label_visibility="collapsed",
        )

        generate_gradcam = True
        generate_gemini = True
        generate_od_params = True

        current_name = getattr(uploaded_file, "name", None) if uploaded_file else None
        if current_name and st.session_state.last_uploaded_name and current_name != st.session_state.last_uploaded_name:
            # reset results
            st.session_state.has_result = False
            st.session_state.last_result = None
            st.session_state.last_cam_path = None
            st.session_state.last_gemini_report = None
            st.session_state.last_od_out = None
            st.session_state.gemini_error = None
            st.session_state.od_error = None

            # reset UI behavior
            st.session_state._scroll_to_results = False
            st.session_state["_active_tab_target"] = None
            st.session_state["_tab_click_token"] = None

            # reset widgets
            for k in ["od_view_mode", "gradcam_width", "od_overlay_width"]:
                if k in st.session_state:
                    del st.session_state[k]

        def request_run():
            if st.session_state.is_processing or uploaded_file is None:
                return

            st.session_state.is_processing = True
            st.session_state.run_requested = True
            st.session_state.gemini_error = None
            st.session_state.od_error = None

            st.session_state.last_uploaded_name = getattr(uploaded_file, "name", None)

            st.session_state.has_result = False
            st.session_state.last_result = None
            st.session_state.last_cam_path = None
            st.session_state.last_gemini_report = None
            st.session_state.last_od_out = None

            # scroll only after run (not on slider changes)
            st.session_state._scroll_to_results = True
            st.session_state["_active_tab_target"] = "📊 Clinical Summary"
            st.session_state["_tab_click_token"] = None

        st.button(
            "🔬 Run Assessment",
            type="primary",
            use_container_width=True,
            on_click=request_run,
            disabled=st.session_state.is_processing or (uploaded_file is None),
        )

        error_placeholder = st.empty()
        status_placeholder = st.empty()

        if uploaded_file is None:
            st.markdown(
                """
                <div class="info-box">
                    <span class="info-icon">ℹ️</span>
                    <div class="info-content">
                        <h4>Getting Started</h4>
                        <p>Upload a macula-centered retinal photograph in JPG, JPEG, or PNG format. For best results, use high-quality fundus images with clear visibility of the optic disc.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        image: Optional[Image.Image] = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            filename = getattr(uploaded_file, "name", "Uploaded image")

            st.markdown('<div class="image-preview animate-fade-in">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown(f'<p class="image-caption">📷 {filename}</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="glass-card empty-state">
                    <div class="empty-icon">👁️</div>
                    <h3 class="empty-title">No Image Selected</h3>
                    <p class="empty-text">Upload a retinal fundus photograph to begin the AI-powered glaucoma assessment analysis.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------
    BASE_DIR = os.path.dirname(__file__)
    GRADCAM_DIR = os.path.join(BASE_DIR, "gradcams")
    os.makedirs(GRADCAM_DIR, exist_ok=True)

    if image is not None and st.session_state.run_requested:
        error_placeholder.empty()

        progress_wrap = st.empty()
        progress_label = st.empty()
        with progress_wrap.container():
            progress = st.progress(0)

        try:
            progress_label.markdown("**🧠 Loading models…**")
            with status_placeholder.container():
                with st.spinner("🧠 Loading models (first time may take a bit)…"):
                    model = load_model_cached()
                    gemini = load_gemini_cached() if generate_gemini else None
                    segmenter = load_segmenter_cached() if generate_od_params else None
            progress.progress(25)

            progress_label.markdown("**🔄 Running DenseNet121 inference…**")
            with status_placeholder.container():
                with st.spinner("🔄 Analysing image with DenseNet121…"):
                    try:
                        result = model.predict(image)
                    except ValueError as e:
                        error_placeholder.error(str(e))
                        progress_wrap.empty()
                        progress_label.empty()
                        status_placeholder.empty()
                        st.session_state.is_processing = False
                        st.session_state.run_requested = False
                        st.stop()
            progress.progress(55)

            cam_path = None
            if generate_gradcam:
                progress_label.markdown("**🧠 Generating Grad-CAM…**")
                with status_placeholder.container():
                    with st.spinner("🧠 Generating DenseNet121 Grad-CAM…"):
                        try:
                            cam_path = model.gradcam(
                                image,
                                GRADCAM_DIR,
                                target_class_index=int(result["pred_class_index"]),
                            )
                        except Exception as e:
                            cam_path = None
                            error_placeholder.warning(f"Grad-CAM generation failed: {e}")
            progress.progress(75)

            od_out = None
            if generate_od_params and segmenter is not None:
                progress_label.markdown("**🧿 Extracting optic disc parameters…**")
                with status_placeholder.container():
                    with st.spinner("🧿 Segmenting disc/cup and computing parameters…"):
                        try:
                            od_out = segmenter.predict_masks_and_outputs(
                                image,
                                disc_thr=0.50,
                                cup_thr=0.20,
                                morph_open=True,
                            )
                        except Exception as e:
                            od_out = None
                            st.session_state.od_error = f"Optic disc parameters failed: {e}"
            progress.progress(88)

            gemini_report = None
            if generate_gemini and gemini is not None:
                progress_label.markdown("**💬 Writing narrative explanation…**")
                with status_placeholder.container():
                    with st.spinner("💬 Generating AI narrative explanation…"):
                        try:
                            gemini_report = gemini.explain(result, cam_path)
                        except Exception as e:
                            st.session_state.gemini_error = str(e)
                            error_placeholder.warning(f"Gemini explanation failed: {e}")

            progress.progress(100)
            st.success("✅ Assessment complete.")
            progress_wrap.empty()
            progress_label.empty()
            status_placeholder.empty()

            st.session_state.has_result = True
            st.session_state.last_result = result
            st.session_state.last_cam_path = cam_path
            st.session_state.last_gemini_report = gemini_report
            st.session_state.last_od_out = od_out

            # Keep scroll for this run only
            st.session_state._scroll_to_results = True

        finally:
            st.session_state.is_processing = False
            st.session_state.run_requested = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Render results (TABS)
    # -------------------------------------------------------------------
    if image is not None and st.session_state.has_result and st.session_state.last_result is not None:
        result = st.session_state.last_result
        cam_path = st.session_state.last_cam_path
        gemini_report = st.session_state.last_gemini_report
        od_out = st.session_state.last_od_out

        p0, p1 = result["probs"]

        st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)

        # Scroll ONLY after running assessment (never for sliders/selectboxes)
        if st.session_state.get("_scroll_to_results", False):
            components.html(
                """
                <script>
                  const doc = window.parent.document;
                  const el = doc.getElementById("results-anchor");
                  if (el) el.scrollIntoView({behavior: "smooth", block: "start"});
                </script>
                """,
                height=0,
                width=0,
            )
            st.session_state._scroll_to_results = False

        tab_summary, tab_disc, tab_gradcam, tab_report = st.tabs(
            ["📊 Clinical Summary", "🧿 Optic Disc Parameters", "🔍 Grad-CAM Visualization", "📄 Report & PDF"]
        )

        # ---- STRICT TAB PINNING (prevents “jump to another tab”) ----
        target = st.session_state.get("_active_tab_target")
        token = f"{target}|{st.session_state.get('od_view_mode')}|{st.session_state.get('od_overlay_width')}|{st.session_state.get('gradcam_width')}"

        if target and st.session_state.get("_tab_click_token") != token:
            safe_target = target.replace('"', '\\"')
            components.html(
                f"""
                <script>
                (function() {{
                    const doc = window.parent.document;
                    const tabs = Array.from(doc.querySelectorAll('button[role="tab"]'));
                    if (!tabs.length) return;

                    const selected = tabs.find(t => t.getAttribute("aria-selected") === "true");
                    const selectedTxt = (selected?.innerText || "").trim();
                    if (selectedTxt.includes("{safe_target}")) return;

                    const want = tabs.find(t => ((t.innerText || "").trim()).includes("{safe_target}"));
                    if (want) want.click();
                }})();
                </script>
                """,
                height=0,
                width=0,
            )
            st.session_state["_tab_click_token"] = token

        # -------------------- Summary --------------------
        with tab_summary:
            st.markdown('<div class="glass-card animate-slide-up">', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value success">{p0:.1%}</div>
                        <div class="metric-label">Normal</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value danger">{p1:.1%}</div>
                        <div class="metric-label">Glaucoma</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title">Overall Assessment</div>', unsafe_allow_html=True)
            st.markdown(risk_label(float(p1)), unsafe_allow_html=True)
            st.caption("Prediction source: DenseNet121")

            st.markdown(
                '<div class="section-title" style="margin-top: 1.5rem;">AI Narrative Explanation</div>',
                unsafe_allow_html=True,
            )
            if gemini_report:
                st.write(gemini_report)
            else:
                st.info("Narrative explanation unavailable.")
                if st.session_state.gemini_error:
                    st.caption(f"Technical note: `{st.session_state.gemini_error}`")

            st.markdown(
                """
                <div class="disclaimer-box">
                    <p><span class="disclaimer-icon">⚠️</span> This is a research prototype. It does not replace clinical examination, OCT, or visual field testing.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # -------------------- Optic Disc Parameters --------------------
        with tab_disc:
            st.markdown('<div class="glass-card animate-slide-up">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title"><span class="section-title-icon">🧿</span> Optic Disc & Cup Parameters</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <p style="margin-bottom: 1.5rem;">
                Parameters are computed from disc/cup segmentation masks.
                <strong>Cup volume and cup depth are not available from 2D fundus images</strong> (these require OCT or stereo imaging).
                </p>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.od_error:
                st.warning(st.session_state.od_error)

            if not od_out:
                st.info("Optic disc parameters are not available.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                metrics = od_out.get("metrics", {})
                ddls = od_out.get("ddls_proxy", {})

                def _od_view_changed():
                    pin_tab("🧿 Optic Disc Parameters")
                    st.session_state["_scroll_to_results"] = False

                view = st.selectbox(
                    "Visualization",
                    ["Disc/Cup overlay (contours + fill)", "Cup heatmap (probability)", "Disc heatmap (probability)"],
                    key="od_view_mode",
                    on_change=_od_view_changed,
                )

                od_w = slider_width(
                    "Display width",
                    key="od_overlay_width",
                    tab_target="🧿 Optic Disc Parameters",
                    default=520,
                )

                # Always render directly (NO base64 caching)
                img_ph = st.empty()
                if view == "Disc/Cup overlay (contours + fill)":
                    img_ph.image(od_out["overlay_pil"], caption="Disc/Cup overlay", width=od_w)
                elif view == "Cup heatmap (probability)":
                    img_ph.image(od_out["cup_heatmap_pil"], caption="Cup probability heatmap", width=od_w)
                else:
                    img_ph.image(od_out["disc_heatmap_pil"], caption="Disc probability heatmap", width=od_w)

                st.markdown("### Core metrics")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Disc area (px²)", f"{metrics.get('disc_area_px2', 0):.0f}")
                c2.metric("Cup area (px²)", f"{metrics.get('cup_area_px2', 0):.0f}")
                c3.metric("Rim area (px²)", f"{metrics.get('rim_area_px2', 0):.0f}")
                c4.metric("C/D area ratio", f"{metrics.get('cd_area_ratio', 0):.3f}")
                c5.metric("Vertical C/D", f"{metrics.get('vertical_cd_ratio', 0):.3f}")

                st.markdown("### Additional parameters")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Horizontal C/D", f"{metrics.get('horizontal_cd_ratio', 0):.3f}")
                a2.metric("Rim/Disc area", f"{metrics.get('rim_to_disc_area_ratio', 0):.3f}")
                a3.metric("Disc vertical (px)", f"{metrics.get('disc_vertical_diameter_px', 0):.0f}")
                a4.metric("Disc horizontal (px)", f"{metrics.get('disc_horizontal_diameter_px', 0):.0f}")

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Cup vertical (px)", f"{metrics.get('cup_vertical_diameter_px', 0):.0f}")
                b2.metric("Cup horizontal (px)", f"{metrics.get('cup_horizontal_diameter_px', 0):.0f}")
                b3.metric("Centroid offset (px)", f"{metrics.get('cup_disc_centroid_offset_px', 0):.1f}")
                b4.metric("DDLS proxy (1–10)", str(ddls.get("ddls_proxy_score_1_to_10", "-")))

                st.markdown("</div>", unsafe_allow_html=True)

        # -------------------- Grad-CAM --------------------
        with tab_gradcam:
            st.markdown('<div class="glass-card animate-slide-up">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title"><span class="section-title-icon">🔍</span> DenseNet121 Grad-CAM</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <p style="margin-bottom: 1.5rem;">
                Grad-CAM highlights regions that contributed most strongly to the DenseNet121 prediction.
                Warmer colors indicate higher contribution.
                </p>
                """,
                unsafe_allow_html=True,
            )

            gc_w = slider_width(
                "Display width",
                key="gradcam_width",
                tab_target="🔍 Grad-CAM Visualization",
                default=520,
            )

            img_ph = st.empty()
            if cam_path:
                # Always render directly (NO base64 caching)
                img_ph.image(cam_path, caption="DenseNet121 Grad-CAM", width=gc_w)
            else:
                st.warning("Grad-CAM was not available.")

            st.markdown(
                """
                <div class="info-box" style="margin-top: 1.5rem;">
                    <span class="info-icon">💡</span>
                    <div class="info-content">
                        <h4>How to interpret Grad-CAM</h4>
                        <p>The heatmap shows which parts of the image the neural network focused on when making its prediction. Red/yellow areas had the highest influence, while blue areas had minimal impact. For glaucoma detection, the model typically focuses on the optic disc region.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # -------------------- Report --------------------
        with tab_report:
            st.markdown('<div class="glass-card animate-slide-up">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><span class="section-title-icon">📄</span> Full Report</div>', unsafe_allow_html=True)

            if result["pred_label"].lower().startswith("glau"):
                classification = "Likely Glaucomatous"
                confidence = float(p1)
                final_assessment = "Likely Glaucomatous"
            else:
                classification = "Likely Normal"
                confidence = float(p0)
                final_assessment = "Likely Normal"

            image_id = getattr(uploaded_file, "name", "unknown").replace(" ", "_")

            explanation_text = gemini_report or (
                "Grad-CAM overlay highlights regions that influenced the model prediction. "
                "This report is generated by a research prototype and is not a clinical diagnosis."
            )

            disclaimer = (
                "Disclaimer: This report is generated with the assistance of artificial "
                "intelligence and should be considered a supplementary tool. It is not a "
                "substitute for professional medical diagnosis or advice. Clinical correlation "
                "and further ophthalmological evaluation are recommended."
            )

            od_block = ""
            if od_out and od_out.get("metrics"):
                m = od_out["metrics"]
                d = od_out.get("ddls_proxy", {})
                od_block = (
                    "\n\nOptic Disc Parameters (from segmentation):\n"
                    f"- Disc area (px²): {m.get('disc_area_px2', 0):.0f}\n"
                    f"- Cup area (px²): {m.get('cup_area_px2', 0):.0f}\n"
                    f"- Rim area (px²): {m.get('rim_area_px2', 0):.0f}\n"
                    f"- C/D area ratio: {m.get('cd_area_ratio', 0):.3f}\n"
                    f"- Vertical C/D ratio: {m.get('vertical_cd_ratio', 0):.3f}\n"
                    f"- Horizontal C/D ratio: {m.get('horizontal_cd_ratio', 0):.3f}\n"
                    f"- DDLS proxy (1–10): {d.get('ddls_proxy_score_1_to_10', '-')}\n"
                    "\nNote: Cup volume/depth require OCT; DDLS is a research proxy in this fundus-only prototype.\n"
                )

            report_preview = (
                f"Image ID: {image_id}\n"
                f"Classification (Model): {classification}\n"
                f"Confidence Score: {confidence:.3f}\n\n"
                f"Explanation:\n{explanation_text}\n\n"
                f"Final Assessment: {final_assessment}\n"
                f"{od_block}\n"
                f"{disclaimer}"
            )

            left_img_col, right_text_col = st.columns([1.25, 1.75], gap="large")

            with left_img_col:
                st.markdown("#### Preview")
                img1_col, img2_col = st.columns(2, gap="small")

                bbox = content_bbox_from_original(image, thr=10, margin=12)
                DISPLAY_SIZE = 768
                orig_norm = crop_same_area_and_square(image, bbox, out_size=DISPLAY_SIZE)

                with img1_col:
                    st.image(orig_norm, caption="Original fundus", use_container_width=True)

                with img2_col:
                    if cam_path:
                        cam_img = Image.open(cam_path).convert("RGB")
                        if cam_img.size != image.size:
                            resample = getattr(Image, "Resampling", Image).LANCZOS
                            cam_img = cam_img.resize(image.size, resample)
                        cam_norm = crop_same_area_and_square(cam_img, bbox, out_size=DISPLAY_SIZE)
                        st.image(cam_norm, caption="Grad-CAM overlay", use_container_width=True)
                    else:
                        st.info("Grad-CAM not available.")

            with right_text_col:
                st.markdown("#### Report Text")
                st.text_area("Report text (preview)", report_preview, height=380, label_visibility="collapsed")

            pdf_bytes = generate_report_pdf_like_template(
                image_id=image_id,
                classification=classification,
                confidence=confidence,
                explanation=explanation_text + (od_block if od_block else ""),
                final_assessment=final_assessment,
                disclaimer=disclaimer,
                original_pil=image,
                gradcam_path=cam_path,
            )

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{image_id}_glaucoma_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

    elif uploaded_file is None and not st.session_state.run_requested:
        pass


if __name__ == "__main__":
    main()
