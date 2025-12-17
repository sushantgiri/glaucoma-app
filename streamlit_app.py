import os
import gc
from io import BytesIO
from typing import Dict, Optional

import streamlit as st
from PIL import Image, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
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
    # Session state flags
    # -------------------------------------------------------------------
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    if "run_requested" not in st.session_state:
        st.session_state.run_requested = False

    if "gemini_error" not in st.session_state:
        st.session_state.gemini_error = None

    # persist results
    if "has_result" not in st.session_state:
        st.session_state.has_result = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_cam_path" not in st.session_state:
        st.session_state.last_cam_path = None
    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = None
    if "last_gemini_report" not in st.session_state:
        st.session_state.last_gemini_report = None

    # NEW: optic disc parameters persistence
    if "last_od_metrics" not in st.session_state:
        st.session_state.last_od_metrics = None
    if "last_od_overlay_path" not in st.session_state:
        st.session_state.last_od_overlay_path = None
    if "od_error" not in st.session_state:
        st.session_state.od_error = None

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
    # Styling
    # -------------------------------------------------------------------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        * { font-family: 'DM Sans', sans-serif; }

        .main { padding-top: 0.5rem; background: linear-gradient(180deg, #020617 0%, #0f172a 100%); }
        .stApp { background: linear-gradient(180deg, #020617 0%, #0f172a 100%); }

        .gl-card {
            padding: 1.5rem 2rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(2, 6, 23, 0.95) 100%);
            border: 1px solid rgba(20, 184, 166, 0.2);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 40px rgba(20, 184, 166, 0.05);
            backdrop-filter: blur(10px);
        }

        .gl-subtitle { color: #94a3b8; font-size: 1rem; max-width: 980px; line-height: 1.6; }
        .gl-section-title { margin-top: 0.5rem; margin-bottom: 0.5rem; color: #e2e8f0; font-weight: 600; }

        .gl-disclaimer {
            font-size: 0.85rem; color: #64748b; padding: 1rem;
            background: rgba(30, 41, 59, 0.5); border-radius: 0.5rem;
            border-left: 3px solid #0d9488; margin-top: 1rem;
        }

        .risk-pill {
            display: inline-flex; align-items: center; padding: 0.4rem 1rem;
            border-radius: 999px; font-size: 0.85rem; font-weight: 600;
            letter-spacing: 0.08em; text-transform: uppercase; gap: 0.5rem;
        }
        .risk-low {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(20, 184, 166, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399; box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
        }
        .risk-mod {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(234, 179, 8, 0.1) 100%);
            border: 1px solid rgba(245, 158, 11, 0.4);
            color: #fbbf24; box-shadow: 0 0 20px rgba(245, 158, 11, 0.15);
        }
        .risk-high {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(248, 113, 113, 0.1) 100%);
            border: 1px solid rgba(239, 68, 68, 0.5);
            color: #f87171; box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);
        }

        .gl-badge {
            display: inline-flex; align-items: center; padding: 0.35rem 0.75rem;
            border-radius: 999px; font-size: 0.7rem; letter-spacing: 0.15em;
            text-transform: uppercase;
            background: linear-gradient(135deg, rgba(20, 184, 166, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%);
            border: 1px solid rgba(20, 184, 166, 0.3);
            color: #14b8a6; font-weight: 500;
        }

        .gl-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
        .gl-icon {
            width: 48px; height: 48px; border-radius: 12px;
            background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 8px 24px rgba(20, 184, 166, 0.3);
        }

        .gl-metric-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
            border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 0.75rem; padding: 1rem; text-align: center;
        }
        .gl-metric-value { font-size: 1.75rem; font-weight: 700; color: #f1f5f9; }
        .gl-metric-label {
            font-size: 0.75rem; color: #64748b;
            text-transform: uppercase; letter-spacing: 0.1em;
        }

        .gl-image-preview {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(2, 6, 23, 0.95) 100%);
            border: 1px solid rgba(20, 184, 166, 0.2);
            border-radius: 1rem; padding: 1rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 40px rgba(20, 184, 166, 0.05);
            overflow: hidden;
            max-width: 100%;
            margin: 0;
        }
        .gl-image-caption {
            text-align: center; color: #64748b; font-size: 0.85rem;
            margin-top: 0.75rem; padding-top: 0.75rem;
            border-top: 1px solid rgba(51, 65, 85, 0.5);
        }

        [data-testid="stImage"] img {
            border-radius: 0.75rem;
            display: block;
            margin: 0 auto;
            object-fit: contain;
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
            border-radius: 0.75rem; padding: 0.25rem; gap: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] { background: transparent; color: #94a3b8; border-radius: 0.5rem; padding: 0.5rem 1rem; }
        .stTabs [aria-selected="true"] { background: rgba(20, 184, 166, 0.2) !important; color: #14b8a6 !important; }

        .stFileUploader {
            background: rgba(15, 23, 42, 0.5);
            border: 2px dashed rgba(20, 184, 166, 0.3);
            border-radius: 1rem; padding: 1rem;
        }

        h1, h2, h3, h4 { color: #f1f5f9 !important; }
        p, span, div { color: #cbd5e1; }

        .status-indicator { display: inline-flex; align-items: center; gap: 0.5rem; }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #14b8a6; animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Paths
    BASE_DIR = os.path.dirname(__file__)
    GRADCAM_DIR = os.path.join(BASE_DIR, "gradcams")
    os.makedirs(GRADCAM_DIR, exist_ok=True)

    OD_DIR = os.path.join(BASE_DIR, "optic_disc")
    os.makedirs(OD_DIR, exist_ok=True)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def build_text_report(result: Dict) -> str:
        p0, p1 = result["probs"]
        pred_label = result["pred_label"]
        lines = [
            "═══════════════════════════════════════════════════════════════",
            "         GLAUCOMA ASSESSMENT – RESEARCH REPORT",
            "═══════════════════════════════════════════════════════════════",
            "",
            "  Prediction Source:     DenseNet121",
            f"  Overall Assessment:    {pred_label.upper()}",
            f"  Normal Probability:    {p0:.3f}",
            f"  Glaucoma Probability:  {p1:.3f}",
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
        return "\n".join(lines)

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
            txt = "✓ Low Suspicion"
        elif prob_glaucoma < 0.50:
            cls = "risk-mod"
            txt = "⚠ Moderate Suspicion"
        else:
            cls = "risk-high"
            txt = "⚠ High Suspicion"
        return f'<span class="risk-pill {cls}">{txt}</span>'

    # -------------------------------------------------------------------
    # Header
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

        generate_gradcam = True
        generate_gemini = True
        generate_od_params = True

        current_name = getattr(uploaded_file, "name", None) if uploaded_file else None
        if current_name and st.session_state.last_uploaded_name and current_name != st.session_state.last_uploaded_name:
            st.session_state.has_result = False
            st.session_state.last_result = None
            st.session_state.last_cam_path = None
            st.session_state.last_gemini_report = None
            st.session_state.last_od_metrics = None
            st.session_state.last_od_overlay_path = None
            st.session_state.od_error = None

        def request_run():
            if st.session_state.is_processing:
                return
            if uploaded_file is None:
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
            st.session_state.last_od_metrics = None
            st.session_state.last_od_overlay_path = None

        st.button(
            "🔬 Run Assessment",
            type="primary",
            use_container_width=True,
            on_click=request_run,
            disabled=st.session_state.is_processing or (uploaded_file is None),
        )

        error_placeholder = st.empty()
        status_placeholder = st.empty()

    with right_col:
        image: Optional[Image.Image] = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            filename = getattr(uploaded_file, "name", "Uploaded image")

            st.markdown('<div class="gl-image-preview">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown(f'<p class="gl-image-caption">📷 {filename}</p>', unsafe_allow_html=True)
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
    # Run inference ONLY when Run button was clicked
    # -------------------------------------------------------------------
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

                    segmenter = None
                    if generate_od_params:
                        try:
                            segmenter = load_segmenter_cached()
                        except Exception as e:
                            segmenter = None
                            st.session_state.od_error = f"Optic disc segmentation unavailable: {e}"

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
                        except ValueError as e:
                            cam_path = None
                            error_placeholder.warning(str(e))
                        except Exception as e:
                            cam_path = None
                            error_placeholder.warning(f"Grad-CAM generation failed and was skipped: {e}")
            progress.progress(75)

            # NEW: optic disc/cup segmentation + metrics
            od_metrics = None
            od_overlay_path = None
            if generate_od_params and segmenter is not None:
                progress_label.markdown("**🧿 Extracting optic disc parameters…**")
                with status_placeholder.container():
                    with st.spinner("🧿 Segmenting disc/cup and computing parameters…"):
                        try:
                            od_out = segmenter.predict_masks_and_metrics(image)
                            od_metrics = od_out["metrics"]
                            overlay_pil = od_out["overlay_pil"]
                            od_overlay_path = os.path.join(OD_DIR, f"disc_cup_overlay_{os.getpid()}.png")
                            overlay_pil.save(od_overlay_path)
                        except Exception as e:
                            st.session_state.od_error = f"Optic disc parameters failed: {e}"
                            od_metrics = None
                            od_overlay_path = None

            progress.progress(88)

            # Gemini narrative
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
            progress_label.markdown("**✅ Completed**")
            st.success("Assessment complete.")
            progress_wrap.empty()
            progress_label.empty()
            status_placeholder.empty()

            # Persist outputs
            st.session_state.has_result = True
            st.session_state.last_result = result
            st.session_state.last_cam_path = cam_path
            st.session_state.last_gemini_report = gemini_report
            st.session_state.last_od_metrics = od_metrics
            st.session_state.last_od_overlay_path = od_overlay_path

        finally:
            st.session_state.is_processing = False
            st.session_state.run_requested = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Render results if we have them
    # -------------------------------------------------------------------
    if image is not None and st.session_state.has_result and st.session_state.last_result is not None:
        result = st.session_state.last_result
        cam_path = st.session_state.last_cam_path
        gemini_report = st.session_state.last_gemini_report
        od_metrics = st.session_state.last_od_metrics
        od_overlay_path = st.session_state.last_od_overlay_path

        p0, p1 = result["probs"]

        tab_summary, tab_od, tab_gradcam, tab_report = st.tabs(
            ["📊 Clinical Summary", "🧿 Optic Disc Parameters", "🔍 Grad-CAM Visualization", "📄 Report & PDF"]
        )

        # Summary tab
        with tab_summary:
            st.markdown('<div class="gl-card">', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="gl-metric-card">
                        <div class="gl-metric-value" style="color: #14b8a6;">{p0:.1%}</div>
                        <div class="gl-metric-label">Normal</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="gl-metric-card">
                        <div class="gl-metric-value" style="color: #f87171;">{p1:.1%}</div>
                        <div class="gl-metric-label">Glaucoma</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<h3 class="gl-section-title">Overall Assessment</h3>', unsafe_allow_html=True)
            st.markdown(risk_label(p1), unsafe_allow_html=True)
            st.caption("Prediction source: DenseNet121")

            st.markdown('<h3 class="gl-section-title">AI Narrative Explanation</h3>', unsafe_allow_html=True)
            if gemini_report:
                st.write(gemini_report)
            else:
                st.info("Narrative explanation unavailable.")
                if st.session_state.gemini_error:
                    st.caption(f"Technical note: Gemini could not be used because of: `{st.session_state.gemini_error}`")

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

        # Optic Disc Parameters tab
        with tab_od:
            st.write("Segmentation debug:", od_out.get("debug"))
            st.markdown('<div class="gl-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="gl-section-title">Optic Disc & Cup Segmentation</h3>', unsafe_allow_html=True)
            st.markdown(
                """
                <p class="gl-subtitle">
                Parameters are computed from disc/cup segmentation masks (not from Grad-CAM).
                Areas below are in <strong>pixel²</strong> unless you add device calibration for mm².
                </p>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.od_error:
                st.warning(st.session_state.od_error)

            if od_overlay_path and os.path.exists(od_overlay_path):
                st.image(od_overlay_path, caption="Disc/Cup overlay (contours + fill)", use_container_width=True)

            if od_metrics:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Disc area (px²)", f"{od_metrics['disc_area_px2']:.0f}")
                c2.metric("Cup area (px²)", f"{od_metrics['cup_area_px2']:.0f}")
                c3.metric("Rim area (px²)", f"{od_metrics['rim_area_px2']:.0f}")
                c4.metric("C/D area ratio", f"{od_metrics['cd_area_ratio']:.3f}")
                c5.metric("Vertical C/D", f"{od_metrics['vertical_cd_ratio']:.3f}")
            else:
                st.info("Optic disc parameters are not available.")

            st.markdown("</div>", unsafe_allow_html=True)

        # Grad-CAM tab
        with tab_gradcam:
            st.markdown('<div class="gl-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="gl-section-title">DenseNet121 Grad-CAM</h3>', unsafe_allow_html=True)
            st.markdown(
                """
                <p class="gl-subtitle">
                This Grad-CAM overlay highlights regions that contributed most strongly to the DenseNet121 prediction.
                Warmer colors indicate higher attention.
                </p>
                """,
                unsafe_allow_html=True,
            )

            if cam_path:
                view_w = st.slider(
                    "Grad-CAM display width",
                    min_value=500,
                    max_value=1400,
                    value=900,
                    step=50,
                    key="gradcam_width",
                )
                st.image(cam_path, caption="DenseNet121", width=view_w)
            else:
                st.warning("Grad-CAM was not available.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Report tab
        with tab_report:
            st.markdown('<div class="gl-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="gl-section-title">Full Report</h3>', unsafe_allow_html=True)

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
            if od_metrics:
                od_block = (
                    "\n\nOptic Disc Parameters (from segmentation):\n"
                    f"- Disc area (px²): {od_metrics['disc_area_px2']:.0f}\n"
                    f"- Cup area (px²): {od_metrics['cup_area_px2']:.0f}\n"
                    f"- Rim area (px²): {od_metrics['rim_area_px2']:.0f}\n"
                    f"- C/D area ratio: {od_metrics['cd_area_ratio']:.3f}\n"
                    f"- Vertical C/D ratio: {od_metrics['vertical_cd_ratio']:.3f}\n"
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

            left_img_col, right_text_col = st.columns([1.15, 1.85], gap="large")
            DISPLAY_SIZE = 512

            with left_img_col:
                st.markdown("#### Preview")
                img1_col, img2_col = st.columns([1, 1], gap="small")

                bbox = content_bbox_from_original(image, thr=10, margin=12)
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

                if od_overlay_path and os.path.exists(od_overlay_path):
                    st.markdown("#### Disc/Cup Overlay")
                    od_img = Image.open(od_overlay_path).convert("RGB")
                    od_norm = crop_same_area_and_square(od_img, bbox, out_size=DISPLAY_SIZE)
                    st.image(od_norm, caption="Optic disc parameters overlay", use_container_width=True)

            with right_text_col:
                st.markdown("#### Text")
                st.text_area("Report text (preview)", report_preview, height=360)

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
        st.info("👆 Upload a fundus photograph to begin the assessment.")


if __name__ == "__main__":
    main()
