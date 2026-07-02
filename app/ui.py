from __future__ import annotations

import os
from typing import Any

import streamlit as st

from .components import render_feature_pill, render_metric, render_result_banner, render_status_chip
from .config import APP_SUBTITLE, APP_TITLE
from .theme import BACKGROUND, BORDER, MUTED, PRIMARY, SURFACE, TEXT


def load_css() -> None:
    with open(os.path.join(os.path.dirname(__file__), "styles.css"), "r", encoding="utf-8") as handle:
        st.markdown(f"<style>{handle.read()}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide")
    load_css()

    st.markdown(
        """
        <div class='hero-card'>
          <div style='display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;'>
            <div>
              <div style='font-size:0.92rem;opacity:0.86;font-weight:700;text-transform:uppercase;letter-spacing:0.16em;'>AI-powered verification suite</div>
              <div style='font-size:2rem;font-weight:800;margin-top:0.2rem;'>🛡️ Face Liveness Detection</div>
              <div style='font-size:1rem;opacity:0.92;margin-top:0.25rem;'>Professional anti-spoofing analysis with a premium experience.</div>
            </div>
            <div style='background:rgba(255,255,255,0.16);padding:0.75rem 1rem;border-radius:20px;border:1px solid rgba(255,255,255,0.2);'>
              <div style='font-size:0.8rem;opacity:0.86;'>System status</div>
              <div style='font-size:1.1rem;font-weight:700;'>Live & Secure</div>
            </div>
          </div>
          <div style='margin-top:1rem;'>
            <span class='feature-pill'>✓ Detect live faces</span>
            <span class='feature-pill'>✓ Detect printed photos</span>
            <span class='feature-pill'>✓ Detect screen replay</span>
            <span class='feature-pill'>✓ Real-time AI inference</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_main_panel() -> None:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### Choose your input")
    st.caption("Capture a fresh image or upload a reference sample for instant liveness analysis.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style='display:flex;align-items:center;gap:0.7rem;padding:0.75rem 0 1rem;'>
              <div style='width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,#2563eb,#1d4ed8);display:flex;align-items:center;justify-content:center;font-size:1.2rem;color:white;'>🛡️</div>
              <div>
                <div style='font-size:1rem;font-weight:800;color:#111827;'>Face Liveness</div>
                <div style='font-size:0.84rem;color:#64748b;'>Anti-spoofing platform</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Navigation")
        st.markdown("- Overview")
        st.markdown("- Verification")
        st.markdown("- Documentation")

        st.divider()
        st.markdown("### Model information")
        st.write("Existing ONNX anti-spoofing model preserved.")
        st.write("Inference logic and preprocessing remain unchanged.")

        st.divider()
        st.markdown("### About")
        st.write("Built for secure liveness checks across camera and uploaded images.")
        st.write("Version 1.0")

        st.divider()
        st.markdown("### Quick links")
        st.link_button("GitHub", "https://github.com")
        st.link_button("Docs", "https://streamlit.io")


def render_input_card() -> tuple[Any, Any]:
    left, right = st.columns([1.03, 0.97], gap="large")
    with left:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("#### 📸 Camera capture")
        st.caption("Use your device camera to capture a fresh verification image.")
        picture = st.camera_input("Capture a photo")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("#### 🖼️ Upload image")
        st.caption("Upload a JPG or PNG file for analysis.")
        uploaded = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)
    return picture, uploaded


def render_preview(image_rgb: Any, source: str, size: tuple[int, int] | None = None) -> None:
    st.markdown("<div class='preview-card'>", unsafe_allow_html=True)
    st.markdown("#### Preview")
    st.image(image_rgb, channels="BGR", use_container_width=True)
    meta_col_1, meta_col_2, meta_col_3 = st.columns(3)
    with meta_col_1:
        render_metric("Source", source.title())
    with meta_col_2:
        render_metric("Size", f"{size[0]} × {size[1]}" if size else "Ready")
    with meta_col_3:
        render_status_chip("Ready for analysis", "info")
    st.markdown("</div>", unsafe_allow_html=True)


def render_analysis_button() -> bool:
    return st.button("🔍 Analyze Face", use_container_width=True, type="primary")


def render_result_card(result: dict) -> None:
    render_result_banner(result)


def render_footer() -> None:
    st.markdown("---")
    footer_cols = st.columns([1, 1, 1])
    with footer_cols[0]:
        st.caption("Powered by ONNX Runtime")
    with footer_cols[1]:
        st.caption("Built with Python + Streamlit")
    with footer_cols[2]:
        st.caption("Made with ❤️ by Parth Savaliya")
