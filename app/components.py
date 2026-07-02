from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from .theme import DANGER, PRIMARY, SUCCESS, WARNING


def render_feature_pill(text: str) -> None:
    st.markdown(f"<span class='feature-pill'>✓ {text}</span>", unsafe_allow_html=True)


def render_status_chip(text: str, tone: str = "info") -> None:
    colors = {"success": SUCCESS, "error": DANGER, "warning": WARNING, "info": PRIMARY}
    st.markdown(
        f"<span style='display:inline-flex;align-items:center;gap:0.4rem;background:{colors[tone]}14;color:{colors[tone]};padding:0.4rem 0.7rem;border-radius:999px;font-weight:700;font-size:0.85rem;'>● {text}</span>",
        unsafe_allow_html=True,
    )


def render_metric(title: str, value: str, suffix: str = "") -> None:
    st.markdown(
        f"""
        <div class='metric-tile'>
          <div style='font-size:0.83rem;color:#64748b;font-weight:600;'>{title}</div>
          <div style='font-size:1.15rem;color:#111827;font-weight:700;margin-top:0.2rem;'>{value}{suffix}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_banner(result: Dict[str, Any]) -> None:
    tone = result.get("tone", "error")
    title = result.get("label", "SPOOF ATTACK")
    confidence = result.get("confidence_percent", 0.0)
    processing_time = result.get("processing_time_ms", 0.0)
    accent = SUCCESS if tone == "success" else DANGER
    icon = "✅" if tone == "success" else "❌"

    st.markdown(
        f"""
        <div class='result-card {('result-live' if tone == 'success' else 'result-spoof')}'>
          <div style='display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;'>
            <div>
              <div style='font-size:0.9rem;font-weight:700;color:{accent};text-transform:uppercase;letter-spacing:0.12em;'>Prediction result</div>
              <div style='font-size:2rem;font-weight:800;color:#111827;margin-top:0.25rem;'>{icon} {title}</div>
              <div style='font-size:0.95rem;color:#475569;margin-top:0.3rem;'>Verified with the existing anti-spoofing model and inference pipeline.</div>
            </div>
            <div style='font-size:2.2rem;color:{accent};'>{icon}</div>
          </div>
          <div style='margin-top:1rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0.8rem;'>
            <div class='metric-tile'><div style='font-size:0.8rem;color:#64748b;'>Confidence</div><div style='font-size:1.1rem;font-weight:700;color:#111827;'>{confidence:.2f}%</div></div>
            <div class='metric-tile'><div style='font-size:0.8rem;color:#64748b;'>Processing time</div><div style='font-size:1.1rem;font-weight:700;color:#111827;'>{processing_time:.2f} ms</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
