from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from app.services import analyze_image
from app.ui import (
    render_analysis_button,
    render_footer,
    render_header,
    render_input_card,
    render_main_panel,
    render_preview,
    render_result_card,
    render_sidebar,
)
from app.utils import format_confidence, format_timestamp, read_image_from_camera, read_image_from_upload, validate_uploaded_image

st.set_option("client.showErrorDetails", False)


def main() -> None:
    render_header()
    render_sidebar()
    render_main_panel()

    picture, uploaded = render_input_card()

    if picture is not None:
        image_bytes = picture.getvalue()
        st.session_state["camera_image"] = image_bytes
        st.session_state["source"] = "camera"

    if uploaded is not None:
        is_valid, error = validate_uploaded_image(uploaded)
        if not is_valid:
            st.error(error)
        else:
            st.session_state["uploaded_image"] = uploaded
            st.session_state["source"] = "upload"

    selected_image = None
    if st.session_state.get("source") == "camera" and st.session_state.get("camera_image"):
        selected_image = read_image_from_camera(st.session_state["camera_image"])
    elif st.session_state.get("source") == "upload" and st.session_state.get("uploaded_image"):
        selected_image = read_image_from_upload(st.session_state["uploaded_image"])

    if selected_image is not None:
        image_rgb = cv2.cvtColor(np.array(selected_image), cv2.COLOR_RGB2BGR)
        image_height, image_width = image_rgb.shape[:2]
        render_preview(image_rgb, st.session_state.get("source", "upload"), (image_width, image_height))

        if render_analysis_button():
            with st.spinner("Analyzing facial features... Checking spoof patterns... Running AI inference..."):
                result = analyze_image(selected_image)

            st.session_state["last_result"] = result
            st.session_state["last_source"] = st.session_state.get("source", "upload")
            st.success("Analysis complete")

    if st.session_state.get("last_result"):
        result = st.session_state["last_result"]
        st.markdown("### Detection result")
        render_result_card(result)

        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.metric("Confidence", format_confidence(result.get("confidence", 0.0)))
        with meta_cols[1]:
            st.metric("Processing Time", f"{result.get('processing_time_ms', 0.0):.2f} ms")
        with meta_cols[2]:
            st.metric("Timestamp", format_timestamp())

        st.caption(f"Input source: {st.session_state.get('last_source', 'unknown').title()}")

    render_footer()


if __name__ == "__main__":
    main()
