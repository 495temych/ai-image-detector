import os
import io
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEMO_DIR = Path(__file__).parent / "demo_images"

st.set_page_config(page_title="AI Image Detector", layout="centered")
st.title("AI Image Detector")
st.caption("Upload an image to find out if it's real or AI-generated.")

tab_upload, tab_demo = st.tabs(["Upload", "Demo Images"])

with tab_upload:
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="upload"
    )
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)
        with st.spinner("Analysing…"):
            try:
                r = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=10,
                )
                r.raise_for_status()
                result = r.json()
                label = result["label"]
                confidence = result["confidence"]
                if label == "real":
                    st.success(f"**REAL** — {confidence:.1%} confidence")
                else:
                    st.error(f"**AI-GENERATED** — {confidence:.1%} confidence")
                st.progress(confidence)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API at " + API_URL + ". Is Docker running?")

with tab_demo:
    if DEMO_DIR.exists():
        demo_files = sorted(DEMO_DIR.glob("*.jpg")) + sorted(DEMO_DIR.glob("*.png"))
        if demo_files:
            selected = st.selectbox(
                "Choose a demo image",
                options=demo_files,
                format_func=lambda p: p.name,
            )
            if selected:
                image = Image.open(selected)
                st.image(image, use_container_width=True)
                if st.button("Analyse"):
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    with st.spinner("Analysing…"):
                        try:
                            r = requests.post(
                                f"{API_URL}/predict",
                                files={"file": (selected.name, buf.getvalue(), "image/jpeg")},
                                timeout=10,
                            )
                            r.raise_for_status()
                            result = r.json()
                            label = result["label"]
                            confidence = result["confidence"]
                            if label == "real":
                                st.success(f"**REAL** — {confidence:.1%} confidence")
                            else:
                                st.error(f"**AI-GENERATED** — {confidence:.1%} confidence")
                            st.progress(confidence)
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach API. Is Docker running?")
        else:
            st.info("No demo images found in app/demo_images/")
    else:
        st.info("app/demo_images/ directory not found.")
