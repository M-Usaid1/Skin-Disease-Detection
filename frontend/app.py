import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Skin Disease Detection using YOLOv11")
st.write("Upload a skin image to detect acne.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    if st.button("Predict"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files
            )

            data = response.json()

            with col2:

                st.subheader("Prediction")

                if data["detections"]:

                    for detection in data["detections"]:

                        st.success(
                            f"{detection['class']} ({detection['confidence']:.2f})"
                        )

                else:

                    st.warning("No acne detected.")

        except Exception as e:

            st.error(str(e))